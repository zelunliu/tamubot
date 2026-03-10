"""Langfuse observability and RAGAS evaluation for TamuBot pipeline.

Uses a minimal REST client (httpx) instead of the official Langfuse SDK to maintain
Python 3.14 compatibility (the SDK's Fern-generated layer depends on pydantic.v1 which
breaks on Python 3.14+).

Provides:
    get_langfuse()          — lazy singleton MinimalLangfuseClient
    compute_ragas_metrics() — Faithfulness + AnswerRelevancy via RAGAS, scores uploaded
    run_ragas_background()  — fire-and-forget wrapper for compute_ragas_metrics()
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("tamubot.observability")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


def _clean(d: dict) -> dict:
    """Remove None-valued keys so the API doesn't reject them."""
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Minimal Langfuse client  (no pydantic v1 dependency)
# ---------------------------------------------------------------------------

class LFGeneration:
    """Represents a Langfuse generation observation."""

    def __init__(self, client: "MinimalLangfuseClient", trace_id: str, gen_id: str):
        self._client = client
        self._trace_id = trace_id
        self.id = gen_id

    def end(
        self,
        output: Any = None,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> None:
        # Normalise usage to the format Langfuse REST API requires for display.
        # Without "unit": "TOKENS" the values are stored but never shown in the UI.
        if usage is not None:
            inp = usage.get("input") or 0
            out = usage.get("output") or 0
            usage = {
                "input": inp,
                "output": out,
                "total": inp + out,
                "unit": "TOKENS",
            }
        body: dict = {
            "id": self.id,
            "traceId": self._trace_id,
            "endTime": _now(),
            "output": output,
            "usage": usage,
            "metadata": metadata,
        }
        if level:
            body["level"] = level
        if status_message:
            body["statusMessage"] = status_message
        self._client._enqueue("generation-update", body)


class LFSpan:
    """Represents a Langfuse span observation. Can nest child spans."""

    def __init__(
        self,
        client: "MinimalLangfuseClient",
        trace_id: str,
        span_id: str,
        parent_id: Optional[str] = None,
    ):
        self._client = client
        self._trace_id = trace_id
        self.id = span_id
        self._parent_id = parent_id

    def span(
        self,
        name: str,
        input: Any = None,
        metadata: Optional[dict] = None,
    ) -> "LFSpan":
        """Create a child span nested under this span."""
        child_id = _uuid()
        self._client._enqueue("span-create", {
            "id": child_id,
            "traceId": self._trace_id,
            "parentObservationId": self.id,
            "name": name,
            "startTime": _now(),
            "input": input,
            "metadata": metadata,
        })
        return LFSpan(self._client, self._trace_id, child_id, parent_id=self.id)

    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        input: Any = None,
        metadata: Optional[dict] = None,
    ) -> "LFGeneration":
        """Create a child generation nested under this span."""
        child_id = _uuid()
        self._client._enqueue("generation-create", {
            "id": child_id,
            "traceId": self._trace_id,
            "parentObservationId": self.id,
            "name": name,
            "model": model,
            "startTime": _now(),
            "input": input,
            "metadata": metadata,
        })
        return LFGeneration(self._client, self._trace_id, child_id)

    def end(
        self,
        output: Any = None,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> None:
        body: dict = {
            "id": self.id,
            "traceId": self._trace_id,
            "endTime": _now(),
            "output": output,
            "metadata": metadata,
        }
        if level:
            body["level"] = level
        if status_message:
            body["statusMessage"] = status_message
        self._client._enqueue("span-update", body)

    def update(
        self,
        output: Any = None,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """Alias for end() — called mid-span to attach metadata."""
        self.end(output=output, usage=usage, metadata=metadata, level=level, status_message=status_message)


class LFTrace:
    """Represents a Langfuse trace (top-level container for a request)."""

    def __init__(self, client: "MinimalLangfuseClient", trace_id: str):
        self._client = client
        self.id = trace_id

    def span(
        self,
        name: str,
        input: Any = None,
        metadata: Optional[dict] = None,
    ) -> LFSpan:
        span_id = _uuid()
        self._client._enqueue("span-create", {
            "id": span_id,
            "traceId": self.id,
            "name": name,
            "startTime": _now(),
            "input": input,
            "metadata": metadata,
        })
        return LFSpan(self._client, self.id, span_id)

    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        input: Any = None,
        metadata: Optional[dict] = None,
    ) -> LFGeneration:
        gen_id = _uuid()
        self._client._enqueue("generation-create", {
            "id": gen_id,
            "traceId": self.id,
            "name": name,
            "model": model,
            "startTime": _now(),
            "input": input,
            "metadata": metadata,
        })
        return LFGeneration(self._client, self.id, gen_id)

    def update(self, output: Any = None, metadata: Optional[dict] = None) -> None:
        """Upsert the trace with final output."""
        self._client._enqueue("trace-create", {
            "id": self.id,
            "output": output,
            "metadata": metadata,
        })


class MinimalLangfuseClient:
    """Minimal Langfuse observability client using the REST ingestion API directly.

    Buffers events in memory and flushes them in a single batch HTTP request.
    Compatible with Python 3.14+ (no pydantic.v1 dependency).
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        self._host = host.rstrip("/")
        self._auth = (public_key, secret_key)
        self._events: list = []
        self._lock = threading.Lock()

    def trace(
        self,
        name: str,
        input: Any = None,
        metadata: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> LFTrace:
        trace_id = _uuid()
        body: dict = {
            "id": trace_id,
            "name": name,
            "input": input,
            "metadata": metadata,
            "timestamp": _now(),
        }
        if user_id:
            body["userId"] = user_id
        self._enqueue("trace-create", body)
        return LFTrace(self, trace_id)

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """Upload a RAGAS (or any) score directly to the scores endpoint."""
        import httpx
        payload: dict = {"traceId": trace_id, "name": name, "value": value}
        if comment:
            payload["comment"] = comment
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.post(
                    f"{self._host}/api/public/scores",
                    json=payload,
                    auth=self._auth,
                )
                resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Langfuse score upload failed ({name}={value}): {e}")

    def flush(self, timeout: float = 15.0) -> None:
        """Send all buffered span/generation events to Langfuse in one batch."""
        import httpx
        with self._lock:
            if not self._events:
                return
            batch = list(self._events)
            self._events.clear()

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{self._host}/api/public/ingestion",
                    json={"batch": batch},
                    auth=self._auth,
                )
                resp.raise_for_status()
            # Log any per-event errors from the 207 Multi-Status response
            try:
                body = resp.json()
                errors = [e for e in body.get("errors", []) if e]
                if errors:
                    logger.warning(f"Langfuse ingestion partial errors: {errors}")
            except Exception:
                pass
            logger.debug(f"Langfuse: flushed {len(batch)} events.")
        except Exception as e:
            logger.warning(f"Langfuse flush failed ({len(batch)} events): {e}")

    def _enqueue(self, event_type: str, body: dict) -> None:
        with self._lock:
            self._events.append({
                "id": _uuid(),
                "type": event_type,
                "timestamp": _now(),
                "body": _clean(body),
            })


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_langfuse_client: Optional[MinimalLangfuseClient] = None


def get_langfuse() -> Optional[MinimalLangfuseClient]:
    """Lazy singleton. Returns None if Langfuse credentials are not configured."""
    global _langfuse_client
    if _langfuse_client is None:
        import config
        if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
            return None
        try:
            _langfuse_client = MinimalLangfuseClient(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_BASE_URL,
            )
            logger.info("Langfuse REST client initialised.")
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
            return None
    return _langfuse_client


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def compute_ragas_metrics(
    question: str,
    contexts: list[str],
    answer: str,
    trace_id: Optional[str] = None,
) -> dict:
    """Compute RAGAS Faithfulness + AnswerRelevancy and upload scores to Langfuse.

    Uses gemini-2.0-flash as the critic LLM to minimise self-evaluation bias.

    Args:
        question:  The original user query.
        contexts:  Retrieved chunk texts that were passed to the generator.
        answer:    The final answer string produced by the generator.
        trace_id:  Langfuse trace ID to attach scores to. Optional.

    Returns:
        Dict of metric_name → float score, or {} on failure.
    """
    try:
        import voyageai
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness

        import config

        critic_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=config.TAMU_MODEL,
                api_key=config.TAMU_API_KEY,
                base_url=config.TAMU_BASE_URL,
                temperature=0,
            )
        )

        # Use Voyage AI for embeddings — already configured in the project,
        # avoids Google embedding API versioning issues on Python 3.14.
        class _VoyageEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
                return client.embed(texts, model="voyage-3", input_type="document").embeddings
            def embed_query(self, text: str) -> list[float]:
                client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
                return client.embed([text], model="voyage-3", input_type="query").embeddings[0]

        critic_embeddings = LangchainEmbeddingsWrapper(_VoyageEmbeddings())

        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
        )
        dataset = EvaluationDataset(samples=[sample])

        metrics = [
            Faithfulness(llm=critic_llm),
            AnswerRelevancy(llm=critic_llm, embeddings=critic_embeddings),
        ]
        result = evaluate(dataset=dataset, metrics=metrics)
        scores: dict = result.to_pandas().iloc[0].to_dict()

        lf = get_langfuse()
        if lf and trace_id:
            import math
            for metric_name, value in scores.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    lf.score(
                        trace_id=trace_id,
                        name=metric_name,
                        value=float(value),
                        comment="RAGAS automated evaluation",
                    )

        logger.info(f"RAGAS scores for trace {trace_id}: {scores}")
        return scores

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {}


def run_ragas_background(
    question: str,
    contexts: list[str],
    answer: str,
    trace_id: Optional[str] = None,
) -> None:
    """Fire-and-forget RAGAS evaluation in a background daemon thread."""
    thread = threading.Thread(
        target=compute_ragas_metrics,
        args=(question, contexts, answer, trace_id),
        daemon=True,
    )
    thread.start()


def score_groundedness(
    question: str,
    contexts: list[str],
    answer: str,
    trace_id: Optional[str] = None,
) -> Optional[float]:
    """Score response groundedness using RAGAS ResponseGroundedness metric.

    Gate 2 (LLM-as-Judge): Uses Gemini 2.5 Flash-Lite as the critic LLM to evaluate
    whether all claims in the response are grounded in the provided contexts.

    Args:
        question:  The original user query.
        contexts:  Retrieved chunk texts passed to the generator.
        answer:    The final answer string produced by the generator.
        trace_id:  Langfuse trace ID to attach the score to. Optional.

    Returns:
        Groundedness score (0.0–1.0) or None on failure.
    """
    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ResponseGroundedness

        import config

        critic_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=config.TAMU_MODEL,
                api_key=config.TAMU_API_KEY,
                base_url=config.TAMU_BASE_URL,
                temperature=0,
            )
        )

        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
        )
        dataset = EvaluationDataset(samples=[sample])

        # Evaluate using ResponseGroundedness metric
        metric = ResponseGroundedness(llm=critic_llm)
        result = evaluate(dataset=dataset, metrics=[metric])
        scores: dict = result.to_pandas().iloc[0].to_dict()

        groundedness_score = scores.get("response_groundedness", None)
        if groundedness_score is None:
            logger.warning("ResponseGroundedness metric returned None")
            return None

        # Upload score to Langfuse
        lf = get_langfuse()
        if lf and trace_id:
            import math
            if isinstance(groundedness_score, (int, float)) and not math.isnan(groundedness_score):
                lf.score(
                    trace_id=trace_id,
                    name="groundedness_score",
                    value=float(groundedness_score),
                    comment="RAGAS ResponseGroundedness (Gate 2 validation)",
                )

        logger.info(f"Groundedness score for trace {trace_id}: {groundedness_score}")
        return float(groundedness_score) if groundedness_score is not None else None

    except Exception as e:
        logger.warning(f"Groundedness scoring failed: {e}")
        return None


def run_groundedness_scoring_background(
    question: str,
    contexts: list[str],
    answer: str,
    trace_id: Optional[str] = None,
) -> None:
    """Fire-and-forget Gate 2 groundedness scoring in a background daemon thread."""
    thread = threading.Thread(
        target=score_groundedness,
        args=(question, contexts, answer, trace_id),
        daemon=True,
    )
    thread.start()
