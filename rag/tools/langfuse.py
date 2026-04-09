"""Langfuse observability and RAGAS evaluation for TamuBot pipeline.

Canonical location: rag/tools/langfuse.py

Uses the official Langfuse SDK (v4+, Pydantic v2, Python 3.14 compatible).

Provides:
    get_langfuse()                       — lazy singleton Langfuse client
    compute_ragas_metrics()              — Faithfulness + AnswerRelevancy via RAGAS
    run_ragas_background()               — fire-and-forget wrapper
    score_groundedness()                 — Gate 2 LLM-as-Judge via RAGAS
    run_groundedness_scoring_background() — fire-and-forget wrapper
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger("tamubot.observability")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_langfuse_client = None


def get_langfuse():
    """Lazy singleton. Returns None if Langfuse credentials are not configured."""
    global _langfuse_client
    if _langfuse_client is None:
        import config
        if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
            return None
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_BASE_URL,
            )
            logger.info("Langfuse SDK client initialised.")
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
        # litellm doesn't know TAMU gateway pricing → assigns huge default cost
        # → trips its own $5 budget cap. Disable budget enforcement.
        try:
            import litellm
            litellm.max_budget = None
        except Exception:
            pass

        result = evaluate(dataset=dataset, metrics=metrics)
        scores: dict = result.to_pandas().iloc[0].to_dict()

        lf = get_langfuse()
        if lf and trace_id:
            import math
            for metric_name, value in scores.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    lf.create_score(
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

        metric = ResponseGroundedness(llm=critic_llm)
        result = evaluate(dataset=dataset, metrics=[metric])
        scores: dict = result.to_pandas().iloc[0].to_dict()

        groundedness_score = scores.get("response_groundedness", None)
        if groundedness_score is None:
            logger.warning("ResponseGroundedness metric returned None")
            return None

        lf = get_langfuse()
        if lf and trace_id:
            import math
            if isinstance(groundedness_score, (int, float)) and not math.isnan(groundedness_score):
                lf.create_score(
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


# ---------------------------------------------------------------------------
# Retrieval evaluation (ContextPrecision + ContextRecall)
# ---------------------------------------------------------------------------

def compute_retrieval_ragas(
    question: str,
    contexts: list[str],
    reference: str,
    trace_id: Optional[str] = None,
) -> dict:
    """Compute RAGAS ContextPrecision + ContextRecall and upload scores to Langfuse.

    Uses TAMU gateway as the critic LLM.  Unlike compute_ragas_metrics() this
    function evaluates *retrieval quality* rather than generation quality, so it
    requires a reference answer instead of a generated answer.

    Args:
        question:  The original user query.
        contexts:  Retrieved chunk texts that were passed to the generator.
        reference: Ground-truth reference answer for precision/recall computation.
        trace_id:  Langfuse trace ID to attach scores to. Optional.

    Returns:
        Dict of metric_name → float score, or {} on failure.
    """
    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextPrecision, ContextRecall

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
            reference=reference,
        )
        dataset = EvaluationDataset(samples=[sample])

        metrics = [
            ContextPrecision(llm=critic_llm),
            ContextRecall(llm=critic_llm),
        ]

        # Disable litellm budget cap to avoid spurious cost-limit errors.
        try:
            import litellm
            litellm.max_budget = None
        except Exception:
            pass

        result = evaluate(dataset=dataset, metrics=metrics)
        scores: dict = result.to_pandas().iloc[0].to_dict()

        lf = get_langfuse()
        if lf and trace_id:
            import math
            for metric_name, value in scores.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    lf.create_score(
                        trace_id=trace_id,
                        name=metric_name,
                        value=float(value),
                        comment="RAGAS retrieval evaluation",
                    )

        logger.info(f"Retrieval RAGAS scores for trace {trace_id}: {scores}")
        return scores

    except Exception as e:
        logger.warning(f"Retrieval RAGAS evaluation failed: {e}")
        return {}
