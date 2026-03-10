"""Unified LLM client — routes calls to TAMU AI gateway or Google Gemini direct.

Public API:
    call_llm(messages, ...)   → LLMResult    # blocking, handles SSE accumulation
    stream_llm(messages, ...) → Iterator[str] # yields text tokens

The backend is selected automatically based on config.USE_TAMU_API:
  - True  → TAMU AI gateway (OpenAI-compat, always stream=True due to SSE quirk)
  - False → Google Gemini direct (google-genai SDK)
"""

from typing import Iterator, NamedTuple

from google.genai import types

import config


class LLMResult(NamedTuple):
    """Return type for call_llm().  Usage fields are None on the TAMU gateway path."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None


def call_llm(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    json_mode: bool = False,
    json_schema: dict | None = None,
    response_schema=None,
    thinking_budget: int = 0,
) -> LLMResult:
    """Make a blocking LLM call, returning an LLMResult with the full response text.

    Handles backend selection and SSE accumulation internally.

    Args:
        messages:        List of {"role": ..., "content": ...} dicts.
                         A leading {"role": "system", ...} is extracted as
                         system_instruction on the Gemini path.
        temperature:     Sampling temperature (0.0 = deterministic).
        max_tokens:      Max output tokens (use >= 4096; thinking tokens consume
                         budget first — 20 tokens → empty response).
        json_mode:       Request JSON-object output.  Sets response_mime_type on Gemini.
        json_schema:     Structured-output schema dict (TAMU json_schema format).
                         Implies json_mode.  Paired with response_schema for Gemini.
        response_schema: Pydantic model for Gemini structured output (parallel to
                         json_schema; ignored on TAMU path).
        thinking_budget: Thinking tokens for Gemini (0 = disabled).
                         Ignored on TAMU gateway path.

    Returns:
        LLMResult with .text (str) and optional token-usage fields (None on TAMU path).
    """
    if config.USE_TAMU_API:
        return _call_tamu(messages, temperature, max_tokens, json_mode, json_schema)
    return _call_gemini(messages, temperature, max_tokens, json_mode, response_schema, thinking_budget)


def stream_llm(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    thinking_budget: int = 0,
    usage_out: list | None = None,
) -> Iterator[str]:
    """Streaming LLM call — yields text tokens as they arrive.

    Not suitable for JSON-schema mode (structured output must be fully received
    before parsing).  Use call_llm() for structured-output calls.

    Args:
        messages:        List of {"role": ..., "content": ...} dicts.
        temperature:     Sampling temperature.
        max_tokens:      Max output tokens.
        thinking_budget: Thinking tokens for Gemini (0 = disabled).
                         Ignored on TAMU gateway path.
        usage_out:       Optional list; after stream ends, populated with
                         [input_tokens, output_tokens, thinking_tokens] on
                         the Gemini path.  TAMU path: not populated.

    Yields:
        str: Text tokens as they arrive.
    """
    if config.USE_TAMU_API:
        yield from _stream_tamu(messages, temperature, max_tokens)
    else:
        yield from _stream_gemini(messages, temperature, max_tokens, thinking_budget, usage_out=usage_out)


# ---------------------------------------------------------------------------
# TAMU AI gateway paths (OpenAI-compatible SDK)
# ---------------------------------------------------------------------------

def _call_tamu(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    json_schema: dict | None,
) -> LLMResult:
    """Blocking call via TAMU gateway.  Always uses stream=True (gateway SSE quirk)."""
    tamu = config.get_tamu_client()
    kwargs: dict = dict(
        model=config.TAMU_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    stream = tamu.chat.completions.create(**kwargs)
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    # TAMU gateway does not expose token counts in SSE
    return LLMResult(text=text)


def _stream_tamu(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> Iterator[str]:
    """Streaming call via TAMU gateway — yields tokens."""
    tamu = config.get_tamu_client()
    stream = tamu.chat.completions.create(
        model=config.TAMU_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


# ---------------------------------------------------------------------------
# Google Gemini direct paths (google-genai SDK)
# ---------------------------------------------------------------------------

def _extract_messages(messages: list[dict]) -> tuple[str | None, str]:
    """Split messages into (system_instruction, user_content).

    Expects at most one system message (first) followed by one user message (last).
    """
    if not messages:
        return None, ""
    system_instruction = None
    user_messages = messages
    if messages[0]["role"] == "system":
        system_instruction = messages[0]["content"]
        user_messages = messages[1:]
    return system_instruction, user_messages[-1]["content"] if user_messages else ""


def _build_genai_config(
    system_instruction: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    response_schema,
    thinking_budget: int,
) -> types.GenerateContentConfig:
    kwargs: dict = dict(
        temperature=temperature,
        max_output_tokens=max_tokens,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if thinking_budget > 0:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    elif json_mode:
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def _call_gemini(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    response_schema,
    thinking_budget: int,
) -> LLMResult:
    """Blocking call via Google Gemini direct."""
    system_instruction, contents = _extract_messages(messages)
    cfg = _build_genai_config(
        system_instruction, temperature, max_tokens, json_mode, response_schema, thinking_budget
    )
    client = config.get_genai_client()
    response = client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=contents,
        config=cfg,
    )
    text = response.text or ""
    usage = response.usage_metadata
    return LLMResult(
        text=text,
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
        thinking_tokens=getattr(usage, "thoughts_token_count", None) or 0,
    )


def _stream_gemini(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    usage_out: list | None = None,
) -> Iterator[str]:
    """Streaming call via Google Gemini direct — yields tokens."""
    system_instruction, contents = _extract_messages(messages)
    cfg = _build_genai_config(
        system_instruction, temperature, max_tokens, False, None, thinking_budget
    )
    client = config.get_genai_client()
    last_chunk = None
    for chunk in client.models.generate_content_stream(
        model=config.GENERATION_MODEL,
        contents=contents,
        config=cfg,
    ):
        last_chunk = chunk
        if chunk.text:
            yield chunk.text
    # Populate usage from final chunk metadata (Gemini only)
    if usage_out is not None and last_chunk is not None:
        meta = getattr(last_chunk, "usage_metadata", None)
        if meta:
            usage_out.extend([
                getattr(meta, "prompt_token_count", None),
                getattr(meta, "candidates_token_count", None),
                getattr(meta, "thoughts_token_count", None) or 0,
            ])
