import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# --- GCP / Vertex AI (legacy, kept as fallback) ---
PROJECT_ID = os.getenv("PROJECT_ID", "glossy-surge-486017-g8")
RETRIEVAL_REGION = os.getenv("RETRIEVAL_REGION", "us-south1")
GENERATION_REGION = os.getenv("GENERATION_REGION", "us-central1")
RAG_CORPUS_RESOURCE_NAME = os.getenv(
    "RAG_CORPUS_RESOURCE_NAME",
    "projects/glossy-surge-486017-g8/locations/us-south1/ragCorpora/2305843009213693952"
)

# --- MongoDB Atlas ---
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "tamubot")

# --- Voyage AI (embeddings + reranking) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2")

# --- Google AI (Gemini for generation + router) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "gemini-2.5-flash-lite")

# --- TAMU AI API (OpenAI-compatible gateway; data privacy + institutional billing) ---
TAMU_API_KEY = os.getenv("TAMU_API_KEY")
TAMU_BASE_URL = os.getenv("TAMU_BASE_URL", "https://chat-api.tamu.ai/openai")
TAMU_MODEL = os.getenv("TAMU_MODEL", "protected.gemini-2.5-flash")
# When set, all RAG LLM calls route through TAMU API instead of direct Google API.
# ingestion_pipeline/process_syllabi.py is excluded (uses PDF multimodal input).
USE_TAMU_API: bool = bool(TAMU_API_KEY)

# --- Thinking token budgets for Gemini 2.5 Flash ---
# metadata_* functions use deterministic extraction (no thinking needed)
THINKING_BUDGET_METADATA = 0
# hybrid_* and semantic_general functions use thinking for complex reasoning
THINKING_BUDGET_SEMANTIC = 1024

# --- Temperature constants for function-based stochasticity ---
# Deterministic (factual extraction): 0.0
TEMP_DETERMINISTIC = 0.0
# Synthesis (advisory reasoning): 0.2 for linguistic fluidity
TEMP_SYNTHESIS = 0.2

# --- Retrieval tuning (global fallbacks for low-confidence paths) ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# Default categories fetched when no specific category is requested.
DEFAULT_SUMMARY_CATEGORIES: list[str] = [
    "COURSE_OVERVIEW", "PREREQUISITES", "LEARNING_OUTCOMES"
]

# category_confidence threshold: >= this → use metadata (exact) path;
# < this → fall back to hybrid search.
CATEGORY_CONFIDENCE_THRESHOLD: float = 0.7

# Per-function retrieval config: retrieve_k = candidates sent to reranker,
# rerank_k = final results kept after reranking (ignored on metadata path).
# For multi-course functions these are *per course*.
FUNCTION_RETRIEVAL_CONFIG: dict[str, dict[str, int]] = {
    "metadata_default":  {"retrieve_k": 10, "rerank_k": 0},
    "metadata_specific": {"retrieve_k": 10, "rerank_k": 0},
    "metadata_combined": {"retrieve_k": 10, "rerank_k": 0},
    "semantic_general":  {"retrieve_k": 30, "rerank_k": 10},
    "hybrid_default":    {"retrieve_k": 12, "rerank_k": 3},
    "hybrid_specific":   {"retrieve_k": 10, "rerank_k": 3},
    "hybrid_combined":   {"retrieve_k": 15, "rerank_k": 4},
}

# Alias used by router._compute_dynamic_k for per-course scaling.
PER_COURSE_K = FUNCTION_RETRIEVAL_CONFIG

# Global caps for scaled multi-course retrieval.
MAX_RETRIEVE_K: int = 60
MAX_RERANK_K: int = 20

# --- Retrieval backend ---
# "mongodb" (default) or "vertex" (legacy fallback)
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "mongodb")

# --- Observability ---
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
APP_MODE = os.getenv("APP_MODE", "test")

# --- Langfuse ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# --- Shared google-genai client (lazy singleton) ---
_genai_client = None


def get_genai_client():
    """Return a shared google.genai.Client instance, creating it on first call."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _genai_client


# --- Shared TAMU OpenAI-compatible client (lazy singleton) ---
_tamu_client = None


def get_tamu_client():
    """Return a shared openai.OpenAI client pointed at the TAMU AI gateway."""
    global _tamu_client
    if _tamu_client is None:
        from openai import OpenAI
        _tamu_client = OpenAI(api_key=TAMU_API_KEY, base_url=TAMU_BASE_URL)
    return _tamu_client
