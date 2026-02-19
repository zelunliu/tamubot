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
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")

# --- Retrieval tuning ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# --- Retrieval backend ---
# "mongodb" (default) or "vertex" (legacy fallback)
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "mongodb")

# --- Observability ---
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
APP_MODE = os.getenv("APP_MODE", "test")

# --- Shared google-genai client (lazy singleton) ---
_genai_client = None


def get_genai_client():
    """Return a shared google.genai.Client instance, creating it on first call."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _genai_client
