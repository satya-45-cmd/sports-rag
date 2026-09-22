"""
Central configuration for the multi-sport RAG project.
Change values here rather than hunting through other files.
"""

import os

try:
    # Optional: lets you keep secrets like GROQ_API_KEY in a local .env file
    # instead of typing `set GROQ_API_KEY=...` every session. Harmless if the
    # package or the file isn't there.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")          # one subfolder per sport
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")    # persistent vector store

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 800       # characters per chunk (rough, not token-exact)
CHUNK_OVERLAP = 150    # overlap between consecutive chunks

# --- Embedding model (runs 100% locally via sentence-transformers) -------
# bge-small-en-v1.5 is small (~130MB), fast on CPU, and strong for its size.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = "sports_rules"

# --- LLM backend -----------------------------------------------------------
# "ollama" = runs on YOUR machine, fully local/offline, no API key, but only
#            reachable while your computer is on and running Ollama.
# "groq"   = a free hosted API that serves open-source models (Llama, etc.)
#            at very fast speed. Needed if you deploy this app publicly
#            (e.g. Streamlit Community Cloud), since a public server can't
#            reach the Ollama running on your personal laptop.
# Set via environment variable / Streamlit secret; defaults to local Ollama
# so nothing changes for your existing local setup.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# --- Local LLM via Ollama (free, runs on your machine, no API key) --------
# Swap this for any model you've pulled with `ollama pull <name>`.
# Good options with a GPU: "llama3.1:8b", "mistral:7b"
# Good options CPU-only / low VRAM: "llama3.2:3b", "phi3:mini"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"

# --- Hosted LLM via Groq (free tier, needed for public deployment) --------
# Get a free key at https://console.groq.com (no local install needed).
# Model list/limits can change -- check https://console.groq.com/docs/models
# Set the key as an environment variable, a local .env file (GROQ_API_KEY=...),
# or a Streamlit Cloud "secret" -- never commit it into the code or GitHub.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Indexing performance ---------------------------------------------------
# These matter once your document collection gets large (full rulebooks,
# many sports, etc.) -- they keep memory use bounded and let you see
# progress instead of the script appearing to hang.
EMBED_BATCH_SIZE = 64          # chunks embedded per batch (lower if RAM-limited)
CHROMA_ADD_BATCH_SIZE = 500    # chunks written to Chroma per .add() call

# --- Retrieval -------------------------------------------------------------
TOP_K = 4              # how many chunks are used to answer, after re-ranking
RERANK_CANDIDATES = 15  # how many chunks the initial vector search retrieves
                         # before the cross-encoder re-ranks them down to TOP_K
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# bge-* embedding models are trained to expect this prefix on queries
# (but NOT on the documents being indexed) — it noticeably improves
# retrieval quality. If you swap to a different embedding model, check
# its model card on Hugging Face for whether it wants something similar.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Sports the UI will offer as filter options. This is derived automatically
# from the folder names under DATA_DIR, so adding a new sport is just:
# mkdir data/<new_sport> and drop files in it.
def list_sports():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(".")
    )
