"""
Core RAG logic: embed the question, retrieve relevant chunks from Chroma,
build a grounded prompt, and generate an answer with a local Ollama model.

Can be run directly as a terminal chat for quick testing:

    python src/rag_chat.py
"""

import json

import chromadb
import ollama
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    LLM_BACKEND,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    QUERY_INSTRUCTION,
    RERANK_CANDIDATES,
    RERANK_MODEL_NAME,
    TOP_K,
    list_sports,
)

SYSTEM_PROMPT = """You are a helpful sports rules and strategy assistant.
Answer the user's question using ONLY the information in the provided context.
If the context does not contain enough information to answer, say so plainly
instead of guessing. Keep answers concise and cite which sport each fact
comes from when it's relevant. Do not mention that you were given "context"
or "chunks" -- just answer naturally, like a knowledgeable friend."""

# Extra words that should point at a sport even though they aren't its folder
# name. Add to these lists as you add sports/documents -- detection is just
# "does any of these words appear in the question," nothing fancier.
SPORT_ALIASES = {
    "basketball": ["nba", "fiba"],
    "nfl": ["american football", "quarterback", "touchdown"],
    "soccer": ["football", "fifa", "premier league"],
    "cricket": ["ipl", "wicket", "bowler"],
    "tennis": ["grand slam", "wimbledon"],
}


def detect_sport(question: str):
    """
    Best-effort guess at which sport a question is about, so retrieval can
    filter to it even when the user hasn't picked one from the dropdown.
    Returns a sport name (matching a data/<sport>/ folder) or None if no
    sport name/alias is found in the question -- callers should treat None
    as "search across everything."
    """
    q = question.lower()
    for sport in list_sports():
        if sport in q:
            return sport
    for sport, aliases in SPORT_ALIASES.items():
        if sport in list_sports() and any(alias in q for alias in aliases):
            return sport
    return None


class RagEngine:
    def __init__(self):
        self._embed_model = None
        self._cross_encoder = None
        self._collection = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embed_model

    @property
    def cross_encoder(self):
        """
        A second, smaller model used only to re-rank an already-retrieved
        shortlist. Vector search (bi-encoder) is fast but approximate --
        it embeds the question and each chunk separately, so it can't see
        how they interact. A cross-encoder reads the question and a chunk
        TOGETHER and scores how relevant that specific pairing is, which is
        slower (so it's only run on a small shortlist) but more accurate --
        the standard "retrieve wide, then re-rank" pattern in production RAG.
        """
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        return self._cross_encoder

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._collection = client.get_or_create_collection(COLLECTION_NAME)
        return self._collection

    def retrieve(self, question: str, sport: str = None, k: int = TOP_K):
        """
        Return the top-k most relevant chunks as a list of
        {text, sport, source, page, distance, rerank_score} dicts.

        If `sport` isn't given (or is "All"), this tries to auto-detect the
        sport from the question's wording via detect_sport() and filters to
        it -- an explicit `sport` argument always overrides that guess.
        """
        effective_sport = sport if sport and sport.lower() != "all" else detect_sport(question)
        where = {"sport": effective_sport} if effective_sport else None

        # Stage 1: fast vector search over a wider candidate pool than we'll
        # actually use, so the re-ranker below has real options to choose from.
        query_vector = self.embed_model.encode(
            [QUERY_INSTRUCTION + question], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=RERANK_CANDIDATES,
            where=where,
        )

        candidates = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            candidates.append(
                {
                    "text": text,
                    "sport": meta.get("sport"),
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "distance": dist,
                }
            )

        if not candidates:
            return []

        # Stage 2: re-rank that pool with the cross-encoder and keep the
        # actual top-k -- vector-search order and true relevance order often
        # differ, especially as your document collection grows.
        pairs = [(question, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:k]

    def build_context(self, hits):
        blocks = []
        for i, h in enumerate(hits, 1):
            page_str = f", p.{h['page']}" if h.get("page") else ""
            blocks.append(f"[{i}] ({h['sport']} / {h['source']}{page_str})\n{h['text']}")
        return "\n\n".join(blocks)

    def ask(self, question: str, sport: str = None, k: int = TOP_K, model: str = None):
        hits = self.retrieve(question, sport=sport, k=k)

        if not hits:
            return {
                "answer": (
                    "I don't have any indexed documents to answer that from yet. "
                    "Add files under data/<sport>/ and run `python src/build_index.py`."
                ),
                "sources": [],
            }

        context = self.build_context(hits)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if LLM_BACKEND == "groq":
            answer = self._ask_groq(messages, model or GROQ_MODEL)
        else:
            answer = self._ask_ollama(messages, model or OLLAMA_MODEL)

        sources = [
            {
                "sport": h["sport"],
                "source": h["source"],
                "page": h.get("page"),
                "text": h["text"],
                "rerank_score": h.get("rerank_score"),
            }
            for h in hits
        ]
        return {"answer": answer, "sources": sources}

    def _ask_ollama(self, messages, model):
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(model=model, messages=messages)
        return response["message"]["content"]

    def _ask_groq(self, messages, model):
        if not GROQ_API_KEY:
            raise RuntimeError(
                "LLM_BACKEND is set to 'groq' but GROQ_API_KEY is missing. "
                "Set it as an environment variable, a local .env file, or a "
                "Streamlit Cloud secret. Get a free key at https://console.groq.com"
            )
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": model, "messages": messages, "temperature": 0.3},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def ask_stream(self, question: str, sport: str = None, k: int = TOP_K, model: str = None):
        """
        Like ask(), but returns (sources, chunks) where `chunks` is a
        generator yielding pieces of the answer as the LLM actually produces
        them, instead of the whole string at once. This is what makes the
        Streamlit UI's answers appear "live" rather than showing up all at
        once after a long pause -- real token streaming, not a replay effect.

        Retrieval happens eagerly before this returns (it's fast -- a vector
        search plus a small cross-encoder re-rank), so `sources` is ready
        immediately; only the slower LLM call is streamed.
        """
        hits = self.retrieve(question, sport=sport, k=k)

        if not hits:
            def _no_docs():
                yield (
                    "I don't have any indexed documents to answer that from yet. "
                    "Add files under data/<sport>/ and run `python src/build_index.py`."
                )

            return [], _no_docs()

        context = self.build_context(hits)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if LLM_BACKEND == "groq":
            chunks = self._stream_groq(messages, model or GROQ_MODEL)
        else:
            chunks = self._stream_ollama(messages, model or OLLAMA_MODEL)

        sources = [
            {
                "sport": h["sport"],
                "source": h["source"],
                "page": h.get("page"),
                "text": h["text"],
                "rerank_score": h.get("rerank_score"),
            }
            for h in hits
        ]
        return sources, chunks

    def _stream_ollama(self, messages, model):
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(model=model, messages=messages, stream=True)

        def _chunks():
            # Matches ollama-python's own documented streaming pattern
            # (chunk['message']['content']) -- every chunk has a "message"
            # key, content is just "" on the final one.
            for part in response:
                piece = part["message"]["content"]
                if piece:
                    yield piece

        return _chunks()

    def _stream_groq(self, messages, model):
        if not GROQ_API_KEY:
            raise RuntimeError(
                "LLM_BACKEND is set to 'groq' but GROQ_API_KEY is missing. "
                "Set it as an environment variable, a local .env file, or a "
                "Streamlit Cloud secret. Get a free key at https://console.groq.com"
            )
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": model, "messages": messages, "temperature": 0.3, "stream": True},
            timeout=30,
            stream=True,
        )
        # Raised here, before any chunk is consumed -- an invalid key or
        # deprecated model name comes back as an immediate 4xx, so callers
        # can still catch it the same way they'd catch a non-streaming error.
        response.raise_for_status()

        def _chunks():
            # Groq's streaming format is Server-Sent Events: each line is
            # either blank (keep-alive) or "data: <json>", ending with the
            # sentinel "data: [DONE]".
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = event.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece

        return _chunks()


def _cli():
    engine = RagEngine()
    sports = list_sports()
    print("Sports available:", ", ".join(sports) if sports else "(none indexed yet)")
    print("Sport is auto-detected from your question by default.")
    print("Type a question, or 'sport:<name>' to force a filter, or 'quit' to exit.\n")

    current_sport = None
    while True:
        raw = input("You: ").strip()
        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            break
        if raw.lower().startswith("sport:"):
            current_sport = raw.split(":", 1)[1].strip() or None
            print(f"(filtering to sport = {current_sport or 'All'})\n")
            continue

        try:
            result = engine.ask(raw, sport=current_sport)
        except Exception as e:
            print(f"\n[Error] {e}")
            print("Is Ollama running? Try `ollama serve` in another terminal.\n")
            continue

        print(f"\nBot: {result['answer']}")
        if result["sources"]:
            src_str = ", ".join(
                f"{s['sport']}/{s['source']}" + (f" p.{s['page']}" if s.get("page") else "")
                for s in result["sources"]
            )
            print(f"(sources: {src_str})")
        print()


if __name__ == "__main__":
    _cli()
