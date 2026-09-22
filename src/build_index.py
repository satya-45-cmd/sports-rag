"""
Build (or rebuild) the local vector index.

Run this whenever you add/change files under data/<sport>/:

    python src/build_index.py

It embeds every chunk with a local sentence-transformers model and stores
the vectors + metadata in a persistent local Chroma database (chroma_db/).
Nothing here calls out to a paid API.
"""

import shutil

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_ADD_BATCH_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)
from ingest import load_all_documents


def build_index(reset: bool = True):
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it, then it's cached locally)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nLoading and chunking documents...")
    chunks = load_all_documents()
    if not chunks:
        print(
            "\nNo documents found. Add .txt, .md, or .pdf files under data/<sport>/ "
            "(a few starter .txt files are already included) and re-run this script."
        )
        return

    if reset:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"\nEmbedding {len(chunks)} chunks (batch size {EMBED_BATCH_SIZE})...")
    texts = [c.text for c in chunks]
    # batch_size controls how many chunks the model processes at once --
    # keeps memory bounded no matter how many documents you throw at this.
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    ids = [f"{c.sport}-{c.source}-p{c.page}-{c.chunk_index}" for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Chroma (and SQLite underneath it) has a practical limit on how many
    # items one .add() call can take, so a large corpus is written in
    # batches rather than a single call.
    total = len(chunks)
    for start in range(0, total, CHROMA_ADD_BATCH_SIZE):
        end = min(start + CHROMA_ADD_BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {end}/{total} chunks...")

    print(f"\nDone. Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
