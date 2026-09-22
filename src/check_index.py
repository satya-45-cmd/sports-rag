"""
Quick sanity check on what actually made it into the vector index.

Run after src/build_index.py:

    python src/check_index.py

Prints, per sport and per source file: how many chunks were indexed and
what page range they cover -- so you can confirm every document you added
(not just the starter basics.txt files) actually got chunked and embedded,
without needing to inspect Chroma's internals by hand.
"""

from collections import defaultdict

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME


def check_index():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"No collection named '{COLLECTION_NAME}' found at {CHROMA_DIR}.")
        print("Run `python src/build_index.py` first.")
        return

    total = collection.count()
    if total == 0:
        print("Index exists but is empty. Run `python src/build_index.py`.")
        return

    # metadatas only -- we don't need the embeddings/text for this report
    data = collection.get(include=["metadatas"])
    metadatas = data["metadatas"]

    # sport -> source filename -> list of page numbers (one per chunk)
    by_sport_source = defaultdict(lambda: defaultdict(list))
    for meta in metadatas:
        sport = meta.get("sport", "?")
        source = meta.get("source", "?")
        page = meta.get("page")
        by_sport_source[sport][source].append(page)

    print(f"Total chunks indexed: {total}\n")

    for sport in sorted(by_sport_source):
        sources = by_sport_source[sport]
        sport_total = sum(len(pages) for pages in sources.values())
        print(f"{sport}  ({sport_total} chunks, {len(sources)} file(s))")
        for source, pages in sorted(sources.items()):
            known_pages = [p for p in pages if p is not None]
            if known_pages:
                page_range = f"pages {min(known_pages)}-{max(known_pages)}"
            else:
                page_range = "no page info"
            print(f"    - {source}: {len(pages)} chunks, {page_range}")
        print()


if __name__ == "__main__":
    check_index()
