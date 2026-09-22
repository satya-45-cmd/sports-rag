"""
Ingestion: walk data/<sport>/ folders, extract text from .txt/.md/.pdf files
page by page, split each page into overlapping chunks, and tag every chunk
with metadata (sport, source file, page number, chunk index) so retrieval
can later filter, cite, or re-rank it.

This is deliberately plain Python (no LangChain) so a beginner can read
every line of the pipeline. Feel free to modify it.
"""

import os
from dataclasses import dataclass, field

from pypdf import PdfReader

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, list_sports


@dataclass
class Chunk:
    text: str
    sport: str
    source: str        # file name the chunk came from
    page: int          # 1-indexed page number (PDFs); 1 for .txt/.md
    chunk_index: int   # position of this chunk within its source file
    metadata: dict = field(default_factory=dict)


def _read_txt_pages(path: str):
    """.txt/.md files have no real pages -- treat the whole file as page 1."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [f.read()]


def _read_pdf_pages(path: str):
    """Return a list of page texts, one entry per PDF page (1-indexed by position)."""
    reader = PdfReader(path)
    if reader.is_encrypted:
        # Many official documents (rulebooks, standards, etc.) are "encrypted"
        # only to restrict editing/printing permissions -- they're still meant
        # to be freely read, with no real password. Try the empty password
        # before giving up; if the file genuinely needs a real password this
        # will fail quietly and extract_text() below will just return "" for
        # every page, same as any other unreadable file.
        try:
            reader.decrypt("")
        except Exception:
            pass
    return [page.extract_text() or "" for page in reader.pages]


def load_file_pages(path: str):
    """Return a list of page texts for any supported file type."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf_pages(path)
    if ext in (".txt", ".md"):
        return _read_txt_pages(path)
    raise ValueError(f"Unsupported file type: {path}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Simple character-based sliding-window chunker with overlap.
    Tries to break on a paragraph or sentence boundary near the target size
    so chunks don't split mid-sentence when it can be avoided.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            # try to end on a sentence boundary within the last 20% of the window
            search_from = start + int(chunk_size * 0.8)
            period = text.rfind(". ", search_from, end)
            if period != -1:
                end = period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)  # ensure forward progress

    return chunks


def load_sport_documents(sport: str):
    """Return a list of Chunk objects for every supported file under data/<sport>/."""
    sport_dir = os.path.join(DATA_DIR, sport)
    chunks = []

    if not os.path.isdir(sport_dir):
        return chunks

    for filename in sorted(os.listdir(sport_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".txt", ".md", ".pdf"):
            continue

        path = os.path.join(sport_dir, filename)
        try:
            pages = load_file_pages(path)
        except Exception as e:
            print(f"  [!] Skipping {filename}: {e}")
            continue

        # Chunk each page separately (rather than joining the whole file into
        # one blob first) so every chunk can be traced back to an exact page
        # -- this is what lets citations say "page 12", not just a filename.
        file_chunk_index = 0
        for page_num, page_text in enumerate(pages, start=1):
            for piece in chunk_text(page_text):
                chunks.append(
                    Chunk(
                        text=piece,
                        sport=sport,
                        source=filename,
                        page=page_num,
                        chunk_index=file_chunk_index,
                        metadata={
                            "sport": sport,
                            "source": filename,
                            "page": page_num,
                            "chunk_index": file_chunk_index,
                        },
                    )
                )
                file_chunk_index += 1

        print(f"  {filename}: {file_chunk_index} chunk(s) across {len(pages)} page(s)")

    return chunks


def load_all_documents():
    """Return a flat list of Chunk objects across every sport folder."""
    all_chunks = []
    sports = list_sports()
    if not sports:
        print(f"No sport folders found under {DATA_DIR}")
        return all_chunks

    for sport in sports:
        print(f"Loading '{sport}'...")
        all_chunks.extend(load_sport_documents(sport))

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_documents()
    print(f"\nTotal chunks loaded: {len(chunks)}")
    if chunks:
        print("\nExample chunk:")
        print(f"  sport={chunks[0].sport} source={chunks[0].source}")
        print(f"  text preview: {chunks[0].text[:200]}...")
