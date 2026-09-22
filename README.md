# Sports Rules & Strategy RAG Bot

A beginner-friendly, end-to-end Retrieval-Augmented Generation (RAG) project:
ask natural-language questions about the rules and strategy of cricket,
soccer, basketball, NFL, and tennis, and get answers grounded in real
documents instead of the model just guessing.

**Everything here is free and open-source and runs on your own machine.**
No API keys, no cloud bills, no signups.

## How it works

```
Your PDFs/text files
        |
        v
   chunk into pieces, page by page (src/ingest.py)
        |
        v
 embed each chunk locally in batches (sentence-transformers)
        |
        v
 store vectors + page numbers in Chroma, a local vector database (src/build_index.py)
        |
        v
  you ask a question  --->  auto-detect the sport  --->  embed the question
  --->  vector search for a wide candidate pool  --->  a cross-encoder
  re-ranks them for true relevance  --->  the real top-k get stuffed into
  a prompt  --->  a local LLM (via Ollama, or Groq if deployed) generates
  an answer grounded in those chunks, cited by sport/file/page
```

This is the same "retrieve wide, then re-rank" pattern used in production
RAG systems — you're just running every piece locally and for free instead
of paying for hosted APIs.

## 1. Prerequisites

- Python 3.10 or newer
- ~5-10 GB free disk space (for the embedding model + an LLM)
- [Ollama](https://ollama.com/download) installed — this runs the LLM
  locally. It auto-detects your GPU (NVIDIA/AMD) if you have one and
  falls back to CPU otherwise.

## 2. Set up the Python environment

```bash
cd sports-rag
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run the embedding step, `sentence-transformers` will
download the embedding model (~130MB) and cache it locally — no account
needed.

## 3. Pull a local LLM with Ollama

With a GPU (recommended, noticeably better answers):

```bash
ollama pull llama3.1:8b
```

If you only have CPU or a small/low-VRAM GPU, use a smaller model instead
and update `OLLAMA_MODEL` in `src/config.py` to match:

```bash
ollama pull llama3.2:3b
# or
ollama pull phi3:mini
```

Then make sure the Ollama server is running (it usually starts
automatically after install; if not, run `ollama serve` in a terminal
and leave it running).

## 4. Add your documents

Starter content is already included under `data/<sport>/basics.txt` for
all five sports, written as short original overviews — enough to try the
whole pipeline immediately with no setup.

For a fuller project, download official rulebooks (PDF or text) and drop
them into the matching folder:

| Sport | Folder | A good free official source |
|---|---|---|
| Cricket | `data/cricket/` | MCC Laws of Cricket — lords.org |
| Soccer | `data/soccer/` | IFAB Laws of the Game — theifab.com |
| Basketball | `data/basketball/` | FIBA Official Basketball Rules — fiba.basketball |
| NFL | `data/nfl/` | NFL Rulebook — operations.nfl.com |
| Tennis | `data/tennis/` | ITF Rules of Tennis — itftennis.com |

You can add as many `.pdf`, `.txt`, or `.md` files per folder as you
like, and add a brand-new sport just by creating a new folder under
`data/` (e.g. `data/hockey/`) — the app picks it up automatically.

## 5. Build the vector index

Every time you add or change documents, rebuild the index:

```bash
python src/build_index.py
```

This embeds all your chunks and (re)writes the local `chroma_db/`
folder. It's safe to delete `chroma_db/` and re-run this at any time.

`chroma_db/` is committed to git (not ignored) on purpose: if you're
deploying this publicly, a committed index means a fresh/woken-up
deployment can start answering immediately instead of re-embedding
everything from scratch first (see "Cold-start speed" under Deploying,
below). **This means that after you rebuild the index, you need to
commit the updated `chroma_db/` folder too** (`git add chroma_db/` or
just commit everything, e.g. via GitHub Desktop's "Commit to master"),
not just the new source documents — otherwise the deployed app keeps
answering from the old index until you do. Because it's a binary
SQLite file, git will show the whole thing as changed on every rebuild
even for a small edit — that's normal, not a sign something broke.

Chunks are tracked page-by-page (for PDFs) so answers can cite exactly
which page a fact came from, and both embedding and writing to Chroma
happen in batches (`EMBED_BATCH_SIZE`, `CHROMA_ADD_BATCH_SIZE` in
`src/config.py`) so this stays fast and memory-bounded even if you drop
in full official rulebooks instead of the small starter files.

## 6. Chat with it

Terminal version (good for quickly checking things work):

```bash
python src/rag_chat.py
```

Web UI version:

```bash
streamlit run app.py
```

This opens a chat interface in your browser with a sidebar dropdown to
filter answers to a specific sport, or search across all of them. Leave
it on "All" and the bot will usually guess the right sport from the
wording of your question (see "Sport auto-detection" below); each answer
comes with an expandable "Sources" section showing which sport/file/page
it used, and a 👍/👎 you can click to log feedback on that answer.

## Project layout

```
sports-rag/
├── app.py                 # Streamlit chat UI
├── requirements.txt
├── data/
│   ├── cricket/
│   ├── soccer/
│   ├── basketball/
│   ├── nfl/
│   └── tennis/            # drop your .pdf/.txt/.md files here
├── src/
│   ├── config.py           # all the knobs you'd want to tune
│   ├── ingest.py           # loads + chunks documents, tracked page-by-page
│   ├── build_index.py      # embeds + stores in Chroma, in batches
│   └── rag_chat.py         # sport detection + retrieve/re-rank + LLM call, plus a CLI chat
├── chroma_db/              # generated by build_index.py -- committed to git so
│                           #   public deployments skip re-indexing on cold start
└── feedback.csv            # generated once you click 👍/👎 in the UI, not committed to git
```

## Deploying so anyone on the internet can use it

Everything above runs only on your own machine, reachable only while it's on.
To get a public link that works 24/7 even when your laptop is off, one piece
has to move off your PC: the LLM. A public web server can't reach Ollama
running on your personal laptop, so this uses a free hosted API that serves
the same kind of open-source models instead. Everything else (chunking,
embeddings, retrieval, Chroma) stays exactly as-is.

### 1. Get a free Groq API key

Groq (console.groq.com) hosts open-source models for free, and is very
fast. Sign up, then create an API key from the dashboard. This project
defaults to the `openai/gpt-oss-20b` model (set in `src/config.py`).
**Groq regularly retires older free-tier models** — if the chat ever shows
a 404 error mentioning the model name, it means the current default has
been deprecated. Check
[console.groq.com/docs/models](https://console.groq.com/docs/models) for
the current list of free-tier models, then either update the default in
`src/config.py` (the `GROQ_MODEL` line) or set `GROQ_MODEL=<new-model-name>`
in your `.env` file / Streamlit secrets to override it without touching
code.

### 2. Test the Groq backend locally first

In your project folder, create a new file named `.env` (no filename before
the dot) with:

```
LLM_BACKEND=groq
GROQ_API_KEY=your-key-here
```

Run `streamlit run app.py` again — the sidebar should now say the model is
running "via Groq's free hosted API" instead of local Ollama, and answers
should come back noticeably fast. If something's wrong, the chat will show
a Groq-specific error message telling you what to check.

### 3. Push the project to GitHub

Streamlit Community Cloud deploys directly from a GitHub repository.

- Create a free GitHub account if you don't have one, and create a new
  (public) repository.
- The easiest path on Windows if you're new to git is
  [GitHub Desktop](https://desktop.github.com/) — install it, sign in,
  "Add local repository" pointing at your `sports-rag` folder, then
  "Publish repository."
- **Never commit your `.env` file or paste your API key into any file that
  gets pushed** — `.env` is already listed in `.gitignore` so this should
  happen automatically, but it's worth double-checking on GitHub's website
  after pushing.

### 4. Deploy on Streamlit Community Cloud

Streamlit Community Cloud (streamlit.io/cloud) hosts Streamlit apps for
free.

- Sign in with your GitHub account.
- Click "New app," pick your `sports-rag` repository and branch, and set
  the main file path to `app.py`.
- Before (or after) deploying, open the app's Settings -> Secrets and add:

  ```
  LLM_BACKEND = "groq"
  GROQ_API_KEY = "your-key-here"
  ```

  This is the secure way to provide the key — it's never stored in your
  code or visible on GitHub.
- Deploy. Because `chroma_db/` is committed to the repo (see step 5
  above), the app can start answering as soon as it's up — it doesn't
  need to rebuild the index from your PDFs on every cold start.
  `app.py` only rebuilds automatically as a fallback, for the rare case
  where `chroma_db/` is missing entirely.

You'll get a public URL (something like
`https://your-app-name.streamlit.app`) that works for anyone, anytime,
without them installing anything.

### Cold-start speed

Streamlit Community Cloud fully stops an app after a period of no
traffic; the next visitor "wakes" it, which means a fresh container:
your dependencies get reinstalled and `app.py` starts from scratch.
Committing `chroma_db/` (a prebuilt index, not just the source PDFs)
means that wake-up doesn't also have to re-embed every chunk of every
rulebook before it can answer — it just loads the existing index. The
embedding and cross-encoder re-ranking models still get downloaded
fresh on a true cold start (they're not part of the repo — see
"First question after adding documents feels slow" in Troubleshooting),
but that's much faster than a full re-index.

Answers themselves also stream in as the model generates them (word by
word) instead of waiting for the whole response before showing anything
— real streaming from Groq/Ollama's API, not a fake typing effect, so
you see the first words almost immediately rather than after a pause.

### Notes on the free tiers

Both Groq's and Streamlit Community Cloud's free tiers can change their
exact limits over time, and Streamlit Cloud apps that get no traffic for a
while go to sleep and take a few seconds to wake back up on the next visit
— normal behavior for a free hosted demo, not a bug.

## Scaling up: bigger documents, smarter retrieval, more features

These three upgrades are already built in — this section explains what
each one does and how to tune or extend it.

### Bigger documents (batched indexing + page citations)

`src/ingest.py` reads PDFs and text files page-by-page instead of as one
big blob, so every chunk remembers exactly which page it came from. You'll
see this show up as `sport / file.pdf, p.12` in the "Sources" section of
the UI and in the CLI's `(sources: ...)` line — much more useful than just
"somewhere in this 40-page PDF" once you swap the small starter files for
full official rulebooks.

`src/build_index.py` embeds chunks and writes them to Chroma in batches
(`EMBED_BATCH_SIZE` and `CHROMA_ADD_BATCH_SIZE` in `src/config.py`)
instead of all at once, so indexing stays memory-bounded and shows
progress as it goes, no matter how many documents or how many sports you
throw at it. If indexing ever feels slow or memory-hungry on your
machine, lowering `EMBED_BATCH_SIZE` is the first knob to try.

### Smarter retrieval (sport auto-detection + re-ranking)

**Sport auto-detection** (`detect_sport()` in `src/rag_chat.py`): when the
sidebar filter is left on "All," the bot guesses which sport your question
is about from its wording, so retrieval only searches that sport's
documents instead of mixing in irrelevant chunks from the others. It's a
deliberately simple, honest heuristic — it just checks whether a sport's
name (e.g. "basketball") or one of a short list of aliases (e.g. "NBA",
"touchdown", "wicket" — see `SPORT_ALIASES`) literally appears in your
question. It won't catch every phrasing (a question that never says a
sport's name or a known alias searches across everything), but it's easy
to extend: just add more aliases to `SPORT_ALIASES` as you notice gaps, or
pick a specific sport from the dropdown to force it.

**Cross-encoder re-ranking**: retrieval is now two stages instead of one.
First, a fast vector search pulls a wider pool of candidates
(`RERANK_CANDIDATES`, default 15) — this is the same embedding-similarity
search as before, just casting a wider net. Then a second, smaller model
(a "cross-encoder," `RERANK_MODEL_NAME` in `src/config.py`) reads the
question together with each candidate chunk and scores how relevant that
specific pairing really is, and only the true top `TOP_K` (default 4)
survive into the prompt. Vector search alone can rank a chunk highly just
because it uses similar words; the cross-encoder catches cases where
wording overlaps but meaning doesn't, which matters more as your document
collection grows. If answers still miss the right chunk, try raising
`RERANK_CANDIDATES` so the re-ranker has more to choose from.

### More features in the app (sources, feedback)

Every answer in the Streamlit UI now comes with an expandable **Sources**
section showing the sport, file, and page each cited chunk came from —
so you can sanity-check where an answer is grounded instead of just
trusting it.

There's also a **👍 / 👎** under each answer. Clicking one appends a row
(timestamp, sport filter, question, answer, rating) to a local
`feedback.csv` file via `log_feedback()` in `app.py`. This is a first,
lightweight step toward real "RAG evaluation" — over time you could
review the 👎 rows to spot patterns (a sport that's consistently
under-served, a chunk size that's cutting off answers) and use them to
tune `CHUNK_SIZE`, `TOP_K`, `RERANK_CANDIDATES`, or the system prompt.
Note that on an ephemeral free host (like a Streamlit Community Cloud
instance that sleeps and restarts), `feedback.csv` won't persist forever
— treat it as a local/demo feature unless you wire up external storage
(a small hosted database, a Google Sheet, etc.).

## Other things worth experimenting with

- **Chunk size/overlap** (`CHUNK_SIZE`, `CHUNK_OVERLAP` in `src/config.py`):
  smaller chunks retrieve more precisely but lose surrounding context;
  larger chunks do the opposite.
- **Swap the embedding model**: try `BAAI/bge-base-en-v1.5` for better
  quality at the cost of speed, or a multilingual model if you want
  non-English rulebooks.
- **Swap the LLM**: any model pulled via `ollama pull <name>` works —
  just update `OLLAMA_MODEL` in `src/config.py`.
- **Formalize evaluation**: write a small set of test questions with
  known correct answers and check whether the right chunk shows up in
  the top-k results, or build a small dashboard over `feedback.csv` — a
  natural next step now that feedback is being logged.

## Troubleshooting

- **"Something went wrong talking to Ollama"**: make sure `ollama serve`
  is running and that you've pulled the model named in `src/config.py`.
- **Out of memory / very slow generation**: switch to a smaller model
  (`llama3.2:3b` or `phi3:mini`) in `src/config.py`.
- **"No documents found"**: check that files are directly inside
  `data/<sport>/`, not in a further subfolder, and that they end in
  `.pdf`, `.txt`, or `.md`.
- **Answers ignore a document you added**: did you re-run
  `python src/build_index.py` after adding it? The index only reflects
  whatever existed the last time you built it.
- **A PDF shows "0 chunks" or prints `[!] Skipping <file>: cryptography>=3.1 is required...`**:
  some official documents (this project hit it with FIBA's basketball rules)
  are AES-encrypted purely to restrict editing/printing, not to hide the
  content — `pypdf` needs the separate `cryptography` package to open those.
  It's already listed in `requirements.txt`; just make sure you've run
  `pip install -r requirements.txt` after pulling the latest code, then
  re-run `python src/build_index.py`.
- **"Something went wrong talking to Groq" / a 404 error mentioning
  `chat/completions`**: this almost always means the model name in
  `src/config.py` (`GROQ_MODEL`) has been deprecated — Groq retires
  free-tier models periodically (this project hit it once already, when
  `llama-3.1-8b-instant` was shut down for free-tier use and the default
  was switched to `openai/gpt-oss-20b`). Check the current free-tier list
  at [console.groq.com/docs/models](https://console.groq.com/docs/models),
  then update `GROQ_MODEL` in `src/config.py` or override it via
  `GROQ_MODEL=<model-name>` in `.env` / Streamlit secrets. Also double
  check `GROQ_API_KEY` is actually set (env var, `.env` file, or Streamlit
  secret).
- **First question after adding documents feels slow**: the first call
  also downloads the cross-encoder re-ranking model (~90MB, one-time,
  cached afterward) alongside the embedding model — this is normal.
- **The bot searched the wrong sport / mixed sports in the sources**:
  auto-detection only matches a sport's name or a listed alias
  (`SPORT_ALIASES` in `src/rag_chat.py`) literally appearing in your
  question — rephrase to include the sport's name, add an alias, or pick
  the sport directly from the sidebar dropdown.
- **`feedback.csv` isn't showing up**: it's only created the first time
  someone clicks 👍 or 👎 in the web UI, and it lives next to `app.py`
  on whatever machine is currently running it (so on a cloud deployment,
  not on your own laptop).
