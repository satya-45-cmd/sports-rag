"""
Streamlit chat UI for the multi-sport rules & strategy RAG bot.

Run with:
    streamlit run app.py

By default this uses local Ollama (LLM_BACKEND=ollama in src/config.py) --
fully free and offline, but only reachable while your own machine is on.
Set LLM_BACKEND=groq (env var, .env file, or Streamlit Cloud secret) to use
Groq's free hosted API instead, which is what public deployments need.
See README.md for the full "deploy so anyone can use it" walkthrough.
"""

import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

# If deployed on Streamlit Community Cloud, secrets set in the app's dashboard
# (Settings -> Secrets) arrive via st.secrets, not environment variables.
# Copy them into the environment so config.py (which is plain Python, no
# Streamlit dependency) picks them up the same way it would locally.
# Locally (no secrets.toml file at all -- the normal case if you're just
# using Ollama) newer Streamlit versions raise instead of treating st.secrets
# as empty, so this is wrapped in a try/except: no secrets file just means
# "nothing to copy," not an error.
try:
    for _key in ("LLM_BACKEND", "GROQ_API_KEY", "GROQ_MODEL"):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from config import list_sports, LLM_BACKEND, OLLAMA_MODEL, GROQ_MODEL, CHROMA_DIR
from rag_chat import RagEngine

st.set_page_config(page_title="Sports Rules & Strategy Bot", page_icon="🏆", layout="centered")

# --- Look & feel -----------------------------------------------------------
# One emoji + one accent color per sport, reused everywhere (the sport pill
# selector, source citations, example questions) so the same sport always
# reads the same way at a glance. New sport folders you add fall back to a
# generic medal icon/gray until you give them their own entry here.
# Colors are saturated and bright -- against the site's near-black,
# stadium-at-night background (see the CSS block below) they need the extra
# vibrancy to read clearly, unlike the muted ink tones a light background
# would call for.
SPORT_ICONS = {
    "cricket": "🏏",
    "soccer": "⚽",
    "basketball": "🏀",
    "nfl": "🏈",
    "tennis": "🎾",
}
SPORT_COLORS = {
    "cricket": "#2DD4BF",
    "soccer": "#4ADE80",
    "basketball": "#FB923C",
    "nfl": "#A78BFA",
    "tennis": "#EAB308",
}
DEFAULT_ICON = "🏅"
DEFAULT_COLOR = "#94A3B8"


def sport_icon(sport: str) -> str:
    return SPORT_ICONS.get(sport.lower(), DEFAULT_ICON)


def sport_color(sport: str) -> str:
    return SPORT_COLORS.get(sport.lower(), DEFAULT_COLOR)


def sport_label(sport: str) -> str:
    """Used as the selectbox's format_func -- turns 'basketball' into '🏀 Basketball'."""
    if sport == "All":
        return "🔎 All sports"
    return f"{sport_icon(sport)} {sport.capitalize()}"


# The site's visual theme: a dark, cinematic "night game under the
# floodlights" look -- a near-black background lit by soft amber glows near
# the top of the page and darkened toward a vignette at the edges, a serif
# display face for headings (Playfair Display) paired with a clean
# sans-serif for body text (Inter), and a warm amber accent standing in for
# stadium floodlights. There is no photograph here: the atmosphere is built
# entirely from CSS gradients, which means it never depends on an external
# image URL that could go missing or need licensing, and it can't ever show
# up broken on a fresh deploy. The actual page background/accent colors come
# from .streamlit/config.toml (Streamlit's native theme, which also colors
# built-in widgets like buttons and the chat input); this CSS layers
# typography, glow, cards, and hover states on top of that using stable
# data-testid selectors rather than Streamlit's internal (and
# frequently-changing) generated class names.
# unsafe_allow_html is safe here because the HTML is fixed by us, not built
# from user input.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #F4F1EA;
        --muted-ink: #9BA3AE;
        --accent: #FFB020;
        --accent-dim: rgba(255, 176, 32, 0.35);
        --gold: #E8C766;
        --hairline: rgba(244, 241, 234, 0.10);
        --card-bg: #12151B;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 860px; }

    /* -- Stadium-at-night atmosphere ------------------------------------ */
    /* Three soft floodlight glows near the top of the page over a
       near-black base, fading into a dark vignette toward the bottom --
       entirely gradients, no image file involved. */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 800px 480px at 12% -8%, rgba(255,176,32,0.14), transparent 60%),
            radial-gradient(ellipse 900px 520px at 50% -12%, rgba(255,176,32,0.11), transparent 65%),
            radial-gradient(ellipse 800px 480px at 88% -8%, rgba(255,176,32,0.14), transparent 60%),
            radial-gradient(ellipse 1300px 800px at 50% 105%, rgba(0,0,0,0.65), transparent 70%),
            #06070A;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* -- Hero header --------------------------------------------------- */
    .hero-mark {
        font-size: 2.1rem;
        line-height: 1;
        margin-bottom: 0.35rem;
        filter: drop-shadow(0 0 14px var(--accent-dim));
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2.6rem;
        letter-spacing: -0.01em;
        color: var(--ink);
        margin: 0;
        text-shadow: 0 0 26px rgba(255, 176, 32, 0.30);
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        color: var(--muted-ink);
        font-size: 1.02rem;
        margin-top: 0.35rem;
    }
    .hero-rule {
        border: none;
        border-top: 2px solid var(--gold);
        width: 64px;
        margin: 1.1rem 0 1.6rem 0;
        box-shadow: 0 0 8px var(--accent-dim);
    }

    /* -- Sport pill selector (main area) --------------------------------- */
    .pill-row-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted-ink);
        margin: 0 0 0.6rem 0;
    }

    /* -- Sidebar --------------------------------------------------------- */
    section[data-testid="stSidebar"] button { text-align: left; }
    .sidebar-brand {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
        color: var(--ink);
        text-transform: uppercase;
        border-bottom: 2px solid var(--gold);
        padding-bottom: 0.6rem;
        margin-bottom: 1.1rem;
    }
    .section-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted-ink);
        margin: 0 0 0.5rem 0;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: none;
        border-bottom: 1px solid var(--hairline);
        border-radius: 0;
        padding: 0.55rem 0.1rem;
        font-size: 0.88rem;
        color: var(--ink);
        transition: color 0.15s ease, padding-left 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        color: var(--accent);
        padding-left: 0.4rem;
        background: transparent;
        border-bottom: 1px solid var(--accent);
    }

    /* -- Buttons (main area) --------------------------------------------- */
    .stButton > button {
        border-radius: 999px;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 14px rgba(0,0,0,0.35);
    }
    /* Native Streamlit button "kind" -- used to give the sport pill row a
       true active/inactive look (primary = lit up, secondary = dim) without
       any extra JS or a Streamlit version bump; both markups (older
       data-testid, newer kind attribute) are covered for resilience. */
    div[data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
    div[data-testid="stAppViewContainer"] [data-testid="baseButton-primary"] {
        box-shadow: 0 0 18px var(--accent-dim);
        border: none;
    }
    div[data-testid="stAppViewContainer"] .stButton > button[kind="secondary"],
    div[data-testid="stAppViewContainer"] [data-testid="baseButton-secondary"] {
        background: rgba(244, 241, 234, 0.04);
        border: 1px solid var(--hairline);
        color: var(--ink);
    }
    div[data-testid="stAppViewContainer"] .stButton > button[kind="secondary"]:hover,
    div[data-testid="stAppViewContainer"] [data-testid="baseButton-secondary"]:hover {
        border-color: var(--accent-dim);
        color: var(--accent);
    }

    /* -- Chat messages as cards -------------------------------------------- */
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border: 1px solid var(--hairline);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    }

    /* -- Source citations -------------------------------------------------- */
    [data-testid="stExpander"] {
        border: 1px solid var(--hairline);
        border-radius: 12px;
        background: var(--card-bg);
    }
    .source-badge {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        color: #06070A;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 8px;
        vertical-align: middle;
    }
    .source-cite {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: var(--ink);
    }

    /* -- Misc -------------------------------------------------------------- */
    hr { border-color: var(--hairline); }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: var(--hairline); border-radius: 999px; }
    ::-webkit-scrollbar-track { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# First run on a fresh deployment (or a fresh clone) won't have a chroma_db
# folder yet -- build it automatically instead of making the visitor guess
# why the bot has no answers.
if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    with st.spinner("Building the knowledge index for the first time (only happens once)..."):
        from build_index import build_index
        build_index()


FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "feedback.csv")


def log_feedback(question, answer, sport_filter, rating):
    """
    Append one row to a local feedback.csv. This is a first, simple step
    toward "RAG evaluation" (see README) -- over time you can look at which
    questions got thumbs-down and use them to tune chunk size, TOP_K, or the
    prompt. Note: on an ephemeral host (like a free Streamlit Cloud instance
    that goes to sleep/restarts), this file doesn't persist forever -- treat
    it as a local/demo feature unless you wire up external storage.
    """
    file_exists = os.path.isfile(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "sport_filter", "question", "answer", "rating"])
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), sport_filter, question, answer, rating]
        )


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page_str = f" · p.{s['page']}" if s.get("page") else ""
            color = sport_color(s["sport"])
            icon = sport_icon(s["sport"])
            st.markdown(
                f"<span class='source-badge' style='background-color:{color}'>"
                f"{icon} {s['sport'].capitalize()}</span>"
                f"<span class='source-cite'>{s['source']}{page_str}</span>",
                unsafe_allow_html=True,
            )
            st.caption(s.get("text", ""))
            st.divider()


def render_feedback(idx, msg):
    if msg.get("rated"):
        icon = "👍" if msg["rated"] == "up" else "👎"
        st.caption(f"Feedback recorded: {icon}")
        return

    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"up_{idx}"):
            log_feedback(msg["question"], msg["content"], msg["sport_filter"], "up")
            msg["rated"] = "up"
            st.rerun()
    with col2:
        if st.button("👎", key=f"down_{idx}"):
            log_feedback(msg["question"], msg["content"], msg["sport_filter"], "down")
            msg["rated"] = "down"
            st.rerun()


@st.cache_resource
def get_engine():
    return RagEngine()


engine = get_engine()

with st.sidebar:
    st.markdown("<div class='sidebar-brand'>Sports Rules Bot</div>", unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Settings</p>", unsafe_allow_html=True)
    st.caption("Pick a sport from the pills up top, or leave it on All sports to auto-detect.")
    if LLM_BACKEND == "groq":
        st.caption(f"Model: `{GROQ_MODEL}` (via Groq's free hosted API)")
    else:
        st.caption(f"Model: `{OLLAMA_MODEL}` (via local Ollama)")
    st.caption("Change the backend/model in src/config.py")

    st.divider()

    with st.expander("How to use this bot"):
        st.markdown(
            "- Ask in plain English -- no special syntax needed.\n"
            "- Leave the sport filter on **All sports** and the bot will "
            "usually guess the right sport from your wording.\n"
            "- Every answer has an expandable **Sources** section showing "
            "exactly which document and page it came from.\n"
            "- Use 👍 / 👎 under an answer to log feedback for later review."
        )

    st.divider()
    st.markdown("<p class='section-label'>Try an example</p>", unsafe_allow_html=True)
    example_questions = [
        ("cricket", "What counts as being stumped in cricket?"),
        ("soccer", "Explain the offside rule in soccer"),
        ("basketball", "What is traveling in basketball?"),
        ("nfl", "What is a false start in the NFL?"),
        ("tennis", "What is a let in tennis?"),
    ]
    for sport_key, example_q in example_questions:
        if st.button(
            f"{sport_icon(sport_key)}  {example_q}",
            key=f"example_{sport_key}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example_q
            st.rerun()

    st.divider()
    asked_count = len(st.session_state.get("messages", [])) // 2
    st.caption(f"{asked_count} question{'s' if asked_count != 1 else ''} asked this session")

    if st.button("Clear conversation"):
        st.session_state.messages = []

st.markdown("<div class='hero-mark'>🏆</div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>Sports Rules &amp; Strategy</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-sub'>Ask about rules, scoring, or strategy for cricket, soccer, "
    "basketball, NFL, or tennis — every answer is grounded in the official rulebooks.</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)

# -- Sport pill selector --------------------------------------------------
# A horizontal row of pill/tab buttons rather than a dropdown. This uses
# Streamlit's native st.button(type="primary"/"secondary") to light up the
# active pill -- no custom JS and no need to bump the pinned Streamlit
# version for a segmented-control widget.
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = "All"

st.markdown("<p class='pill-row-label'>Filter by sport</p>", unsafe_allow_html=True)
pill_options = ["All"] + list_sports()
pill_cols = st.columns(len(pill_options))
for col, sport in zip(pill_cols, pill_options):
    with col:
        is_active = st.session_state.selected_sport == sport
        if st.button(
            sport_label(sport),
            key=f"pill_{sport}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_sport = sport
            st.rerun()
selected_sport = st.session_state.selected_sport
st.caption("'All' auto-detects the sport from your question's wording; pick one to force it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg["role"] == "assistant":
            render_feedback(idx, msg)

typed_question = st.chat_input("Ask a question about the rules or strategy...")
# An example-question button in the sidebar can't fill st.chat_input directly
# (Streamlit doesn't support pre-filling it), so it stashes the question in
# session_state instead and we pick it up here as a fallback.
question = typed_question or st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []
        try:
            # Retrieval (embed the question, vector search, cross-encoder
            # re-rank) happens here and is quick -- the spinner covers that
            # brief gap. The LLM call itself is streamed below via
            # st.write_stream, so the answer appears live, word by word, as
            # Groq/Ollama actually generate it -- no waiting for the whole
            # response before anything shows up.
            with st.spinner("Searching documents..."):
                sources, answer_chunks = engine.ask_stream(question, sport=selected_sport)
            answer = st.write_stream(answer_chunks)
        except Exception as e:
            if LLM_BACKEND == "groq":
                answer = (
                    f"Something went wrong talking to Groq: {e}\n\n"
                    "Check that GROQ_API_KEY is set correctly (env var, "
                    ".env file, or Streamlit secret) and that the model "
                    "name in src/config.py is still valid at "
                    "console.groq.com/docs/models."
                )
            else:
                answer = (
                    f"Something went wrong talking to Ollama: {e}\n\n"
                    "Make sure Ollama is installed and running (`ollama serve`), "
                    "and that the model in src/config.py has been pulled "
                    "(`ollama pull llama3.1:8b`)."
                )
            sources = []
            st.markdown(answer)

        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "question": question,
            "sport_filter": selected_sport,
        }
    )
    st.rerun()  # immediately show the feedback buttons via the history loop above
