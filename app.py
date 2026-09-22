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
# One emoji + one accent color per sport, reused everywhere (the sidebar
# dropdown, source citations, example questions) so the same sport always
# reads the same way at a glance. New sport folders you add fall back to a
# generic medal icon/gray until you give them their own entry here.
SPORT_ICONS = {
    "cricket": "🏏",
    "soccer": "⚽",
    "basketball": "🏀",
    "nfl": "🏈",
    "tennis": "🎾",
}
SPORT_COLORS = {
    "cricket": "#0F766E",
    "soccer": "#16A34A",
    "basketball": "#EA580C",
    "nfl": "#7C3AED",
    "tennis": "#CA8A04",
}
DEFAULT_ICON = "🏅"
DEFAULT_COLOR = "#475569"


def sport_icon(sport: str) -> str:
    return SPORT_ICONS.get(sport.lower(), DEFAULT_ICON)


def sport_color(sport: str) -> str:
    return SPORT_COLORS.get(sport.lower(), DEFAULT_COLOR)


def sport_label(sport: str) -> str:
    """Used as the selectbox's format_func -- turns 'basketball' into '🏀 Basketball'."""
    if sport == "All":
        return "🔎 All sports"
    return f"{sport_icon(sport)} {sport.capitalize()}"


# A little CSS polish -- none of this changes behavior, just spacing/styling.
# unsafe_allow_html is safe here because the HTML is fixed by us, not built
# from user input.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 850px; }
    section[data-testid="stSidebar"] button { text-align: left; }
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        vertical-align: middle;
    }
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
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            page_str = f" · p.{s['page']}" if s.get("page") else ""
            color = sport_color(s["sport"])
            icon = sport_icon(s["sport"])
            st.markdown(
                f"<span class='source-badge' style='background-color:{color}'>"
                f"{icon} {s['sport'].capitalize()}</span>"
                f"<strong>{s['source']}{page_str}</strong>",
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
    st.header("⚙️ Settings")
    sports = ["All"] + list_sports()
    selected_sport = st.selectbox("Filter by sport", sports, index=0, format_func=sport_label)
    st.caption("'All' auto-detects the sport from your question's wording; pick one to force it.")
    if LLM_BACKEND == "groq":
        st.caption(f"Model: `{GROQ_MODEL}` (via Groq's free hosted API)")
    else:
        st.caption(f"Model: `{OLLAMA_MODEL}` (via local Ollama)")
    st.caption("Change the backend/model in src/config.py")

    st.divider()

    with st.expander("ℹ️ How to use this bot"):
        st.markdown(
            "- Ask in plain English -- no special syntax needed.\n"
            "- Leave the sport filter on **All sports** and the bot will "
            "usually guess the right sport from your wording.\n"
            "- Every answer has an expandable **Sources** section showing "
            "exactly which document and page it came from.\n"
            "- Use 👍 / 👎 under an answer to log feedback for later review."
        )

    st.divider()
    st.subheader("💡 Try an example")
    example_questions = [
        ("cricket", "What counts as being stumped in cricket?"),
        ("soccer", "Explain the offside rule in soccer"),
        ("basketball", "What is traveling in basketball?"),
        ("nfl", "What is a false start in the NFL?"),
        ("tennis", "What is a let in tennis?"),
    ]
    for sport_key, example_q in example_questions:
        if st.button(
            f"{sport_icon(sport_key)} {example_q}",
            key=f"example_{sport_key}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example_q
            st.rerun()

    st.divider()
    asked_count = len(st.session_state.get("messages", [])) // 2
    st.caption(f"💬 {asked_count} question{'s' if asked_count != 1 else ''} asked this session")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []

st.title("🏆 Sports Rules & Strategy Bot")
st.caption("Ask about rules, scoring, or strategy for cricket, soccer, basketball, NFL, or tennis.")

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
