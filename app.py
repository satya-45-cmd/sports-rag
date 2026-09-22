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

st.set_page_config(page_title="Sports Rules & Strategy Bot", page_icon="🏆")

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
            page_str = f", p.{s['page']}" if s.get("page") else ""
            st.markdown(f"**{s['sport']} / {s['source']}{page_str}**")
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
    st.header("Settings")
    sports = ["All"] + list_sports()
    selected_sport = st.selectbox("Filter by sport", sports, index=0)
    st.caption("'All' auto-detects the sport from your question's wording; pick one to force it.")
    if LLM_BACKEND == "groq":
        st.caption(f"Model: `{GROQ_MODEL}` (via Groq's free hosted API)")
    else:
        st.caption(f"Model: `{OLLAMA_MODEL}` (via local Ollama)")
    st.caption("Change the backend/model in src/config.py")

    if st.button("Clear chat"):
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

question = st.chat_input("Ask a question about the rules or strategy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = engine.ask(question, sport=selected_sport)
                answer = result["answer"]
                sources = result["sources"]
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
