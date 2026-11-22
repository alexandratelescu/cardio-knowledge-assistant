"""
Web interface for the assistant.

Run from the project folder:
    streamlit run app.py
"""
import html
import re
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))   # so the modules in src/ can be imported
import assistant  # noqa: E402
from search import HybridSearch  # noqa: E402

GITHUB_URL = "https://github.com/alexandratelescu/cardio-knowledge-assistant"
EXAMPLES = [
    "Can Kardia detect a heart attack?",
    "How do I change the battery in KardiaMobile 6L?",
    "Does coffee affect AFib?",
]

MODEL = assistant.OLLAMA_MODEL if assistant.BACKEND == "ollama" else assistant.GROQ_MODEL

st.set_page_config(page_title="Cardio Knowledge Assistant", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stMarkdown, .stButton button, .stChatInput textarea { font-family: 'Inter', sans-serif; }
.block-container { max-width: 820px; padding-top: 2.5rem; }

.hero-eyebrow { font-size: .76rem; letter-spacing: .12em; text-transform: uppercase;
                color: #D7263D; font-weight: 600; }
.hero-title { font-size: 2.1rem; font-weight: 700; color: #1B2430; margin: .15rem 0 .35rem; line-height: 1.15; }
.hero-sub { color: #5B6675; font-size: 1rem; margin-bottom: 1rem; }
.stats { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .8rem; }
.stat { background: #F6F7F9; border: 1px solid #E6E9EE; border-radius: 999px;
        padding: .28rem .8rem; font-size: .8rem; color: #3A4554; }
a.stat-link { text-decoration: none !important; color: #3A4554 !important; }
a.stat-link:hover { border-color: #D7263D; color: #D7263D !important; }
.disclaimer { font-size: .76rem; color: #8A94A3; margin-bottom: 1.6rem; }

.stButton button { border-radius: 999px; border: 1px solid #E6E9EE; background: #fff;
                   color: #3A4554; font-size: .84rem; padding: .45rem .9rem; }
.stButton button:hover { border-color: #D7263D; color: #D7263D; }

.cite { display: inline-block; background: #FDECEE; color: #D7263D; font-size: .68rem; font-weight: 600;
        border-radius: 4px; padding: 0 .32rem; margin-left: .12rem; vertical-align: super; line-height: 1.25; }
.sources-label { font-size: .72rem; letter-spacing: .09em; text-transform: uppercase;
                 color: #8A94A3; font-weight: 600; margin: 1rem 0 .3rem; }
a.source-card { display: flex; gap: .75rem; align-items: flex-start; border: 1px solid #E6E9EE;
                border-radius: 10px; padding: .6rem .8rem; margin: .35rem 0; background: #fff;
                text-decoration: none !important; transition: border-color .15s, box-shadow .15s; }
a.source-card:hover { border-color: #D7263D; box-shadow: 0 2px 10px rgba(215, 38, 61, .08); }
.source-num { background: #FDECEE; color: #D7263D; font-weight: 600; font-size: .76rem;
              border-radius: 6px; padding: .08rem .45rem; white-space: nowrap; }
.source-title { color: #1B2430; font-weight: 500; font-size: .9rem; line-height: 1.3; }
.source-kind { color: #8A94A3; font-size: .74rem; margin-top: .12rem; }
.refusal { border-left: 3px solid #B8C0CC; background: #F6F7F9; padding: .75rem .95rem;
           border-radius: 6px; color: #3A4554; }
.refusal small { display: block; color: #8A94A3; margin-top: .3rem; }
.emergency { border-left: 3px solid #D7263D; background: #FDECEE; padding: .75rem .95rem;
             border-radius: 6px; color: #8E1526; }
.meta { font-size: .74rem; color: #8A94A3; margin-top: .7rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource            # load the model and indexes once, not on every question
def load_searcher():
    return HybridSearch()


REASONS = {
    "low similarity": "The question seems to be outside AliveCor's content.",
    "not in sources": "The closest passages don't contain the answer.",
    "no citations": "An answer was drafted but couldn't be tied to a source, so it isn't shown.",
}


def source_kind(url):
    if "zendesk.com" in url:
        return "Help Center"
    if "/blog/" in url:
        return "Kardia Blog"
    return "kardia.com"


def render(result, seconds=None):
    if result["abstained"] and result["reason"] == "emergency":
        st.markdown(f'<div class="emergency">🚨 <b>{html.escape(result["answer"])}</b></div>',
                    unsafe_allow_html=True)
    elif result["abstained"] and result["reason"] == "medical advice":
        st.markdown(f'<div class="refusal"><b>{html.escape(result["answer"])}</b></div>',
                    unsafe_allow_html=True)
    elif result["abstained"]:
        # on-topic questions without an answer may still be about someone's health
        doctor = ("<br>For questions about your symptoms, results or medication, please talk to your doctor."
                  if result["reason"] != "low similarity" else "")
        st.markdown(
            f'<div class="refusal"><b>I don\'t know.</b> I couldn\'t find this in AliveCor\'s content.{doctor}'
            f'<small>{REASONS.get(result["reason"], "")}</small></div>',
            unsafe_allow_html=True,
        )
    else:
        # citation numbers like [2] become small badges
        text = re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', html.escape(result["answer"]))
        st.markdown(text, unsafe_allow_html=True)

        cards = "".join(
            f'<a class="source-card" href="{html.escape(s["url"])}" target="_blank">'
            f'<span class="source-num">{"·".join(str(n) for n in s["n"])}</span>'
            f'<div><div class="source-title">{html.escape(s["title"])}</div>'
            f'<div class="source-kind">{source_kind(s["url"])}</div></div></a>'
            for s in result["sources"]
        )
        st.markdown(f'<div class="sources-label">Sources</div>{cards}', unsafe_allow_html=True)

    meta = f"{MODEL} · best match similarity {result['best_similarity']}"
    if seconds is not None:
        meta = f"{seconds:.1f}s · " + meta
    st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)


searcher = load_searcher()
n_docs = len({c["doc_id"] for c in searcher.chunks.values()})

st.markdown(f"""
<div class="hero-eyebrow">AliveCor · Kardia knowledge base</div>
<div class="hero-title">Cardio Knowledge Assistant</div>
<div class="hero-sub">Ask about Kardia EKG devices and heart health. Every answer cites its sources.</div>
<div class="stats">
  <span class="stat">{n_docs} documents</span>
  <span class="stat">{len(searcher.chunks)} passages</span>
  <a class="stat stat-link" href="{GITHUB_URL}" target="_blank">Code on GitHub ↗</a>
</div>
<div class="disclaimer">Not medical advice. Always talk to a doctor about your heart health.</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []          # list of (question, result, seconds), kept between reruns

for question, result, seconds in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render(result, seconds)

clicked = None
examples = st.empty()                      # a slot we can clear as soon as a question comes in
if not st.session_state.history:           # example questions only on an empty chat
    with examples.container():
        for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            if column.button(example, use_container_width=True):
                clicked = example

question = st.chat_input("Ask a question about Kardia or heart health...") or clicked
if question:
    examples.empty()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching AliveCor's content..."):
            start = time.time()
            result = assistant.answer(question, searcher)
            seconds = time.time() - start
        render(result, seconds)
    st.session_state.history.append((question, result, seconds))