"""
The assistant: hybrid search -> LLM answer with citations, or "I don't know".

Two layers of protection against made-up answers:
  1. if even the closest chunk is not similar enough, refuse without calling the LLM
  2. the LLM may only use the numbered sources and must cite them; an answer
     without valid citations is not shown

LLM backend (set with environment variables):
  LLM_BACKEND=ollama  (default) local model, OLLAMA_MODEL (default qwen2.5:7b)
  LLM_BACKEND=groq    hosted API, needs GROQ_API_KEY; GROQ_MODEL (default llama-3.3-70b-versatile)

Try it:  python src/assistant.py
"""
import os
import re
import time

import requests

from search import HybridSearch

TOP_K = 5                 # chunks given to the LLM
SIM_THRESHOLD = 0.6       # below this, the question is off-topic: refuse right away
IDK = "I don't know. I couldn't find this in AliveCor's content."

BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You answer questions about AliveCor's Kardia devices and heart health, \
using ONLY the numbered sources you are given.

Rules:
- If the user describes severe symptoms happening now (such as chest pain, fainting, \
severe shortness of breath, or signs of a stroke), reply exactly: \
"This could be an emergency. Call your local emergency number (112 in Europe, 911 in the US) now."
- Use only information from the sources. Never use outside knowledge.
- After each sentence, cite the source(s) it comes from, like [1] or [2][3].
- If the question asks for personal medical advice (whether to take, stop or change a medication \
or treatment, or what a result means for the user's own health), start with exactly: \
"I can't give medical advice. Please talk to your doctor." Then, if the sources contain relevant \
general information (for example, what a result means in general), add it in 1 to 3 sentences, \
with citations. Never reply "I don't know" to such a question.
- Otherwise, if the sources do not contain the answer, reply exactly: I don't know.
- If the question is about symptoms, add that the user should talk to a doctor.
- Keep the answer short: 2 to 5 sentences."""

MEDICAL = "I can't give medical advice. Please talk to your doctor."
EMERGENCY = "This could be an emergency. Call your local emergency number (112 in Europe, 911 in the US) now."


def generate(system, user):
    """Sends the prompt to the chosen LLM and returns its reply as text."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if BACKEND == "ollama":
        import ollama
        reply = ollama.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": 0})
        return reply["message"]["content"]

    if BACKEND == "groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    raise ValueError(f"unknown LLM_BACKEND: {BACKEND!r} (use 'ollama' or 'groq')")


def build_prompt(question, chunks):
    sources = "\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(chunks, start=1))
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def refuse(reason, best_similarity, chunks=()):
    return {"answer": IDK, "sources": [], "abstained": True, "reason": reason,
            "best_similarity": best_similarity, "retrieved": [c["chunk_id"] for c in chunks]}


def answer(question, searcher):
    found = searcher.search(question, k=TOP_K)
    chunks, best = found["results"], found["best_similarity"]

    # layer 1: off-topic question
    if best is None or best < SIM_THRESHOLD:
        return refuse("low similarity", best, chunks)

    reply = generate(SYSTEM_PROMPT, build_prompt(question, chunks)).strip()

    # safety first: a possible emergency gets the emergency message and nothing else
    if reply.lower().startswith("this could be an emergency"):
        return {**refuse("emergency", best, chunks), "answer": EMERGENCY}

    # layer 2: the LLM says the sources don't contain the answer
    if reply.lower().startswith("i don't know") or reply.lower().startswith("i do not know"):
        return refuse("not in sources", best, chunks)

    # citations like [2] must point to sources that exist; made-up numbers are removed from the text
    reply = re.sub(r"\s?\[(\d+)\]",
                   lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(chunks) else "", reply)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply)})
    if not cited:
        # a medical-advice redirect with nothing to cite is still a safe, useful reply
        if reply.lower().startswith("i can't give medical advice"):
            return {**refuse("medical advice", best, chunks), "answer": MEDICAL}
        return refuse("no citations", best, chunks)

    # several chunks can come from the same page: list each page once, with all its numbers
    by_url = {}
    for n in cited:
        c = chunks[n - 1]
        by_url.setdefault(c["url"], {"n": [], "title": c["title"], "url": c["url"]})["n"].append(n)
    sources = list(by_url.values())
    return {"answer": reply, "sources": sources, "abstained": False, "reason": None,
            "best_similarity": best, "retrieved": [c["chunk_id"] for c in chunks]}


if __name__ == "__main__":
    print(f"LLM: {BACKEND} ({OLLAMA_MODEL if BACKEND == 'ollama' else GROQ_MODEL})")
    searcher = HybridSearch()
    while question := input("\nquestion (empty to quit): ").strip():
        start = time.time()
        result = answer(question, searcher)
        print(f"\n{result['answer']}")
        for s in result["sources"]:
            print(f"  {''.join(f'[{n}]' for n in s['n'])} {s['title']} - {s['url']}")
        if result["abstained"]:
            print(f"  (refused: {result['reason']}, best similarity {result['best_similarity']})")
        print(f"  ({time.time() - start:.1f}s)")