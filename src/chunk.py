"""
Splits the cleaned documents into chunks for search.
Input:  data/docs_dedup.jsonl
Output: data/chunks.jsonl (one chunk per line, with the metadata of its document)
"""
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

DOCS = Path("data/docs_dedup.jsonl")
OUT = Path("data/chunks.jsonl")

MAX_WORDS = 300          # a chunk is at most ~300 words
MIN_WORDS = 80           # a new section starts a new chunk only if the current one has 80+ words
HEADING_MAX_WORDS = 12   # a short line without final punctuation is treated as a heading


def n_words(text):
    return len(text.split())


def is_heading(line):
    return n_words(line) <= HEADING_MAX_WORDS and not line.endswith((".", "!", "?", ":", ";", ","))


def split_long(paragraph):
    """Splits a paragraph longer than MAX_WORDS at the ends of sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current = [], []
    for s in sentences:
        if current and n_words(" ".join(current + [s])) > MAX_WORDS:
            parts.append(" ".join(current))
            current = []
        current.append(s)
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(text):
    # 1. paragraphs: one per line, very long ones split at sentence ends
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        paragraphs.extend(split_long(line) if n_words(line) > MAX_WORDS else [line])

    # 2. group paragraphs into chunks
    chunks, current = [], []
    for p in paragraphs:
        size = n_words("\n".join(current))
        starts_section = is_heading(p) and size >= MIN_WORDS
        too_big = bool(current) and size + n_words(p) > MAX_WORDS
        if starts_section or too_big:
            # don't leave a heading at the end of a chunk, separated from its text
            carry = [current.pop()] if too_big and len(current) > 1 and is_heading(current[-1]) else []
            chunks.append("\n".join(current))
            current = carry
        current.append(p)

    # 3. a very small last piece is added to the previous chunk
    if current:
        last = "\n".join(current)
        if chunks and n_words(last) < MIN_WORDS // 2:
            chunks[-1] += "\n" + last
        else:
            chunks.append(last)
    return chunks


docs = [json.loads(line) for line in DOCS.read_text(encoding="utf-8").splitlines() if line.strip()]

chunks = []
for d in docs:
    header = f"{d['title']} ({d['section']})" if d.get("section") else d["title"]
    for i, body in enumerate(chunk_text(d["text"])):
        chunks.append({
            "chunk_id": f"{d['id']}#{i}",
            "doc_id": d["id"],
            "source": d["source"],
            "url": d["url"],
            "title": d["title"],
            "section": d.get("section"),
            "date": d.get("date"),
            "position": i,
            "text": body,                              # shown to the LLM and the user
            "context_text": f"{header}\n\n{body}",     # used for search: knows what it belongs to
        })

with OUT.open("w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

sizes = [n_words(c["text"]) for c in chunks]
per_doc = Counter(c["doc_id"] for c in chunks)
print(f"documents:           {len(docs)}")
print(f"chunks:              {len(chunks)}")
print(f"chunks per source:   {dict(Counter(c['source'] for c in chunks))}")
print(f"words per chunk:     avg {mean(sizes):.0f}, min {min(sizes)}, max {max(sizes)}")
print(f"docs split in 2+:    {sum(n > 1 for n in per_doc.values())}")

example = max(chunks, key=lambda c: per_doc[c["doc_id"]])
print("\nexample chunk:\n" + "-" * 60)
print(example["context_text"][:600])