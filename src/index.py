"""
Builds the two search indexes over data/chunks.jsonl:
  - vector index (semantic search): bge-small embeddings stored in Chroma (data/chroma/)
  - keyword index (BM25): tokenized chunks saved to data/bm25.pkl
Rebuilds everything from scratch on every run.
"""
import json
import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/chunks.jsonl")
CHROMA_DIR = "data/chroma"
BM25_FILE = Path("data/bm25.pkl")
COLLECTION = "chunks"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models work best when the question (not the documents) starts with this instruction
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "at", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "i", "you", "your", "we", "our", "my", "me", "do", "does", "did", "can", "will",
    "would", "should", "could", "have", "has", "had", "not", "no", "so", "what", "how", "when",
    "which", "who", "why", "there", "their", "they", "them", "about", "into", "than", "then",
}

METADATA_FIELDS = ("doc_id", "source", "url", "title", "section", "date", "position")


def tokenize(text):
    """Lowercase words and numbers, without stopwords. Used for BM25 (here and in search.py)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def load_chunks():
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vector_index(chunks, model):
    embeddings = model.encode(
        [c["context_text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,    # unit-length vectors: distance then depends only on the angle
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)   # start fresh every time
    except Exception:
        pass                                   # first run: nothing to delete
    collection = client.create_collection(COLLECTION)

    # Chroma does not accept None in metadata, so empty values become ""
    metadatas = [{k: ("" if c.get(k) is None else c[k]) for k in METADATA_FIELDS} for c in chunks]

    batch = 500
    for start in range(0, len(chunks), batch):
        end = start + batch
        collection.add(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            embeddings=embeddings[start:end].tolist(),
            documents=[c["text"] for c in chunks[start:end]],
            metadatas=metadatas[start:end],
        )
    return collection


def build_bm25(chunks):
    data = {"ids": [c["chunk_id"] for c in chunks],
            "tokens": [tokenize(c["context_text"]) for c in chunks]}
    with BM25_FILE.open("wb") as f:
        pickle.dump(data, f)
    return BM25Okapi(data["tokens"]), data["ids"]


def main():
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    print(f"chunks: {len(chunks)}")

    print(f"loading embedding model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    collection = build_vector_index(chunks, model)
    print(f"vector index: {collection.count()} chunks in {CHROMA_DIR}")

    bm25, ids = build_bm25(chunks)
    print(f"BM25 index:   {len(ids)} chunks in {BM25_FILE}")

    # quick sanity check: the same question through both indexes
    question = "Can Kardia detect a heart attack?"
    print(f"\nsanity check: {question!r}")

    q_emb = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=3)
    print("  vector top 3:")
    for chunk_id in result["ids"][0]:
        print(f"    {by_id[chunk_id]['title'][:70]}")

    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(ids)), key=lambda i: -scores[i])[:3]
    print("  BM25 top 3:")
    for i in top:
        print(f"    {by_id[ids[i]]['title'][:70]}")


if __name__ == "__main__":
    main()