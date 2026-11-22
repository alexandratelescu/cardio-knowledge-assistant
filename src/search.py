"""
Hybrid search over the chunks: semantic search (Chroma) + keyword search (BM25),
combined with Reciprocal Rank Fusion (RRF).

Try it from the command line:
    python src/search.py "Can Kardia detect a heart attack?"
    python src/search.py            (then type questions, empty line to quit)
"""
import pickle
import sys

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from index import BM25_FILE, CHROMA_DIR, COLLECTION, MODEL_NAME, QUERY_PREFIX, load_chunks, tokenize

CANDIDATES = 20   # how many results each method contributes before fusion
RRF_K = 60        # standard RRF constant: dampens the gap between rank 1 and rank 2


class HybridSearch:
    def __init__(self):
        self.chunks = {c["chunk_id"]: c for c in load_chunks()}
        self.model = SentenceTransformer(MODEL_NAME)
        self.collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
        with BM25_FILE.open("rb") as f:
            data = pickle.load(f)
        self.bm25_ids = data["ids"]
        self.bm25 = BM25Okapi(data["tokens"])

    def vector_search(self, question, n=CANDIDATES):
        """Returns [(chunk_id, cosine similarity)], best first."""
        emb = self.model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=n)
        # Chroma returns squared L2 distance; for unit vectors d = 2 - 2*cos, so cos = 1 - d/2
        return [(cid, 1 - d / 2) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def keyword_search(self, question, n=CANDIDATES):
        """Returns [(chunk_id, BM25 score)], best first; chunks with no matching word are left out."""
        scores = self.bm25.get_scores(tokenize(question))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [(self.bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def search(self, question, k=5, mode="hybrid"):
        """
        mode: "hybrid" (both methods), "vector" or "bm25" (one method, for comparisons in eval).
        Returns {"results": top k chunks, "best_similarity": cosine of the closest chunk}.
        """
        vec = self.vector_search(question) if mode in ("hybrid", "vector") else []
        kw = self.keyword_search(question) if mode in ("hybrid", "bm25") else []

        # RRF: each list gives every chunk 1 / (RRF_K + its rank); the scores are added up
        fused = {}
        for results in (vec, kw):
            for rank, (cid, _) in enumerate(results, start=1):
                fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)

        similarity = dict(vec)
        ranked = sorted(fused, key=lambda cid: -fused[cid])[:k]
        results = [{
            **self.chunks[cid],
            "rrf_score": round(fused[cid], 4),
            "similarity": round(similarity[cid], 3) if cid in similarity else None,
        } for cid in ranked]

        return {"results": results, "best_similarity": round(vec[0][1], 3) if vec else None}


def show(searcher, question):
    out = searcher.search(question)
    print(f"\n{question!r}   (best similarity: {out['best_similarity']})")
    for i, r in enumerate(out["results"], start=1):
        sim = f"{r['similarity']:.3f}" if r["similarity"] is not None else "  -  "
        print(f"{i}. rrf {r['rrf_score']:.4f} | sim {sim} | {r['source']:10} | {r['title'][:60]}")
        print(f"   {r['url']}")
        print(f"   {r['text'][:150].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    searcher = HybridSearch()
    if len(sys.argv) > 1:
        show(searcher, " ".join(sys.argv[1:]))
    else:
        while question := input("\nquestion (empty to quit): ").strip():
            show(searcher, question)