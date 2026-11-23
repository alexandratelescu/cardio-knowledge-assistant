"""
Evaluates the assistant against eval/evaluation_questions.csv.

For every question with has_answer=yes, it checks retrieval accuracy:
is the correct source URL among the top-5 chunks, separately for
vector-only, BM25-only and hybrid search (search.py already supports
all three modes).

For every question, it then runs the full assistant (the same hybrid
pipeline app.py uses) and checks answer accuracy:
  - has_answer=yes  -> did it answer (not abstain) AND cite the
                       correct source URL?
  - has_answer=no, category=medical_advice -> did the reply start
                       with the medical-advice redirect?
  - has_answer=no, category=emergency      -> did it give the
                       emergency reply?
  - has_answer=no, no category (out of scope) -> did it abstain?

Writes eval/retrieval_results.csv and eval/answer_results.csv (one
row per question, so you can see exactly which ones failed) and
prints a summary to the terminal.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import assistant
from search import HybridSearch

QUESTIONS = Path(__file__).parent / "evaluation_questions.csv"
MODES = ["vector", "bm25", "hybrid"]


def load_questions():
    with open(QUESTIONS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["has_answer"].strip().lower() == "yes" and not r["source_url"].strip():
            sys.exit(f"missing source_url for: {r['question']}")
    return rows


def evaluate_retrieval(searcher, questions):
    answerable = [g for g in questions if g["has_answer"].strip().lower() == "yes"]
    hits = {mode: 0 for mode in MODES}
    rows = []

    for g in answerable:
        row = {"question": g["question"], "expected_url": g["source_url"]}
        for mode in MODES:
            result = searcher.search(g["question"], k=5, mode=mode)
            found = any(r["url"] == g["source_url"] for r in result["results"])
            row[mode] = "yes" if found else "no"
            hits[mode] += int(found)
        rows.append(row)

    with open(Path(__file__).parent / "retrieval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected_url"] + MODES)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== retrieval accuracy (correct source in top-5) ===")
    for mode in MODES:
        n = len(answerable)
        pct = 100 * hits[mode] / n if n else 0
        print(f"  {mode:8s}: {hits[mode]}/{n}  ({pct:.0f}%)")
    return rows


def evaluate_answers(searcher, questions):
    rows = []
    correct_answer = 0
    correct_abstain = 0
    total_answerable = 0
    total_should_abstain = 0

    for g in questions:
        result = assistant.answer(g["question"], searcher)
        has_answer = g["has_answer"].strip().lower() == "yes"
        category = g.get("category", "").strip().lower()

        row = {
            "question": g["question"],
            "has_answer": g["has_answer"],
            "category": category,
            "abstained": result["abstained"],
            "reason": result["reason"],
            "answer": result["answer"],
            "cited_urls": "; ".join(s["url"] for s in result["sources"]),
        }

        if has_answer:
            total_answerable += 1
            ok = (not result["abstained"]) and g["source_url"] in [s["url"] for s in result["sources"]]
            if ok:
                correct_answer += 1
        else:
            total_should_abstain += 1
            if category == "medical_advice":
                ok = result["answer"].startswith(assistant.MEDICAL)
            elif category == "emergency":
                ok = result["answer"].startswith(assistant.EMERGENCY)
            else:
                ok = result["abstained"]
            if ok:
                correct_abstain += 1

        row["correct"] = "yes" if ok else "no"
        rows.append(row)

    with open(Path(__file__).parent / "answer_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== answer accuracy ===")
    if total_answerable:
        print(f"  answerable questions, correctly answered + cited: "
              f"{correct_answer}/{total_answerable} ({100*correct_answer/total_answerable:.0f}%)")
    if total_should_abstain:
        print(f"  should-abstain questions, handled correctly:       "
              f"{correct_abstain}/{total_should_abstain} ({100*correct_abstain/total_should_abstain:.0f}%)")

    failed = [r for r in rows if r["correct"] == "no"]
    if failed:
        print(f"\n{len(failed)} question(s) marked 'no' - see answer_results.csv for the full text:")
        for r in failed:
            print(f"  - {r['question']}")
    return rows


if __name__ == "__main__":
    questions = load_questions()
    print(f"loaded {len(questions)} evaluation questions")
    searcher = HybridSearch()
    evaluate_retrieval(searcher, questions)
    evaluate_answers(searcher, questions)