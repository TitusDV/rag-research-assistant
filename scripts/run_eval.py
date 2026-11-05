"""Compare BM25, dense and hybrid retrieval at several chunk sizes on the approved questions.

  python scripts/run_eval.py             the generated, reviewed questions -> eval/results.md and results.json
  python scripts/run_eval.py --heldout   hand-written questions in eval/heldout.csv -> eval/results_heldout.md
  add --rerank to either                 also score hybrid retrieval reranked by a cross-encoder (about 90 MB download)

The held-out questions were never used to choose a setup, so they give the fairer final score.
The first run downloads the embedding model (about 90 MB).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from research_rag.chunking import load_papers  # noqa: E402
from research_rag.evaluate import compare_to, markdown_table, run_grid  # noqa: E402
from research_rag.rerank import DEFAULT_MODEL, cross_encoder_scorer  # noqa: E402
from research_rag.questions import load_jsonl  # noqa: E402

CHUNK_SIZES = [None, 1000, 500, 250]  # None = whole pages
KINDS = ["bm25", "dense", "hybrid"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
APP_SETUP = "hybrid/500"  # what app.py uses


def load_heldout(path: Path) -> list[dict]:
    """Load held-out questions. CSV page numbers start at 1, as in the PDF viewer."""
    questions = []

    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            question = row["question"].strip()
            if not question:
                continue

            questions.append({
                "id": f"heldout-{i}",
                "question": question,
                "paper": row["paper"].strip(),
                "page": int(row["page"]) - 1,
                "evidence": (row.get("evidence") or "").strip(),
            })

    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", action="store_true", help="score the hand-written questions in eval/heldout.csv")
    parser.add_argument("--rerank", action="store_true", help=f"also score hybrid reranked by {DEFAULT_MODEL}")
    args = parser.parse_args()

    if args.heldout:
        questions = load_heldout(ROOT / "eval" / "heldout.csv")
        if not questions:
            sys.exit("eval/heldout.csv has no questions yet.")
    else:
        questions = [q for q in load_jsonl(ROOT / "eval" / "questions.jsonl") if q["status"] == "approved"]
        if not questions:
            sys.exit("No approved questions yet; run scripts/review_questions.py first.")

    pages = load_papers(ROOT / "data" / "papers")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Scoring {(len(KINDS) + args.rerank) * len(CHUNK_SIZES)} setups on {len(questions)} questions ...")
    scorer = cross_encoder_scorer() if args.rerank else None
    rows, ranks = run_grid(pages, questions, embeddings, CHUNK_SIZES, KINDS, scorer=scorer)

    table = markdown_table(rows, len(questions)) + "\n" + compare_to(ranks, APP_SETUP)
    if args.heldout:
        (ROOT / "eval" / "results_heldout.md").write_text(
            f"# Retrieval results on held-out questions\n\nHand-written questions, never used to choose a setup.\n"
            f"Embedding model: {EMBEDDING_MODEL}\n\n{table}", encoding="utf-8")
        print(table)
        print("Saved eval/results_heldout.md")
        return
    (ROOT / "eval" / "results.md").write_text(f"# Retrieval results\n\nEmbedding model: {EMBEDDING_MODEL}\n\n{table}",
                                              encoding="utf-8")
    per_question = [{"id": q["id"], "question": q["question"], **{s: r[i] for s, r in ranks.items()}}
                    for i, q in enumerate(questions)]
    (ROOT / "eval" / "results.json").write_text(json.dumps(per_question, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table)
    print("Saved eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
