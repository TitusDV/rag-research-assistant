"""Compare BM25, dense and hybrid retrieval at several chunk sizes on the approved questions.

  python scripts/run_eval.py             the generated, reviewed questions -> eval/results.md and results.json
  python scripts/run_eval.py --heldout   held-out questions in eval/heldout.csv -> eval/results_heldout.md
  add --rerank to either                 also score hybrid retrieval reranked by a cross-encoder (about 90 MB download)

The held-out CSV is an additional evaluation set. Record its provenance when publishing results.
The first run downloads the embedding model (about 90 MB).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from research_rag.answer_eval import load_questions, validate_questions  # noqa: E402
from research_rag.chunking import load_papers  # noqa: E402
from research_rag.evaluate import compare_to, markdown_table, run_grid  # noqa: E402
from research_rag.rerank import DEFAULT_MODEL, cross_encoder_scorer  # noqa: E402
from research_rag.questions import load_jsonl  # noqa: E402

CHUNK_SIZES = [None, 1000, 500, 250]  # None = whole pages
KINDS = ["bm25", "dense", "hybrid"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
APP_SETUP = "hybrid/500"  # what app.py uses


def load_heldout(path: Path) -> list[dict]:
    """Answerable held-out questions, with the page converted to the zero-based index the labels use."""
    return [{**q, "page": q["page"] - 1} for q in load_questions(path) if q["answerable"] and q["page"] is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", action="store_true", help="score the held-out questions in eval/heldout.csv")
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
    validation_rows = [{**q, "answerable": True, "page": q["page"] + 1,
                        "required_facts": q.get("required_facts", []),
                        "evidence": q.get("evidence", "")} for q in questions]
    validate_questions(validation_rows, pages)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Scoring {(len(KINDS) + args.rerank) * len(CHUNK_SIZES)} setups on {len(questions)} questions ...")
    scorer = cross_encoder_scorer() if args.rerank else None
    evidence_ranks = {}
    rows, ranks = run_grid(pages, questions, embeddings, CHUNK_SIZES, KINDS, scorer=scorer,
                           evidence_ranks_by_setup=evidence_ranks)

    details = [
        {"id": q["id"], "question": q["question"],
         "setups": {name: {"page_rank": values[i], "evidence_rank": evidence_ranks[name][i]}
                    for name, values in ranks.items()}}
        for i, q in enumerate(questions)
    ]
    suffix = "_heldout" if args.heldout else ""
    details_path = ROOT / "eval" / f"retrieval_details{suffix}.json"
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {details_path.relative_to(ROOT)}")
    table = markdown_table(rows, len(questions)) + "\n" + compare_to(ranks, APP_SETUP)
    if args.heldout:
        (ROOT / "eval" / "results_heldout.md").write_text(
            f"# Retrieval results on held-out questions\n\nAI-assisted questions with source-checked reference quotes, originally reserved from setup selection. Subsequent reuse should be reported.\n"
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
