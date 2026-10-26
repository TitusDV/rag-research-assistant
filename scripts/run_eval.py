"""Compare BM25, dense and hybrid retrieval at several chunk sizes on the approved questions.

Writes eval/results.md (summary table) and eval/results.json (every question's rank per setup).
The first run downloads the embedding model (about 90 MB).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from research_rag.chunking import load_papers  # noqa: E402
from research_rag.evaluate import markdown_table, run_grid  # noqa: E402
from research_rag.questions import load_jsonl  # noqa: E402

CHUNK_SIZES = [None, 1000, 500, 250]  # None = whole pages
KINDS = ["bm25", "dense", "hybrid"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    questions = [q for q in load_jsonl(ROOT / "eval" / "questions.jsonl") if q["status"] == "approved"]
    if not questions:
        sys.exit("No approved questions yet; run scripts/review_questions.py first.")

    pages = load_papers(ROOT / "data" / "papers")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Scoring {len(KINDS) * len(CHUNK_SIZES)} setups on {len(questions)} questions ...")
    rows, ranks = run_grid(pages, questions, embeddings, CHUNK_SIZES, KINDS)

    table = markdown_table(rows, len(questions))
    (ROOT / "eval" / "results.md").write_text(f"# Retrieval results\n\nEmbedding model: {EMBEDDING_MODEL}\n\n{table}",
                                              encoding="utf-8")
    per_question = [{"id": q["id"], "question": q["question"], **{s: r[i] for s, r in ranks.items()}}
                    for i, q in enumerate(questions)]
    (ROOT / "eval" / "results.json").write_text(json.dumps(per_question, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table)
    print("Saved eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
