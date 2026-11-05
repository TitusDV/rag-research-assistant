"""Run the assistant, exactly as the app does, on the held-out questions and save everything needed to review it.

  python scripts/run_answer_eval.py           writes eval/answers_heldout.jsonl and eval/answer_review.csv
  python scripts/run_answer_eval.py --force   also overwrite an existing review sheet (loses your verdicts)

Fill in the verdict columns of eval/answer_review.csv, then run scripts/score_answers.py.
Uses MODEL and OPENAI_API_KEY from .env; each question is one model call.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from research_rag.answer import Assistant, make_answerer  # noqa: E402
from research_rag.answer_eval import audit, load_questions, review_row, save_jsonl, save_review_sheet, validate_questions  # noqa: E402
from research_rag.chunking import load_papers  # noqa: E402
from research_rag.llm import build_chat_model  # noqa: E402

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE, K = 500, 6  # the app's setup


def git_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return (rev or "unknown") + (" plus uncommitted changes" if dirty else "")
    except OSError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "heldout.csv",
                        help="question CSV, for example eval/answer_questions.csv")
    parser.add_argument("--force", action="store_true", help="overwrite an existing eval/answer_review.csv")
    args = parser.parse_args()

    review_path = ROOT / "eval" / "answer_review.csv"
    output_paths = [review_path, ROOT / "eval" / "answers_heldout.jsonl", ROOT / "eval" / "answer_config.json"]
    if any(p.exists() for p in output_paths) and not args.force:
        sys.exit("Answer evaluation output already exists. Archive the three output files before a new run, or use --force to overwrite them.")

    load_dotenv(ROOT / ".env")
    questions = load_questions(args.questions)
    if not questions:
        sys.exit("eval/heldout.csv has no questions yet.")

    pages = load_papers(ROOT / "data" / "papers")
    validate_questions(questions, pages, require_facts=True)
    model = os.getenv("MODEL", "llama3")
    config = {"model": model, "retrieval": "hybrid", "chunk_size": CHUNK_SIZE, "passages": K,
              "embeddings": EMBEDDING_MODEL, "code": git_revision(),
              "question_ids": [q["id"] for q in questions],
              "dataset_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest()}
    assistant = Assistant(pages, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
                          make_answerer(build_chat_model(model)), chunk_size=CHUNK_SIZE, k=K)

    records = []
    for q in questions:
        try:
            records.append(audit(q, assistant.ask_detailed(q["question"])))
        except Exception as err:  # keep going; a failure is recorded, never counted as a refusal
            records.append(audit(q, None, error=repr(err)))
        r = records[-1]
        status = "ERROR" if r["error"] else ("answered" if r["final"]["answered"] else "declined")
        print(f"{q['id']:<12} {status:<9} {q['question'][:70]}")

    save_jsonl(records, ROOT / "eval" / "answers_heldout.jsonl")
    save_review_sheet([review_row(r) for r in records], review_path)
    (ROOT / "eval" / "answer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSaved eval/answers_heldout.jsonl (full audit), eval/answer_review.csv (to fill in) and eval/answer_config.json")


if __name__ == "__main__":
    main()
