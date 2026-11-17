"""Check eval/heldout.csv (papers, pages and quotes) without running retrieval or calling a model."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_rag.answer_eval import load_questions, validate_questions
from research_rag.chunking import load_papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "heldout.csv")
    parser.add_argument("--answers", action="store_true", help="also require reference facts for answerable questions")
    args = parser.parse_args()
    try:
        questions = load_questions(args.questions)
        if not questions:
            raise ValueError("no questions in the file")
        validate_questions(questions, load_papers(ROOT / "data" / "papers"), require_facts=args.answers)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(str(error))
    for q in questions:
        label = f"{q['paper']} | PDF page {q['page']}" if q["answerable"] else "unanswerable label"
        print(f"OK: {q['id']} | {label}")
    print(f"{len(questions)} questions checked: papers, pages and quotes are fine.")
    print("This doesn't check whether the questions are answerable or the required facts are right. That still needs a person.")
    missing = sum(q["answerable"] and not q["evidence"] for q in questions)
    if missing:
        print(f"{missing} answerable questions have no quote, so Evidence@k will show n/a.")


if __name__ == "__main__":
    main()
