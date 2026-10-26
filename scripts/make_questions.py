"""Write eval/questions.jsonl: model-generated questions, each tied to a paper and page.

Usage: python scripts/make_questions.py [--per-paper 8] [--seed 0]
The model comes from MODEL in .env (default llama3 through Ollama).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from research_rag.chunking import load_papers  # noqa: E402
from research_rag.llm import build_chat_model  # noqa: E402
from research_rag.questions import generate_questions, make_ask, sample_pages, save_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-paper", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model = os.getenv("MODEL", "llama3")
    out = ROOT / "eval" / "questions.jsonl"
    if out.exists():
        sys.exit(f"{out} already exists; delete it first if you want to regenerate (this discards reviews)")

    pages = sample_pages(load_papers(ROOT / "data" / "papers"), args.per_paper, args.seed)
    print(f"Writing questions for {len(pages)} pages with {model} ...")
    kept, dropped = generate_questions(pages, make_ask(build_chat_model(model)))

    out.parent.mkdir(exist_ok=True)
    save_jsonl(kept, out)
    save_jsonl(dropped, ROOT / "eval" / "dropped.jsonl")
    print(f"Kept {len(kept)}, dropped {len(dropped)} (see eval/dropped.jsonl). Next: python scripts/review_questions.py")


if __name__ == "__main__":
    main()
