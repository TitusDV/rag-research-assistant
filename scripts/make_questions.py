"""Write eval/questions.jsonl: model-generated questions, each tied to a paper and page.

Usage:
  python scripts/make_questions.py [--per-paper 8] [--seed 0]
  python scripts/make_questions.py --append [--per-paper 8]   add questions from pages not used yet

The model comes from MODEL in .env (default llama3 through Ollama).
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from research_rag.chunking import load_papers  # noqa: E402
from research_rag.llm import build_chat_model  # noqa: E402
from research_rag.questions import generate_questions, load_jsonl, make_ask, sample_pages, save_jsonl  # noqa: E402

# Harmless version clash between langchain-openai 0.3 and the OpenAI SDK; the parsed result is still correct.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-paper", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="add to the existing file, using only unused pages")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model = os.getenv("MODEL", "llama3")
    out, dropped_out = ROOT / "eval" / "questions.jsonl", ROOT / "eval" / "dropped.jsonl"

    used = set()
    if out.exists():
        if not args.append:
            sys.exit(f"{out} already exists. Use --append to add more questions, "
                     "or delete it to start again (this discards your reviews).")
        for path in (out, dropped_out):
            if path.exists():
                used |= {(r["paper"], r["page"]) for r in load_jsonl(path)}

    pages = sample_pages(load_papers(ROOT / "data" / "papers"), args.per_paper, args.seed, exclude=used)
    print(f"Writing questions for {len(pages)} new pages with {model} ...")
    kept, dropped = generate_questions(pages, make_ask(build_chat_model(model)))

    out.parent.mkdir(exist_ok=True)
    save_jsonl(kept, out, append=args.append)
    save_jsonl(dropped, dropped_out, append=args.append)
    print(f"Kept {len(kept)}, dropped {len(dropped)} (see eval/dropped.jsonl). Next: python scripts/review_questions.py")


if __name__ == "__main__":
    main()
