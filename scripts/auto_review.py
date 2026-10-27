"""Review questions automatically with an LLM judge.

  python scripts/auto_review.py --check   measure the judge against your own reviews (changes nothing)
  python scripts/auto_review.py           decide every unreviewed question

The judge model is JUDGE_MODEL in .env, or MODEL if JUDGE_MODEL is not set.
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
from research_rag.papers import PAPERS  # noqa: E402
from research_rag.questions import load_jsonl, save_jsonl  # noqa: E402
from research_rag.review import auto_review, check_against_humans, make_judge  # noqa: E402

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
PATH = ROOT / "eval" / "questions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare the judge with your reviews; change nothing")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model = os.getenv("JUDGE_MODEL") or os.getenv("MODEL", "llama3")
    judge = make_judge(build_chat_model(model))
    abstracts = {p.metadata["paper"]: p.page_content for p in load_papers(ROOT / "data" / "papers")
                 if p.metadata["page"] == 0}
    rows = load_jsonl(PATH)

    if args.check:
        result = check_against_humans(rows, judge, PAPERS, abstracts)
        lines = [f"# Judge check ({model})", "",
                 f"Agreement with human review: {result['agreement']:.0%} on {result['n']} questions.", "",
                 "| | Human approved | Human rejected |", "|---|---|---|",
                 f"| Judge approved | {result['both_approve']} | {result['judge_only_approves']} |",
                 f"| Judge rejected | {result['judge_only_rejects']} | {result['both_reject']} |", "",
                 "## Disagreements", ""]
        lines += [f"- **{d['id']}** (you: {d['human']}, judge: {d['judge_reason']}): {d['question']}  \n"
                  f"  Judge: {d['judge_explanation']}" for d in result["disagreements"]] or ["None."]
        (ROOT / "eval" / "judge_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines[:9]))
        print("\nFull report: eval/judge_check.md")
        return

    decided = auto_review(rows, judge, PAPERS, abstracts)
    save_jsonl(rows, PATH)
    approved = sum(r["status"] == "approved" for r in decided)
    print(f"Judge decided {len(decided)}: {approved} approved, {len(decided) - approved} rejected.")
    for r in decided:
        print(f"  {'APPROVE' if r['status'] == 'approved' else 'reject '}  {r['judge_reason']:<15} {r['question'][:80]}")


if __name__ == "__main__":
    main()
