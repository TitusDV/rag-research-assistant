"""Approve or reject each unreviewed question in eval/questions.jsonl.

For each question: a = approve, r = reject, s = skip for now, q = save and quit.
Progress is saved after every answer, so you can stop and come back.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_rag.papers import PAPERS  # noqa: E402
from research_rag.questions import load_jsonl, save_jsonl  # noqa: E402

PATH = ROOT / "eval" / "questions.jsonl"


def main() -> None:
    rows = load_jsonl(PATH)
    todo = [r for r in rows if r["status"] == "unreviewed"]
    print(f"{len(todo)} to review. A good question is specific, stands on its own, and is answered by the evidence.\n")
    for n, row in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {PAPERS.get(row['paper'], row['paper'])}, page {row['page'] + 1}")
        print(f"  Q: {row['question']}\n  A: {row['answer']}\n  Evidence: \"{row['evidence']}\"")
        choice = ""
        while choice not in ("a", "r", "s", "q"):
            choice = input("  a/r/s/q > ").strip().lower()
        if choice == "q":
            break
        if choice != "s":
            row["status"] = "approved" if choice == "a" else "rejected"
            save_jsonl(rows, PATH)
        print()
    counts = {s: sum(r["status"] == s for r in rows) for s in ("approved", "rejected", "unreviewed")}
    print(counts)


if __name__ == "__main__":
    main()
