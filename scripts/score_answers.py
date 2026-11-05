"""Turn the filled-in review sheet into the answer-quality results.

  python scripts/score_answers.py   reads eval/answer_review.csv, writes eval/answer_results.md
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_rag.answer_eval import markdown_report, summarise  # noqa: E402


def main() -> None:
    with (ROOT / "eval" / "answer_review.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    config_path = ROOT / "eval" / "answer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    expected = config.get("question_ids")
    if expected is not None and [r["id"] for r in rows] != expected:
        sys.exit("Review IDs do not match this run. Restore missing rows and the original order.")
    try:
        summary = summarise(rows)
    except ValueError as error:
        sys.exit(f"Review incomplete or invalid: {error}")
    report_config = {k: v for k, v in config.items() if k != "question_ids"}
    report = markdown_report(summary, report_config)
    (ROOT / "eval" / "answer_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
