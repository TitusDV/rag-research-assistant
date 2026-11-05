"""End-to-end answer evaluation on held-out questions: correctness, citation support and abstention.

Retrieval metrics say whether the right passage was found. These say whether the user got a correct,
verifiable answer, and whether the assistant declined questions the papers can't answer.

Flow: run_answer_eval.py saves an audit record per question (passages, raw model output, final answer) and a
review sheet. A person fills in the verdict columns. score_answers.py turns the sheet into the results table.
"""
import csv
import json
from pathlib import Path

from .evaluate import holds_evidence
from .metrics import bootstrap_ci

REVIEW_COLUMNS = [
    "id", "question", "answerable", "required_facts", "gold", "gold_page_rank", "evidence_rank",
    "final_answered", "final_answer", "cited",
    # filled in by the reviewer:
    "correct", "claims_total", "claims_supported", "enough_info_in_passages", "notes",
]


def _yes(value: str) -> bool:
    text = str(value).strip().lower()
    if text in ("y", "yes", "true", "1"):
        return True
    if text in ("n", "no", "false", "0"):
        return False
    raise ValueError(f"Expected yes or no, received {value!r}")


def load_questions(path: Path) -> list[dict]:
    """Read CSV questions with one-based PDF pages and optional answer labels."""
    out = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "question" not in headers or len(headers) != len(set(headers)):
            raise ValueError("CSV needs a question header and unique column names.")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CSV row {row_number}: extra fields. Check commas and quoting.")
            if not (row.get("question") or "").strip():
                if any((v or "").strip() for v in row.values()):
                    raise ValueError(f"CSV row {row_number}: question is blank.")
                continue
            try:
                answerable = _yes(row.get("answerable") or "yes")
                paper = (row.get("paper") or "").strip()
                page = (row.get("page") or "").strip()
                if answerable:
                    if not paper or not page:
                        raise ValueError("answerable questions require paper and page.")
                    page_number = int(page)
                    if page_number < 1:
                        raise ValueError("page must be a positive PDF viewer page number.")
                else:
                    page_number = None
            except ValueError as error:
                raise ValueError(f"CSV row {row_number}: {error}") from error
            out.append({
                "id": f"heldout-{len(out) + 1}",
                "question": row["question"].strip(),
                "answerable": answerable,
                "paper": paper if answerable else "",
                "page": page_number,
                "required_facts": [x.strip() for x in (row.get("required_facts") or "").split(";") if x.strip()],
                "evidence": (row.get("evidence") or "").strip(),
            })
    return out


def validate_questions(questions: list[dict], pages: list, require_facts: bool = False) -> None:
    """Check labels and supplied quotes before model calls or retrieval scoring."""
    page_text = {(p.metadata["paper"], p.metadata["page"] + 1): p.page_content for p in pages}
    errors = []
    for q in questions:
        if not q["answerable"]:
            continue
        key = (q["paper"], q["page"])
        if key not in page_text:
            errors.append(f"{q['id']}: paper or PDF page does not exist.")
        elif q["evidence"] and not holds_evidence(page_text[key], q["evidence"]):
            errors.append(f"{q['id']}: evidence quote does not match the labelled page.")
        if require_facts and not q["required_facts"]:
            errors.append(f"{q['id']}: required_facts is empty.")
    if errors:
        raise ValueError("\n".join(errors))


def audit(question: dict, detailed: dict | None, error: str | None = None) -> dict:
    """One saved record: the question, what the model saw, what it returned, and where the gold page ranked."""
    record = {**question, "error": error, "passages": [], "raw": None, "final": None,
              "gold_page_rank": None, "evidence_rank": None}
    if detailed is None:
        return record
    record.update(passages=detailed["passages"], raw=detailed["raw"], final=detailed["final"])
    if question["answerable"] and question["page"] is not None:
        on_gold = [p for p in detailed["passages"] if (p["paper"], p["page"]) == (question["paper"], question["page"])]
        record["gold_page_rank"] = on_gold[0]["n"] if on_gold else None
        if question["evidence"]:
            hit = next((p["n"] for p in on_gold if holds_evidence(p["text"], question["evidence"])), None)
            record["evidence_rank"] = hit
    return record


def review_row(record: dict) -> dict:
    final = record.get("final") or {}
    cited = "; ".join(f"[{s['n']}] {s['paper']} p.{s['page']} (passage {','.join(map(str, s.get('passages', [])))})"
                      for s in final.get("sources", []))
    return {
        "id": record["id"],
        "question": record["question"],
        "answerable": "yes" if record["answerable"] else "no",
        "required_facts": "; ".join(record["required_facts"]),
        "gold": f"{record['paper']} p.{record['page']}" if record["answerable"] else "",
        "gold_page_rank": record["gold_page_rank"] or "",
        "evidence_rank": record["evidence_rank"] or "",
        "final_answered": "ERROR" if record["error"] else ("yes" if final.get("answered") else "no"),
        "final_answer": record["error"] or final.get("answer", ""),
        "cited": cited,
        "correct": "", "claims_total": "", "claims_supported": "", "enough_info_in_passages": "", "notes": "",
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def save_review_sheet(rows: list[dict], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _claim_counts(row: dict) -> tuple[int, int]:
    try:
        total = int(str(row["claims_total"]).strip())
        supported = int(str(row["claims_supported"]).strip())
    except (ValueError, KeyError) as error:
        raise ValueError("claims_total and claims_supported need explicit whole numbers.") from error
    if not 0 <= supported <= total:
        raise ValueError("Claim counts must satisfy 0 <= claims_supported <= claims_total.")
    return total, supported


def summarise(rows: list[dict]) -> dict:
    """Score completed reviews. Every attempted question remains in its denominator."""
    if not rows:
        raise ValueError("The review sheet contains no questions.")
    seen = set()
    counts = dict(answerable=0, correct=0, wrong_abstentions=0, answered_wrong=0,
                  unanswerable=0, declined=0, claims_total=0, claims_supported=0,
                  errors=0, answerable_errors=0, unanswerable_errors=0)
    retrieval_miss, generation_miss, unknown_cause = [], [], []
    for row in rows:
        identifier = row["id"].strip()
        if not identifier or identifier in seen:
            raise ValueError(f"Blank or duplicate review ID: {identifier!r}")
        seen.add(identifier)
        try:
            answerable = _yes(row["answerable"])
            status = row["final_answered"].strip().lower()
            if status not in ("yes", "no", "error"):
                raise ValueError("final_answered must be yes, no or ERROR.")
            counts["answerable" if answerable else "unanswerable"] += 1
            if status == "error":
                counts["errors"] += 1
                counts["answerable_errors" if answerable else "unanswerable_errors"] += 1
                continue
            correct = False
            if status == "yes":
                total, supported = _claim_counts(row)
                counts["claims_total"] += total
                counts["claims_supported"] += supported
                if answerable:
                    correct = _yes(row["correct"])
                    counts["correct" if correct else "answered_wrong"] += 1
            elif answerable:
                counts["wrong_abstentions"] += 1
            else:
                counts["declined"] += 1

            if answerable and not correct:
                enough = (row.get("enough_info_in_passages") or "").strip().lower()
                if enough == "unknown":
                    unknown_cause.append(identifier)
                elif _yes(enough):
                    generation_miss.append(identifier)
                else:
                    retrieval_miss.append(identifier)
        except (ValueError, KeyError) as error:
            raise ValueError(f"Review {identifier}: {error}") from error

    n = counts["answerable"]
    hits = counts["correct"]
    counts.update(
        correct_ci=bootstrap_ci([1.0] * hits + [0.0] * (n - hits)) if n else None,
        retrieval_miss=retrieval_miss, generation_miss=generation_miss,
        unknown_cause=unknown_cause, unreviewed=[],
    )
    return counts


def markdown_report(s: dict, config: dict) -> str:
    def frac(a, b):
        return f"{a}/{b} ({a / b:.0%})" if b else "n/a"

    corr = frac(s["correct"], s["answerable"])
    if s["correct_ci"]:
        corr += f", empirical 95% bootstrap interval {s['correct_ci'][0]:.0%} to {s['correct_ci'][1]:.0%}"
    lines = ["# Answer quality on held-out questions", "",
             "Configuration " + ", ".join(f"{k} {v}" for k, v in config.items()), "",
             "| Measure | Result | Denominator |", "|---|---|---|",
             f"| Answer correctness | {corr} | all attempted answerable questions, including run errors |",
             f"| Wrong abstentions | {s['wrong_abstentions']} | answerable questions deliberately declined |",
             f"| Answered but wrong | {s['answered_wrong']} | answerable questions with an incorrect or incomplete answer |",
             f"| Citation support | {frac(s['claims_supported'], s['claims_total'])} | all reviewed factual claims in returned answers |",
             f"| Correct abstention | {frac(s['declined'], s['unanswerable'])} | all attempted unanswerable questions, including run errors |",
             f"| Run errors | {s['errors']} | {s['answerable_errors']} answerable and {s['unanswerable_errors']} unanswerable |", "",
             "Empirical intervals resample these questions. An all-success or all-failure sample yields a zero-width interval and does not establish certainty about future questions.", "",
             "## Review of failed answers", "",
             f"- Retrieved passages lacked enough information according to the reviewer: {len(s['retrieval_miss'])} ({', '.join(s['retrieval_miss'])})",
             f"- Retrieved passages had enough information but the final answer failed: {len(s['generation_miss'])} ({', '.join(s['generation_miss'])})",
             f"- Evidence sufficiency unresolved: {len(s['unknown_cause'])} ({', '.join(s['unknown_cause'])})"]
    return "\n".join(lines) + "\n"
