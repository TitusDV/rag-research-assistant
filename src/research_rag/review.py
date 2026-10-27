"""Automatic review of test questions by an LLM judge, using the same checklist a person uses.

The judge's decisions are checked against human decisions (see check_against_humans),
so its agreement with a person is measured rather than assumed.
"""
from typing import Callable, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

Reason = Literal[
    "ok", "citation", "layout", "example_trivia", "headline", "duplicate", "vague", "beyond_evidence"
]


class Verdict(BaseModel):
    reason: Reason = Field(description="'ok' if the question is good, otherwise the main problem")
    explanation: str = Field(description="One sentence explaining the decision")
    approve: bool = Field(description="True only when reason is 'ok'")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You review test questions for a search system over research papers. Each question was written "
     "from one page, and the search system is scored on whether it finds that exact page. "
     "Reject a question if ANY of these apply, using the first that fits as the reason:\n"
     "- citation: it is about a reference-list entry (a cited paper's title, authors, venue, year or pages).\n"
     "- layout: it is about the paper's structure, such as a section title or which table or figure shows what.\n"
     "- example_trivia: it asks about the content of a worked example or prompt (e.g. who directed a film "
     "in a sample passage) rather than what the paper itself says.\n"
     "- headline: the answer is one of the paper's main claims and is stated in the abstract given below, "
     "so it is not specific to one page. Facts tied to a particular model, setting, dataset or number are "
     "not headline even if the abstract mentions the topic.\n"
     "- duplicate: it asks for the same fact as one of the already approved questions listed below.\n"
     "- vague: it is not about a specific fact, or the answer just restates the question.\n"
     "- beyond_evidence: the answer states facts (names, numbers, claims) that are not in the evidence quote. "
     "A short explanation of a term is fine.\n"
     "Otherwise approve with reason 'ok'."),
    ("human",
     "Paper: {title}\nQuestion page: {page}\n\nAbstract (page 1):\n{abstract}\n\n"
     "Already approved questions from this paper:\n{approved}\n\n"
     "Question: {question}\nAnswer: {answer}\nEvidence quote from the page: \"{evidence}\""),
])

Judge = Callable[[dict, str, str, list[str]], Verdict]


def make_judge(chat_model) -> Judge:
    chain = PROMPT | chat_model.with_structured_output(Verdict)

    def judge(q: dict, title: str, abstract: str, approved: list[str]) -> Verdict:
        return chain.invoke({
            "title": title, "page": q["page"] + 1, "abstract": abstract[:3000],
            "approved": "\n".join(f"- {a}" for a in approved) or "(none)",
            "question": q["question"], "answer": q["answer"], "evidence": q["evidence"],
        })

    return judge


def approved_questions(rows: list[dict], paper: str, exclude_id: str | None = None) -> list[str]:
    return [r["question"] for r in rows
            if r["paper"] == paper and r["status"] == "approved" and r["id"] != exclude_id]


def auto_review(rows: list[dict], judge: Judge, titles: dict, abstracts: dict) -> list[dict]:
    """Decide every unreviewed question in place; later questions see earlier approvals (for duplicates)."""
    decided = []
    for row in rows:
        if row["status"] != "unreviewed":
            continue
        v = judge(row, titles.get(row["paper"], row["paper"]), abstracts.get(row["paper"], ""),
                  approved_questions(rows, row["paper"]))
        approve = v.approve and v.reason == "ok"  # a reason other than ok always means reject
        row.update(status="approved" if approve else "rejected", reviewed_by="judge",
                   judge_reason=v.reason, judge_explanation=v.explanation)
        decided.append(row)
    return decided


def check_against_humans(rows: list[dict], judge: Judge, titles: dict, abstracts: dict) -> dict:
    """Run the judge on human-reviewed questions without changing them, and measure agreement."""
    human = [r for r in rows if r["status"] in ("approved", "rejected") and r.get("reviewed_by", "human") == "human"]
    if not human:
        raise ValueError("no human-reviewed questions to check against")
    counts = {"both_approve": 0, "both_reject": 0, "judge_only_approves": 0, "judge_only_rejects": 0}
    disagreements = []
    for row in human:
        v = judge(row, titles.get(row["paper"], row["paper"]), abstracts.get(row["paper"], ""),
                  approved_questions(rows, row["paper"], exclude_id=row["id"]))
        judge_ok = v.approve and v.reason == "ok"
        human_ok = row["status"] == "approved"
        key = {(True, True): "both_approve", (False, False): "both_reject",
               (True, False): "judge_only_approves", (False, True): "judge_only_rejects"}[(judge_ok, human_ok)]
        counts[key] += 1
        if judge_ok != human_ok:
            disagreements.append({"id": row["id"], "question": row["question"], "human": row["status"],
                                  "judge_reason": v.reason, "judge_explanation": v.explanation})
    agreement = (counts["both_approve"] + counts["both_reject"]) / len(human)
    return {"n": len(human), "agreement": agreement, **counts, "disagreements": disagreements}
