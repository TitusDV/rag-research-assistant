import pytest

from research_rag.review import Verdict, approved_questions, auto_review, check_against_humans


def q(id_, paper, status, question, reviewed_by=None):
    row = {"id": id_, "paper": paper, "page": 3, "question": question, "answer": "A.", "evidence": "e", "status": status}
    if reviewed_by:
        row["reviewed_by"] = reviewed_by
    return row


def judge_that_rejects_duplicates(row, title, abstract, approved):
    if row["question"] in approved:
        return Verdict(reason="duplicate", explanation="Same fact.", approve=False)
    if "Table" in row["question"]:
        return Verdict(reason="layout", explanation="About layout.", approve=False)
    return Verdict(reason="ok", explanation="Fine.", approve=True)


def test_auto_review_decides_only_unreviewed_and_sees_earlier_approvals():
    rows = [q("p-1", "p", "approved", "What is X?"),
            q("p-2", "p", "unreviewed", "What is X?"),
            q("p-3", "p", "unreviewed", "What does Table 2 show?"),
            q("p-4", "p", "unreviewed", "What is Y?"),
            q("p-5", "p", "unreviewed", "What is Y?")]
    decided = auto_review(rows, judge_that_rejects_duplicates, {}, {})
    assert [r["id"] for r in decided] == ["p-2", "p-3", "p-4", "p-5"]
    assert [r["status"] for r in rows] == ["approved", "rejected", "rejected", "approved", "rejected"]
    assert rows[2]["judge_reason"] == "layout" and rows[3]["reviewed_by"] == "judge"


def test_a_problem_reason_always_rejects_even_if_approve_is_true():
    rows = [q("p-1", "p", "unreviewed", "Q?")]
    auto_review(rows, lambda *a: Verdict(reason="vague", explanation="x", approve=True), {}, {})
    assert rows[0]["status"] == "rejected"


def test_check_measures_agreement_on_human_reviews_only_and_changes_nothing():
    rows = [q("p-1", "p", "approved", "What is X?"),                 # judge agrees
            q("p-2", "p", "rejected", "What does Table 1 show?"),    # judge agrees
            q("p-3", "p", "rejected", "What is Z?"),                 # judge disagrees (approves)
            q("p-4", "p", "approved", "What is W?", reviewed_by="judge"),  # ignored: not human
            q("p-5", "p", "unreviewed", "What is V?")]               # ignored
    before = [dict(r) for r in rows]
    result = check_against_humans(rows, judge_that_rejects_duplicates, {}, {})
    assert rows == before
    assert result["n"] == 3 and result["agreement"] == pytest.approx(2 / 3)
    assert result["judge_only_approves"] == 1 and result["disagreements"][0]["id"] == "p-3"


def test_approved_questions_excludes_the_question_being_checked():
    rows = [q("p-1", "p", "approved", "A?"), q("p-2", "p", "approved", "B?"), q("o-1", "o", "approved", "C?")]
    assert approved_questions(rows, "p", exclude_id="p-1") == ["B?"]
