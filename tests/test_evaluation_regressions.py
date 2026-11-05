import csv

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.answer_eval import load_questions, summarise, validate_questions
from research_rag.evaluate import holds_evidence, run_grid


@pytest.mark.parametrize("passage,quote", [
    ("180% accuracy", "80% accuracy"),
    ("unfiltered TriviaQA", "filtered TriviaQA"),
    ("ten documents", "ten document"),
    ("the exact match score on SQuAD v1.1 improves from above 80% to more than 90%",
     "the exact match score on SQuAD v1.1 drops from above 80% to less than 40%"),
    ("the score", "the exact match score drops from 80% to 40%"),
])
def test_evidence_false_positives(passage, quote):
    assert not holds_evidence(passage, quote)


def test_normalised_quote_can_have_surrounding_punctuation():
    assert holds_evidence('Results: (80% accuracy).', '80% accuracy')
    assert holds_evidence('An unﬁl-\ntered test.', 'unfiltered test.')


@pytest.mark.parametrize("paper,page,answerable", [
    ("p", "1", "yees"), ("p", "", "yes"), ("", "1", "yes"),
    ("p", "0", "yes"), ("p", "1.0", "yes"),
])
def test_invalid_csv_rows_are_not_silently_dropped(tmp_path, paper, page, answerable):
    path = tmp_path / "questions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question", "paper", "page", "answerable"])
        writer.writerow(["What?", paper, page, answerable])
    with pytest.raises(ValueError, match="row 2"):
        load_questions(path)


def review(identifier="q1", answerable="yes", status="yes", correct="yes",
           total="1", supported="1", enough="yes"):
    return dict(id=identifier, answerable=answerable, final_answered=status, correct=correct,
                claims_total=total, claims_supported=supported,
                enough_info_in_passages=enough, gold_page_rank="1")


def test_errors_remain_in_both_question_denominators():
    result = summarise([review(), review("q2", status="ERROR"),
                        review("q3", answerable="no", status="no"),
                        review("q4", answerable="no", status="ERROR")])
    assert (result["correct"], result["answerable"]) == (1, 2)
    assert (result["declined"], result["unanswerable"]) == (1, 2)
    assert result["errors"] == 2


@pytest.mark.parametrize("total,supported", [("", ""), ("1", "2"), ("-1", "0"), ("1.5", "1")])
def test_claim_review_must_be_complete_and_consistent(total, supported):
    with pytest.raises(ValueError, match="Review q1"):
        summarise([review(total=total, supported=supported)])


def test_blank_correctness_cannot_be_published():
    with pytest.raises(ValueError, match="yes or no"):
        summarise([review(correct="")])


def test_failure_diagnosis_uses_evidence_sufficiency():
    result = summarise([review(correct="no", enough="no"),
                        review("q2", correct="no", enough="yes"),
                        review("q3", status="no", enough="unknown")])
    assert result["retrieval_miss"] == ["q1"]
    assert result["generation_miss"] == ["q2"]
    assert result["unknown_cause"] == ["q3"]


def test_duplicate_review_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        summarise([review(), review()])


def test_gold_page_and_required_facts_are_validated():
    question = dict(id="q1", answerable=True, paper="p", page=1,
                    evidence="the result", required_facts=[])
    pages = [Document(page_content="the result is 10", metadata={"paper": "p", "page": 0})]
    validate_questions([question], pages)
    with pytest.raises(ValueError, match="required_facts"):
        validate_questions([question], pages, require_facts=True)
    with pytest.raises(ValueError, match="does not exist"):
        validate_questions([{**question, "page": 2}], pages)
    with pytest.raises(ValueError, match="quote"):
        validate_questions([{**question, "evidence": "another result"}], pages)


def test_grid_retains_evidence_ranks_and_scores_at_six():
    pages = [Document(page_content="known target evidence", metadata={"paper": "p", "page": 0})]
    questions = [dict(question="target", paper="p", page=0, evidence="target evidence")]
    details = {}
    rows, _ = run_grid(pages, questions, DeterministicFakeEmbedding(size=8),
                       [None], ["bm25"], evidence_ranks_by_setup=details)
    assert rows[0]["evidence@6"] == 1
    assert details == {"bm25/page": [1]}


def test_results_table_header_and_separator_have_the_same_number_of_columns():
    # GitHub only renders a Markdown table when the separator row has one cell per header cell.
    from research_rag.evaluate import markdown_table
    row = {"retriever": "hybrid", "chunk": 500, "chunks": 10, "hit@1": 0.5, "hit@3": 0.6, "hit@5": 0.7,
           "hit@6": 0.7, "mrr@10": 0.6, "ndcg@10": 0.6, "evidence@5": 0.5, "evidence@6": 0.5}
    table = [line for line in markdown_table([row], 1).splitlines() if line.startswith("|")]
    header, separator, body = table
    assert header.count("|") == separator.count("|") == body.count("|")
