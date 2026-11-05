from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.answer import Assistant, GeneratedAnswer
from research_rag.answer_eval import audit, load_questions, markdown_report, review_row, summarise
from research_rag.evaluate import holds_evidence

PASSAGES = [
    Document(page_content="DPR processes 995 questions per second when returning the top 100 passages.",
             metadata={"paper": "2004.04906", "page": 6}),
    Document(page_content="BM25 is a strong lexical baseline.", metadata={"paper": "2004.04906", "page": 2}),
]


class Fixed:
    def invoke(self, question):
        return PASSAGES


def assistant(answer, citations, answered=True):
    a = Assistant(PASSAGES, DeterministicFakeEmbedding(size=8),
                  lambda q, c: GeneratedAnswer(answered=answered, answer=answer, citations=citations), chunk_size=200, k=6)
    a.retriever = Fixed()
    return a


def test_ask_detailed_keeps_passages_and_raw_output_before_renumbering():
    d = assistant("995 per second [1] [9].", [1, 9]).ask_detailed("How fast is DPR?")
    assert [(p["n"], p["page"]) for p in d["passages"]] == [(1, 7), (2, 3)]   # pages shown as in the PDF viewer
    assert d["raw"]["citations"] == [1, 9]                                   # the model's original numbers
    assert d["final"]["answer"] == "995 per second [1]."                     # what the user saw
    assert d["final"]["sources"][0]["passages"] == [1]                       # which passage backs card [1]


def test_load_questions_reads_answerable_and_unanswerable_rows(tmp_path):
    f = tmp_path / "heldout.csv"
    f.write_text("question,paper,page,answerable,required_facts,evidence\n"
                 "How fast is DPR?,2004.04906,7,yes,995 questions per second; top 100,995 questions per second\n"
                 "What GPU did Ragas use?,,,no,,\n\n", encoding="utf-8")
    q = load_questions(f)
    assert q[0]["answerable"] and q[0]["page"] == 7 and q[0]["required_facts"] == ["995 questions per second", "top 100"]
    assert not q[1]["answerable"] and q[1]["page"] is None


def test_audit_finds_the_gold_page_and_evidence_among_the_passages():
    q = {"id": "h1", "question": "How fast?", "answerable": True, "paper": "2004.04906", "page": 7,
         "required_facts": ["995"], "evidence": "995 questions per second"}
    r = audit(q, assistant("995 [1].", [1]).ask_detailed("How fast?"))
    assert r["gold_page_rank"] == 1 and r["evidence_rank"] == 1
    assert review_row(r)["cited"].startswith("[1] 2004.04906 p.7")


def test_errors_are_kept_apart_from_refusals():
    q = {"id": "h2", "question": "x", "answerable": True, "paper": "p", "page": 1, "required_facts": [], "evidence": ""}
    assert review_row(audit(q, None, error="Timeout()"))["final_answered"] == "ERROR"


def test_ligatures_in_pdf_text_still_match_a_typed_quote():
    assert holds_evidence("results on unﬁltered TriviaQA", "unfiltered TriviaQA")


def row(id_, answerable, answered, correct="", gold_rank="", claims=("", ""), enough="yes"):
    return {"id": id_, "answerable": answerable, "final_answered": answered, "correct": correct,
            "gold_page_rank": gold_rank, "claims_total": claims[0], "claims_supported": claims[1],
            "enough_info_in_passages": enough}


def test_summary_counts_and_failure_causes():
    rows = [row("a", "yes", "yes", "y", "1", (2, 2)),
            row("b", "yes", "yes", "n", "2", (3, 1)),     # page was retrieved, answer wrong: generation
            row("c", "yes", "no", gold_rank="", enough="no"),           # declined, page missing: retrieval
            row("d", "no", "no"),                          # correctly declined
            row("e", "no", "yes", claims=(1, 0)),          # answered an unanswerable question
            row("f", "yes", "ERROR")]
    s = summarise(rows)
    assert (s["answerable"], s["correct"], s["wrong_abstentions"], s["answered_wrong"]) == (4, 1, 1, 1)
    assert (s["unanswerable"], s["declined"], s["errors"]) == (2, 1, 1)
    assert (s["claims_supported"], s["claims_total"]) == (3, 6)
    assert s["retrieval_miss"] == ["c"] and s["generation_miss"] == ["b"]
    report = markdown_report(s, {"model": "m"})
    assert "| Answer correctness | 1/4 (25%)" in report and "| Correct abstention | 1/2 (50%)" in report
