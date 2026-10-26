from langchain_core.documents import Document

from research_rag.questions import (
    GeneratedQuestion,
    evidence_on_page,
    generate_questions,
    load_jsonl,
    sample_pages,
    save_jsonl,
)

TEXT = ("Language models are better at using relevant information that occurs at the very begin-\n"
        "ning or end of the input context. " * 20)


def page(paper, n, text=TEXT):
    return Document(page_content=text, metadata={"paper": paper, "page": n})


def test_evidence_check_ignores_case_spacing_and_line_hyphens():
    assert evidence_on_page("at the very  beginning or END of the input", TEXT)
    assert not evidence_on_page("in the middle of the input", TEXT)
    assert not evidence_on_page("   ", TEXT)


def test_sample_pages_is_balanced_reproducible_and_skips_bad_pages():
    pages = [page("a", i) for i in range(5)] + [page("b", i) for i in range(5)]
    pages += [page("a", 9, "short"), page("b", 9, "References\n" + TEXT)]
    picked = sample_pages(pages, per_paper=2, seed=1)
    assert sorted(p.metadata["paper"] for p in picked) == ["a", "a", "b", "b"]
    assert all(p.metadata["page"] != 9 for p in picked)
    assert picked == sample_pages(pages, per_paper=2, seed=1)


def test_generate_keeps_grounded_questions_and_drops_the_rest():
    def ask(title, text):
        if text is TEXT:
            return GeneratedQuestion(question="Where in the context do models use information best?",
                                     answer="At the beginning or end.",
                                     evidence="at the very beginning or end of the input context")
        return GeneratedQuestion(question="Made up?", answer="Yes.", evidence="this sentence is not on the page")

    kept, dropped = generate_questions([page("2307.03172", 3), page("2307.03172", 4, TEXT + " ")], ask)
    assert [k["id"] for k in kept] == ["2307.03172-p3"]
    assert kept[0]["status"] == "unreviewed"
    assert dropped[0]["reason"] == "evidence not found on page"


def test_model_errors_are_recorded_not_raised():
    def ask(title, text):
        raise RuntimeError("timeout")

    kept, dropped = generate_questions([page("x", 0)], ask)
    assert kept == [] and "timeout" in dropped[0]["reason"]


def test_jsonl_round_trip(tmp_path):
    rows = [{"id": "a", "question": "Q?"}, {"id": "b", "question": "Ā?"}]
    save_jsonl(rows, tmp_path / "q.jsonl")
    assert load_jsonl(tmp_path / "q.jsonl") == rows
