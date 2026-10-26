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
    pages += [page("a", 9, "short"), page("b", 9, "References\n" + TEXT)]  # too short; reference list
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



def ref_page(paper, n):
    return page(paper, n, "\n".join(f"Author {k}. {2015 + k % 9}. Title {k}. In Proc. of ACL {2016 + k % 7}." for k in range(30)))


def test_reference_list_runs_from_heading_to_appendix():
    from research_rag.questions import reference_pages
    end_of_body = page("p", 9, TEXT + "\nConclusion\nWe did well.\nR EFERENCES\nAuthor A. 2020. A title.")
    appendix = page("p", 13, "A Appendix\nMore details about training. " * 30)
    body_with_many_citations = page("q", 6, TEXT + " (Clark et al. 2018; Lee 2019; Kim 2020; Ho 2021)" * 20)
    pages = [page("p", 8), end_of_body, ref_page("p", 10), ref_page("p", 11),
             page("p", 12, "References\n" + ref_page("p", 12).page_content), appendix, body_with_many_citations]
    assert reference_pages(pages) == {("p", 10), ("p", 11), ("p", 12)}


def test_heading_at_top_skips_that_page_too():
    from research_rag.questions import reference_pages
    pages = [page("r", 5, "Preprint.\nReferences\n" + ref_page("r", 5).page_content), page("r", 6, "Appendix " * 200)]
    assert reference_pages(pages) == {("r", 5)}


def test_sample_pages_skips_excluded_pages():
    pages = [page("a", i) for i in range(4)]
    picked = sample_pages(pages, per_paper=4, exclude={("a", 0), ("a", 1)})
    assert sorted(p.metadata["page"] for p in picked) == [2, 3]


def test_save_jsonl_can_append(tmp_path):
    path = tmp_path / "q.jsonl"
    save_jsonl([{"id": "a"}], path)
    save_jsonl([{"id": "b"}], path, append=True)
    assert [r["id"] for r in load_jsonl(path)] == ["a", "b"]
