from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.answer import NOT_COVERED, Assistant, GeneratedAnswer, excerpt, format_context

PAGES = [
    Document(page_content="Index hot-swapping lets RAG update its knowledge at test time without retraining. " * 8,
             metadata={"paper": "2005.11401", "page": 6}),
    Document(page_content="DPR uses two independent BERT encoders and takes the CLS token as output. " * 8,
             metadata={"paper": "2004.04906", "page": 2}),
]

# Three retrieved passages: [1] and [3] come from the same page.
PASSAGES = [
    Document(page_content="Hot-swapping the index updates RAG without retraining.", metadata={"paper": "2005.11401", "page": 6}),
    Document(page_content="DPR trains two BERT encoders with in-batch negatives.", metadata={"paper": "2004.04906", "page": 2}),
    Document(page_content="The swapped index covers a newer Wikipedia dump.", metadata={"paper": "2005.11401", "page": 6}),
]


class FixedRetriever:
    def invoke(self, question):
        return PASSAGES


def assistant(answerer, fixed=True):
    a = Assistant(PAGES, DeterministicFakeEmbedding(size=16), answerer, chunk_size=200, k=4)
    if fixed:
        a.retriever = FixedRetriever()
    return a


def ask(answer, citations, answered=True):
    return assistant(lambda q, c: GeneratedAnswer(answered=answered, answer=answer, citations=citations)).ask("q")


def test_context_numbers_passages_and_names_paper_and_page():
    text = format_context(PAGES)
    assert text.startswith("[1] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, page 7\n")
    assert "[2] Dense Passage Retrieval for Open-Domain Question Answering, page 3" in text


def test_real_retriever_end_to_end_gives_matching_badges_and_cards():
    result = assistant(lambda q, c: GeneratedAnswer(answered=True, answer="It swaps the index [1].", citations=[1]),
                       fixed=False).ask("How does hot-swapping work?")
    assert result["answered"] and result["answer"] == "It swaps the index [1]."
    assert [s["n"] for s in result["sources"]] == [1]


def test_every_badge_matches_one_card_and_pages_are_not_repeated():
    result = ask("Swap the index [3]. DPR uses two encoders [2]. No retraining [1].", [1, 2, 3])
    # [3] and [1] are the same page, so both become badge 1; DPR becomes badge 2.
    assert result["answer"] == "Swap the index [1]. DPR uses two encoders [2]. No retraining [1]."
    assert [(s["n"], s["paper"], s["page"]) for s in result["sources"]] == [(1, "2005.11401", 7), (2, "2004.04906", 3)]


def test_markers_with_no_passage_are_removed():
    result = ask("It swaps the index [1] [99].", [1, 99])
    assert result["answer"] == "It swaps the index [1]."
    assert [s["n"] for s in result["sources"]] == [1]


def test_repeated_markers_for_one_page_collapse():
    assert ask("It swaps the index [1][3].", [1, 3])["answer"] == "It swaps the index [1]."


def test_citation_list_is_used_when_the_text_has_no_markers():
    result = ask("It swaps the index.", [2])
    assert result["answered"] and [s["paper"] for s in result["sources"]] == ["2004.04906"]


def test_answer_with_no_valid_citation_is_treated_as_unsupported():
    assert ask("It swaps the index [7].", [8]) == {"answered": False, "answer": NOT_COVERED, "sources": []}
    assert ask("It swaps the index.", []) == {"answered": False, "answer": NOT_COVERED, "sources": []}


def test_contradictory_outputs_fail_safe():
    assert ask("anything [1]", [1], answered=False)["answered"] is False
    assert ask(NOT_COVERED, [1])["answered"] is False  # says answered but the text is the refusal
    assert ask("   ", [1])["answered"] is False


def test_sources_carry_a_quote_from_the_cited_passage():
    result = ask("DPR uses two encoders [2].", [2])
    assert result["sources"][0]["quote"] == "DPR trains two BERT encoders with in-batch negatives."


def test_excerpt_is_one_line_and_cut_at_a_word():
    text = "word " * 100
    short = excerpt("a\n  b\tc")
    long = excerpt(text, limit=23)
    assert short == "a b c"
    assert long == "word word word word…"
