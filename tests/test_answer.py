from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.answer import NOT_COVERED, Assistant, GeneratedAnswer, format_context

PAGES = [
    Document(page_content="Index hot-swapping lets RAG update its knowledge at test time without retraining. " * 8,
             metadata={"paper": "2005.11401", "page": 6}),
    Document(page_content="DPR uses two independent BERT encoders and takes the CLS token as output. " * 8,
             metadata={"paper": "2004.04906", "page": 2}),
]


def assistant(answerer):
    return Assistant(PAGES, DeterministicFakeEmbedding(size=16), answerer, chunk_size=200, k=4)


def test_context_numbers_passages_and_names_paper_and_page():
    text = format_context(PAGES)
    assert text.startswith("[1] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, page 7\n")
    assert "[2] Dense Passage Retrieval for Open-Domain Question Answering, page 3" in text


def test_answer_cites_real_passages_only_and_once_per_page():
    seen_context = {}

    def answerer(question, context):
        seen_context["text"] = context
        return GeneratedAnswer(answered=True, answer="It swaps the index [1][2].", citations=[1, 2, 1, 99])

    result = assistant(answerer).ask("How does hot-swapping work?")
    assert result["answered"] and result["answer"] == "It swaps the index [1][2]."
    assert len(result["sources"]) == len({(s["paper"], s["page"]) for s in result["sources"]})
    assert all(s["n"] in (1, 2) for s in result["sources"])  # 99 is not a real passage
    assert "[1] " in seen_context["text"]


def test_unanswerable_question_returns_the_fixed_message_and_no_sources():
    result = assistant(lambda q, c: GeneratedAnswer(answered=False, answer="anything", citations=[1])).ask("Who won the 2026 Rugby World Cup?")
    assert result == {"answered": False, "answer": NOT_COVERED, "sources": []}
