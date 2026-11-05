from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.evaluate import run_grid
from research_rag.rerank import RerankingRetriever

DOCS = [Document(page_content=t, metadata={"paper": "p", "page": i})
        for i, t in enumerate(["about cats", "about dogs and training", "about birds"])]


class Fixed:
    def invoke(self, question):
        return DOCS


def keyword_scorer(question, passages):
    return [sum(w in p for w in question.split()) for p in passages]


def test_reranker_puts_the_best_scored_passage_first_and_keeps_top_n():
    out = RerankingRetriever(Fixed(), keyword_scorer, candidates=3, top_n=2).invoke("dogs training")
    assert [d.metadata["page"] for d in out] == [1, 0]  # page 1 scores 2; ties keep the first-stage order


def test_reranker_only_rereads_the_first_candidates():
    seen = []
    RerankingRetriever(Fixed(), lambda q, ps: seen.extend(ps) or [0] * len(ps), candidates=2).invoke("x")
    assert seen == ["about cats", "about dogs and training"]


def test_grid_adds_reranked_setups_and_timing():
    pages = [Document(page_content=f"topic {i} " * 30, metadata={"paper": "p", "page": i}) for i in range(4)]
    qs = [{"id": str(i), "question": f"topic {i}", "paper": "p", "page": i} for i in range(4)]
    rows, ranks = run_grid(pages, qs, DeterministicFakeEmbedding(size=8), [None], ["bm25", "hybrid"], scorer=keyword_scorer)
    assert [r["retriever"] for r in rows] == ["bm25", "hybrid", "hybrid+rerank"]
    assert "hybrid+rerank/page" in ranks and all(r["ms"] >= 0 for r in rows)
