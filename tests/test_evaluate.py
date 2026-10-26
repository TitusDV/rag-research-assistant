from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.evaluate import TOP_K, markdown_table, rank_questions, run_grid
from research_rag.retrievers import build_retriever

PAGES = [
    Document(page_content="Index hot-swapping lets RAG update its knowledge at test time. " * 5, metadata={"paper": "a", "page": 0}),
    Document(page_content="Dense passage retrieval uses two BERT encoders and the CLS token. " * 5, metadata={"paper": "b", "page": 3}),
    Document(page_content="Performance saturates long before retriever recall does. " * 5, metadata={"paper": "c", "page": 9}),
]
QUESTIONS = [
    {"id": "b-p3", "question": "Which BERT encoders and token does dense passage retrieval use?", "paper": "b", "page": 3},
    {"id": "a-p0", "question": "How does index hot-swapping update knowledge?", "paper": "a", "page": 0},
]
EMB = DeterministicFakeEmbedding(size=16)


def test_bm25_ranks_the_gold_page_first():
    ranks = rank_questions(build_retriever("bm25", PAGES, EMB, k=TOP_K), QUESTIONS)
    assert ranks == [1, 1]


def test_grid_scores_every_setup():
    rows, ranks = run_grid(PAGES, QUESTIONS, EMB, chunk_sizes=[None, 60], kinds=["bm25", "dense", "hybrid"])
    assert len(rows) == 6 and set(ranks) == {f"{k}/{c}" for k in ["bm25", "dense", "hybrid"] for c in ["page", 60]}
    assert all(0 <= r["hit@1"] <= r["hit@3"] <= r["hit@5"] <= 1 for r in rows)
    assert rows[0]["chunks"] == 3 and rows[3]["chunks"] > 3


def test_markdown_table_is_sorted_by_mrr():
    rows = [{"retriever": "bm25", "chunk": 500, "chunks": 9, "hit@1": .5, "hit@3": .6, "hit@5": .7, "mrr@10": .55},
            {"retriever": "hybrid", "chunk": 500, "chunks": 9, "hit@1": .6, "hit@3": .7, "hit@5": .8, "mrr@10": .65}]
    table = markdown_table(rows, n_questions=2)
    assert table.index("| hybrid |") < table.index("| bm25 |")
