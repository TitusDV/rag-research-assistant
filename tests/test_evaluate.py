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


def test_evidence_needs_the_whole_quote_word_for_word():
    from research_rag.evaluate import holds_evidence
    quote = "the exact match score on SQuAD v1.1 drops from above 80% to less than 40%"
    assert holds_evidence("As shown, The exact match score on SQuAD v1.1\ndrops from above 80% to less than 40%.", quote)
    assert not holds_evidence("the score", quote)                                          # generic words only
    assert not holds_evidence("the score improves from above 80% to more than 90%", quote)  # contradicts it
    assert not holds_evidence("Index hot-swapping updates knowledge.", quote)


def test_evidence_matching_normalises_line_breaks_and_ligatures():
    from research_rag.evaluate import holds_evidence
    assert holds_evidence("we use the unﬁl-\ntered TriviaQA version", "the unfiltered TriviaQA version")


def test_grid_reports_ndcg_evidence_and_intervals():
    qs = [dict(q, evidence=e) for q, e in zip(QUESTIONS, ["two BERT encoders and the CLS token", "hot-swapping lets RAG update"])]
    rows, ranks = run_grid(PAGES, qs, EMB, chunk_sizes=[None], kinds=["bm25"])
    r = rows[0]
    assert r["ndcg@10"] == 1.0 and r["evidence@5"] == 1.0
    assert r["hit@5_ci"][0] <= r["hit@5"] <= r["hit@5_ci"][1]


def test_comparison_says_when_setups_cannot_be_told_apart():
    from research_rag.evaluate import compare_to
    text = compare_to({"a/500": [1, 1, None, 2], "b/500": [1, 2, None, 1]}, "a/500")
    assert "| b/500 |" in text and text.rstrip().endswith("| no |")
