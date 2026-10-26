import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from research_rag.chunking import chunk_pages, load_papers, location
from research_rag.retrievers import build_retriever

PAGES = [
    Document(page_content="RAG combines a retriever with a seq2seq generator.", metadata={"paper": "2005.11401", "page": 0}),
    Document(page_content="Dense passage retrieval encodes questions and passages with two BERT encoders.", metadata={"paper": "2004.04906", "page": 1}),
    Document(page_content="Models use information at the start or end of a long context better than the middle.", metadata={"paper": "2307.03172", "page": 0}),
]
EMB = DeterministicFakeEmbedding(size=16)


def test_chunks_keep_their_location():
    chunks = chunk_pages(PAGES, chunk_size=30)
    assert len(chunks) > len(PAGES)
    assert {location(c) for c in chunks} == {location(p) for p in PAGES}


def test_none_chunk_size_keeps_whole_pages():
    assert chunk_pages(PAGES, chunk_size=None) is PAGES


@pytest.mark.parametrize("kind", ["bm25", "dense", "hybrid"])
def test_each_retriever_returns_located_documents(kind):
    docs = build_retriever(kind, PAGES, EMB, k=2).invoke("dense passage encoders")
    assert 1 <= len(docs) <= 4  # hybrid merges two lists of 2
    assert all(location(d) for d in docs)


def test_bm25_finds_the_keyword_page():
    top = build_retriever("bm25", PAGES, EMB, k=1).invoke("long context middle")[0]
    assert location(top) == ("2307.03172", 0)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        build_retriever("magic", PAGES, EMB)


def test_missing_papers_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_papers"):
        load_papers(tmp_path)


def test_load_papers_labels_pages_with_arxiv_id(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    pdf = fitz.open()
    for text in ("first page", "second page"):
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(tmp_path / "2005.11401.pdf")

    pages = load_papers(tmp_path)
    assert [location(p) for p in pages] == [("2005.11401", 0), ("2005.11401", 1)]
    assert "second page" in pages[1].page_content
