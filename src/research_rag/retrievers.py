from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

KINDS = ("bm25", "dense", "hybrid")


def build_retriever(kind: str, docs: list[Document], embeddings: Embeddings, k: int = 4) -> BaseRetriever:
    """Build one of three retrievers over the same chunks.

    bm25:   keyword search
    dense:  embedding search (DocArrayInMemorySearch)
    hybrid: both, merged by EnsembleRetriever with equal weights
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    if kind in ("bm25", "hybrid"):
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = k
    if kind in ("dense", "hybrid"):
        dense = DocArrayInMemorySearch.from_documents(docs, embedding=embeddings).as_retriever(
            search_kwargs={"k": k}
        )

    if kind == "bm25":
        return bm25
    if kind == "dense":
        return dense
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])
