"""Second-stage reranking: a cross-encoder rereads the question with each candidate passage and reorders them.

The first-stage retriever (BM25, dense or hybrid) is fast but scores the question and passage separately.
A cross-encoder reads them together, which is slower but usually better at putting the best passage first.
"""
from typing import Callable

from langchain_core.documents import Document

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
Scorer = Callable[[str, list[str]], list[float]]


def cross_encoder_scorer(model_name: str = DEFAULT_MODEL) -> Scorer:
    """Load a sentence-transformers cross-encoder (downloads about 90 MB the first time)."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu")
    return lambda question, passages: [float(s) for s in model.predict([(question, p) for p in passages])]


class RerankingRetriever:
    """Fetch `candidates` passages with a first-stage retriever, then keep the `top_n` the scorer rates highest."""

    def __init__(self, first_stage, scorer: Scorer, candidates: int = 20, top_n: int = 10):
        self.first_stage = first_stage
        self.scorer = scorer
        self.candidates = candidates
        self.top_n = top_n

    def invoke(self, question: str) -> list[Document]:
        docs = self.first_stage.invoke(question)[: self.candidates]
        if not docs:
            return []
        scores = self.scorer(question, [d.page_content for d in docs])
        ranked = sorted(zip(scores, range(len(docs)), docs), key=lambda t: (-t[0], t[1]))  # ties keep first-stage order
        return [d for _, _, d in ranked][: self.top_n]
