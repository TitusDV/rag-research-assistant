"""Answer questions from the papers, citing paper and page, and saying so when the papers don't cover it."""
from typing import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .chunking import chunk_pages, location
from .papers import PAPERS
from .retrievers import build_retriever

NOT_COVERED = "The papers don't cover this."


class GeneratedAnswer(BaseModel):
    answered: bool = Field(description="False if the passages do not contain the answer")
    answer: str = Field(description="The answer, using only the passages, with [n] citations after each claim")
    citations: list[int] = Field(description="The passage numbers the answer relies on")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about research papers using ONLY the numbered passages given. "
     "Put the passage number in square brackets after each claim, like [2]. "
     "Do not use outside knowledge. If the passages do not contain the answer, set answered to false "
     f"and reply exactly: {NOT_COVERED}"),
    ("human", "Passages:\n{context}\n\nQuestion: {question}"),
])

Answerer = Callable[[str, str], GeneratedAnswer]


def make_answerer(chat_model) -> Answerer:
    chain = PROMPT | chat_model.with_structured_output(GeneratedAnswer)
    return lambda question, context: chain.invoke({"question": question, "context": context})


def format_context(docs: list[Document]) -> str:
    blocks = []
    for n, doc in enumerate(docs, start=1):
        paper, page = location(doc)
        blocks.append(f"[{n}] {PAPERS.get(paper, paper)}, page {page + 1}\n{doc.page_content}")
    return "\n\n".join(blocks)


class Assistant:
    """Hybrid retrieval over 500-character chunks (see eval/results.md), then a grounded answer."""

    def __init__(self, pages: list[Document], embeddings: Embeddings, answerer: Answerer,
                 chunk_size: int = 500, k: int = 6):
        chunks = chunk_pages(pages, chunk_size, chunk_overlap=chunk_size // 10)
        self.retriever = build_retriever("hybrid", chunks, embeddings, k=k)
        self.answerer = answerer
        self.k = k

    def ask(self, question: str, docs: list[Document] | None = None) -> dict:
        """Pass `docs` to skip retrieval, when it has already been done (the web app does this)."""
        if docs is None:
            docs = self.retriever.invoke(question)
        docs = docs[: self.k]
        result = self.answerer(question, format_context(docs))
        if not result.answered:
            return {"answered": False, "answer": NOT_COVERED, "sources": []}

        sources, seen = [], set()
        for n in result.citations:
            if 1 <= n <= len(docs):  # ignore citation numbers that don't match a passage
                paper, page = location(docs[n - 1])
                if (paper, page) not in seen:
                    seen.add((paper, page))
                    sources.append({"n": n, "paper": paper, "title": PAPERS.get(paper, paper), "page": page + 1})
        return {"answered": True, "answer": result.answer, "sources": sources}
