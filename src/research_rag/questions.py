"""Build the test set: questions written from individual pages of the papers.

A model reads one page and writes a question that the page answers, plus a short
exact quote from the page as evidence. A question is kept only if that quote
really appears on the page, which catches questions the model made up.
Kept questions start as 'unreviewed'; a person then approves or rejects each one
(scripts/review_questions.py) and only approved questions are used for scoring.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .chunking import location
from .papers import PAPERS


class GeneratedQuestion(BaseModel):
    question: str = Field(description="A specific question that this page answers, understandable on its own")
    answer: str = Field(description="A short answer, one or two sentences")
    evidence: str = Field(description="An exact quote of 5 to 30 words copied from the page that supports the answer")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write test questions for a search system over research papers. "
     "Given one page from a paper, write one question that this page answers. "
     "The question must make sense on its own to someone who has not seen the page: "
     "name the method or concept instead of saying 'this paper', 'this page' or 'the authors'. "
     "Ask about a specific fact, result, definition or design choice, not a vague summary. "
     "Copy the evidence word for word from the page."),
    ("human", "Paper: {title}\n\nPage text:\n{page_text}"),
])

Ask = Callable[[str, str], GeneratedQuestion]


def make_ask(chat_model) -> Ask:
    """Wrap a chat model so it returns a validated GeneratedQuestion."""
    chain = PROMPT | chat_model.with_structured_output(GeneratedQuestion)
    return lambda title, page_text: chain.invoke({"title": title, "page_text": page_text})


def usable(page: Document, min_chars: int = 800) -> bool:
    """Skip near-empty pages and reference lists, which make poor questions."""
    text = page.page_content.strip()
    return len(text) >= min_chars and not text.lower().startswith("references")


def sample_pages(pages: list[Document], per_paper: int, seed: int = 0) -> list[Document]:
    """Pick the same number of usable pages from each paper, reproducibly."""
    by_paper = defaultdict(list)
    for page in pages:
        if usable(page):
            by_paper[page.metadata["paper"]].append(page)
    rng = random.Random(seed)
    picked = []
    for paper in sorted(by_paper):
        candidates = by_paper[paper]
        picked += rng.sample(candidates, min(per_paper, len(candidates)))
    return picked


def _normalise(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)  # join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_on_page(evidence: str, page_text: str) -> bool:
    return bool(evidence.strip()) and _normalise(evidence) in _normalise(page_text)


def generate_questions(pages: list[Document], ask: Ask) -> tuple[list[dict], list[dict]]:
    """Return (kept, dropped). Dropped questions failed the evidence check or errored."""
    kept, dropped = [], []
    for page in pages:
        paper, page_no = location(page)
        record = {"paper": paper, "page": page_no}
        try:
            q = ask(PAPERS.get(paper, paper), page.page_content)
        except Exception as err:  # one bad page should not stop the whole run
            dropped.append({**record, "reason": f"model error: {err}"})
            continue
        record.update(q.model_dump())
        if evidence_on_page(q.evidence, page.page_content):
            kept.append({"id": f"{paper}-p{page_no}", **record, "status": "unreviewed"})
        else:
            dropped.append({**record, "reason": "evidence not found on page"})
    return kept, dropped


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
