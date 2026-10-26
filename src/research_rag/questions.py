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


YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")


def year_density(text: str) -> float:
    """Years per 1000 characters. Reference lists have one in nearly every entry."""
    return len(YEAR.findall(text)) * 1000 / max(len(text), 1)


def _is_references_heading(line: str) -> bool:
    # PDFs often break small-caps headings into pieces, e.g. "R EFERENCES".
    return line.replace(" ", "").lower() in ("references", "bibliography")


def reference_pages(pages: list[Document], end_below: float = 2.0) -> set[tuple[str, int]]:
    """Find the pages that hold each paper's reference list.

    The list starts at the "References" heading. The page with the heading is
    kept unless the heading is at its top, because that page usually ends the
    main text. The list carries on until the first page whose year density
    drops below end_below: in these papers every reference page is at 2.4 or
    above and every appendix page is at 1.4 or below.
    """
    skipped, in_refs, paper = set(), False, None
    for page in pages:  # pages arrive in order, paper by paper
        if page.metadata["paper"] != paper:
            paper, in_refs = page.metadata["paper"], False
        text = page.page_content.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if in_refs:
            if year_density(text) < end_below:
                in_refs = False
            else:
                skipped.add(location(page))
                continue
        heading_at = next((i for i, line in enumerate(lines) if _is_references_heading(line)), None)
        if heading_at is not None:
            in_refs = True
            if heading_at <= 1:  # heading at the top (after a running header such as "Preprint.")
                skipped.add(location(page))
    return skipped


def usable(page: Document, min_chars: int = 800) -> bool:
    """Skip near-empty pages, which make poor questions."""
    return len(page.page_content.strip()) >= min_chars


def sample_pages(pages: list[Document], per_paper: int, seed: int = 0,
                 exclude: set[tuple[str, int]] = frozenset()) -> list[Document]:
    """Pick the same number of usable pages from each paper, reproducibly.

    Reference-list pages and pages in exclude (already used) are never picked.
    """
    skip = reference_pages(pages) | set(exclude)
    by_paper = defaultdict(list)
    for page in pages:
        if usable(page) and location(page) not in skip:
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


def save_jsonl(rows: list[dict], path: str | Path, append: bool = False) -> None:
    with Path(path).open("a" if append else "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
