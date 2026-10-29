"""HTML for the web app's answer panel, and the daily question limit."""
import datetime
import html
import re
import threading

from .papers import PAPERS


class DailyLimit:
    """Allows a fixed number of questions per UTC day across all visitors. Resets when the app restarts."""

    def __init__(self, limit: int):
        self.limit = limit
        self._day = None
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self._day:
            self._day, self._used = today, 0

    def take(self) -> bool:
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return self.limit - self._used


def _with_citations(answer: str) -> str:
    """Escape the answer and turn [2] into a small badge."""
    text = html.escape(answer)
    text = re.sub(r"[ \t]*\[(\d+)\]", r'&nbsp;<sup class="cite">\1</sup>', text)
    return "".join(f"<p>{part}</p>" for part in text.split("\n") if part.strip())


def render_empty() -> str:
    return ('<div class="panel empty"><p class="label">Answer</p>'
            "<p>Ask a question, or pick one of the examples. Answers come only from the five papers, "
            "with the page each claim came from.</p></div>")


def render_message(text: str) -> str:
    return f'<div class="panel empty"><p class="label">Answer</p><p>{html.escape(text)}</p></div>'


def render_result(result: dict) -> str:
    if not result["answered"]:
        return ('<div class="panel not-covered"><p class="label">Not in the papers</p>'
                f"<p>{html.escape(result['answer'])} The assistant only answers from the five papers "
                "listed on this page, so it won't guess.</p></div>")

    cards = "".join(
        f'<a class="source" href="https://arxiv.org/pdf/{html.escape(s["paper"])}#page={s["page"]}" '
        f'target="_blank" rel="noopener">'
        f'<span class="num">{s["n"]}</span>'
        f'<span class="meta"><strong>{html.escape(s["title"])}</strong>'
        f'<span>Page {s["page"]} · arXiv {html.escape(s["paper"])}</span></span></a>'
        for s in result["sources"]
    )
    sources = f'<p class="label">Sources</p><div class="sources">{cards}</div>' if cards else ""
    return f'<div class="panel"><p class="label">Answer</p>{_with_citations(result["answer"])}{sources}</div>'


def render_papers() -> str:
    items = "".join(
        f'<li><a href="https://arxiv.org/abs/{pid}" target="_blank" rel="noopener">{html.escape(title)}</a></li>'
        for pid, title in PAPERS.items()
    )
    return f'<ol class="papers">{items}</ol>'
