from research_rag.ui import DailyLimit, render_result


def test_daily_limit_stops_at_the_limit():
    limit = DailyLimit(2)
    assert limit.take() and limit.take()
    assert not limit.take()
    assert limit.remaining() == 0


def test_answer_is_escaped_and_citations_become_badges():
    html = render_result({"answered": True, "answer": "Use <b>BM25</b> [1].",
                          "sources": [{"n": 1, "paper": "2005.11401", "title": "RAG", "page": 3}]})
    assert "&lt;b&gt;" in html and "<b>" not in html
    assert '<sup class="cite">1</sup>' in html
    assert "arxiv.org/pdf/2005.11401#page=3" in html


def test_not_covered_has_no_sources():
    html = render_result({"answered": False, "answer": "The papers don't cover this.", "sources": []})
    assert "not-covered" in html and "source" not in html.replace("not-covered", "")
