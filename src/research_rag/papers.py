"""The papers the assistant answers questions about.

They are downloaded from arXiv at runtime (scripts/download_papers.py) and never
committed, because arXiv's default licence does not allow redistributing them.
"""
import time
import urllib.request
from pathlib import Path

PAPERS = {
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts",
    "2309.15217": "Ragas: Automated Evaluation of Retrieval Augmented Generation",
    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
}


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def download(out: Path) -> None:
    """Fetch any papers not already in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    for arxiv_id, title in PAPERS.items():
        target = out / f"{arxiv_id}.pdf"
        if target.exists():
            print(f"have  {arxiv_id}  {title}")
            continue
        request = urllib.request.Request(pdf_url(arxiv_id), headers={"User-Agent": "rag-research-assistant"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"got   {arxiv_id}  {title}")
        time.sleep(3)  # arXiv asks automated clients to pause between requests
