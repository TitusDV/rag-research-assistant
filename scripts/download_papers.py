"""Download the papers listed in research_rag/papers.py into data/papers/."""
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_rag.papers import PAPERS, pdf_url  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "papers"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for arxiv_id, title in PAPERS.items():
        target = OUT / f"{arxiv_id}.pdf"
        if target.exists():
            print(f"have  {arxiv_id}  {title}")
            continue
        request = urllib.request.Request(pdf_url(arxiv_id), headers={"User-Agent": "rag-research-assistant"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"got   {arxiv_id}  {title}")
        time.sleep(3)  # arXiv asks automated clients to pause between requests


if __name__ == "__main__":
    main()
