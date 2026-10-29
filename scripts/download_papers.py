"""Download the papers listed in research_rag/papers.py into data/papers/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_rag.papers import download  # noqa: E402

if __name__ == "__main__":
    download(ROOT / "data" / "papers")
