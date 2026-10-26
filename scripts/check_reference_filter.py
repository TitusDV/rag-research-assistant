"""List the pages the reference filter skips, so you can confirm it is right."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_rag.chunking import load_papers, location  # noqa: E402
from research_rag.questions import reference_pages, year_density  # noqa: E402

pages = load_papers(ROOT / "data" / "papers")
skipped = reference_pages(pages)
for page in pages:
    text = page.page_content.strip()
    flag = "SKIP" if location(page) in skipped else "    "
    start = " ".join(text.split())[:60]
    print(f"{flag}  {page.metadata['paper']}  page {page.metadata['page'] + 1:>2}  "
          f"years/1000 chars {year_density(text):4.1f}  {start}")
