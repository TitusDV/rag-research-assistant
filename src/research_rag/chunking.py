from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_papers(paper_dir: str | Path) -> list[Document]:
    """Load every PDF in paper_dir as one Document per page.

    Each page gets metadata 'paper' (the arXiv id, taken from the file name) and
    'page' (0-based), which together say exactly where a passage came from.
    """
    pages = []
    for pdf in sorted(Path(paper_dir).glob("*.pdf")):
        for page in PyPDFLoader(str(pdf)).load():
            page.metadata = {"paper": pdf.stem, "page": page.metadata["page"]}
            pages.append(page)
    if not pages:
        raise FileNotFoundError(f"no PDFs in {paper_dir}; run scripts/download_papers.py first")
    return pages


def chunk_pages(pages: list[Document], chunk_size: int | None, chunk_overlap: int = 0) -> list[Document]:
    """Split pages into chunks; chunk_size=None keeps whole pages.

    The splitter copies each page's metadata onto its chunks, so every chunk
    still knows its paper and page. The evaluation relies on that.
    """
    if chunk_size is None:
        return pages
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def location(doc: Document) -> tuple[str, int]:
    """Where a chunk came from: (paper id, page)."""
    return doc.metadata["paper"], doc.metadata["page"]
