"""The papers the assistant answers questions about.

They are downloaded from arXiv at runtime (scripts/download_papers.py) and never
committed, because arXiv's default licence does not allow redistributing them.
"""

PAPERS = {
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts",
    "2309.15217": "Ragas: Automated Evaluation of Retrieval Augmented Generation",
    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
}


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"
