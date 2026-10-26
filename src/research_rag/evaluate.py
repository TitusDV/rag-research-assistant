"""Score every retriever and chunk size on the approved test questions."""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .chunking import chunk_pages, location
from .metrics import first_relevant_rank, hit_rate_at_k, mean_reciprocal_rank
from .retrievers import build_retriever

TOP_K = 10  # every retriever is cut to its top 10 results, so all are judged on the same list length


def rank_questions(retriever, questions: list[dict]) -> list[int | None]:
    """For each question, the rank of the first result from its gold page (None if not in the top 10)."""
    ranks = []
    for q in questions:
        retrieved = [location(d) for d in retriever.invoke(q["question"])][:TOP_K]
        ranks.append(first_relevant_rank(retrieved, (q["paper"], q["page"])))
    return ranks


def run_grid(pages: list[Document], questions: list[dict], embeddings: Embeddings,
             chunk_sizes: list[int | None], kinds: list[str]) -> tuple[list[dict], dict]:
    """Return (summary rows, per-question ranks keyed by 'kind/chunk')."""
    rows, ranks_by_setup = [], {}
    for size in chunk_sizes:
        chunks = chunk_pages(pages, size, chunk_overlap=(size // 10 if size else 0))
        for kind in kinds:
            ranks = rank_questions(build_retriever(kind, chunks, embeddings, k=TOP_K), questions)
            label = f"{kind}/{size or 'page'}"
            ranks_by_setup[label] = ranks
            rows.append({
                "retriever": kind,
                "chunk": size or "page",
                "chunks": len(chunks),
                "hit@1": hit_rate_at_k(ranks, 1),
                "hit@3": hit_rate_at_k(ranks, 3),
                "hit@5": hit_rate_at_k(ranks, 5),
                "mrr@10": mean_reciprocal_rank(ranks),
            })
    return rows, ranks_by_setup


def markdown_table(rows: list[dict], n_questions: int) -> str:
    lines = [f"Approved questions: {n_questions}. Results are cut to the top {TOP_K}.", "",
             "| Retriever | Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR@10 |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["mrr@10"]):
        lines.append(f"| {r['retriever']} | {r['chunk']} | {r['chunks']} | {r['hit@1']:.2f} | "
                     f"{r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['mrr@10']:.3f} |")
    return "\n".join(lines) + "\n"
