"""Score every retriever and chunk size on the approved test questions."""
import re
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .chunking import chunk_pages, location
from .metrics import bootstrap_ci, paired_difference_ci, first_relevant_rank, hit_rate_at_k, mean_reciprocal_rank, ndcg_at_k
from .rerank import RerankingRetriever, Scorer
from .retrievers import build_retriever

RERANK_CANDIDATES = 20  # how many first-stage results the reranker rereads
TOP_K = 10  # every retriever is cut to its top 10 results, so all are judged on the same list length


EVIDENCE_COVERAGE = 0.8  # share of the evidence quote's words a passage must contain to count as holding it


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def holds_evidence(passage: str, evidence: str) -> bool:
    """True if the passage contains most of the evidence quote.

    Measured against the shorter of the two, so a small chunk that lies wholly inside a long quote counts too.
    """
    ev, words = _words(evidence), set(_words(passage))
    if not ev:
        return False
    found = sum(w in words for w in ev)
    return found / min(len(ev), len(words) or 1) >= EVIDENCE_COVERAGE


def rank_questions(retriever, questions: list[dict]) -> list[int | None]:
    """For each question, the rank of the first result from its gold page (None if not in the top 10)."""
    return _ranks(retriever, questions)[0]


def _ranks(retriever, questions: list[dict]) -> tuple[list[int | None], list[int | None]]:
    """Page ranks, and evidence ranks: the first result from the gold page that also holds the evidence quote."""
    page_ranks, evidence_ranks = [], []
    for q in questions:
        docs = retriever.invoke(q["question"])[:TOP_K]
        page_ranks.append(first_relevant_rank([location(d) for d in docs], (q["paper"], q["page"])))
        evidence = q.get("evidence", "")
        evidence_ranks.append(next((i for i, d in enumerate(docs, start=1)
                                    if location(d) == (q["paper"], q["page"]) and holds_evidence(d.page_content, evidence)),
                                   None))
    return page_ranks, evidence_ranks


def per_question(ranks: list[int | None], k: int | None = None) -> list[float]:
    """Each question's score: 1/rank (for MRR), or 1/0 for a hit within k."""
    if k is None:
        return [1 / r if r else 0.0 for r in ranks]
    return [1.0 if r is not None and r <= k else 0.0 for r in ranks]


def run_grid(pages: list[Document], questions: list[dict], embeddings: Embeddings,
             chunk_sizes: list[int | None], kinds: list[str], scorer: Scorer | None = None,
             rerank_kinds: tuple[str, ...] = ("hybrid",)) -> tuple[list[dict], dict]:
    """Return (summary rows, per-question ranks keyed by 'kind/chunk').

    With a scorer, each kind in rerank_kinds is also scored with a cross-encoder reranking its top 20,
    labelled 'kind+rerank'.
    """
    rows, ranks_by_setup = [], {}
    has_evidence = all(q.get("evidence") for q in questions)
    for size in chunk_sizes:
        chunks = chunk_pages(pages, size, chunk_overlap=(size // 10 if size else 0))
        setups = []
        for kind in kinds:
            setups.append((kind, build_retriever(kind, chunks, embeddings, k=TOP_K)))
            if scorer is not None and kind in rerank_kinds:
                wide = build_retriever(kind, chunks, embeddings, k=RERANK_CANDIDATES)
                setups.append((f"{kind}+rerank", RerankingRetriever(wide, scorer, RERANK_CANDIDATES, TOP_K)))
        for name, retriever in setups:
            start = time.perf_counter()
            ranks, ev_ranks = _ranks(retriever, questions)
            ms = (time.perf_counter() - start) * 1000 / len(questions)
            ranks_by_setup[f"{name}/{size or 'page'}"] = ranks
            rows.append({
                "retriever": name,
                "chunk": size or "page",
                "chunks": len(chunks),
                "hit@1": hit_rate_at_k(ranks, 1),
                "hit@3": hit_rate_at_k(ranks, 3),
                "hit@5": hit_rate_at_k(ranks, 5),
                "mrr@10": mean_reciprocal_rank(ranks),
                "ndcg@10": ndcg_at_k(ranks, TOP_K),
                "evidence@5": hit_rate_at_k(ev_ranks, 5) if has_evidence else None,
                "hit@5_ci": bootstrap_ci(per_question(ranks, 5)),
                "mrr_ci": bootstrap_ci(per_question(ranks)),
                "ms": ms,
            })
    return rows, ranks_by_setup


def markdown_table(rows: list[dict], n_questions: int) -> str:
    lines = [f"Questions: {n_questions}. Results are cut to the top {TOP_K}. Brackets: 95% bootstrap interval.", "",
             "| Retriever | Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | Evidence@5 | MRR@10 | nDCG@10 | ms/question |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["mrr@10"]):
        ev = f"{r['evidence@5']:.2f}" if r.get("evidence@5") is not None else "n/a"
        h_ci, m_ci = r.get("hit@5_ci"), r.get("mrr_ci")
        hit5 = f"{r['hit@5']:.2f}" + (f" ({h_ci[0]:.2f}-{h_ci[1]:.2f})" if h_ci else "")
        mrr = f"{r['mrr@10']:.3f}" + (f" ({m_ci[0]:.2f}-{m_ci[1]:.2f})" if m_ci else "")
        ndcg = f"{r['ndcg@10']:.3f}" if "ndcg@10" in r else "n/a"
        ms = f"{r['ms']:.0f}" if "ms" in r else "n/a"
        lines.append(f"| {r['retriever']} | {r['chunk']} | {r['chunks']} | {r['hit@1']:.2f} | "
                     f"{r['hit@3']:.2f} | {hit5} | {ev} | {mrr} | {ndcg} | {ms} |")
    return "\n".join(lines) + "\n"


def compare_to(ranks_by_setup: dict, baseline: str) -> str:
    """How every setup differs from the baseline on the same questions, with 95% intervals.

    An interval that includes 0 means the questions can't tell the two setups apart.
    """
    base = ranks_by_setup[baseline]
    lines = [f"## Compared with {baseline} (the setup the app uses)", "",
             "Difference = other setup minus this one, on the same questions. "
             "If the interval includes 0, the difference could be chance.", "",
             "| Setup | Hit@5 difference | MRR difference | Clear difference? |", "|---|---|---|---|"]
    for label, ranks in ranks_by_setup.items():
        if label == baseline:
            continue
        h, hl, hh = paired_difference_ci(per_question(ranks, 5), per_question(base, 5))
        m, ml, mh = paired_difference_ci(per_question(ranks), per_question(base))
        clear = [name for name, lo, hi in (("Hit@5", hl, hh), ("MRR", ml, mh)) if lo > 0 or hi < 0]
        lines.append(f"| {label} | {h:+.2f} ({hl:+.2f} to {hh:+.2f}) | {m:+.3f} ({ml:+.2f} to {mh:+.2f}) | "
                     f"{', '.join(clear) if clear else 'no'} |")
    return "\n".join(lines) + "\n"
