"""Retrieval metrics.

Each test question has one gold location: the (paper, page) its answer was
written from. A retrieved chunk counts as relevant if it comes from that
location. Judging by page rather than by chunk keeps the gold label the same
when the chunk size changes, so chunk sizes are compared fairly.

With one gold location per question, recall@k and hit rate@k are the same
number, so we report hit rate@k and MRR.
"""

Location = tuple[str, int]


def first_relevant_rank(retrieved: list[Location], gold: Location) -> int | None:
    """1-based rank of the first chunk from the gold location, or None if it was not retrieved."""
    for rank, loc in enumerate(retrieved, start=1):
        if loc == gold:
            return rank
    return None


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose gold location appears in the top k results."""
    if not ranks:
        raise ValueError("no questions to score")
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank of the first relevant result (0 when none was retrieved)."""
    if not ranks:
        raise ValueError("no questions to score")
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
