"""Retrieval metrics.

Each test question has one gold location: the (paper, page) its answer was
written from. A retrieved chunk counts as relevant if it comes from that
location. Judging by page rather than by chunk keeps the gold label the same
when the chunk size changes, so chunk sizes are compared fairly.

With one gold location per question, recall@k and hit rate@k are the same
number, so we report hit rate@k and MRR.
"""

import math
import random

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


def cohen_kappa(both_yes: int, only_a_yes: int, only_b_yes: int, both_no: int) -> float:
    """Agreement between two raters on yes/no decisions, corrected for agreement expected by chance.

    1 is perfect agreement, 0 is no better than chance. Counts come from the 2x2 table.
    """
    n = both_yes + only_a_yes + only_b_yes + both_no
    if n == 0:
        raise ValueError("no decisions to compare")
    observed = (both_yes + both_no) / n
    a_yes = (both_yes + only_a_yes) / n
    b_yes = (both_yes + only_b_yes) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if expected == 1:  # both raters gave the same single answer every time
        return 1.0
    return (observed - expected) / (1 - expected)


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """Normalised discounted cumulative gain with one relevant page per question.

    A hit at rank r scores 1/log2(r + 1): 1.0 at rank 1, 0.63 at rank 2, 0.5 at rank 3. Unlike MRR,
    lower ranks are discounted gently, so it rewards "near the top" rather than only "first".
    """
    if not ranks:
        raise ValueError("no questions to score")
    return sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / len(ranks)


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the mean of per-question scores, by resampling questions with replacement."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_resamples))
    tail = (1 - level) / 2
    return means[int(tail * n_resamples)], means[int((1 - tail) * n_resamples) - 1]


def paired_difference_ci(a: list[float], b: list[float], **kwargs) -> tuple[float, float, float]:
    """Mean of a - b over the same questions, with a bootstrap confidence interval.

    If the interval includes 0, the data can't tell the two setups apart.
    """
    if len(a) != len(b):
        raise ValueError("a and b must score the same questions")
    diffs = [x - y for x, y in zip(a, b)]
    low, high = bootstrap_ci(diffs, **kwargs)
    return sum(diffs) / len(diffs), low, high
