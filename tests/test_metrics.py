import pytest

from research_rag.metrics import first_relevant_rank, hit_rate_at_k, mean_reciprocal_rank


def test_first_relevant_rank_needs_paper_and_page_to_match():
    retrieved = [("2004.04906", 3), ("2005.11401", 2), ("2005.11401", 3)]
    assert first_relevant_rank(retrieved, gold=("2005.11401", 3)) == 3
    assert first_relevant_rank(retrieved, gold=("2307.03172", 3)) is None


def test_hit_rate_and_mrr_by_hand():
    ranks = [1, 2, None, 4]
    assert hit_rate_at_k(ranks, k=1) == 0.25
    assert hit_rate_at_k(ranks, k=3) == 0.5
    # (1 + 1/2 + 0 + 1/4) / 4
    assert mean_reciprocal_rank(ranks) == pytest.approx(0.4375)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])


def test_cohen_kappa_corrects_for_chance():
    from research_rag.metrics import cohen_kappa
    assert cohen_kappa(10, 0, 0, 10) == 1.0             # perfect agreement
    assert abs(cohen_kappa(5, 5, 5, 5)) < 1e-9           # 50% agreement with 50/50 raters: chance level
    # 90% raw agreement, but almost everything is "yes", so kappa is much lower than 0.9
    assert round(cohen_kappa(17, 1, 1, 1), 2) == 0.44


def test_ndcg_discounts_lower_ranks_gently():
    from research_rag.metrics import ndcg_at_k
    assert ndcg_at_k([1, 1], 10) == 1.0
    assert round(ndcg_at_k([3, None], 10), 3) == 0.25     # rank 3 scores 0.5, a miss scores 0
    assert ndcg_at_k([11], 10) == 0.0


def test_bootstrap_interval_contains_the_mean_and_narrows_with_more_questions():
    from research_rag.metrics import bootstrap_ci
    few = [1.0, 0.0] * 5
    many = [1.0, 0.0] * 200
    lo_few, hi_few = bootstrap_ci(few)
    lo_many, hi_many = bootstrap_ci(many)
    assert lo_few <= 0.5 <= hi_few and lo_many <= 0.5 <= hi_many
    assert (hi_many - lo_many) < (hi_few - lo_few)


def test_paired_difference_spots_a_clear_gap_and_a_tie():
    from research_rag.metrics import paired_difference_ci
    mean, lo, hi = paired_difference_ci([1.0] * 30, [0.0] * 30)
    assert mean == 1.0 and lo > 0
    mean, lo, hi = paired_difference_ci([1.0, 0.0] * 15, [0.0, 1.0] * 15)
    assert mean == 0.0 and lo < 0 < hi
