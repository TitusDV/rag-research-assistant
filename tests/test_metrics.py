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
