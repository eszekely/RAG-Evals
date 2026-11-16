import pytest

from rag.eval.threshold import candidates, rates, recommend, sweep

# (best retrieval score, answerable)
SEPARABLE = [(0.88, True), (0.89, True), (0.87, True), (0.84, False), (0.86, False)]


def test_candidates_are_midpoints_plus_both_ends():
    assert candidates([0.80, 0.90, 0.80]) == [0.799, 0.85, 0.901]


def test_rates_refuse_strictly_below_threshold():
    r = rates(SEPARABLE, 0.865)
    assert (r["false_refusal_rate"], r["unanswerable_caught_rate"]) == (0.0, 1.0)


def test_recommend_catches_all_when_separable():
    chosen = recommend(sweep(SEPARABLE), max_false_refusal=0.0)
    assert chosen["threshold"] == pytest.approx(0.865)  # midpoint of 0.86 and 0.87
    assert chosen["unanswerable_caught_rate"] == 1.0


def test_recommend_respects_false_refusal_limit_when_overlapping():
    # One unanswerable question scores higher than every answerable one.
    rows = [(0.88, True), (0.87, True), (0.84, False), (0.90, False)]
    chosen = recommend(sweep(rows), max_false_refusal=0.0)
    assert chosen["false_refusal_rate"] == 0.0
    assert chosen["unanswerable_caught_rate"] == 0.5
    assert chosen["threshold"] == pytest.approx(0.855)


def test_lowest_threshold_wins_ties():
    rows = [(0.9, True), (0.5, False)]
    assert recommend(sweep(rows), max_false_refusal=0.0)["threshold"] == pytest.approx(0.7)


def test_no_recommendation_without_unanswerable_dev_questions():
    rows = [(0.88, True), (0.89, True)]
    assert recommend(sweep(rows), max_false_refusal=0.05) is None
