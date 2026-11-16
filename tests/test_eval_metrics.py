"""Metric functions on toy data with answers worked out by hand."""

import pytest

from rag.eval.answer_metrics import (
    citation_correctness,
    citation_validity,
    cites_gold,
    refusal_metrics,
)
from rag.eval.retrieval_metrics import (
    doc_hit,
    first_hit_rank,
    page_hit,
    retrieval_report,
    summarize_ranks,
)
from rag.eval.stats import bootstrap_ci


def chunk(doc: str, page: int) -> dict:
    return {"doc": doc, "page": page}


# --- retrieval ---------------------------------------------------------------


def test_first_hit_rank_page_versus_doc_level(make_question):
    q = make_question(gold_docs=["a.pdf"], gold_pages=[5])
    retrieved = [chunk("b.pdf", 5), chunk("a.pdf", 2), chunk("a.pdf", 5)]
    assert first_hit_rank(retrieved, q, page_hit) == 3  # b.pdf p5 is the wrong document
    assert first_hit_rank(retrieved, q, doc_hit) == 2  # right document, wrong page
    assert first_hit_rank([chunk("b.pdf", 1)], q, page_hit) is None


def test_hit_at_k_and_mrr_by_hand():
    s = summarize_ranks([1, 3, None, 2])
    assert (s["hit@1"], s["hit@3"], s["hit@5"], s["hit@10"]) == (0.25, 0.75, 0.75, 0.75)
    assert s["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert summarize_ranks([]) == {"n": 0}


def test_report_per_type_skips_unanswerable(make_question):
    questions = [
        make_question(id="a", type="number"),
        make_question(id="b", type="single_fact"),
        make_question(id="c", type="unanswerable"),
    ]
    report = retrieval_report(questions, {"a": 1, "b": None})
    assert report["overall"]["n"] == 2
    assert report["overall"]["hit@1"] == 0.5
    assert set(report["by_type"]) == {"number", "single_fact"}


# --- answers -------------------------------------------------------------------


def test_refusal_metrics_by_hand():
    # (answerable, refused): correct, false refusal, caught, missed
    m = refusal_metrics([(True, False), (True, True), (False, True), (False, False)])
    assert m["refusal_accuracy"] == 0.5
    assert m["false_refusal_rate"] == 0.5
    assert m["unanswerable_caught_rate"] == 0.5


def test_citation_validity_counts_citations_not_answers():
    lists = [[{"verified": True}, {"verified": False}], [{"verified": True}], []]
    assert citation_validity(lists) == {"n_citations": 3, "rate": 2 / 3}


def test_citation_correctness_needs_a_verified_gold_citation(make_question):
    q = make_question(gold_docs=["a.pdf"], gold_pages=[5])
    gold = {"doc": "a.pdf", "pdf_page": 5, "verified": True}
    wrong_page = {"doc": "a.pdf", "pdf_page": 6, "verified": True}
    assert cites_gold([wrong_page, gold], q)
    assert not cites_gold([wrong_page], q)
    assert citation_correctness([(q, [gold]), (q, []), (q, [wrong_page])])["rate"] == 1 / 3


# --- bootstrap -----------------------------------------------------------------


def test_bootstrap_is_deterministic_with_fixed_seed():
    values = [1, 0, 1, 1, 0, 1, 0, 1]
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)


def test_bootstrap_interval_contains_mean_and_constant_data_has_no_width():
    ci = bootstrap_ci([1, 0, 1, 1, 0])
    assert ci["low"] <= ci["mean"] == 0.6 <= ci["high"]
    assert bootstrap_ci([1, 1, 1]) == {"mean": 1.0, "low": 1.0, "high": 1.0, "n": 3}
    assert bootstrap_ci([])["mean"] is None
