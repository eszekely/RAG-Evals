"""End-to-end pipeline with the fake embedder and fake LLM (no network, no model)."""

import pytest

from rag.answer import answer_question
from rag.config import Settings
from rag.index import build_index
from rag.llm import FakeLLMClient
from rag.prompt import IDK

OPEN = Settings(min_score=0.0)  # threshold never triggers


@pytest.fixture
def setup(tiny_corpus, fake_embedder):
    return build_index(tiny_corpus, fake_embedder), tiny_corpus, fake_embedder


def ask(setup, llm, settings=OPEN, question="glacier mass loss", **kwargs):
    index, chunks, embedder = setup
    return answer_question(question, index, chunks, embedder, llm, settings, **kwargs)


def test_low_retrieval_score_refuses_without_calling_llm(setup):
    llm = FakeLLMClient("should never be used")
    result = ask(setup, llm, Settings(min_score=0.99))

    assert llm.calls == []
    assert result.refused and result.refusal_reason == "low_retrieval_score"
    assert result.answer == IDK
    assert result.retrieved  # still reported, for debugging and logging


def test_end_to_end_answer_with_verified_citation(setup):
    # tiny_corpus: glacier chunk is r.pdf PDF page 1, printed label "11".
    llm = FakeLLMClient("Glaciers lost mass [r.pdf, p.11].")
    result = ask(setup, llm, k=2)

    assert len(llm.calls) == 1
    assert "glacier mass loss" in llm.calls[0]["user"]
    assert not result.refused
    assert result.citations == [
        {"doc": "r.pdf", "page": "11", "pdf_page": 1, "verified": True, "problem": None}
    ]
    assert result.warnings == []
    assert [h["rank"] for h in result.retrieved] == [1, 2]
    assert all(h["in_context"] for h in result.retrieved)
    assert "text" not in result.retrieved[0]  # result never repeats chunk text


def test_unverified_citation_produces_warning(setup):
    result = ask(setup, FakeLLMClient("Claim [r.pdf, p.99]."))
    assert not result.citations[0]["verified"]
    assert "possible hallucination" in result.warnings[0]


def test_answer_without_citations_is_warned(setup):
    result = ask(setup, FakeLLMClient("Glaciers lost mass."))
    assert result.warnings == ["answer contains no citations"]


def test_model_idk_is_reported_as_refusal(setup):
    result = ask(setup, FakeLLMClient(IDK))
    assert result.refused and result.refusal_reason == "model_said_idk"
    assert result.warnings == []


def test_cost_only_for_claude_backend(setup):
    result = ask(setup, FakeLLMClient("x [r.pdf, p.11]"))
    assert result.cost_usd is None  # fake backend, and "fake-llm" has no price


def test_per_doc_option_is_used(setup):
    result = ask(setup, FakeLLMClient(IDK), per_doc=1)
    assert len(result.retrieved) == 1  # tiny corpus has one document
