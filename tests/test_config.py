import pytest

from rag.config import PRICES_PER_MTOK, Settings, estimate_cost_usd


def test_defaults():
    s = Settings.from_env({})
    assert s.claude_model == "claude-haiku-5-5"
    assert s.min_score == Settings().min_score


def test_env_overrides():
    s = Settings.from_env({"RAG_MIN_SCORE": "0.87", "RAG_CLAUDE_MODEL": "claude-sonnet-5-5"})
    assert (s.min_score, s.claude_model) == (0.87, "claude-sonnet-5-5")


def test_cost_uses_price_table():
    price_in, price_out = PRICES_PER_MTOK["claude-haiku-5-5"]
    expected = (2_000 * price_in + 100 * price_out) / 1_000_000
    assert estimate_cost_usd("claude-haiku-5-5", 2_000, 100) == pytest.approx(expected)


def test_cost_unknown_model_or_tokens_is_none():
    assert estimate_cost_usd("some-local-model", 10, 10) is None
    assert estimate_cost_usd("claude-haiku-5-5", None, 10) is None
