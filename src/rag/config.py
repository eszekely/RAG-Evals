"""All settings, read from environment variables (with defaults) in one place.

Everything tunable lives here so the code has no magic numbers, and a deployment
(Docker, CI) changes behaviour through environment variables, not code. Every
variable is listed in .env.example.

Env var                 Default                 Meaning
RAG_INDEX_DIR           data/processed/index    FAISS index directory (read-only at runtime)
RAG_BACKEND             claude                  default LLM backend for /ask: claude | ollama
RAG_MIN_SCORE           0.80 (placeholder)      best score below this: "I don't know", no LLM call
RAG_CLAUDE_MODEL        claude-haiku-5-5        Claude model id
RAG_OLLAMA_MODEL        llama3.2                local Ollama model name
RAG_OLLAMA_URL          http://localhost:11434
RAG_MAX_CONTEXT_TOKENS  6000                    cap on retrieved text sent to the LLM (estimated)
RAG_MAX_ANSWER_TOKENS   16000                   output cap; generous, Claude's thinking counts too
RAG_LLM_TIMEOUT_S       60                      LLM call timeout; exceeded -> HTTP 504
RAG_API_KEY             (unset)                 if set, /ask requires it in the X-API-Key header
RAG_LOG_LEVEL           INFO

ANTHROPIC_API_KEY is deliberately NOT a setting: only the Anthropic client reads it,
so it never sits in a settings object that could be printed or logged. RAG_API_KEY
(this service's own access key) is excluded from the settings' repr for the same reason.
"""

import os
from dataclasses import dataclass, field

BACKENDS = ("claude", "ollama")

# USD per million tokens (input, output), Anthropic first-party API prices as of
# 2026-11. Haiku 5.5's price applies to prompts up to 100K tokens; our prompts are
# far smaller. A model missing from this table gets no cost estimate rather than a guess.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass(frozen=True)
class Settings:
    index_dir: str = "data/processed/index"
    default_backend: str = "claude"
    # Placeholder until calibrated. Stage 2 measured top scores of 0.88-0.89 for
    # answerable questions and 0.843 for a nonsense query, so 0.80 refuses almost
    # nothing yet; the model's own "I don't know" is the backstop.
    min_score: float = 0.80
    claude_model: str = "claude-haiku-5-5"
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    max_context_tokens: int = 6000
    max_answer_tokens: int = 16000
    llm_timeout_s: float = 60.0
    api_key: str | None = field(default=None, repr=False)
    log_level: str = "INFO"

    def __post_init__(self):
        if self.default_backend not in BACKENDS:
            raise ValueError(f"RAG_BACKEND must be one of {BACKENDS}, got {self.default_backend!r}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        d = cls()
        return cls(
            index_dir=env.get("RAG_INDEX_DIR", d.index_dir),
            default_backend=env.get("RAG_BACKEND", d.default_backend),
            min_score=float(env.get("RAG_MIN_SCORE", d.min_score)),
            claude_model=env.get("RAG_CLAUDE_MODEL", d.claude_model),
            ollama_model=env.get("RAG_OLLAMA_MODEL", d.ollama_model),
            ollama_url=env.get("RAG_OLLAMA_URL", d.ollama_url),
            max_context_tokens=int(env.get("RAG_MAX_CONTEXT_TOKENS", d.max_context_tokens)),
            max_answer_tokens=int(env.get("RAG_MAX_ANSWER_TOKENS", d.max_answer_tokens)),
            llm_timeout_s=float(env.get("RAG_LLM_TIMEOUT_S", d.llm_timeout_s)),
            api_key=env.get("RAG_API_KEY") or None,  # empty string = no key
            log_level=env.get("RAG_LOG_LEVEL", d.log_level),
        )


def estimate_cost_usd(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Cost from the price table, or None if the model or token counts are unknown."""
    if model not in PRICES_PER_MTOK or input_tokens is None or output_tokens is None:
        return None
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
