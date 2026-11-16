"""LLM backends behind one tiny interface: generate(system, user, max_tokens).

- ClaudeClient: the official anthropic SDK. The API key is read only from the
  ANTHROPIC_API_KEY environment variable and is never logged or printed.
- OllamaClient: a local model through Ollama's HTTP API (no data leaves the machine).
- FakeLLMClient: deterministic, no network; for tests.

Every failure is raised as LLMError with a short, actionable message, so callers
(CLI now, API in Stage 4) handle one exception type. Messages never contain the key.
"""

import os
import time
from dataclasses import dataclass
from typing import Protocol

import anthropic
import requests


class LLMError(RuntimeError):
    """A backend could not produce an answer (configuration, network or API error)."""


class LLMTimeoutError(LLMError):
    """The backend did not answer in time (the API maps this to 504, other errors to 503)."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    stop_reason: str | None = None


class LLMClient(Protocol):
    backend: str

    def generate(self, system: str, user: str, max_tokens: int) -> LLMResult: ...


class ClaudeClient:
    backend = "claude"

    def __init__(
        self, model: str, client: anthropic.Anthropic | None = None, timeout: float = 60.0
    ):
        self.model = model
        if client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Export it or use: uv run --env-file .env ..."
                )
            # Passing the key explicitly means only this env var is used, never another
            # credential source the SDK might otherwise find. One retry keeps a slow
            # request's worst case near 2 x timeout instead of the SDK default 3 x.
            client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=1)
        self.client = client

    def generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        start = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude API rejected the API key (401). Check ANTHROPIC_API_KEY.") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError("Claude API: this API key may not use this model (403).") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Claude API: model {self.model!r} not found (404).") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude API rate limit reached (429). Try again later.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APITimeoutError as e:  # subclass of APIConnectionError: catch first
            raise LLMTimeoutError("The Claude API did not answer in time.") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Cannot reach the Claude API (network error).") from e

        # Only text blocks form the answer; thinking blocks (if any) are skipped.
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - start,
            stop_reason=response.stop_reason,
        )


class OllamaClient:
    backend = "ollama"

    def __init__(self, model: str, url: str = "http://localhost:11434", session=None, timeout=300):
        self.model = model
        self.url = url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        start = time.perf_counter()
        try:
            response = self.session.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self.url}. Is it running? Start: ollama serve"
            ) from e
        except requests.Timeout as e:
            raise LLMTimeoutError(f"Ollama did not answer within {self.timeout}s.") from e
        if response.status_code == 404:
            raise LLMError(
                f"Ollama model {self.model!r} not found. Pull it with: ollama pull {self.model}"
            )
        if response.status_code != 200:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")

        data = response.json()
        return LLMResult(
            text=data["message"]["content"],
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_s=time.perf_counter() - start,
            stop_reason=data.get("done_reason"),
        )


class FakeLLMClient:
    """Returns a fixed reply and records every call, so tests can check if it was called."""

    backend = "fake"

    def __init__(self, reply: str = "I don't know based on the provided documents."):
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return LLMResult(
            text=self.reply,
            model="fake-llm",
            input_tokens=len(user) // 4,
            output_tokens=len(self.reply) // 4,
            latency_s=0.0,
            stop_reason="end_turn",
        )
