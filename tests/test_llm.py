"""LLM backends with mocked HTTP: no network, no real API key."""

import json

import anthropic
import httpx2
import pytest
import requests

from rag.llm import ClaudeClient, FakeLLMClient, LLMError, LLMTimeoutError, OllamaClient

FAKE_KEY = "sk-ant-test-THIS-IS-SECRET"


def claude_with(handler) -> ClaudeClient:
    """A ClaudeClient whose SDK sends every request to `handler` instead of the network."""
    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    sdk = anthropic.Anthropic(
        api_key=FAKE_KEY, base_url="http://test", http_client=http, max_retries=0
    )
    return ClaudeClient("claude-haiku-5-5", client=sdk)


def message_json(text: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


def test_claude_success_returns_text_usage_and_sends_prompt():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=message_json("Answer [a.pdf, p.1]"))

    result = claude_with(handler).generate("SYSTEM", "USER", max_tokens=500)

    assert result.text == "Answer [a.pdf, p.1]"  # thinking block skipped
    assert (result.input_tokens, result.output_tokens) == (120, 30)
    assert result.model == "claude-haiku-5-5"
    assert seen["body"]["system"] == "SYSTEM"
    assert seen["body"]["messages"] == [{"role": "user", "content": "USER"}]
    assert seen["body"]["max_tokens"] == 500


def error_handler(status: int, error_type: str):
    body = {"type": "error", "error": {"type": error_type, "message": "nope"}}
    return lambda request: httpx2.Response(status, json=body)


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (401, "authentication_error", "API key"),
        (404, "not_found_error", "not found"),
        (429, "rate_limit_error", "rate limit"),
        (500, "api_error", "500"),
    ],
)
def test_claude_http_errors_become_clear_llm_errors(status, error_type, expected):
    with pytest.raises(LLMError, match=expected) as info:
        claude_with(error_handler(status, error_type)).generate("s", "u", 10)
    assert FAKE_KEY not in str(info.value)


def test_claude_connection_error():
    def handler(request):
        raise httpx2.ConnectError("boom", request=request)

    with pytest.raises(LLMError, match="Cannot reach"):
        claude_with(handler).generate("s", "u", 10)


def test_claude_missing_key_fails_early(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        ClaudeClient("claude-haiku-5-5")


# --- Ollama ----------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, data: dict | None = None, text: str = ""):
        self.status_code, self._data, self.text = status_code, data, text

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.posts = response, error, []

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        if self.error:
            raise self.error
        return self.response


def test_ollama_success():
    data = {
        "model": "llama3.2",
        "message": {"content": "Hi"},
        "prompt_eval_count": 50,
        "eval_count": 5,
    }
    session = FakeSession(FakeResponse(200, data))

    result = OllamaClient("llama3.2", session=session).generate("SYS", "USER", 100)

    assert (result.text, result.input_tokens, result.output_tokens) == ("Hi", 50, 5)
    url, payload = session.posts[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["messages"][0] == {"role": "system", "content": "SYS"}
    assert payload["stream"] is False


def test_ollama_not_running_gives_clear_message():
    session = FakeSession(error=requests.ConnectionError("refused"))
    with pytest.raises(LLMError, match="Is it running"):
        OllamaClient("llama3.2", session=session).generate("s", "u", 10)


def test_ollama_missing_model_suggests_pull():
    session = FakeSession(FakeResponse(404, text="model not found"))
    with pytest.raises(LLMError, match="ollama pull llama3.2"):
        OllamaClient("llama3.2", session=session).generate("s", "u", 10)


def test_fake_llm_records_calls():
    llm = FakeLLMClient("reply")
    assert llm.generate("s", "u", 5).text == "reply"
    assert llm.calls == [{"system": "s", "user": "u", "max_tokens": 5}]


def test_claude_timeout_becomes_llm_timeout_error():
    def handler(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeoutError):
        claude_with(handler).generate("s", "u", 10)


def test_ollama_timeout_becomes_llm_timeout_error():
    session = FakeSession(error=requests.Timeout("slow"))
    with pytest.raises(LLMTimeoutError):
        OllamaClient("llama3.2", session=session).generate("s", "u", 10)
