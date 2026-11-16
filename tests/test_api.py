"""API tests: fake embedder, tiny index, fake LLM. No network, no model download.

TestClient is used without `with`, so the real lifespan (which loads the model and
the index) never runs; get_resources is overridden with fakes instead.
"""

import io
import logging

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from rag.api import Resources, app, get_resources
from rag.config import Settings
from rag.index import build_index
from rag.llm import FakeLLMClient, LLMError, LLMTimeoutError
from rag.logging import LOGGER_NAME, JsonFormatter, log_event

ANSWER = "Glaciers lost mass [r.pdf, p.11]."


class FailingLLM:
    backend = "claude"

    def __init__(self, error: Exception):
        self.error = error

    def generate(self, system, user, max_tokens):
        raise self.error


@pytest.fixture
def make_client(tiny_corpus, fake_embedder):
    def _make(llm=None, **settings_kwargs) -> tuple[TestClient, Resources]:
        settings = Settings(min_score=0.0, **settings_kwargs)
        resources = Resources(
            settings=settings,
            index=build_index(tiny_corpus, fake_embedder),
            chunks=tiny_corpus,
            index_config={"model_name": "fake-bag-of-words", "built_at": "2026-11-13T00:00:00"},
            embedder=fake_embedder,
            llms={"claude": llm or FakeLLMClient(ANSWER)},
            unavailable={"ollama": "not configured in tests"},
        )
        app.dependency_overrides[get_resources] = lambda: resources
        return TestClient(app), resources

    yield _make
    app.dependency_overrides.clear()


def ask(client, question="glacier mass loss", **extra):
    return client.post("/ask", json={"question": question, **extra})


def test_valid_request(make_client):
    client, _ = make_client()
    response = ask(client, k=2)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ANSWER
    assert body["refused"] is False
    assert body["citations"][0] | {} == {
        "doc": "r.pdf",
        "page": "11",
        "pdf_page": 1,
        "verified": True,
        "problem": None,
    }
    assert len(body["retrieved"]) == 2
    assert {"doc", "page", "chunk_id", "score"} <= body["retrieved"][0].keys()
    assert body["backend"] == "fake" and body["latency_ms"] >= 0
    assert response.headers["X-Request-ID"] == body["request_id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {"question": "ok", "k": 0},
        {"question": "ok", "k": 11},
        {"question": "ok", "backend": "gpt"},
        {"question": "ok", "unexpected": 1},
        {},
    ],
)
def test_invalid_input_returns_422(make_client, payload):
    client, resources = make_client()
    assert client.post("/ask", json=payload).status_code == 422
    assert resources.llms["claude"].calls == []


def test_refused_path_does_not_call_llm(make_client):
    client, resources = make_client()
    resources.settings = Settings(min_score=0.99)

    body = ask(client).json()

    assert body["refused"] is True
    assert body["refusal_reason"] == "low_retrieval_score"
    assert resources.llms["claude"].calls == []


def test_api_key_required_when_set(make_client):
    client, _ = make_client(api_key="s3cret")
    assert ask(client).status_code == 401
    assert (
        client.post("/ask", json={"question": "q"}, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )
    ok = client.post("/ask", json={"question": "glacier"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # only /ask is protected


def test_no_api_key_needed_when_unset(make_client):
    client, _ = make_client()
    assert ask(client).status_code == 200


def test_health(make_client):
    client, _ = make_client()
    body = client.get("/health").json()
    assert body["index_loaded"] is True
    assert body["n_chunks"] == 3
    assert body["embedding_model"] == "fake-bag-of-words"
    assert body["index_built_at"] == "2026-11-13T00:00:00"
    assert body["backends"] == {"claude": "configured", "ollama": "unavailable"}


def sample(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_count_requests_refusals_and_unverified_citations(make_client):
    client, resources = make_client(llm=FakeLLMClient("Claim [r.pdf, p.99]."))
    requests_before = sample("rag_requests_total", {"endpoint": "/ask", "status": "200"})
    unverified_before = sample("rag_unverified_citations_total", {})
    refusals_before = sample("rag_refusals_total", {"reason": "low_retrieval_score"})

    ask(client)  # answered, with one unverified citation
    resources.settings = Settings(min_score=0.99)
    ask(client)  # refused before the LLM

    assert (
        sample("rag_requests_total", {"endpoint": "/ask", "status": "200"}) == requests_before + 2
    )
    assert sample("rag_unverified_citations_total", {}) == unverified_before + 1
    assert sample("rag_refusals_total", {"reason": "low_retrieval_score"}) == refusals_before + 1
    text = client.get("/metrics").text
    assert 'rag_requests_total{endpoint="/ask",status="200"}' in text
    assert "rag_request_latency_seconds_bucket" in text
    assert 'rag_llm_tokens_total{backend="fake",direction="input"}' in text


@pytest.mark.parametrize(
    ("error", "status"),
    [(LLMError("backend down"), 503), (LLMTimeoutError("too slow"), 504)],
)
def test_llm_failures_map_to_503_and_504(make_client, error, status):
    client, _ = make_client(llm=FailingLLM(error))
    response = ask(client)
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_unavailable_backend_returns_503(make_client):
    client, _ = make_client()
    response = ask(client, backend="ollama")
    assert response.status_code == 503
    assert "not configured in tests" in response.json()["detail"]


@pytest.fixture
def captured_logs():
    """Capture the 'rag' logger's JSON output exactly as it would be written."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield stream
    logger.removeHandler(handler)


def test_question_answer_and_chunk_text_never_logged(make_client, captured_logs):
    secret_question = "glacier mass loss SECRET-QUESTION-7f3a"
    secret_answer = "SECRET-ANSWER-91bc [r.pdf, p.11]"
    client, _ = make_client(llm=FakeLLMClient(secret_answer))

    assert ask(client, question=secret_question).status_code == 200
    ask(client, question="SECRET-QUESTION-7f3a again", backend="ollama")  # 503 path too

    logs = captured_logs.getvalue()
    assert '"event": "ask"' in logs and '"event": "ask_failed"' in logs  # logging happened
    assert "SECRET-QUESTION" not in logs
    assert "SECRET-ANSWER" not in logs
    assert "glaciers melted" not in logs  # retrieved chunk text
    assert '"retrieved": [{"doc": "r.pdf", "page": 1}' in logs  # ids and pages are logged


def test_formatter_drops_fields_not_on_the_allow_list(captured_logs):
    log_event("test", request_id="abc", question="LEAK", api_key="LEAK", text="LEAK")
    logs = captured_logs.getvalue()
    assert '"request_id": "abc"' in logs
    assert "LEAK" not in logs
