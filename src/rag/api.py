"""HTTP API: POST /ask, GET /health, GET /metrics.

Run:  uv run --env-file .env uvicorn rag.api:app

Design:
- The lifespan handler loads the index, the embedder and the LLM clients ONCE at
  startup; requests get them through the `get_resources` dependency, which tests
  override with fakes.
- Endpoints are plain `def`: embedding and FAISS search are CPU-bound and the LLM
  calls are blocking HTTP, so FastAPI runs them in its thread pool instead of
  blocking the event loop.
- A missing index stops startup (nothing can work; the healthcheck fails visibly).
  A missing backend does not: /health reports it and /ask returns 503 for it.
- Logs and metrics never contain question, answer or document text (see rag.logging).
"""

import hmac
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import faiss
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from rag.answer import AnswerResult, answer_question
from rag.config import Settings
from rag.embed import TextEmbedder
from rag.index import load_index
from rag.llm import ClaudeClient, LLMClient, LLMError, LLMTimeoutError, OllamaClient
from rag.logging import log_event, setup_logging
from rag.metrics import LATENCY, REFUSALS, REQUESTS, TOKENS, UNVERIFIED_CITATIONS

ENDPOINTS = ("/ask", "/health", "/metrics")


# --- resources loaded once ---------------------------------------------------


@dataclass
class Resources:
    settings: Settings
    index: faiss.Index
    chunks: list[dict]
    index_config: dict
    embedder: TextEmbedder
    llms: dict[str, LLMClient]
    unavailable: dict[str, str] = field(default_factory=dict)  # backend -> reason


def load_resources(settings: Settings) -> Resources:
    index, chunks, config = load_index(Path(settings.index_dir))

    from rag.embed import Embedder  # slow import (PyTorch)

    llms: dict[str, LLMClient] = {}
    unavailable: dict[str, str] = {}
    try:
        llms["claude"] = ClaudeClient(settings.claude_model, timeout=settings.llm_timeout_s)
    except LLMError as e:
        unavailable["claude"] = str(e)
    llms["ollama"] = OllamaClient(
        settings.ollama_model, settings.ollama_url, timeout=settings.llm_timeout_s
    )
    return Resources(
        settings, index, chunks, config, Embedder(config["model_name"]), llms, unavailable
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    resources = load_resources(settings)
    app.state.resources = resources
    log_event(
        "startup",
        n_chunks=len(resources.chunks),
        index_built_at=resources.index_config.get("built_at"),
        backends=sorted(resources.llms),
    )
    yield


def get_resources(request: Request) -> Resources:
    return request.app.state.resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]


def require_api_key(
    resources: ResourcesDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """If RAG_API_KEY is set, /ask needs it in X-API-Key.

    Demo-level access control: one shared static key, no users, no rotation or rate
    limiting. compare_digest takes the same time whether or not the first characters
    match, so the key can't be guessed one character at a time by timing responses.
    """
    expected = resources.settings.api_key
    if expected is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# --- request / response models ----------------------------------------------

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    k: int = Field(default=5, ge=1, le=10)
    backend: Literal["claude", "ollama"] | None = None  # None: RAG_BACKEND


class Citation(BaseModel):
    doc: str
    page: str
    pdf_page: int | None
    verified: bool
    problem: str | None


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    doc: str
    page: int
    page_label: str | None
    score: float
    in_context: bool


class AskResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    backend: str
    model: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    warnings: list[str]


# --- app ---------------------------------------------------------------------

app = FastAPI(title="RAG over public reports", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Request id, request counter and latency histogram for every request."""
    request.state.request_id = uuid.uuid4().hex
    endpoint = request.url.path if request.url.path in ENDPOINTS else "other"
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    finally:
        REQUESTS.labels(endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(time.perf_counter() - start)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(body: AskRequest, request: Request, resources: ResourcesDep):
    request_id = request.state.request_id
    backend = body.backend or resources.settings.default_backend
    start = time.perf_counter()

    def fail(status: int, error_type: str, detail: str):
        log_event(
            "ask_failed",
            request_id=request_id,
            status=status,
            backend=backend,
            k=body.k,
            error_type=error_type,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        raise HTTPException(status_code=status, detail=detail)

    llm = resources.llms.get(backend)
    if llm is None:
        fail(
            503,
            "backend_unavailable",
            f"Backend {backend!r} unavailable: "
            f"{resources.unavailable.get(backend, 'not configured')}",
        )
    try:
        result = answer_question(
            body.question,
            resources.index,
            resources.chunks,
            resources.embedder,
            llm,
            resources.settings,
            k=body.k,
        )
    except LLMTimeoutError as e:
        fail(504, "llm_timeout", str(e))
    except LLMError as e:
        fail(503, "llm_error", str(e))

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    record_metrics(result)
    log_ask(request_id, body.k, result, latency_ms)
    return AskResponse(request_id=request_id, latency_ms=latency_ms, **response_fields(result))


def response_fields(result: AnswerResult) -> dict:
    data = result.to_dict()
    data.pop("latency_s")
    data.pop("llm_latency_s")
    return data


def record_metrics(result: AnswerResult) -> None:
    if result.refused:
        REFUSALS.labels(result.refusal_reason).inc()
    UNVERIFIED_CITATIONS.inc(sum(not c["verified"] for c in result.citations))
    if result.input_tokens is not None:
        TOKENS.labels(result.backend, "input").inc(result.input_tokens)
    if result.output_tokens is not None:
        TOKENS.labels(result.backend, "output").inc(result.output_tokens)


def log_ask(request_id: str, k: int, result: AnswerResult, latency_ms: float) -> None:
    """One log line per answered request: ids, pages, scores and counts, never text."""
    log_event(
        "ask",
        request_id=request_id,
        status=200,
        latency_ms=latency_ms,
        backend=result.backend,
        model=result.model,
        k=k,
        retrieved=[{"doc": h["doc"], "page": h["page"]} for h in result.retrieved],
        top_score=round(max((h["score"] for h in result.retrieved), default=0.0), 4),
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        unverified_citations=sum(not c["verified"] for c in result.citations),
    )


@app.get("/health")
def health(resources: ResourcesDep) -> dict:
    config = resources.index_config
    return {
        "status": "ok",
        "index_loaded": True,
        "n_chunks": len(resources.chunks),
        "embedding_model": config["model_name"],
        "index_built_at": config.get("built_at"),
        "default_backend": resources.settings.default_backend,
        # "configured" means the client exists; Ollama's server is not probed here.
        "backends": {
            name: ("configured" if name in resources.llms else "unavailable")
            for name in ("claude", "ollama")
        },
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
