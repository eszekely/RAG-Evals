"""Question -> cited answer: retrieve -> threshold -> generate -> verify.

1. Retrieve the top-k chunks (or the top M per document with per_doc).
2. If the best score is below settings.min_score, answer "I don't know" WITHOUT
   calling the LLM: weak retrieval means no grounding, and the call would cost money.
3. Fill the prompt with the highest-scoring chunks that fit the context budget.
4. Generate, then check every citation against the chunks the model was shown.
"""

import time
from dataclasses import asdict, dataclass, field

import faiss

from rag.citations import verify_citations
from rag.config import Settings, estimate_cost_usd
from rag.embed import TextEmbedder
from rag.llm import LLMClient
from rag.prompt import IDK, SYSTEM_PROMPT, build_user_message, select_chunks
from rag.search import search, search_per_doc


@dataclass
class AnswerResult:
    answer: str
    refused: bool
    refusal_reason: str | None  # "low_retrieval_score" | "no_context_fits" | "model_said_idk"
    citations: list[dict]
    retrieved: list[dict]
    backend: str
    model: str | None
    latency_s: float
    llm_latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def retrieve(
    question: str,
    index: faiss.Index,
    chunks: list[dict],
    embedder: TextEmbedder,
    k: int,
    per_doc: int | None,
) -> list[dict]:
    if per_doc:
        return search_per_doc(question, index, chunks, embedder, per_doc)
    return search(question, index, chunks, embedder, k)


def summarize_hits(hits: list[dict], context_ids: set[str]) -> list[dict]:
    """Retrieved chunks without their text, marking which ones went into the prompt."""
    keys = ("rank", "chunk_id", "doc", "page", "page_label", "score")
    return [
        {**{key: h[key] for key in keys}, "in_context": h["chunk_id"] in context_ids} for h in hits
    ]


def answer_question(
    question: str,
    index: faiss.Index,
    chunks: list[dict],
    embedder: TextEmbedder,
    llm: LLMClient,
    settings: Settings,
    k: int = 5,
    per_doc: int | None = None,
) -> AnswerResult:
    start = time.perf_counter()
    hits = retrieve(question, index, chunks, embedder, k, per_doc)
    best = max((h["score"] for h in hits), default=0.0)

    def refuse(reason: str, context_ids: set[str] = frozenset()) -> AnswerResult:
        return AnswerResult(
            answer=IDK,
            refused=True,
            refusal_reason=reason,
            citations=[],
            retrieved=summarize_hits(hits, context_ids),
            backend=llm.backend,
            model=None,
            latency_s=time.perf_counter() - start,
        )

    if best < settings.min_score:
        return refuse("low_retrieval_score")
    context = select_chunks(hits, settings.max_context_tokens)
    if not context:
        return refuse("no_context_fits")

    result = llm.generate(
        SYSTEM_PROMPT, build_user_message(question, context), settings.max_answer_tokens
    )
    citations = verify_citations(result.text, context)
    said_idk = IDK in result.text
    return AnswerResult(
        answer=result.text,
        refused=said_idk,
        refusal_reason="model_said_idk" if said_idk else None,
        citations=citations,
        retrieved=summarize_hits(hits, {c["chunk_id"] for c in context}),
        backend=llm.backend,
        model=result.model,
        latency_s=time.perf_counter() - start,
        llm_latency_s=result.latency_s,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=(
            estimate_cost_usd(result.model, result.input_tokens, result.output_tokens)
            if llm.backend == "claude"
            else None
        ),
        warnings=make_warnings(result.text, result.stop_reason, citations, said_idk),
    )


def make_warnings(
    text: str, stop_reason: str | None, citations: list[dict], said_idk: bool
) -> list[str]:
    warnings = [
        f"unverified citation [{c['doc']}, p.{c['page']}]: {c['problem']} (possible hallucination)"
        for c in citations
        if not c["verified"]
    ]
    if not said_idk and not citations:
        warnings.append("answer contains no citations")
    if stop_reason == "max_tokens":
        warnings.append("answer was cut off at the max_tokens limit")
    if stop_reason == "refusal":
        warnings.append("the model declined to answer (stop_reason=refusal)")
    return warnings
