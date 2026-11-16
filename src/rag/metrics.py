"""Prometheus metrics, defined once per process.

Label values come from small fixed sets (route template, status code, backend,
refusal reason), never from user input, so the number of time series stays bounded.
"""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("rag_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "rag_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
REFUSALS = Counter("rag_refusals_total", "Answers that were refusals", ["reason"])
UNVERIFIED_CITATIONS = Counter(
    "rag_unverified_citations_total", "Citations not matching any chunk in the context"
)
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["backend", "direction"])
