"""Retrieval metrics over answerable questions.

- Page-level hit: a retrieved chunk is from a gold document AND a gold page.
- Doc-level hit: a retrieved chunk is from a gold document (any page). Reported
  separately: "right report, wrong page" is a different failure from "wrong report".
- hit@k: share of questions with a hit in the top k.
- MRR: mean of 1/rank of the first hit (0 if none in the retrieved list, so this
  is MRR@depth, with depth = how many chunks were retrieved, 10 in rag.eval.run).
"""

from collections.abc import Callable

from rag.eval.schema import Question

KS = (1, 3, 5, 10)


def page_hit(chunk: dict, q: Question) -> bool:
    return chunk["doc"] in q.gold_docs and chunk["page"] in q.gold_pages


def doc_hit(chunk: dict, q: Question) -> bool:
    return chunk["doc"] in q.gold_docs


def first_hit_rank(retrieved: list[dict], q: Question, hit: Callable = page_hit) -> int | None:
    """1-based rank of the first hit in `retrieved` (in order), or None."""
    return next((rank for rank, c in enumerate(retrieved, start=1) if hit(c, q)), None)


def hit_values(ranks: list[int | None], k: int) -> list[float]:
    return [1.0 if r is not None and r <= k else 0.0 for r in ranks]


def reciprocal_ranks(ranks: list[int | None]) -> list[float]:
    return [1.0 / r if r else 0.0 for r in ranks]


def summarize_ranks(ranks: list[int | None]) -> dict:
    """hit@k for every k in KS, and MRR, for one group of questions."""
    if not ranks:
        return {"n": 0}
    summary = {"n": len(ranks)}
    for k in KS:
        values = hit_values(ranks, k)
        summary[f"hit@{k}"] = sum(values) / len(values)
    summary["mrr"] = sum(reciprocal_ranks(ranks)) / len(ranks)
    return summary


def retrieval_report(questions: list[Question], ranks: dict[str, int | None]) -> dict:
    """Overall and per-type summaries; `ranks` maps question id -> first-hit rank."""
    answerable = [q for q in questions if q.answerable]
    by_type: dict[str, list] = {}
    for q in answerable:
        by_type.setdefault(q.type, []).append(ranks[q.id])
    return {
        "overall": summarize_ranks([ranks[q.id] for q in answerable]),
        "by_type": {t: summarize_ranks(r) for t, r in sorted(by_type.items())},
    }
