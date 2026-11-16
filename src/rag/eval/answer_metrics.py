"""Answer-level metrics: refusals and citations.

- Refusal accuracy: a question is handled correctly when the system refuses exactly
  when the question is unanswerable. Reported with its two error rates:
  false-refusal rate (refused an answerable question) and the share of unanswerable
  questions caught (refused).
- Citation validity: share of all citations that match a chunk the model was shown.
- Citation correctness: share of answerable questions whose answer cites at least
  one gold (document, page); a refused answer counts as not citing gold.
"""

from rag.eval.schema import Question


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def refusal_correct(answerable: bool, refused: bool) -> bool:
    return refused != answerable


def refusal_metrics(rows: list[tuple[bool, bool]]) -> dict:
    """rows: (answerable, refused) per question."""
    answerable = [refused for ok, refused in rows if ok]
    unanswerable = [refused for ok, refused in rows if not ok]
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "refusal_accuracy": _rate([refusal_correct(a, r) for a, r in rows]),
        "false_refusal_rate": _rate(answerable),
        "unanswerable_caught_rate": _rate(unanswerable),
    }


def citation_validity(citation_lists: list[list[dict]]) -> dict:
    citations = [c for cs in citation_lists for c in cs]
    return {"n_citations": len(citations), "rate": _rate([c["verified"] for c in citations])}


def cites_gold(citations: list[dict], q: Question) -> bool:
    """True if a verified citation points at a gold (document, PDF page)."""
    return any(
        c["verified"] and c["doc"] in q.gold_docs and c["pdf_page"] in q.gold_pages
        for c in citations
    )


def citation_correctness(pairs: list[tuple[Question, list[dict]]]) -> dict:
    """pairs: (question, citations) for answerable questions."""
    values = [cites_gold(cs, q) for q, cs in pairs if q.answerable]
    return {"n": len(values), "rate": _rate(values)}
