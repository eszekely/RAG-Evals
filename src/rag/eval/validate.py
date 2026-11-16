"""Check evaluation labels against the extracted text, before they corrupt the metrics.

Usage:
    uv run python -m rag.eval.validate [--questions eval/questions.jsonl]
                                       [--pages data/processed/pages.jsonl]

For every question with gold labels (verified or not):
- each gold document exists in the processed data,
- each gold page exists in at least one gold document,
- evidence_snippet appears on at least one gold (document, page); figure_only
  questions have no text evidence, so only their document and page are checked.
Prints every failure and exits non-zero if there is any.

Matching ignores case, all whitespace (including the special spaces PDFs put in
numbers like "3 250") and hyphens (line-break hyphenation, soft hyphens), so
"consecu-\\ntive" matches "consecutive".
"""

import argparse
import re
import sys
from pathlib import Path

from rag.eval.schema import Question, load_questions
from rag.jsonl import read_jsonl

IGNORED = re.compile(r"[\s\-\u00ad\u2010\u2011]+")  # whitespace, hyphen, soft/non-breaking hyphens


def match_key(text: str) -> str:
    return IGNORED.sub("", text.lower())


def page_index(pages: list[dict]) -> dict[str, dict[int, str]]:
    """{doc: {page: match_key(text)}}"""
    index: dict[str, dict[int, str]] = {}
    for p in pages:
        index.setdefault(p["doc"], {})[p["page"]] = match_key(p["text"])
    return index


def check_question(q: Question, index: dict[str, dict[int, str]]) -> list[str]:
    """Human-readable failures for one question (empty list = labels look right)."""
    failures = [
        f"{q.id}: gold document not in processed data: {d}" for d in q.gold_docs if d not in index
    ]
    docs = [d for d in q.gold_docs if d in index]
    for page in q.gold_pages:
        if docs and not any(page in index[d] for d in docs):
            failures.append(f"{q.id}: gold page {page} does not exist in {docs}")
    if q.evidence_snippet is None:  # unanswerable, or figure_only (no text evidence)
        return failures
    snippet = match_key(q.evidence_snippet)
    found = [(d, p) for d in docs for p in q.gold_pages if snippet in index[d].get(p, "")]
    if docs and not found:
        failures.append(
            f"{q.id}: evidence_snippet not found on any gold page: {q.evidence_snippet!r}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate evaluation labels.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--pages", type=Path, default=Path("data/processed/pages.jsonl"))
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)  # raises on malformed lines
    index = page_index(read_jsonl(args.pages))
    failures = [f for q in questions for f in check_question(q, index)]
    for failure in failures:
        print("FAIL", failure)
    answerable = sum(q.answerable for q in questions)
    print(
        f"{len(questions)} questions ({answerable} answerable, "
        f"{sum(q.verified for q in questions)} verified): {len(failures)} label failure(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
