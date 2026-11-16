"""Sanity checks on the real evaluation set (no PDFs or model needed, so it runs in CI).

Label correctness against the PDF text is checked by `python -m rag.eval.validate`,
which needs data/processed and so does not run in CI.
"""

from pathlib import Path

import pytest

from rag.eval.schema import load_questions
from rag.manifest import read_manifest

ROOT = Path(__file__).parent.parent
QUESTIONS = load_questions(ROOT / "eval" / "questions.jsonl")  # raises on malformed lines
MANIFEST_DOCS = {row["filename"] for row in read_manifest(ROOT / "manifest.csv")}


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.id)
def test_gold_documents_are_in_the_manifest(q):
    assert set(q.gold_docs) <= MANIFEST_DOCS


def test_both_splits_have_answerable_and_unanswerable_questions():
    for split in ("dev", "test"):
        types = {q.answerable for q in QUESTIONS if q.split == split}
        assert types == {True, False}, f"{split} needs both kinds for refusal metrics"
