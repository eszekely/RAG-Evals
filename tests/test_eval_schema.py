import json

import pytest
from pydantic import ValidationError

from rag.eval.schema import Question, load_questions

GOOD = {
    "id": "q1",
    "question": "Q?",
    "language": "en",
    "type": "number",
    "gold_docs": ["a.pdf"],
    "gold_pages": [3],
    "answer": "42",
    "evidence_snippet": "42 stations",
    "author": "human",
    "verified": True,
    "split": "dev",
}
UNANSWERABLE = {
    **GOOD,
    "id": "q2",
    "type": "unanswerable",
    "gold_docs": [],
    "gold_pages": [],
    "answer": None,
    "evidence_snippet": None,
    "verified": False,
    "split": "test",
}


def write(tmp_path, rows) -> object:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(r if isinstance(r, str) else json.dumps(r) + "\n" for r in rows))
    return path


def test_loads_and_filters_by_split_and_verified(tmp_path):
    path = write(tmp_path, [GOOD, UNANSWERABLE])
    assert [q.id for q in load_questions(path)] == ["q1", "q2"]
    assert [q.id for q in load_questions(path, split="test")] == ["q2"]
    assert [q.id for q in load_questions(path, verified_only=True)] == ["q1"]


def test_answerable_property(tmp_path):
    good, unanswerable = load_questions(write(tmp_path, [GOOD, UNANSWERABLE]))
    assert good.answerable and not unanswerable.answerable


def test_malformed_json_names_the_line(tmp_path):
    path = write(tmp_path, [GOOD, "{not json\n"])
    with pytest.raises(ValueError, match=r"questions.jsonl:2"):
        load_questions(path)


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "type": "opinion"},  # unknown type
        {**GOOD, "split": "train"},  # unknown split
        {**GOOD, "evidence_snippet": None},  # answerable needs evidence
        {**GOOD, "gold_pages": []},  # answerable needs a gold page
        {**GOOD, "gold_pages": [0]},  # pages are 1-based
        {**UNANSWERABLE, "gold_docs": ["a.pdf"]},  # unanswerable must have no gold
        {**UNANSWERABLE, "answer": "something"},
        {**GOOD, "surprise": 1},  # unknown field
    ],
)
def test_invalid_questions_are_rejected(bad):
    with pytest.raises(ValidationError):
        Question(**bad)


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(write(tmp_path, [GOOD, GOOD]))


FIGURE_ONLY = {**GOOD, "type": "figure_only", "evidence_snippet": None, "answer": None}


def test_figure_only_keeps_gold_page_but_expects_refusal():
    q = Question(**FIGURE_ONLY)
    assert not q.answerable  # the text-only system should refuse
    assert (q.gold_docs, q.gold_pages) == (["a.pdf"], [3])
    assert Question(**{**FIGURE_ONLY, "answer": "2012"}).answer == "2012"  # true answer may be kept


@pytest.mark.parametrize(
    "bad",
    [
        {**FIGURE_ONLY, "evidence_snippet": "some quote"},  # no text evidence by definition
        {**FIGURE_ONLY, "gold_pages": []},  # the figure's page is required
    ],
)
def test_invalid_figure_only_questions_are_rejected(bad):
    with pytest.raises(ValidationError):
        Question(**bad)
