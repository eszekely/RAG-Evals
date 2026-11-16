"""The evaluation question format and its loader.

eval/questions.jsonl holds one question per line. Pydantic validates every field,
so a malformed line fails loudly with its line number instead of silently skewing
the metrics. Only questions with verified=true count in reported metrics.

A question is a hit target when a retrieved chunk is from one of `gold_docs` AND on
one of `gold_pages` (PDF page numbers, 1-based). Limitation of this flat format:
for a question with several gold documents, every gold page counts for every gold
document, so a page number that happens to exist in the "wrong" report also counts.

Two kinds of question are NOT answerable from the extracted text, so the correct
behaviour is to refuse ("I don't know"):
- "unanswerable": the reports do not contain the answer at all (no gold).
- "figure_only": the answer is visible only in a chart or map (bar heights, colours),
  which text extraction does not capture. These honesty tests keep their gold page
  (where the figure is) and may record the true answer for a future multimodal pipeline.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

QuestionType = Literal[
    "single_fact", "number", "table", "cross_year", "unanswerable", "multi_hop", "figure_only"
]
NOT_ANSWERABLE_FROM_TEXT = ("unanswerable", "figure_only")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = Field(min_length=2)
    type: QuestionType
    gold_docs: list[str]
    gold_pages: list[int]
    answer: str | None
    evidence_snippet: str | None
    author: Literal["human", "llm"]
    verified: bool
    split: Literal["dev", "test"]
    notes: str | None = None  # free text for labellers; not used by any metric
    # Labeller's estimate of how hard the question is; optional, not used by any metric yet.
    difficulty: Literal["easy", "medium", "hard"] | None = None

    @property
    def answerable(self) -> bool:
        """Answerable from the extracted text (False = the system should refuse)."""
        return self.type not in NOT_ANSWERABLE_FROM_TEXT

    @model_validator(mode="after")
    def check_consistency(self) -> "Question":
        if self.type == "unanswerable":
            if self.gold_docs or self.gold_pages or self.answer is not None:
                raise ValueError(
                    "unanswerable questions need gold_docs=[], gold_pages=[], answer=null"
                )
            return self
        if not self.gold_docs or not self.gold_pages:
            raise ValueError(f"{self.type} questions need at least one gold doc and gold page")
        if any(p < 1 for p in self.gold_pages):
            raise ValueError("gold_pages are 1-based PDF page numbers")
        if self.type == "figure_only":
            if self.evidence_snippet is not None:
                raise ValueError("figure_only questions need evidence_snippet=null")
        elif not self.answer or not self.evidence_snippet:
            raise ValueError("answerable questions need an answer and an evidence_snippet")
        return self


def load_questions(
    path: Path,
    split: str | None = None,
    verified_only: bool = False,
) -> list[Question]:
    """Load and validate every line; raise ValueError naming the first bad line."""
    questions: list[Question] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                questions.append(Question.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"{path}:{lineno}: invalid question: {e}") from e
    ids = [q.id for q in questions]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"{path}: duplicate question ids: {duplicates}")
    return [
        q
        for q in questions
        if (split is None or q.split == split) and (q.verified or not verified_only)
    ]
