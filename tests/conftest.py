"""Shared test fixtures."""

import zlib
from pathlib import Path

import numpy as np
import pymupdf
import pytest


class FakeEmbedder:
    """Deterministic stand-in for the real Embedder: no model, no download.

    Bag of words hashed into 64 dimensions (crc32, unlike hash(), is stable
    across runs), then normalised. Texts sharing words get higher cosine
    similarity, which is enough to test that search orders results correctly.
    """

    model_name = "fake-bag-of-words"
    dim = 64

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)[np.newaxis, :]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def tiny_corpus() -> list[dict]:
    """Three chunks on clearly different topics."""
    texts = [
        "glaciers melted fast and glacier mass loss continued",
        "river discharge was below normal during the drought",
        "groundwater levels fell in many aquifers",
    ]
    return [
        {
            "chunk_id": f"r.pdf:p{i}:c0",
            "doc": "r.pdf",
            "page": i,
            "page_label": str(i + 10),
            "text": t,
        }
        for i, t in enumerate(texts, start=1)
    ]


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Factory: make_pdf("name.pdf", ["page 1 text", "", ...]) -> Path.

    An empty string produces a blank page. PDFs are generated on the fly so
    no binary test files need to be committed.
    """

    def _make(name: str, page_texts: list[str]) -> Path:
        path = tmp_path / name
        doc = pymupdf.open()
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(path)
        doc.close()
        return path

    return _make


@pytest.fixture
def make_question():
    """Factory for eval Questions with sensible defaults; override any field."""
    from rag.eval.schema import Question

    def _make(**overrides) -> Question:
        fields = {
            "id": "q1",
            "question": "glacier mass loss",
            "language": "en",
            "type": "single_fact",
            "gold_docs": ["r.pdf"],
            "gold_pages": [1],
            "answer": "Glaciers lost mass.",
            "evidence_snippet": "glaciers melted",
            "author": "human",
            "verified": True,
            "split": "dev",
        }
        if overrides.get("type") == "unanswerable":
            fields.update(gold_docs=[], gold_pages=[], answer=None, evidence_snippet=None)
        fields.update(overrides)
        return Question(**fields)

    return _make
