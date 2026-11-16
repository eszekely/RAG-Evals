"""Tests for the Embedder wrapper, using a stub model (no download, no PyTorch)."""

import numpy as np

from rag.embed import Embedder


class StubModel:
    """Records what the Embedder sends to the model."""

    max_seq_length = 512

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 4), dtype=np.float64)

    def tokenizer(self, texts, truncation):
        return {"input_ids": [t.split() for t in texts]}


def test_passages_get_passage_prefix_and_normalisation():
    model = StubModel()
    Embedder("m", model=model).embed_passages(["a", "b"])
    texts, kwargs = model.calls[0]
    assert texts == ["passage: a", "passage: b"]
    assert kwargs["normalize_embeddings"] is True


def test_query_gets_query_prefix_and_shape_1_by_dim():
    model = StubModel()
    vector = Embedder("m", model=model).embed_query("wie viel Wasser?")
    assert model.calls[0][0] == ["query: wie viel Wasser?"]
    assert vector.shape == (1, 4)


def test_vectors_are_float32_for_faiss():
    assert Embedder("m", model=StubModel()).embed_passages(["a"]).dtype == np.float32


def test_token_count_includes_prefix():
    # The stub tokenizer splits on spaces: "passage: one two" -> 3 tokens.
    assert Embedder("m", model=StubModel()).count_passage_tokens(["one two"]) == [3]


def test_max_tokens_comes_from_model():
    assert Embedder("m", model=StubModel()).max_tokens == 512
