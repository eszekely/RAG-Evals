"""Turn text into vectors with a multilingual sentence-transformers model (CPU).

e5 models were trained with role prefixes: documents as "passage: ..." and
questions as "query: ...". Leaving the prefixes out silently hurts retrieval,
so the Embedder adds them; callers pass plain text.

Vectors are L2-normalised, so the inner product of two vectors equals their
cosine similarity (used by the FAISS IndexFlatIP in rag.index).
"""

from typing import Protocol

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class TextEmbedder(Protocol):
    """What index building and search need from an embedder.

    The real Embedder and the fake one used in tests both satisfy this.
    """

    model_name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Embedder:
    """Wraps a sentence-transformers model, adding e5 prefixes and normalising."""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32, model=None):
        self.model_name = model_name
        self.batch_size = batch_size
        if model is None:
            # Imported here so that importing rag.embed (e.g. in tests) stays fast:
            # sentence-transformers pulls in PyTorch, which takes seconds to load.
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device="cpu")
        self.model = model

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)  # FAISS requires float32

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents. Returns an array of shape (len(texts), dim)."""
        return self._encode([PASSAGE_PREFIX + t for t in texts], show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question. Returns an array of shape (1, dim)."""
        return self._encode([QUERY_PREFIX + text])

    @property
    def max_tokens(self) -> int:
        """Longest input the model reads; anything beyond is silently truncated."""
        return self.model.max_seq_length

    def count_passage_tokens(self, texts: list[str]) -> list[int]:
        """Token count of each passage as the model sees it (prefix + special tokens)."""
        encoded = self.model.tokenizer([PASSAGE_PREFIX + t for t in texts], truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]
