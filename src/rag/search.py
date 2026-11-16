"""Find the chunks most similar to a question.

Usage:
    uv run python -m rag.search "your question" [--k 5] [--index data/processed/index]

Scores are cosine similarities (higher = more similar). With e5 they tend to sit in
a narrow, high band, so compare them relative to each other, not to 0 or 1.

Not here yet, deliberately (add only if the Stage 5 evaluation shows a need):
- a reranker: a cross-encoder that rescores the top ~20 results inside `search`;
- hybrid search: BM25 keyword scores merged with these vector scores, helping
  with exact terms such as report numbers;
- a vector database: only worth it at far larger scale or with frequent updates.
"""

import argparse
from pathlib import Path

import faiss

from rag.embed import TextEmbedder
from rag.index import load_index


def check_same_model(config: dict, embedder: TextEmbedder) -> None:
    """Queries must be embedded by the model that embedded the index.

    A mismatch raises no error by itself; it just returns meaningless results.
    """
    if embedder.model_name != config["model_name"]:
        raise ValueError(
            f"Index was built with {config['model_name']!r}, "
            f"but the embedder is {embedder.model_name!r}"
        )


def search(
    query: str, index: faiss.Index, chunks: list[dict], embedder: TextEmbedder, k: int = 5
) -> list[dict]:
    """Return up to `k` chunks, best first, each with its score and rank."""
    scores, ids = index.search(embedder.embed_query(query), k)
    return [
        {"rank": rank, "score": float(score), **chunks[i]}
        for rank, (score, i) in enumerate(zip(scores[0], ids[0], strict=True), start=1)
        if i != -1  # FAISS pads with -1 when k is larger than the index
    ]


def search_per_doc(
    query: str, index: faiss.Index, chunks: list[dict], embedder: TextEmbedder, per_doc: int
) -> list[dict]:
    """Best `per_doc` chunks from EACH document, merged by score and re-ranked.

    Lets a question comparing years pull from several reports even when one report
    dominates the global top-k. Exact search makes scoring every chunk cheap
    (milliseconds for a few thousand), so this simply ranks everything and filters.
    """
    kept, count = [], {}
    for hit in search(query, index, chunks, embedder, k=index.ntotal):
        if count.get(hit["doc"], 0) < per_doc:
            count[hit["doc"]] = count.get(hit["doc"], 0) + 1
            kept.append(hit)
    return [{**hit, "rank": rank} for rank, hit in enumerate(kept, start=1)]


def format_hit(hit: dict, chars: int) -> str:
    """One result as text, e.g. '1. 0.842  report.pdf  printed p.33 (PDF p.47)'."""
    printed = hit["page_label"] if hit["page_label"] is not None else "-"
    text = hit["text"] if len(hit["text"]) <= chars else hit["text"][:chars] + " ..."
    return (
        f"{hit['rank']}. {hit['score']:.3f}  {hit['doc']}  "
        f"printed p.{printed} (PDF p.{hit['page']})  [{hit['chunk_id']}]\n   {text}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search the index for a question.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chars", type=int, default=300, help="text shown per hit")
    args = parser.parse_args(argv)

    index, chunks, config = load_index(args.index)

    from rag.embed import Embedder  # slow import (PyTorch), only needed here

    embedder = Embedder(config["model_name"])  # the index's model, not a code default
    for hit in search(args.query, index, chunks, embedder, args.k):
        print(format_hit(hit, args.chars))


if __name__ == "__main__":
    main()
