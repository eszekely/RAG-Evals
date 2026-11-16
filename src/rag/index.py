"""Build, save and load the FAISS index over all chunks.

Usage:
    uv run python -m rag.index [--chunks data/processed/chunks.jsonl]
                               [--out data/processed/index] [--force]

The index directory holds three files:
    faiss.index     the vectors (row i = chunk i)
    metadata.jsonl  a snapshot of the chunks in the same order: chunk_id, doc, page,
                    page_label, text. FAISS returns row numbers; this maps them back
                    to citations. It is a copy, not a link to chunks.jsonl, so that
                    re-chunking without rebuilding can never mismatch rows and vectors.
    config.json     model name, chunking parameters, build date, ...: how to reproduce it

An existing index is never overwritten unless --force is given, so a rebuild
only happens when explicitly requested.

IndexFlatIP = exact search by inner product. With normalised vectors this is
cosine similarity. Exact search over a few thousand vectors takes milliseconds;
approximate indexes (IVF, HNSW) or a vector database only pay off at far larger scale.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from rag.chunking import params_path
from rag.embed import TextEmbedder
from rag.jsonl import read_jsonl, write_jsonl

INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
CONFIG_FILE = "config.json"


def build_index(chunks: list[dict], embedder: TextEmbedder) -> faiss.Index:
    """Embed every chunk's text and put the vectors in an exact inner-product index."""
    if not chunks:
        raise ValueError("No chunks to index")
    vectors = embedder.embed_passages([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def make_config(embedder: TextEmbedder, index: faiss.Index, chunk_params: dict) -> dict:
    """Everything needed to tell how this index was built."""
    return {
        "model_name": embedder.model_name,
        "dim": index.d,
        "n_chunks": index.ntotal,
        **chunk_params,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def save_index(
    out_dir: Path, index: faiss.Index, chunks: list[dict], config: dict, force: bool = False
) -> None:
    """Write the three index files. Refuses to overwrite an existing index unless `force`."""
    if (out_dir / CONFIG_FILE).exists() and not force:
        raise FileExistsError(f"An index already exists in {out_dir}. Use --force to rebuild.")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    write_jsonl(chunks, out_dir / METADATA_FILE)
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict], dict]:
    """Load (index, chunks, config) saved by `save_index`."""
    if not (index_dir / CONFIG_FILE).exists():
        raise FileNotFoundError(f"No index in {index_dir}. Build it with: python -m rag.index")
    index = faiss.read_index(str(index_dir / INDEX_FILE))
    chunks = read_jsonl(index_dir / METADATA_FILE)
    config = json.loads((index_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise ValueError(f"Index has {index.ntotal} vectors but {len(chunks)} chunks")
    return index, chunks, config


def token_report(counts: list[int], limit: int) -> dict:
    """Summarise chunk token counts against the model's limit (truncation check)."""
    arr = np.array(counts)
    return {
        "max_tokens": limit,
        "chunks_over_limit": int((arr > limit).sum()),
        "token_count_median": int(np.median(arr)),
        "token_count_max": int(arr.max()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index over all chunks.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing index")
    args = parser.parse_args(argv)

    if (args.out / CONFIG_FILE).exists() and not args.force:
        raise SystemExit(f"An index already exists in {args.out}. Use --force to rebuild.")

    from rag.embed import Embedder  # slow import (PyTorch), only needed here

    chunks = read_jsonl(args.chunks)
    chunk_params = json.loads(params_path(args.chunks).read_text(encoding="utf-8"))
    embedder = Embedder()

    tokens = token_report(
        embedder.count_passage_tokens([c["text"] for c in chunks]), embedder.max_tokens
    )
    start = time.perf_counter()
    index = build_index(chunks, embedder)
    seconds = round(time.perf_counter() - start, 1)

    config = make_config(embedder, index, chunk_params) | {"embed_seconds": seconds, **tokens}
    save_index(args.out, index, chunks, config, force=args.force)
    print(json.dumps(config, indent=2))
    print(f"Saved index to {args.out}")


if __name__ == "__main__":
    main()
