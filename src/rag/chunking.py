"""Split page texts into overlapping chunks.

Usage:
    uv run python -m rag.chunking [--pages data/processed/pages.jsonl]
                                  [--out data/processed/chunks.jsonl]

Design:
- Chunks never cross a page boundary, so every chunk has exactly one page
  and citations ("doc, page N") are always exact.
- Size is measured in words (whitespace split) as a cheap, dependency-free
  proxy for tokens. The first choice, 300 words, measured with the e5 tokenizer
  gave a median of 428 tokens, but 60 of 531 body-text chunks exceeded the
  model's 512-token limit (their ends were never embedded). 220 words leaves
  headroom; `python -m rag.index` reports how many chunks still exceed it.
- Consecutive chunks on the same page share `overlap` words, so a sentence cut
  at a chunk boundary still appears whole in at least one chunk.
"""

import argparse
import json
from pathlib import Path

from rag.jsonl import read_jsonl, write_jsonl

CHUNK_SIZE = 220  # words
OVERLAP = 40  # words


def split_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Split `words` into windows of `size` words, each sharing `overlap` with the previous.

    The last window may be shorter. No window is produced for an empty list.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    step = size - overlap
    windows: list[list[str]] = []
    start = 0
    while start < len(words):
        windows.append(words[start : start + size])
        if start + size >= len(words):  # this window reached the end
            break
        start += step
    return windows


def make_chunk_id(doc: str, page: int, index: int) -> str:
    """Stable, human-readable id, e.g. 'report.pdf:p12:c0'."""
    return f"{doc}:p{page}:c{index}"


def chunk_page(page: dict, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """Chunk one page record {"doc", "page", "page_label", "text"}. Empty pages give no chunks.

    The chunk id uses the PDF page index because it is always present and unique;
    printed labels can be missing or repeated (e.g. "1" in two sections).
    """
    windows = split_words(page["text"].split(), size, overlap)
    return [
        {
            "chunk_id": make_chunk_id(page["doc"], page["page"], i),
            "doc": page["doc"],
            "page": page["page"],
            "page_label": page.get("page_label"),
            "text": " ".join(window),
        }
        for i, window in enumerate(windows)
    ]


def chunk_pages(pages: list[dict], size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """Chunk every page, preserving document and page order."""
    return [chunk for page in pages for chunk in chunk_page(page, size, overlap)]


def params_path(chunks_path: Path) -> Path:
    """Where the chunking parameters are stored: chunks.jsonl -> chunks.config.json."""
    return chunks_path.with_suffix(".config.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Split pages into overlapping chunks.")
    parser.add_argument("--pages", type=Path, default=Path("data/processed/pages.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    args = parser.parse_args(argv)

    chunks = chunk_pages(read_jsonl(args.pages), args.size, args.overlap)
    write_jsonl(chunks, args.out)
    # Record the parameters next to the chunks so the index can store what was really used.
    params = {"chunk_size_words": args.size, "overlap_words": args.overlap}
    params_path(args.out).write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {args.out} (size={args.size}, overlap={args.overlap})")


if __name__ == "__main__":
    main()
