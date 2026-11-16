"""Export the extracted page text as plain-text files, 20 pages per file.

Usage:
    uv run python scripts/export_text.py [--pages data/processed/pages.jsonl]
                                         [--out data/processed/text] [--part-size 20]

For each document, writes <doc>_pp<start>-<end>.txt (doc name without ".pdf"), where
start-end is a fixed range of PDF pages (1-20, 21-40, ...; the last part ends at the
document's last page). Inside, each page starts with a marker line:

    === PAGE <n> ===

<n> is the PDF page index (1-based), the same `page` number stored in chunk metadata
and chunk ids (e.g. doc.pdf:p47:c0) - not the printed page label. Pages with fewer
than 200 characters (covers, blank or near-empty pages) are skipped, so a part can
have gaps; a part with no pages left is not written.

The output contains text from copyrighted reports: data/processed/ is git-ignored.
"""

import argparse
from pathlib import Path

from rag.jsonl import read_jsonl

MIN_CHARS = 200
PART_SIZE = 20


def group_by_doc(pages: list[dict]) -> dict[str, list[dict]]:
    """{doc: pages sorted by page number}, in first-seen document order."""
    docs: dict[str, list[dict]] = {}
    for page in pages:
        docs.setdefault(page["doc"], []).append(page)
    return {doc: sorted(ps, key=lambda p: p["page"]) for doc, ps in docs.items()}


def part_ranges(last_page: int, size: int = PART_SIZE) -> list[tuple[int, int]]:
    """[(1, 20), (21, 40), ..., (k, last_page)]"""
    return [(start, min(start + size - 1, last_page)) for start in range(1, last_page + 1, size)]


def render_part(pages: list[dict]) -> str:
    return "".join(f"=== PAGE {p['page']} ===\n{p['text'].strip()}\n\n" for p in pages)


def export(
    pages: list[dict], out_dir: Path, size: int = PART_SIZE, min_chars: int = MIN_CHARS
) -> list[tuple[Path, int]]:
    """Write all part files; return (path, number of characters) for each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.txt"):  # a re-run never leaves stale parts behind
        old.unlink()
    written = []
    for doc, doc_pages in group_by_doc(pages).items():
        kept = [p for p in doc_pages if len(p["text"].strip()) >= min_chars]
        for start, end in part_ranges(doc_pages[-1]["page"], size):
            part = [p for p in kept if start <= p["page"] <= end]
            if not part:
                continue
            text = render_part(part)
            path = out_dir / f"{Path(doc).stem}_pp{start}-{end}.txt"
            path.write_text(text, encoding="utf-8")
            written.append((path, len(text)))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export page text to .txt files.")
    parser.add_argument("--pages", type=Path, default=Path("data/processed/pages.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/text"))
    parser.add_argument("--part-size", type=int, default=PART_SIZE)
    args = parser.parse_args(argv)

    written = export(read_jsonl(args.pages), args.out, args.part_size)
    for path, chars in written:
        print(f"{path.name:45} {chars:>8,} chars  ~{chars // 4:>7,} tokens")
    total = sum(chars for _, chars in written)
    print(f"{len(written)} files, {total:,} chars, ~{total // 4:,} tokens (chars / 4)")


if __name__ == "__main__":
    main()
