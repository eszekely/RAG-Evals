"""Extract text per page from the PDFs listed in manifest.csv.

Usage:
    uv run python -m rag.ingest [--manifest manifest.csv] [--raw data/raw]
                                [--out data/processed/pages.jsonl]

Only files listed in the manifest are ingested, so stray or renamed PDFs left in
data/raw/ are never indexed. A listed file that is missing is an error (run
scripts/download.py first) rather than being silently skipped.

Output: one JSONL record per page:
    {"doc": "<pdf filename>", "page": <1-based PDF index>, "page_label": "<printed>", "text": "..."}

Two page numbers are stored because they often differ: `page` is the position in
the PDF file (what a PDF viewer's page box shows), `page_label` is the number
printed on the page (e.g. "xii" or "33"), taken from the PDF's own page-label
table. If the PDF defines no labels, `page_label` is None rather than a guess.

Empty pages are kept (with text "") so the record faithfully mirrors the PDF;
the chunker decides what to skip.
"""

import argparse
from pathlib import Path

import pymupdf

from rag.jsonl import write_jsonl
from rag.manifest import read_manifest


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return one record per page of `pdf_path`.

    Page numbers are 1-based, matching what a reader sees in a PDF viewer,
    so citations like "page 12" can be checked by hand.
    """
    with pymupdf.open(pdf_path) as doc:
        return [
            {
                "doc": pdf_path.name,
                "page": number,
                "page_label": page.get_label() or None,  # "" means no label defined
                "text": page.get_text(),
            }
            for number, page in enumerate(doc, start=1)
        ]


def ingest_from_manifest(manifest_path: Path, raw_dir: Path) -> list[dict]:
    """Extract pages from each PDF listed in the manifest, in manifest order.

    Raises FileNotFoundError naming every listed PDF that is missing from `raw_dir`.
    """
    paths = [raw_dir / row["filename"] for row in read_manifest(manifest_path)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {raw_dir}: {', '.join(missing)}. Run scripts/download.py first."
        )
    return [page for path in paths for page in extract_pages(path)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract per-page text from PDFs.")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/pages.jsonl"))
    args = parser.parse_args(argv)

    pages = ingest_from_manifest(args.manifest, args.raw)
    write_jsonl(pages, args.out)
    docs = {p["doc"] for p in pages}
    empty = sum(1 for p in pages if not p["text"].strip())
    print(f"Wrote {len(pages)} pages from {len(docs)} PDFs to {args.out} ({empty} empty pages)")


if __name__ == "__main__":
    main()
