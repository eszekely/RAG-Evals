"""Download the PDFs listed in manifest.csv into data/raw/.

Usage:
    uv run python scripts/download.py [--manifest manifest.csv] [--out data/raw]

For each row: skip if the file already exists, otherwise download it,
check that it really is a PDF (starts with b"%PDF"), and print a short SHA-256
so the exact file version can be recorded.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from rag.manifest import read_manifest

USER_AGENT = "rag-downloader/0.1"
PDF_MAGIC = b"%PDF"


def is_pdf(path: Path) -> bool:
    """True if the file starts with the PDF magic bytes."""
    with path.open("rb") as f:
        return f.read(len(PDF_MAGIC)) == PDF_MAGIC


def short_sha256(path: Path, length: int = 12) -> str:
    """First `length` hex chars of the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def fetch(url: str, dest: Path, timeout: float = 60) -> None:
    """Download `url` to `dest` via a temporary file.

    Writing to a .part file first means an interrupted download never leaves
    a half-written file that would later be "skipped" as already present.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as f:
        while block := response.read(1 << 20):
            f.write(block)
    tmp.replace(dest)


def download_one(row: dict[str, str], out_dir: Path) -> str:
    """Download one manifest row. Returns a one-line status message.

    Raises ValueError if the downloaded file is not a PDF (the file is removed).
    """
    dest = out_dir / row["filename"]
    if dest.exists():
        return f"skip      {dest.name}  sha256:{short_sha256(dest)}"
    fetch(row["url"], dest)
    if not is_pdf(dest):
        dest.unlink()
        raise ValueError(f"{dest.name}: downloaded file is not a PDF (url={row['url']})")
    return f"download  {dest.name}  sha256:{short_sha256(dest)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    if not rows:
        print(f"No entries in {args.manifest}.")
        return 0

    failures = 0
    for row in rows:
        try:
            print(download_one(row, args.out))
        except Exception as exc:  # keep going; report all failures at the end
            failures += 1
            print(f"FAILED    {row['filename']}: {exc}", file=sys.stderr)
    print(f"{len(rows) - failures}/{len(rows)} files OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
