"""Read manifest.csv, the single list of source documents.

Columns: filename, title, year, language, publisher, url, licence_note.
Both the download script and ingestion use it, so only listed files are ever indexed.
"""

import csv
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Return manifest rows as dicts keyed by column name."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
