"""Read and write JSON Lines files (one JSON object per line).

JSONL is used for all intermediate data: it is human-readable, streamable,
diff-friendly, and needs no extra dependency.
"""

import json
from collections.abc import Iterable
from pathlib import Path


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    """Write records to `path` (creating parent dirs). Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # ensure_ascii=False keeps German/French accents readable in the file.
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from `path`, ignoring blank lines."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
