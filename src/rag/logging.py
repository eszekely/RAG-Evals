"""Structured JSON logs, one object per line, with an allow-list of fields.

Privacy by construction: the JSON formatter writes only the fields named in
ALLOWED_FIELDS, so question text, answer text, retrieved chunk text or keys
cannot end up in the logs, even if someone passes them by mistake.

(This module is rag.logging; inside it, `import logging` still means the standard
library, because Python 3 imports are absolute.)
"""

import json
import logging
from datetime import UTC, datetime

LOGGER_NAME = "rag"

ALLOWED_FIELDS = frozenset(
    {
        "event",
        "request_id",
        "endpoint",
        "status",
        "latency_ms",
        "backend",
        "model",
        "k",
        "retrieved",  # list of {"doc", "page"}: ids only, never text
        "top_score",
        "refused",
        "refusal_reason",
        "input_tokens",
        "output_tokens",
        "unverified_citations",
        "error_type",
        "n_chunks",
        "index_built_at",
        "backends",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
        }
        fields = getattr(record, "fields", {})
        entry |= {key: value for key, value in fields.items() if key in ALLOWED_FIELDS}
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Send the 'rag' logger's records to stderr as JSON (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False  # no duplicate plain-text copies via the root logger
    return logger


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """Log an event with structured fields (non-allowed fields are dropped by the formatter)."""
    logging.getLogger(LOGGER_NAME).log(level, event, extra={"fields": {"event": event, **fields}})
