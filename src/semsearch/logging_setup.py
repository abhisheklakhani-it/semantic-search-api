"""Structured logging: every log line is one JSON object, easy to search in Loki, CloudWatch or `jq`."""

import json
import logging
import sys
from datetime import datetime, timezone

# Attributes every LogRecord has; anything else was passed through `extra=` and belongs in the JSON.
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Route uvicorn's own logs through the same JSON handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers[:] = []
        logger.propagate = True
    # Our middleware already logs every request with more detail, so drop uvicorn's access line.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Hugging Face download progress and HTTP client chatter are noise in production logs.
    for name in ("httpx", "httpcore", "huggingface_hub", "fastembed"):
        logging.getLogger(name).setLevel(logging.WARNING)
