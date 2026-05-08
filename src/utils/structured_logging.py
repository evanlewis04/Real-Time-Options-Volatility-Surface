"""Small structured logging helpers for provider and dashboard events."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Render log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default, sort_keys=True)


def configure_structured_logging(settings: Any) -> None:
    """Configure root logging once from typed settings."""
    level_name = str(getattr(settings, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    log_file = Path(getattr(settings, "log_file", "volatility_system.log"))
    structured = bool(getattr(settings, "structured", True))
    formatter: logging.Formatter
    if structured:
        formatter = StructuredJsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file.resolve()
               for handler in root.handlers):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root.addHandler(handler)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured info-level event."""
    logger.info(event, extra={"structured": {"event": event, **fields}})


def _json_default(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)
