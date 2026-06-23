"""Structured logging for CServe.

All CServe components use this module for logging.  Output is structured
JSON in production (parseable by any log aggregator) and human-readable
colored text during development.

Usage:
    from cserve.common.logging import get_logger
    log = get_logger("scheduler")
    log.info("assigned job", job_id=job_id, replica=replica_id)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


class _StructuredFormatter(logging.Formatter):
    """Emits log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "_structured_extra", None)
        if extra:
            entry.update(extra)
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class _HumanFormatter(logging.Formatter):
    """Human-readable colored output for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        component = getattr(record, "component", record.name)
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        msg = record.getMessage()

        extra = getattr(record, "_structured_extra", None)
        extra_str = ""
        if extra:
            parts = [f"{k}={v}" for k, v in extra.items()]
            extra_str = f"  [{', '.join(parts)}]"

        line = f"{color}{ts} {record.levelname:<5}{self.RESET} [{component}] {msg}{extra_str}"
        if record.exc_info and record.exc_info[1]:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that supports structured key-value pairs.

    log.info("message", key=value, key2=value2)
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        # Pull out our structured extras from kwargs
        extra = kwargs.get("extra", {})
        extra["component"] = self.extra.get("component", "cserve")

        # Everything that's not a standard logging kwarg goes into structured extra
        structured = {}
        standard_keys = {"exc_info", "stack_info", "stacklevel", "extra"}
        for k, v in list(kwargs.items()):
            if k not in standard_keys:
                structured[k] = v
                del kwargs[k]

        extra["_structured_extra"] = structured
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(component: str) -> _StructuredLogger:
    """Get a structured logger for a CServe component.

    Args:
        component: name of the component (e.g. "scheduler", "gateway",
                   "node_agent", "autoscaler").
    """
    logger = logging.getLogger(f"cserve.{component}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)

        log_format = os.environ.get("CSERVE_LOG_FORMAT", "human")
        if log_format == "json":
            handler.setFormatter(_StructuredFormatter())
        else:
            handler.setFormatter(_HumanFormatter())

        logger.addHandler(handler)
        logger.setLevel(
            getattr(logging, os.environ.get("CSERVE_LOG_LEVEL", "INFO").upper(), logging.INFO)
        )
        logger.propagate = False

    return _StructuredLogger(logger, {"component": component})
