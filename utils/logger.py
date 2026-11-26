"""Logging utilities with sensitive credential redaction."""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable


class SensitiveDataFilter(logging.Filter):
    """Redacts configured API keys and common token patterns from log records."""

    _TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
        re.compile(r"AQ\.[0-9A-Za-z\-_]{20,}"),
        re.compile(r"ghp_[0-9A-Za-z]{20,}"),
        re.compile(r"github_pat_[0-9A-Za-z_]{20,}"),
    )

    def _get_secrets(self) -> Iterable[str]:
        for env_name in ("GEMINI_API_KEY", "GITHUB_TOKEN"):
            val = os.getenv(env_name, "").strip()
            if len(val) >= 8:
                yield val

    def _redact(self, text: str) -> str:
        masked = text
        for secret in self._get_secrets():
            if secret in masked:
                masked = masked.replace(secret, "[REDACTED]")
        for pattern in self._TOKEN_PATTERNS:
            masked = pattern.sub("[REDACTED]", masked)
        return masked

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: self._redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a configured logger with secret redaction."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
