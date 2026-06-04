"""Unified logging setup. Called once at application startup."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

_configured = False


def setup_logging(level: int = logging.INFO):
    global _configured
    if _configured:
        return
    _configured = True

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")

    # File (rotating) — the only handler; stdout is not used to avoid
    # duplicate lines when the process's stdout is redirected to the same log file.
    log_dir = Path.home() / "stock-analysis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
