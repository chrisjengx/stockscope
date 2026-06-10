"""Unified logging setup. Called once at application startup."""
import logging
import sys
from pathlib import Path

_configured = False


def setup_logging(level: int = logging.INFO):
    global _configured
    if _configured:
        return
    _configured = True

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")

    log_dir = Path.home() / "stock-analysis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
