"""Centralized logger with file rotation and console output."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_configured = False


def setup_logging(log_file: str, level_str: str, max_bytes: int, backup_count: int):
    global _configured
    if _configured:
        return
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    level = getattr(logging, level_str.upper(), logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
