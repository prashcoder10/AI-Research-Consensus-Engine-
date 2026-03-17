import logging
import logging.handlers
import os
from typing import Optional
from src.utils.config_loader import load_config

# Global logger cache to avoid duplicate initialization
_LOGGERS = {}

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    Ensures singleton behavior per logger name.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    config = load_config()
    log_config = config.get("logging", {})

    logger = logging.getLogger(name)
    logger.setLevel(_get_log_level(log_config.get("level", "INFO")))
    logger.propagate = False  # prevent duplicate logs

    # Prevent adding handlers multiple times
    if not logger.handlers:
        formatter = _get_formatter(config)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_config.get("log_to_file", False):
            file_handler = _get_file_handler(log_config, formatter)
            logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger

# ==========================================
# Helper Functions
# ==========================================

def _get_log_level(level_str: str) -> int:
    return getattr(logging, level_str.upper(), logging.INFO)

def _get_formatter(config: dict) -> logging.Formatter:
    """
    Returns a structured or standard formatter.
    Can be extended to JSON logging easily.
    """
    env = config.get("app", {}).get("env", "development")

    if env == "production":
        # Structured log format (machine-friendly)
        format_str = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","message":"%(message)s"}'
        )
    else:
        # Human-readable format
        format_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    return logging.Formatter(format_str)

def _get_file_handler(config: dict, formatter: logging.Formatter) -> logging.Handler:
    """
    Creates a rotating file handler.
    """
    log_file = config.get("log_file_path", "./logs/app.log")

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Rotate logs: 10MB per file, keep 5 backups
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    handler.setFormatter(formatter)

    return handler