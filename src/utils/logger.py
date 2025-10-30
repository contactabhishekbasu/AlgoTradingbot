"""Logging configuration for the trading system."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from pythonjsonlogger import jsonlogger

from .config import settings


def setup_logging() -> structlog.BoundLogger:
    """
    Set up structured logging with JSON output.

    Returns:
        Configured structlog logger
    """
    # Ensure log directory exists
    log_dir = Path(settings.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure standard logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create formatters
    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        timestamp=True,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if settings.debug:
        # Human-readable format for development
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
    else:
        # JSON format for production
        console_handler.setFormatter(json_formatter)

    # File handler
    file_handler = logging.FileHandler(log_dir / "trading.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(json_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.debug
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()
    logger.info(
        "logging_initialized",
        level=settings.log_level,
        environment=settings.environment,
        debug=settings.debug,
    )

    return logger


def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask sensitive information in log data.

    Args:
        data: Dictionary potentially containing sensitive data

    Returns:
        Dictionary with sensitive values masked
    """
    sensitive_keys = {
        "password",
        "api_key",
        "secret_key",
        "token",
        "authorization",
        "secret",
    }

    masked_data = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            masked_data[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value)
        else:
            masked_data[key] = value

    return masked_data


# Global logger instance
logger = setup_logging()
