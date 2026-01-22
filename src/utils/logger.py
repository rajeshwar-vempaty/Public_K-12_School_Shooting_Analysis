"""
Logging Module
Comprehensive logging framework with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from .config import get_config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file. If None, uses config default
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable console logging

    Returns:
        Configured logger instance
    """
    try:
        config = get_config()
    except Exception as e:
        # Config may not be available (e.g., during early initialization)
        # Use default logging settings in this case
        config = None

    # Get logging configuration
    if config:
        log_level = level or config.get("logging.level", "INFO")
        log_format = config.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        file_enabled = config.get("logging.file.enabled", True)
        console_enabled = config.get("logging.console.enabled", True) and console
        max_bytes = config.get("logging.file.max_bytes", 10485760)
        backup_count = config.get("logging.file.backup_count", 5)

        if log_file is None:
            log_file = config.get("logging.file.filename", "logs/app.log")
    else:
        # Fallback defaults if config not available
        log_level = level or "INFO"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_enabled = True
        console_enabled = console
        max_bytes = 10485760
        backup_count = 5
        log_file = log_file or "logs/app.log"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    if file_enabled and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = ColoredFormatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


# Module-level convenience functions
def debug(message: str, logger_name: str = "app") -> None:
    """Log debug message."""
    get_logger(logger_name).debug(message)


def info(message: str, logger_name: str = "app") -> None:
    """Log info message."""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: str = "app") -> None:
    """Log warning message."""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: str = "app") -> None:
    """Log error message."""
    get_logger(logger_name).error(message)


def critical(message: str, logger_name: str = "app") -> None:
    """Log critical message."""
    get_logger(logger_name).critical(message)
