"""Centralized logging configuration for ankigen."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

# Default settings
DEFAULT_LOG_DIR = "./logs"
DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_LOG_RETENTION = -1  # days (-1 = keep forever)


def get_log_dir() -> Path:
    """Get the log directory from environment or default."""
    return Path(os.getenv("ANKIGEN_LOG_DIR", DEFAULT_LOG_DIR))


def get_log_level() -> int:
    """Get the file log level from environment or default."""
    level_name = os.getenv("ANKIGEN_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.DEBUG)


def get_log_retention() -> int:
    """
    Get the log retention period in days from environment or default.

    Returns:
        Number of days to keep logs, or -1 for infinite retention
    """
    try:
        return int(os.getenv("ANKIGEN_LOG_RETENTION", DEFAULT_LOG_RETENTION))
    except ValueError:
        return DEFAULT_LOG_RETENTION


def cleanup_old_logs(log_dir: Path, retention_days: int) -> None:
    """
    Remove log files older than the retention period.

    Args:
        log_dir: Directory containing log files
        retention_days: Days to keep logs (-1 = keep forever)
    """
    # Skip cleanup if retention is infinite
    if retention_days < 0:
        return

    if not log_dir.exists():
        return

    cutoff = datetime.now() - timedelta(days=retention_days)

    for log_file in log_dir.glob("ankigen_*.log"):
        try:
            # Parse date from filename: ankigen_YYYYMMDD.log
            date_str = log_file.stem.replace("ankigen_", "")
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff:
                log_file.unlink()
                logging.debug("Removed old log file: %s", log_file.name)
        except (ValueError, OSError):
            # Skip files that don't match the expected pattern or can't be deleted
            pass


def setup_logging(*, verbose: bool = False) -> None:
    """
    Configure logging with console and file handlers.

    Args:
        verbose: If True, console shows DEBUG level; otherwise INFO
    """
    # Get root logger for ankigen
    root_logger = logging.getLogger("ankigen")
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler: clean output
    console_handler = logging.StreamHandler()
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler: detailed output with timestamps
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Daily log file: ankigen_YYYYMMDD.log
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"ankigen_{today}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(get_log_level())
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Cleanup old logs
    cleanup_old_logs(log_dir, get_log_retention())

    # Log startup
    root_logger.debug("Logging initialized: console=%s, file=%s", console_level, log_file)
