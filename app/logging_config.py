import logging
from pathlib import Path

#   Logs are written to both the console and a log file.

def configure_logging(
    log_file: str = "logs/ghostbusters.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Logs are written to both the console and a log file.
    """

    log_path = Path(log_file)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("ghostbusters")

    # Prevent duplicate handlers if configure_logging()
    # is called more than once.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler.
    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger