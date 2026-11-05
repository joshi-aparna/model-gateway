import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):

    def format(self, record):
        event = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        if hasattr(record, "event_data"):
            event.update(record.event_data)

        if record.exc_info:
            event["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(event)


def create_logger(
    name: str = "llm_gateway",
    level: str = "INFO",
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
):
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = JsonFormatter()

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger