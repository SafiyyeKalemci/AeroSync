import logging
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(*, log_file: Path | None = None) -> Path | None:
    """Konsol + kalıcı dosya loglaması kurar.

    Sunucu GET /prediction/ ile geri okumaya kapalı olduğundan, gönderilen
    tahminlerin tek kalıcı kaydı bu dosyadır (itiraz süreci için de kullanılır).
    """
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    resolved_file = log_file
    if resolved_file is None:
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        resolved_file = logs_dir / f"run_{timestamp}.log"
    handlers.append(logging.FileHandler(resolved_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)
    return resolved_file
