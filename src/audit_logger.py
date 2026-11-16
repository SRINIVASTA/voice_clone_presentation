import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("enterprise_voice_audit")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_generation_event(user_id: str, engine_mode: str, segment_count: int, status: str, detail: str = "") -> None:
    """Emits structured JSON compliance logs for enterprise observability."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "engine_mode": engine_mode,
        "segment_count": segment_count,
        "status": status,
        "detail": detail,
    }
    logger.info(json.dumps(payload))
