import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.core.config import settings


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _encode(value: Any) -> str | int | float:
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, default=str, separators=(",", ":"))


async def set_job_state(
    redis: Redis,
    job_id: str,
    *,
    status: str,
    stage: str,
    progress: int,
    retries: int | None = None,
    error: str | None = None,
    **extra: Any,
) -> None:
    mapping: dict[str, str | int | float] = {
        "status": status,
        "stage": stage,
        "progress": max(0, min(100, progress)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if retries is not None:
        mapping["retries"] = retries
    if error is not None:
        mapping["error"] = error[:2000]
    mapping.update({key: _encode(value) for key, value in extra.items()})
    try:
        await redis.hset(job_key(job_id), mapping=mapping)
        await redis.expire(job_key(job_id), settings.job_ttl_seconds)
    except Exception:
        return
