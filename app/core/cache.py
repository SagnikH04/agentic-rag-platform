import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from redis.asyncio import Redis

from app.core.config import settings

T = TypeVar("T")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


async def redis_client() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


async def knowledge_revision(redis: Redis, workspace_id: str) -> int:
    raw = await redis.get(f"ws:{stable_hash(workspace_id)}:revision")
    return int(raw or "0")


async def bump_knowledge_revision(redis: Redis, workspace_id: str) -> int:
    return int(await redis.incr(f"ws:{stable_hash(workspace_id)}:revision"))


def embedding_key(text: str) -> str:
    return f"emb:{stable_hash({'model': settings.embedding_model, 'text': normalize_text(text)})}"


def retrieval_key(workspace_id: str, revision: int, question: str, filters: dict, top_k: int) -> str:
    return "ret:" + stable_hash(
        {"workspace": workspace_id, "revision": revision, "question": normalize_text(question), "filters": filters, "top_k": top_k}
    )


def answer_key(workspace_id: str, revision: int, session_hash: str, question: str) -> str:
    return "ans:" + stable_hash(
        {
            "workspace": workspace_id,
            "revision": revision,
            "model": settings.groq_model,
            "prompt": settings.groq_prompt_version,
            "session": session_hash,
            "question": normalize_text(question),
        }
    )


async def cached_json(redis: Redis, key: str) -> Any | None:
    raw = await redis.get(key)
    return json.loads(raw) if raw else None


async def set_json(redis: Redis, key: str, value: Any, ttl: int) -> None:
    await redis.set(key, json.dumps(value, separators=(",", ":"), default=str), ex=ttl)


async def with_lock(redis: Redis, key: str, factory: Callable[[], Awaitable[T]], ttl: int = 90) -> T:
    lock = redis.lock(f"lock:{key}", timeout=ttl, blocking_timeout=ttl)
    async with lock:
        return await factory()
