import hashlib
import json
from typing import Literal

from redis.asyncio import Redis

from app.core.config import settings


class ConversationMemory:
    def _key(self, session_id: str) -> str:
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        return f"session:{digest}:turns"

    async def recent(self, session_id: str) -> list[dict[str, str]]:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            values = await redis.lrange(self._key(session_id), 0, settings.session_max_turns * 2 - 1)
            turns = [json.loads(value) for value in reversed(values)]
            return [turn for turn in turns if isinstance(turn, dict) and "role" in turn and "content" in turn]
        except Exception:
            return []
        finally:
            await redis.aclose()

    async def append(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            key = self._key(session_id)
            await redis.lpush(key, json.dumps({"role": role, "content": content[:6000]}))
            await redis.ltrim(key, 0, settings.session_max_turns * 2 - 1)
            await redis.expire(key, settings.session_ttl_seconds)
        except Exception:
            return
        finally:
            await redis.aclose()
