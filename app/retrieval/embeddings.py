import asyncio
import json
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.cache import embedding_key, redis_client


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]
        encoded = model().encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors.extend(encoded.tolist())
    return vectors


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    redis = await redis_client()
    try:
        keys = [embedding_key(text) for text in texts]
        cached = await redis.mget(keys)
        missing_indexes = [index for index, raw in enumerate(cached) if raw is None]
        results: list[list[float] | None] = [
            json.loads(raw) if raw is not None else None for raw in cached
        ]
        if missing_indexes:
            vectors = await asyncio.to_thread(embed_texts, [texts[index] for index in missing_indexes])
            pipe = redis.pipeline()
            for index, vector in zip(missing_indexes, vectors, strict=True):
                results[index] = vector
                pipe.set(keys[index], json.dumps(vector, separators=(",", ":")), ex=settings.embedding_cache_ttl_seconds)
            await pipe.execute()
        return [vector or [] for vector in results]
    finally:
        await redis.aclose()
