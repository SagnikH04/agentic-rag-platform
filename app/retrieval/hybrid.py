"""Scalable workspace-scoped hybrid retrieval.

Dense candidates come directly from Qdrant and sparse candidates from
PostgreSQL FTS.  They are hydrated together, so neither path is constrained by
the other or by a recent-10K in-memory corpus.
"""

import re
from collections import defaultdict
from time import perf_counter

from sqlalchemy import func, select

from app.core.cache import cached_json, knowledge_revision, redis_client, retrieval_key, set_json
from app.core.config import settings
from app.core.db import Chunk, Document, SessionLocal
from app.models.schemas import Citation
from app.retrieval.embeddings import embed_texts_async
from app.retrieval.vector_store import search_vectors


def _nested_value(metadata: dict, path: str):
    value = metadata
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _matches_filters(metadata: dict, filters: dict) -> bool:
    return all((expected in actual if isinstance(actual := _nested_value(metadata, key), list) else actual == expected) for key, expected in filters.items())


async def _sparse_candidates(question: str, workspace_id: str, limit: int) -> dict[str, float]:
    """Use indexed PostgreSQL full-text search, not per-query BM25 rebuilds."""
    tsquery = func.websearch_to_tsquery("simple", question)
    document = func.to_tsvector("simple", Chunk.text)
    rank = func.ts_rank_cd(document, tsquery)
    async with SessionLocal() as session:
        rows = (await session.execute(select(Chunk.id, rank.label("rank")).join(Document, Chunk.document_id == Document.id)
            .where(Document.workspace_id == workspace_id, Document.status == "indexed", document.op("@@")(tsquery))
            .order_by(rank.desc()).limit(limit))).all()
    return {chunk_id: float(score) for chunk_id, score in rows}


async def _hydrate(chunk_ids: set[str], workspace_id: str, filters: dict) -> dict[str, Citation]:
    if not chunk_ids:
        return {}
    async with SessionLocal() as session:
        rows = (await session.execute(select(Chunk, Document).join(Document, Chunk.document_id == Document.id).where(
            Chunk.workspace_id == workspace_id, Chunk.id.in_(chunk_ids), Document.status == "indexed"))).all()
    result = {}
    for chunk, document in rows:
        metadata = {**(document.doc_metadata or {}), **(chunk.chunk_metadata or {})}
        if _matches_filters(metadata, filters):
            result[chunk.id] = Citation(document_id=document.id, chunk_id=chunk.id, score=0.0, text=chunk.text, metadata=metadata,
                source=(document.doc_metadata or {}).get("file_name") or document.source, content_type=document.content_type)
    return result


def _rrf(ids: list[str]) -> dict[str, float]:
    return {chunk_id: 1.0 / (60 + rank) for rank, chunk_id in enumerate(ids, 1)}


async def _citations_from_ids(ids: list[str], scores: dict[str, float], workspace_id: str) -> list[Citation]:
    records = await _hydrate(set(ids), workspace_id, {})
    return [records[item].model_copy(update={"score": float(scores.get(item, 0.0))}) for item in ids if item in records]


async def hybrid_search(question: str, top_k: int, filters: dict | None = None, workspace_id: str = "default") -> list[Citation]:
    filters = filters or {}
    redis = await redis_client()
    try:
        revision = await knowledge_revision(redis, workspace_id)
        key = retrieval_key(workspace_id, revision, question, filters, top_k)
        if cached := await cached_json(redis, key):
            return await _citations_from_ids(cached["chunk_ids"], cached.get("scores", {}), workspace_id)
        started = perf_counter()
        candidate_k = max(top_k * 4, settings.retrieval_candidate_k)
        sparse = await _sparse_candidates(question, workspace_id, max(candidate_k, settings.sparse_candidate_limit)) if settings.retrieval_mode != "dense" else {}
        dense: dict[str, float] = {}
        if settings.retrieval_mode != "sparse":
            vector = (await embed_texts_async([question]))[0]
            for hit in await search_vectors(vector, candidate_k, filters, workspace_id):
                payload = hit.payload or {}
                dense[str(payload.get("chunk_id") or hit.id)] = float(hit.score)
        ids = set(sparse) | set(dense)
        records = await _hydrate(ids, workspace_id, filters)
        dense_rank = _rrf([key for key, _ in sorted(dense.items(), key=lambda item: item[1], reverse=True)])
        sparse_rank = _rrf([key for key, _ in sorted(sparse.items(), key=lambda item: item[1], reverse=True)])
        ranked = []
        for chunk_id, citation in records.items():
            # RRF deliberately fuses ranks rather than incomparable raw scores.
            score = dense_rank.get(chunk_id, 0.0) + sparse_rank.get(chunk_id, 0.0)
            ranked.append(citation.model_copy(update={"score": score}))
        by_document, selected, seen = defaultdict(int), [], set()
        for citation in sorted(ranked, key=lambda item: item.score, reverse=True):
            fingerprint = re.sub(r"\W+", "", citation.text.lower())[:240]
            if fingerprint in seen or by_document[citation.document_id] >= settings.max_chunks_per_document_in_response:
                continue
            by_document[citation.document_id] += 1; seen.add(fingerprint); selected.append(citation)
            if len(selected) == top_k:
                break
        await set_json(redis, key, {"chunk_ids": [item.chunk_id for item in selected], "scores": {item.chunk_id: item.score for item in selected},
            "retrieval_ms": round((perf_counter() - started) * 1000, 2)}, settings.retrieval_cache_ttl_seconds)
        return selected
    finally:
        await redis.aclose()
