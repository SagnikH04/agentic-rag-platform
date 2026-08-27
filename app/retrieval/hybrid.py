import asyncio
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.core.cache import cached_json, knowledge_revision, redis_client, retrieval_key, set_json
from app.core.config import settings
from app.core.db import Chunk, Document, SessionLocal
from app.models.schemas import Citation
from app.retrieval.embeddings import embed_texts_async
from app.retrieval.vector_store import search_vectors

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\b[\w-]+\b", text.lower()) if token not in STOPWORDS]


def _nested_value(metadata: dict, path: str):
    value = metadata
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _matches_filters(metadata: dict, filters: dict) -> bool:
    for key, expected in filters.items():
        actual = _nested_value(metadata, key)
        if isinstance(actual, list):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (score - low) / (high - low) for key, score in scores.items()}


async def _load_sparse_rows(filters: dict, workspace_id: str):
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "indexed", Document.workspace_id == workspace_id)
                .order_by(Chunk.created_at.desc())
                .limit(settings.sparse_candidate_limit)
            )
        ).all()
    return [
        (chunk, document)
        for chunk, document in rows
        if _matches_filters({**(document.doc_metadata or {}), **(chunk.chunk_metadata or {})}, filters)
    ]


async def _citations_from_ids(chunk_ids: list[str], scores: dict[str, float], workspace_id: str) -> list[Citation]:
    if not chunk_ids:
        return []
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.workspace_id == workspace_id, Chunk.id.in_(chunk_ids), Document.status == "indexed")
            )
        ).all()
    by_id = {chunk.id: (chunk, document) for chunk, document in rows}
    citations = []
    for chunk_id in chunk_ids:
        if chunk_id not in by_id:
            continue
        chunk, document = by_id[chunk_id]
        citations.append(
            Citation(
                document_id=document.id,
                chunk_id=chunk.id,
                score=float(scores.get(chunk.id, 0.0)),
                text=chunk.text,
                metadata={**(document.doc_metadata or {}), **(chunk.chunk_metadata or {})},
                source=document.source,
                content_type=document.content_type,
            )
        )
    return citations


async def hybrid_search(question: str, top_k: int, filters: dict | None = None, workspace_id: str = "default") -> list[Citation]:
    filters = filters or {}
    redis = await redis_client()
    revision = await knowledge_revision(redis, workspace_id)
    cache_key = retrieval_key(workspace_id, revision, question, filters, top_k)
    cached = await cached_json(redis, cache_key)
    if cached:
        await redis.aclose()
        return await _citations_from_ids(cached["chunk_ids"], cached.get("scores", {}), workspace_id)
    candidate_k = max(top_k * 4, settings.retrieval_candidate_k)
    embedding_task = asyncio.create_task(embed_texts_async([question]))
    sparse_rows_task = asyncio.create_task(_load_sparse_rows(filters, workspace_id))
    query_vector = (await embedding_task)[0]
    dense_hits_task = asyncio.create_task(search_vectors(query_vector, candidate_k, filters, workspace_id))
    rows = await sparse_rows_task
    dense_hits = await dense_hits_task

    if not rows:
        await redis.aclose()
        return []

    records = {
        chunk.id: {
            "chunk": chunk,
            "document": document,
            "metadata": {**(document.doc_metadata or {}), **(chunk.chunk_metadata or {})},
        }
        for chunk, document in rows
    }
    corpus = [tokenize(chunk.text) for chunk, _ in rows]
    bm25 = BM25Okapi(corpus) if corpus else None
    sparse_scores = bm25.get_scores(tokenize(question)) if bm25 else []
    sparse_by_id = {rows[index][0].id: float(score) for index, score in enumerate(sparse_scores)}
    sparse_ranked = sorted(sparse_by_id, key=sparse_by_id.get, reverse=True)[:candidate_k]

    dense_scores: dict[str, float] = {}
    for hit in dense_hits:
        payload = hit.payload or {}
        chunk_id = str(payload.get("chunk_id") or hit.id)
        if chunk_id in records:
            dense_scores[chunk_id] = float(hit.score)

    candidate_ids = set(sparse_ranked) | set(dense_scores)
    sparse_scores_for_candidates = {chunk_id: sparse_by_id.get(chunk_id, 0.0) for chunk_id in candidate_ids}
    dense_normalized = _normalize(dense_scores)
    sparse_normalized = _normalize(sparse_scores_for_candidates)
    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(sorted(dense_scores, key=dense_scores.get, reverse=True), 1)}
    sparse_ranks = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ranked, 1)}

    merged: list[Citation] = []
    for chunk_id in candidate_ids:
        record = records[chunk_id]
        rrf = 0.0
        if chunk_id in dense_ranks:
            rrf += 1.0 / (60 + dense_ranks[chunk_id])
        if chunk_id in sparse_ranks:
            rrf += 1.0 / (60 + sparse_ranks[chunk_id])
        score = (
            0.50 * dense_normalized.get(chunk_id, 0.0)
            + 0.30 * sparse_normalized.get(chunk_id, 0.0)
            + 0.20 * min(1.0, rrf * 60)
        )
        chunk = record["chunk"]
        document = record["document"]
        merged.append(
            Citation(
                document_id=document.id,
                chunk_id=chunk_id,
                score=score,
                text=chunk.text,
                metadata=record["metadata"],
                source=document.source,
                content_type=document.content_type,
            )
        )

    query_terms = set(tokenize(question))
    reranked = []
    for citation in merged:
        text_terms = set(tokenize(citation.text))
        lexical_coverage = len(query_terms & text_terms) / max(1, len(query_terms))
        reranked.append(citation.model_copy(update={"score": 0.85 * citation.score + 0.15 * lexical_coverage}))

    by_document: dict[str, int] = defaultdict(int)
    selected: list[Citation] = []
    seen_text: set[str] = set()
    for citation in sorted(reranked, key=lambda item: item.score, reverse=True):
        fingerprint = re.sub(r"\W+", "", citation.text.lower())[:240]
        if fingerprint in seen_text:
            continue
        if by_document[citation.document_id] >= settings.max_chunks_per_document_in_response:
            continue
        by_document[citation.document_id] += 1
        seen_text.add(fingerprint)
        selected.append(citation)
        if len(selected) >= top_k:
            break
    await set_json(
        redis,
        cache_key,
        {"chunk_ids": [item.chunk_id for item in selected], "scores": {item.chunk_id: item.score for item in selected}},
        settings.retrieval_cache_ttl_seconds,
    )
    await redis.aclose()
    return selected
