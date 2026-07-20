import asyncio
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sqlalchemy import select

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


async def _load_sparse_rows(filters: dict):
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "indexed")
                .order_by(Chunk.created_at.desc())
                .limit(settings.sparse_candidate_limit)
            )
        ).all()
    return [
        (chunk, document)
        for chunk, document in rows
        if _matches_filters({**(document.doc_metadata or {}), **(chunk.chunk_metadata or {})}, filters)
    ]


async def hybrid_search(question: str, top_k: int, filters: dict | None = None) -> list[Citation]:
    filters = filters or {}
    candidate_k = max(top_k * 4, settings.retrieval_candidate_k)
    embedding_task = asyncio.create_task(embed_texts_async([question]))
    sparse_rows_task = asyncio.create_task(_load_sparse_rows(filters))
    query_vector = (await embedding_task)[0]
    dense_hits_task = asyncio.create_task(search_vectors(query_vector, candidate_k, filters))
    rows = await sparse_rows_task
    dense_hits = await dense_hits_task

    if not rows:
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
    return selected

    async with SessionLocal() as session:
        rows = (await session.execute(select(Chunk).limit(5000))).scalars().all()

    corpus = [tokenize(row.text) for row in rows]
    bm25 = BM25Okapi(corpus) if corpus else None
    sparse_scores = bm25.get_scores(tokenize(question)) if bm25 else []
    sparse_by_id = {rows[i].id: float(score) for i, score in enumerate(sparse_scores)}

    merged = {}
    for hit in dense_hits:
        payload = hit.payload or {}
        chunk_id = str(payload.get("chunk_id"))
        dense_score = float(hit.score)
        sparse_score = sparse_by_id.get(chunk_id, 0.0)
        merged[chunk_id] = Citation(
            document_id=str(payload.get("document_id")),
            chunk_id=chunk_id,
            score=0.75 * dense_score + 0.25 * sparse_score,
            text=str(payload.get("text", "")),
            metadata=payload.get("metadata", {}),
        )

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]
