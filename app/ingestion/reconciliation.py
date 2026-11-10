"""Repair divergence between PostgreSQL's canonical chunks and Qdrant."""

import asyncio
import logging
import uuid

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import select

from app.core.config import settings
from app.core.db import Chunk, Document, OutboxEvent, SessionLocal
from app.models.schemas import ContentType, IngestionMessage
from app.retrieval.vector_store import COLLECTION, client as qdrant_client, delete_chunk_vectors

logger = logging.getLogger(__name__)


async def _schedule_rebuild(document: Document) -> None:
    metadata = {key: value for key, value in (document.doc_metadata or {}).items() if key != "ingestion"}
    message = IngestionMessage(
        job_id=document.id,
        document_id=document.id,
        workspace_id=document.workspace_id,
        source_uri=document.source,
        content_type=ContentType(document.content_type),
        metadata=metadata,
        document_version=document.version,
    )
    async with SessionLocal() as session:
        session.add(OutboxEvent(
            id=str(uuid.uuid4()), topic=settings.kafka_ingest_topic, message_key=document.id,
            event_payload=message.model_dump(mode="json"),
        ))
        await session.commit()


async def reconcile_once(limit: int = 100) -> dict[str, int]:
    """Delete orphan Qdrant points and enqueue rebuilds for missing vectors.

    PostgreSQL is the source of truth. A missing vector cannot be recreated
    from a chunk without preserving the embedding model's output, so the safe
    repair is a version-fenced reindex through the ordinary outbox path.
    """
    repaired = orphaned = 0
    async with SessionLocal() as session:
        documents = (await session.execute(
            select(Document).where(Document.status == "indexed", Document.active_version > 0).limit(limit)
        )).scalars().all()
        for document in documents:
            chunk_ids = set((await session.execute(select(Chunk.id).where(
                Chunk.document_id == document.id,
                Chunk.workspace_id == document.workspace_id,
                Chunk.document_version == document.active_version,
            ))).scalars().all())
            points, _ = await qdrant_client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document.id)),
                    FieldCondition(key="workspace_id", match=MatchValue(value=document.workspace_id)),
                ]),
                limit=10_000,
                with_payload=["chunk_id", "document_version"],
            )
            point_ids = {str((point.payload or {}).get("chunk_id") or point.id) for point in points}
            extras = point_ids - chunk_ids
            if extras:
                await delete_chunk_vectors(list(extras))
                orphaned += len(extras)
            if chunk_ids - point_ids:
                await _schedule_rebuild(document)
                repaired += 1
    return {"rebuilds_enqueued": repaired, "orphan_vectors_deleted": orphaned}


async def reconciliation_worker(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await reconcile_once()
        except Exception:
            logger.exception("Qdrant reconciliation pass failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.reconciliation_poll_seconds)
        except TimeoutError:
            pass
