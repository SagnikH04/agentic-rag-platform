"""At-least-once Kafka worker with idempotent, version-fenced indexing."""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from redis.asyncio import Redis
from sqlalchemy import delete

from app.core.cache import bump_knowledge_revision
from app.core.config import settings
from app.core.db import Chunk, Document, OutboxEvent, SessionLocal
from app.core.telemetry import configure_telemetry, tracer
from app.ingestion.loaders import chunk_text, load_source
from app.ingestion.state import set_job_state
from app.models.schemas import IngestionMessage
from app.retrieval.embeddings import embed_texts_async
from app.retrieval.vector_store import delete_document_vectors, delete_obsolete_document_vectors, upsert_chunks

logger = logging.getLogger(__name__)


def _chunk_id(document_id: str, version: int, index: int, content_hash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{version}:{index}:{content_hash}"))


async def _current(message: IngestionMessage) -> bool:
    async with SessionLocal() as session:
        doc = await session.get(Document, message.document_id)
        return bool(doc and doc.workspace_id == message.workspace_id and doc.status != "deleted" and doc.version == message.document_version)


async def _state(redis: Redis, message: IngestionMessage, *, status: str, stage: str, progress: int, error: str | None = None) -> None:
    await set_job_state(redis, message.job_id, status=status, stage=stage, progress=progress, retries=message.attempt, error=error)
    async with SessionLocal() as session:
        doc = await session.get(Document, message.document_id)
        if doc and doc.version == message.document_version and doc.status != "deleted":
            doc.status, doc.pipeline_stage, doc.progress, doc.attempts = status, stage, progress, message.attempt
            if error:
                doc.error_message = error[:2000]
            elif status not in {"failed", "retrying"}:
                doc.error_message = None
            await session.commit()


def _chunks(loaded) -> list[tuple[str, dict]]:
    if loaded.segments:
        return [(chunk, location) for text, location in loaded.segments for chunk in chunk_text(text)]
    return [(chunk, {}) for chunk in chunk_text(loaded.text)]


async def _replace_document_index(message: IngestionMessage, chunks: list[tuple[str, dict]], vectors: list[list[float]], content_hash: str, loader_metadata: dict) -> bool:
    if not await _current(message):
        return False
    points, db_chunks = [], []
    for index, ((chunk, location), vector) in enumerate(zip(chunks, vectors, strict=True)):
        chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
        chunk_id = _chunk_id(message.document_id, message.document_version, index, chunk_hash)
        metadata = {**message.metadata, **location, "content_type": message.content_type.value, "chunk_index": index, "document_version": message.document_version}
        db_chunks.append(Chunk(id=chunk_id, workspace_id=message.workspace_id, document_id=message.document_id, chunk_index=index,
            content_hash=chunk_hash, text=chunk, chunk_metadata=metadata))
        points.append({"id": chunk_id, "vector": vector, "payload": {"workspace_id": message.workspace_id, "document_id": message.document_id,
            "chunk_id": chunk_id, "content_type": message.content_type.value, "document_version": message.document_version, "text": chunk, "metadata": metadata}})
    # Replayed events replace only their own generation. Existing valid
    # generations remain query-safe until the metadata switch below succeeds.
    await delete_document_vectors(message.document_id, message.workspace_id, message.document_version)
    await upsert_chunks(points)
    async with SessionLocal() as session:
        doc = await session.get(Document, message.document_id, with_for_update=True)
        if not doc or doc.status == "deleted" or doc.version != message.document_version:
            await delete_document_vectors(message.document_id, message.workspace_id, message.document_version)
            return False
        await session.execute(delete(Chunk).where(Chunk.document_id == message.document_id, Chunk.workspace_id == message.workspace_id))
        session.add_all(db_chunks)
        doc.status, doc.pipeline_stage, doc.progress, doc.attempts = "indexed", "complete", 100, message.attempt
        doc.content_hash, doc.error_message = content_hash, None
        doc.doc_metadata = {**message.metadata, "ingestion": loader_metadata, "document_version": message.document_version}
        await session.commit()
    await delete_obsolete_document_vectors(message.document_id, message.workspace_id, message.document_version)
    return True


async def process_message(raw_message: dict, redis: Redis) -> bool:
    message = IngestionMessage.model_validate(raw_message)
    if not await _current(message):
        return True
    with tracer.start_as_current_span("ingestion.pipeline") as span:
        span.set_attribute("document.id", message.document_id)
        span.set_attribute("workspace.id_hash", hashlib.sha256(message.workspace_id.encode()).hexdigest())
        await _state(redis, message, status="running", stage="loading", progress=10)
        loaded = await load_source(message.source_uri, message.content_type.value)
        if not loaded.text.strip():
            raise ValueError("source produced no extractable text")
        content_hash = hashlib.sha256(loaded.text.encode()).hexdigest()
        parts = _chunks(loaded)
        if not parts:
            raise ValueError("no chunks were produced from source")
        if len(parts) > settings.max_chunks_per_document:
            raise ValueError("document exceeds MAX_CHUNKS_PER_DOCUMENT")
        await _state(redis, message, status="running", stage="embedding", progress=55)
        vectors = await embed_texts_async([chunk for chunk, _ in parts])
        await _state(redis, message, status="running", stage="indexing", progress=80)
        if await _replace_document_index(message, parts, vectors, content_hash, loaded.metadata):
            await set_job_state(redis, message.job_id, status="indexed", stage="complete", progress=100, retries=message.attempt)
            await bump_knowledge_revision(redis, message.workspace_id)
    return True


async def _publish(producer: AIOKafkaProducer, topic: str, message: IngestionMessage) -> None:
    await producer.send_and_wait(topic, message.model_dump_json().encode(), key=message.document_id.encode())


async def _schedule_retry(message: IngestionMessage, error: Exception) -> None:
    """Persist delay in the outbox; never sleep in a Kafka consumer loop."""
    delay = min(settings.retry_base_seconds * (2 ** message.attempt), settings.retry_max_seconds)
    retry = message.model_copy(update={"attempt": message.attempt + 1})
    async with SessionLocal() as session:
        session.add(OutboxEvent(
            id=str(uuid.uuid4()), topic=settings.kafka_retry_topic, message_key=retry.document_id,
            event_payload=retry.model_dump(mode="json"),
            available_at=datetime.now(timezone.utc) + timedelta(seconds=delay),
        ))
        await session.commit()


async def _consume_kafka() -> None:
    consumer = AIOKafkaConsumer(settings.kafka_ingest_topic, settings.kafka_retry_topic, bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group, enable_auto_commit=False, auto_offset_reset="earliest",
        max_poll_interval_ms=settings.worker_max_poll_interval_ms, client_id=f"{settings.app_name}-{os.getpid()}")
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers, acks="all", enable_idempotence=True)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await consumer.start(); await producer.start()
    try:
        async for record in consumer:
            try:
                message = IngestionMessage.model_validate_json(record.value)
                await process_message(message.model_dump(mode="json"), redis)
            except Exception as exc:
                try:
                    failed = IngestionMessage.model_validate_json(record.value)
                    if failed.attempt < settings.max_retries:
                        retry = failed.model_copy(update={"attempt": failed.attempt + 1})
                        await _state(redis, retry, status="retrying", stage="retry_scheduled", progress=0, error=str(exc))
                        await _schedule_retry(failed, exc)
                    else:
                        await _state(redis, failed, status="failed", stage="dead_lettered", progress=0, error=str(exc))
                        await _publish(producer, settings.kafka_dlq_topic, failed)
                except Exception:
                    logger.exception("Unable to make failed Kafka event durable")
                    continue  # Do not commit: Kafka will replay the original event.
            await consumer.commit()
    finally:
        await consumer.stop(); await producer.stop(); await redis.aclose()


async def main() -> None:
    configure_telemetry()
    if settings.queue_backend != "kafka":
        raise RuntimeError("Redis Streams is a legacy backend; set QUEUE_BACKEND=kafka")
    await _consume_kafka()


if __name__ == "__main__":
    asyncio.run(main())
