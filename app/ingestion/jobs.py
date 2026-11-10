"""Durable ingestion acceptance and transactional-outbox publishing."""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aiokafka import AIOKafkaProducer
from redis.asyncio import Redis
from sqlalchemy import select

from app.core.cache import bump_knowledge_revision
from app.core.config import settings
from app.core.db import Document, OutboxEvent, SessionLocal
from app.ingestion.state import set_job_state
from app.models.schemas import IngestRequest, IngestionMessage

logger = logging.getLogger(__name__)
INGEST_TOPIC = "documents.ingest"
DLQ_TOPIC = "documents.ingest.dlq"


@dataclass(frozen=True)
class EnqueueResult:
    job_id: str
    status: str
    deduplicated: bool


def source_fingerprint(workspace_id: str, payload: IngestRequest) -> str:
    identity = payload.external_id or payload.source
    metadata = json.dumps(payload.metadata, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{workspace_id}\0{payload.content_type.value}\0{identity}\0{metadata}".encode()).hexdigest()


def _topic() -> str:
    return settings.kafka_ingest_topic if settings.queue_backend == "kafka" else settings.redis_stream_ingest


async def enqueue_ingestion(payload: IngestRequest, workspace_id: str) -> EnqueueResult:
    """Accept a job only after its document and outbox record commit together."""
    fingerprint = source_fingerprint(workspace_id, payload)
    job_id = str(uuid.uuid4())
    async with SessionLocal() as session:
        existing = await session.scalar(select(Document).where(Document.workspace_id == workspace_id, Document.source_fingerprint == fingerprint))
        if existing and not payload.force_reindex and existing.status not in {"failed", "deleted"}:
            return EnqueueResult(existing.id, existing.status, True)
        if existing:
            job_id, version = existing.id, existing.version + 1
            existing.version = version
            existing.external_id, existing.source, existing.content_type = payload.external_id, payload.source, payload.content_type.value
            existing.status, existing.pipeline_stage, existing.progress, existing.attempts = "queued", "accepted", 0, 0
            existing.error_message, existing.deleted_at, existing.doc_metadata = None, None, dict(payload.metadata)
        else:
            version = 1
            session.add(Document(id=job_id, workspace_id=workspace_id, external_id=payload.external_id, source=payload.source,
                source_fingerprint=fingerprint, content_type=payload.content_type.value, status="queued", pipeline_stage="accepted",
                progress=0, attempts=0, version=version, doc_metadata=dict(payload.metadata)))
        message = IngestionMessage(job_id=job_id, document_id=job_id, workspace_id=workspace_id, source_uri=payload.source,
            content_type=payload.content_type, metadata=payload.metadata, force_reindex=payload.force_reindex, document_version=version)
        session.add(OutboxEvent(id=str(uuid.uuid4()), topic=_topic(), message_key=job_id, event_payload=message.model_dump(mode="json")))
        await session.commit()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await set_job_state(redis, job_id, status="queued", stage="accepted", progress=0, retries=0)
        await bump_knowledge_revision(redis, workspace_id)
    finally:
        await redis.aclose()
    return EnqueueResult(job_id, "queued", False)


class Publisher:
    def __init__(self) -> None:
        self.producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        if settings.queue_backend == "kafka":
            self.producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers, acks="all", enable_idempotence=True)
            await self.producer.start()

    async def stop(self) -> None:
        if self.producer:
            await self.producer.stop()
            self.producer = None

    async def publish(self, topic: str, key: str, payload: dict) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode()
        if settings.queue_backend == "kafka":
            if not self.producer:
                raise RuntimeError("Kafka publisher is not started")
            await self.producer.send_and_wait(topic, encoded, key=key.encode())
            return
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            await redis.xadd(topic, {"payload": encoded.decode()}, maxlen=10_000, approximate=True)
        finally:
            await redis.aclose()


publisher = Publisher()


async def publish_pending_outbox(limit: int = 100) -> int:
    """Publish committed rows. Duplicate publish is safe because workers fence by version."""
    now = datetime.now(timezone.utc)
    async with SessionLocal() as session:
        rows = (await session.execute(select(OutboxEvent).where(OutboxEvent.published_at.is_(None), OutboxEvent.available_at <= now)
            .order_by(OutboxEvent.created_at).limit(limit).with_for_update(skip_locked=True))).scalars().all()
        for row in rows:
            try:
                await publisher.publish(row.topic, row.message_key or row.id, row.event_payload)
                row.published_at, row.last_error = now, None
            except Exception as exc:
                row.attempts += 1
                row.last_error = str(exc)[:2000]
                row.available_at = now + timedelta(seconds=min(settings.retry_base_seconds * (2 ** min(row.attempts, 8)), settings.retry_max_seconds))
                logger.warning("Outbox publish failed for %s: %s", row.id, type(exc).__name__)
        await session.commit()
    return len(rows)


async def outbox_dispatcher(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await publish_pending_outbox()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.outbox_poll_seconds)
        except TimeoutError:
            pass
