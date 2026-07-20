import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aiokafka import AIOKafkaProducer
from redis.asyncio import Redis
from sqlalchemy import select

from app.core.config import settings
from app.core.db import Document, OutboxEvent, SessionLocal
from app.ingestion.state import set_job_state
from app.models.schemas import IngestRequest, IngestionMessage

INGEST_TOPIC = "documents.ingest"
DLQ_TOPIC = "documents.ingest.dlq"


class KafkaPublisher:
    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._producer is not None:
            return
        async with self._lock:
            if self._producer is not None:
                return
            producer = AIOKafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                acks="all",
                enable_idempotence=True,
            )
            await producer.start()
            self._producer = producer

    async def publish(self, topic: str, payload: dict, key: str | None = None) -> None:
        await self.start()
        if self._producer is None:
            raise RuntimeError("Kafka producer was not initialized")
        await self._producer.send_and_wait(
            topic,
            json.dumps(payload, separators=(",", ":"), default=str).encode(),
            key=key.encode() if key else None,
        )

    async def stop(self) -> None:
        async with self._lock:
            if self._producer is not None:
                await self._producer.stop()
                self._producer = None


publisher = KafkaPublisher()


@dataclass(frozen=True)
class EnqueueResult:
    job_id: str
    status: str
    deduplicated: bool


def source_fingerprint(payload: IngestRequest) -> str:
    identity = payload.external_id or payload.source
    metadata = json.dumps(payload.metadata, sort_keys=True, separators=(",", ":"), default=str)
    raw = f"{payload.content_type.value}\0{identity}\0{metadata}".encode()
    return hashlib.sha256(raw).hexdigest()


async def enqueue_ingestion(payload: IngestRequest) -> EnqueueResult:
    fingerprint = source_fingerprint(payload)
    job_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    message: IngestionMessage

    async with SessionLocal() as session:
        existing = await session.scalar(
            select(Document).where(Document.source_fingerprint == fingerprint)
        )
        if existing and not payload.force_reindex and existing.status not in {"failed", "deleted"}:
            return EnqueueResult(existing.id, existing.status, True)

        if existing:
            job_id = existing.id
            existing.external_id = payload.external_id
            existing.source = payload.source
            existing.content_type = payload.content_type.value
            existing.status = "queued"
            existing.pipeline_stage = "accepted"
            existing.progress = 0
            existing.attempts = 0
            existing.error_message = None
            existing.doc_metadata = dict(payload.metadata)
        else:
            session.add(
                Document(
                    id=job_id,
                    external_id=payload.external_id,
                    source=payload.source,
                    source_fingerprint=fingerprint,
                    content_type=payload.content_type.value,
                    status="queued",
                    pipeline_stage="accepted",
                    progress=0,
                    attempts=0,
                    doc_metadata=dict(payload.metadata),
                )
            )

        message = IngestionMessage(
            job_id=job_id,
            source=payload.source,
            content_type=payload.content_type,
            metadata=payload.metadata,
            force_reindex=payload.force_reindex,
        )
        session.add(
            OutboxEvent(
                id=event_id,
                topic=INGEST_TOPIC,
                message_key=job_id,
                event_payload=message.model_dump(mode="json"),
            )
        )
        await session.commit()

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await set_job_state(
            redis,
            job_id,
            status="queued",
            stage="accepted",
            progress=0,
            retries=0,
        )
    finally:
        await redis.aclose()

    try:
        await publish_outbox_event(event_id)
    except Exception:
        pass

    return EnqueueResult(job_id, "queued", False)


async def publish_outbox_event(event_id: str) -> None:
    async with SessionLocal() as session:
        event = await session.get(OutboxEvent, event_id)
        if event is None or event.published_at is not None:
            return
        try:
            await publisher.publish(event.topic, event.event_payload, event.message_key)
        except Exception as exc:
            event.attempts += 1
            delay = min(
                settings.retry_base_seconds * (2 ** min(event.attempts, 8)),
                settings.retry_max_seconds,
            )
            event.available_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            event.last_error = str(exc)[:2000]
            await session.commit()
            raise
        event.published_at = datetime.now(timezone.utc)
        event.last_error = None
        await session.commit()


async def publish_pending_outbox(limit: int = 100) -> int:
    now = datetime.now(timezone.utc)
    async with SessionLocal() as session:
        event_ids = (
            await session.execute(
                select(OutboxEvent.id)
                .where(
                    OutboxEvent.published_at.is_(None),
                    OutboxEvent.available_at <= now,
                )
                .order_by(OutboxEvent.created_at)
                .limit(limit)
            )
        ).scalars().all()

    published = 0
    for event_id in event_ids:
        try:
            await publish_outbox_event(event_id)
            published += 1
        except Exception:
            continue
    return published


async def outbox_dispatcher(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await publish_pending_outbox()
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.outbox_poll_seconds)
        except TimeoutError:
            continue
