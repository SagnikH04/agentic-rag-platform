import hashlib
import json
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy import select

from app.core.cache import bump_knowledge_revision
from app.core.config import settings
from app.core.db import Document, SessionLocal
from app.ingestion.state import set_job_state
from app.models.schemas import IngestRequest, IngestionMessage

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
    raw = f"{workspace_id}\0{payload.content_type.value}\0{identity}\0{metadata}".encode()
    return hashlib.sha256(raw).hexdigest()


async def _publish(redis: Redis, message: IngestionMessage) -> None:
    await redis.xadd(
        settings.redis_stream_ingest,
        {"payload": message.model_dump_json()},
        maxlen=10_000,
        approximate=True,
    )


async def enqueue_ingestion(payload: IngestRequest, workspace_id: str) -> EnqueueResult:
    fingerprint = source_fingerprint(workspace_id, payload)
    job_id = str(uuid.uuid4())

    async with SessionLocal() as session:
        existing = await session.scalar(
            select(Document).where(
                Document.workspace_id == workspace_id,
                Document.source_fingerprint == fingerprint,
            )
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
                    workspace_id=workspace_id,
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
        await session.commit()

    message = IngestionMessage(
        job_id=job_id,
        workspace_id=workspace_id,
        source=payload.source,
        content_type=payload.content_type,
        metadata=payload.metadata,
        force_reindex=payload.force_reindex,
    )
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await set_job_state(redis, job_id, status="queued", stage="accepted", progress=0, retries=0)
        await _publish(redis, message)
        await bump_knowledge_revision(redis, workspace_id)
    finally:
        await redis.aclose()

    return EnqueueResult(job_id, "queued", False)


async def publish_pending_outbox(limit: int = 100) -> int:
    return 0


async def outbox_dispatcher(stop_event) -> None:
    await stop_event.wait()


class NoopPublisher:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


publisher = NoopPublisher()
