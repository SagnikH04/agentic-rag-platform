import asyncio
import hashlib
import json
import logging
import time
import uuid

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from redis.asyncio import Redis
from sqlalchemy import delete

from app.core.config import settings
from app.core.db import Chunk, Document, SessionLocal
from app.core.telemetry import configure_telemetry, tracer
from app.ingestion.jobs import DLQ_TOPIC, INGEST_TOPIC
from app.ingestion.loaders import chunk_text, load_source
from app.ingestion.state import set_job_state
from app.models.schemas import IngestionMessage
from app.retrieval.embeddings import embed_texts_async
from app.retrieval.vector_store import delete_document_vectors, upsert_chunks

logger = logging.getLogger(__name__)
RETRY_ZSET = "ingestion:retry:schedule"


async def _update_document(
    job_id: str,
    *,
    status: str,
    stage: str,
    progress: int,
    attempts: int | None = None,
    error: str | None = None,
) -> None:
    async with SessionLocal() as session:
        document = await session.get(Document, job_id)
        if document is None:
            return
        document.status = status
        document.pipeline_stage = stage
        document.progress = progress
        if attempts is not None:
            document.attempts = attempts
        if error is not None:
            document.error_message = error[:2000]
        elif status not in {"failed", "retrying"}:
            document.error_message = None
        await session.commit()


async def _set_pipeline_state(
    redis: Redis,
    job_id: str,
    *,
    status: str,
    stage: str,
    progress: int,
    attempts: int | None = None,
    error: str | None = None,
) -> None:
    await set_job_state(
        redis,
        job_id,
        status=status,
        stage=stage,
        progress=progress,
        retries=attempts,
        error=error,
    )
    await _update_document(
        job_id,
        status=status,
        stage=stage,
        progress=progress,
        attempts=attempts,
        error=error,
    )


def _chunk_id(document_id: str, index: int, content_hash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}:{content_hash}"))


async def _replace_document_index(
    message: IngestionMessage,
    chunks: list[str],
    vectors: list[list[float]],
    content_hash: str,
    loader_metadata: dict,
) -> None:
    db_chunks: list[Chunk] = []
    points: list[dict] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
        chunk_id = _chunk_id(message.job_id, index, chunk_hash)
        metadata = {
            **message.metadata,
            "content_type": message.content_type.value,
            "chunk_index": index,
        }
        db_chunks.append(
            Chunk(
                id=chunk_id,
                document_id=message.job_id,
                chunk_index=index,
                content_hash=chunk_hash,
                text=chunk,
                chunk_metadata=metadata,
            )
        )
        points.append(
            {
                "id": chunk_id,
                "vector": vector,
                "payload": {
                    "document_id": message.job_id,
                    "chunk_id": chunk_id,
                    "content_type": message.content_type.value,
                    "source": message.source,
                    "text": chunk,
                    "metadata": metadata,
                },
            }
        )

    await delete_document_vectors(message.job_id)
    await upsert_chunks(points)

    async with SessionLocal() as session:
        await session.execute(delete(Chunk).where(Chunk.document_id == message.job_id))
        session.add_all(db_chunks)
        document = await session.get(Document, message.job_id)
        if document is None:
            raise ValueError("document disappeared during indexing")
        document.status = "indexed"
        document.pipeline_stage = "complete"
        document.progress = 100
        document.attempts = message.attempt
        document.content_hash = content_hash
        document.error_message = None
        document.doc_metadata = {
            **message.metadata,
            "ingestion": loader_metadata,
        }
        await session.commit()


async def _schedule_retry(redis: Redis, message: IngestionMessage, error: Exception) -> None:
    next_attempt = message.attempt + 1
    retry_message = message.model_copy(update={"attempt": next_attempt})
    delay = min(settings.retry_base_seconds * (2 ** message.attempt), settings.retry_max_seconds)
    await redis.zadd(
        RETRY_ZSET,
        {retry_message.model_dump_json(): time.time() + delay},
    )
    await _set_pipeline_state(
        redis,
        message.job_id,
        status="retrying",
        stage="retry_scheduled",
        progress=0,
        attempts=next_attempt,
        error=str(error),
    )


async def _send_dead_letter(producer: AIOKafkaProducer, message: IngestionMessage, error: Exception) -> None:
    payload = {
        **message.model_dump(mode="json"),
        "error": str(error)[:2000],
        "error_type": type(error).__name__,
    }
    await producer.send_and_wait(
        DLQ_TOPIC,
        json.dumps(payload, separators=(",", ":"), default=str).encode(),
        key=message.job_id.encode(),
    )


async def process_message(raw_message: dict, redis: Redis, producer: AIOKafkaProducer) -> None:
    message = IngestionMessage.model_validate(raw_message)
    try:
        with tracer.start_as_current_span("ingestion.pipeline") as span:
            span.set_attribute("document.id", message.job_id)
            span.set_attribute("ingestion.attempt", message.attempt)
            async with SessionLocal() as session:
                document = await session.get(Document, message.job_id)
            if document is None or document.status == "deleted":
                return

            await _set_pipeline_state(
                redis,
                message.job_id,
                status="running",
                stage="loading",
                progress=10,
                attempts=message.attempt,
            )
            loaded = await load_source(message.source, message.content_type.value)
            if not loaded.text.strip():
                raise ValueError("source produced no extractable text")
            content_hash = hashlib.sha256(loaded.text.encode()).hexdigest()
            if document.status == "indexed" and document.content_hash == content_hash and not message.force_reindex:
                await set_job_state(
                    redis,
                    message.job_id,
                    status="indexed",
                    stage="already_indexed",
                    progress=100,
                    retries=message.attempt,
                )
                return

            await _set_pipeline_state(
                redis,
                message.job_id,
                status="running",
                stage="chunking",
                progress=30,
                attempts=message.attempt,
            )
            chunks = chunk_text(loaded.text)
            if not chunks:
                raise ValueError("no chunks were produced from source")
            if len(chunks) > settings.max_chunks_per_document:
                raise ValueError("document exceeds MAX_CHUNKS_PER_DOCUMENT")

            await _set_pipeline_state(
                redis,
                message.job_id,
                status="running",
                stage="embedding",
                progress=55,
                attempts=message.attempt,
            )
            vectors = await embed_texts_async(chunks)

            await _set_pipeline_state(
                redis,
                message.job_id,
                status="running",
                stage="indexing",
                progress=80,
                attempts=message.attempt,
            )
            await _replace_document_index(message, chunks, vectors, content_hash, loaded.metadata)
            await set_job_state(
                redis,
                message.job_id,
                status="indexed",
                stage="complete",
                progress=100,
                retries=message.attempt,
            )
    except Exception as exc:
        logger.exception("Ingestion failed for job %s", message.job_id)
        if message.attempt < settings.max_retries:
            await _schedule_retry(redis, message, exc)
            return
        await _send_dead_letter(producer, message, exc)
        await _set_pipeline_state(
            redis,
            message.job_id,
            status="failed",
            stage="dead_lettered",
            progress=0,
            attempts=message.attempt,
            error=str(exc),
        )


async def retry_dispatcher(redis: Redis, producer: AIOKafkaProducer) -> None:
    while True:
        try:
            due = await redis.zrangebyscore(RETRY_ZSET, "-inf", time.time(), start=0, num=100)
            for raw in due:
                payload = json.loads(raw)
                job_id = str(payload["job_id"])
                await producer.send_and_wait(
                    INGEST_TOPIC,
                    json.dumps(payload, separators=(",", ":"), default=str).encode(),
                    key=job_id.encode(),
                )
                await redis.zrem(RETRY_ZSET, raw)
                await set_job_state(
                    redis,
                    job_id,
                    status="queued",
                    stage="retry_dispatched",
                    progress=0,
                    retries=int(payload.get("attempt", 0)),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Retry dispatcher failed")
        await asyncio.sleep(settings.retry_poll_seconds)


async def main():
    configure_telemetry()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    consumer = AIOKafkaConsumer(
        INGEST_TOPIC,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="ingestion-workers",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_interval_ms=settings.worker_max_poll_interval_ms,
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        acks="all",
        enable_idempotence=True,
    )

    await consumer.start()
    await producer.start()
    dispatcher = asyncio.create_task(retry_dispatcher(redis, producer))
    try:
        async for record in consumer:
            try:
                await process_message(json.loads(record.value.decode()), redis, producer)
            except Exception:
                logger.exception("Unhandled ingestion record failure; record will be retried")
                continue
            await consumer.commit()
    finally:
        dispatcher.cancel()
        await asyncio.gather(dispatcher, return_exceptions=True)
        await consumer.stop()
        await producer.stop()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
