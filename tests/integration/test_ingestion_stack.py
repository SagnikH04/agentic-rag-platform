"""Black-box integration coverage for the durable ingestion path.

These tests run only in the disposable Docker Compose profile.  They use the
network API plus real Kafka/PostgreSQL/Redis/Qdrant clients to validate the
contracts that unit tests cannot cover.
"""

import asyncio
import hashlib
import json
import os
import time
import uuid

import httpx
import pytest
import pytest_asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from qdrant_client.models import FieldCondition, Filter, MatchValue
from redis.asyncio import Redis
from sqlalchemy import func, select

from app.core.config import settings
from app.core.db import Chunk, Document, SessionLocal
from app.models.schemas import IngestionMessage
from app.ingestion.reconciliation import reconcile_once
from app.retrieval.vector_store import COLLECTION, client as qdrant_client, delete_chunk_vectors


pytestmark = pytest.mark.integration


def _require_integration_stack() -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 and use docker-compose.integration.yml")


async def _wait_for_status(client: httpx.AsyncClient, job_id: str, expected: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/documents/{job_id}/status")
        response.raise_for_status()
        last = response.json()
        if last["status"] == expected:
            return last
        if last["status"] == "failed":
            raise AssertionError(f"ingestion failed: {last}")
        await asyncio.sleep(1)
    raise AssertionError(f"timed out waiting for {expected}: {last}")


@pytest_asyncio.fixture
async def api_client():
    _require_integration_stack()
    async with httpx.AsyncClient(base_url=os.getenv("API_BASE_URL", "http://api:8000"), timeout=15.0) as client:
        deadline = time.monotonic() + 90
        while True:
            try:
                response = await client.get("/health/ready")
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError("API never became ready")
            await asyncio.sleep(1)
        yield client


async def _document_and_chunk_count(job_id: str) -> tuple[Document, int]:
    async with SessionLocal() as session:
        document = await session.get(Document, job_id)
        assert document is not None
        count = await session.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == job_id))
        return document, int(count or 0)


async def _wait_for_index_runs(job_id: str, minimum: int, timeout: float = 180.0) -> Document:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        document, _ = await _document_and_chunk_count(job_id)
        if document.index_runs >= minimum:
            return document
        await asyncio.sleep(1)
    raise AssertionError(f"worker did not process replay for {job_id}")


@pytest.mark.asyncio
async def test_http_outbox_kafka_worker_indexes_and_deletes(api_client: httpx.AsyncClient):
    marker = f"integration-evidence-{uuid.uuid4()}"
    accepted = await api_client.post(
        "/v1/documents/ingest",
        json={"source": f"{marker} is durable end-to-end test content.", "content_type": "text", "metadata": {"suite": "integration"}},
    )
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]

    indexed = await _wait_for_status(api_client, job_id, "indexed")
    assert indexed["progress"] == 100
    document, chunk_count = await _document_and_chunk_count(job_id)
    assert chunk_count >= 1

    points, _ = await qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=job_id))]),
        limit=100,
        with_payload=True,
    )
    assert len(points) == chunk_count

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        assert (await redis.hgetall(f"job:{job_id}"))["status"] == "indexed"
    finally:
        await redis.aclose()

    deleted = await api_client.delete(f"/v1/documents/{job_id}")
    assert deleted.status_code == 200, deleted.text
    assert (await api_client.get(f"/v1/documents/{job_id}/status")).json()["status"] == "deleted"
    _, chunk_count = await _document_and_chunk_count(job_id)
    assert chunk_count == 0
    points, _ = await qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=job_id))]),
        limit=1,
    )
    assert points == []


@pytest.mark.asyncio
async def test_duplicate_delivery_is_idempotent(api_client: httpx.AsyncClient):
    accepted = await api_client.post(
        "/v1/documents/ingest",
        json={"source": f"duplicate-delivery-{uuid.uuid4()}", "content_type": "text"},
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    await _wait_for_status(api_client, job_id, "indexed")
    document, initial_count = await _document_and_chunk_count(job_id)
    initial_runs = document.index_runs

    event = IngestionMessage(
        job_id=job_id,
        document_id=job_id,
        workspace_id=document.workspace_id,
        source_uri=document.source,
        content_type=document.content_type,
        document_version=document.version,
    )
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    await producer.start()
    try:
        await producer.send_and_wait(settings.kafka_ingest_topic, event.model_dump_json().encode(), key=job_id.encode())
    finally:
        await producer.stop()
    await _wait_for_index_runs(job_id, initial_runs + 1)
    _, replay_count = await _document_and_chunk_count(job_id)
    assert replay_count == initial_count


@pytest.mark.asyncio
async def test_force_reindex_keeps_an_active_generation_searchable(api_client: httpx.AsyncClient):
    source = f"active-generation-{uuid.uuid4()}"
    accepted = await api_client.post("/v1/documents/ingest", json={"source": source, "content_type": "text"})
    job_id = accepted.json()["job_id"]
    await _wait_for_status(api_client, job_id, "indexed")
    initial, initial_chunks = await _document_and_chunk_count(job_id)

    replacement = await api_client.post("/v1/documents/ingest", json={"source": source, "content_type": "text", "force_reindex": True})
    assert replacement.status_code == 202
    during_reindex, visible_chunks = await _document_and_chunk_count(job_id)
    assert during_reindex.status == "indexed"
    assert during_reindex.active_version >= initial.active_version
    assert visible_chunks >= 1

    await _wait_for_index_runs(job_id, initial.index_runs + 1)
    completed, completed_chunks = await _document_and_chunk_count(job_id)
    assert completed.active_version == completed.version
    assert completed_chunks == initial_chunks


@pytest.mark.asyncio
async def test_reconciler_rebuilds_a_missing_qdrant_vector(api_client: httpx.AsyncClient):
    accepted = await api_client.post(
        "/v1/documents/ingest", json={"source": f"reconciliation-{uuid.uuid4()}", "content_type": "text"},
    )
    job_id = accepted.json()["job_id"]
    await _wait_for_status(api_client, job_id, "indexed")
    document, _ = await _document_and_chunk_count(job_id)
    initial_runs = document.index_runs
    points, _ = await qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=job_id))]),
        limit=1,
    )
    assert points
    await delete_chunk_vectors([str(points[0].id)])

    result = await reconcile_once()
    assert result["rebuilds_enqueued"] >= 1
    await _wait_for_index_runs(job_id, initial_runs + 1)
    repaired, _ = await qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=job_id))]),
        limit=1,
    )
    assert repaired


@pytest.mark.asyncio
async def test_malformed_kafka_record_is_quarantined_to_dlq():
    _require_integration_stack()
    raw = f"not-json-{uuid.uuid4()}".encode()
    expected_hash = hashlib.sha256(raw).hexdigest()
    consumer = AIOKafkaConsumer(
        settings.kafka_dlq_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=f"integration-dlq-{uuid.uuid4()}",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    await consumer.start()
    await producer.start()
    try:
        await producer.send_and_wait(settings.kafka_ingest_topic, raw, key=expected_hash.encode())
        async with asyncio.timeout(45):
            async for record in consumer:
                payload = json.loads(record.value)
                if payload.get("raw_event_sha256") == expected_hash:
                    assert payload["error"] == "malformed_ingestion_event"
                    break
            else:  # pragma: no cover - async Kafka consumer never naturally exhausts
                raise AssertionError("DLQ consumer stopped before receiving the record")
    finally:
        await producer.stop()
        await consumer.stop()
