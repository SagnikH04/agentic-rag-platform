import asyncio
import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy import delete, text

from app.agents.orchestrator import AgentOrchestrator
from app.core.config import settings
from app.core.db import Chunk, Document, SessionLocal, engine
from app.ingestion.jobs import enqueue_ingestion, publisher
from app.ingestion.state import job_key, set_job_state
from app.models.schemas import (
    DeleteResponse,
    DocumentStatusResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.retrieval.vector_store import client as qdrant_client
from app.retrieval.vector_store import delete_document_vectors

router = APIRouter()
orchestrator = AgentOrchestrator()


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(payload: IngestRequest):
    result = await enqueue_ingestion(payload)
    return IngestResponse(
        job_id=result.job_id,
        status=result.status,
        deduplicated=result.deduplicated,
    )


@router.get("/documents/{job_id}/status", response_model=DocumentStatusResponse)
async def document_status(job_id: str):
    async with SessionLocal() as session:
        document = await session.get(Document, job_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document job was not found")

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        state = await redis.hgetall(job_key(job_id))
    finally:
        await redis.aclose()
    return DocumentStatusResponse(
        job_id=job_id,
        status=state.get("status", document.status),
        stage=state.get("stage", document.pipeline_stage),
        progress=int(state.get("progress", document.progress)),
        attempts=int(state.get("retries", document.attempts)),
        error=state.get("error", document.error_message),
        metadata=document.doc_metadata or {},
    )


@router.delete("/documents/{job_id}", response_model=DeleteResponse)
async def delete_document(job_id: str):
    async with SessionLocal() as session:
        document = await session.get(Document, job_id)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document was not found")
        await delete_document_vectors(job_id)
        await session.execute(delete(Chunk).where(Chunk.document_id == job_id))
        document.status = "deleted"
        document.pipeline_stage = "deleted"
        document.progress = 100
        await session.commit()

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await set_job_state(redis, job_id, status="deleted", stage="deleted", progress=100)
    finally:
        await redis.aclose()
    return DeleteResponse(document_id=job_id, status="deleted")


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    return await orchestrator.answer(payload.question, payload.session_id, payload.filters, payload.top_k)


@router.post("/query/stream")
async def query_stream(payload: QueryRequest):
    async def events():
        try:
            response = await orchestrator.answer(payload.question, payload.session_id, payload.filters, payload.top_k)
            citations = [citation.model_dump() for citation in response.citations]
            yield f"event: citations\ndata: {json.dumps(citations, default=str)}\n\n"
            for token in response.answer.split(" "):
                yield f"event: token\ndata: {json.dumps(token + ' ')}\n\n"
                await asyncio.sleep(0)
            yield f"event: complete\ndata: {response.model_dump_json()}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/health/live")
async def live_health():
    return {"status": "ok"}


@router.get("/health/ready")
@router.get("/health")
async def readiness_health():
    checks: dict[str, str] = {}

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception:
        checks["postgres"] = "unavailable"

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"
    finally:
        await redis.aclose()

    try:
        await qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception:
        checks["qdrant"] = "unavailable"

    try:
        await publisher.start()
        checks["kafka"] = "ok"
    except Exception:
        checks["kafka"] = "unavailable"

    if any(result != "ok" for result in checks.values()):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "degraded", "checks": checks})
    return {"status": "ok", "checks": checks}
