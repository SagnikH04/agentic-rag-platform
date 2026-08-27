import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy import delete, select, text

from app.agents.orchestrator import AgentOrchestrator
from app.core.config import settings
from app.core.db import Chunk, Document, SessionLocal, engine
from app.core.cache import bump_knowledge_revision
from app.core.workspace import workspace_from_request
from app.ingestion.jobs import enqueue_ingestion
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


UPLOAD_ROOT = Path("/data/uploads")


def _v1(path: str) -> str:
    return f"/v1{path}"


@router.post(_v1("/documents/ingest"), response_model=IngestResponse)
@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(payload: IngestRequest, request: Request):
    result = await enqueue_ingestion(payload, workspace_from_request(request))
    return IngestResponse(
        job_id=result.job_id,
        status=result.status,
        deduplicated=result.deduplicated,
    )


@router.post(_v1("/documents/upload"), response_model=IngestResponse)
async def upload_document(request: Request, file: UploadFile = File(...), force_reindex: bool = False):
    workspace_id = workspace_from_request(request)
    suffix = Path(file.filename or "").suffix.lower()
    kind = "pdf" if suffix == ".pdf" else "office" if suffix in {".docx", ".pptx", ".xlsx", ".xlsm"} else "image" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff"} else "text"
    target_dir = UPLOAD_ROOT / workspace_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4()}{suffix or '.txt'}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    payload = IngestRequest(
        source=str(target),
        content_type=kind,
        metadata={"file_name": file.filename or target.name, "upload": True},
        force_reindex=force_reindex,
    )
    result = await enqueue_ingestion(payload, workspace_id)
    return IngestResponse(job_id=result.job_id, status=result.status, deduplicated=result.deduplicated)


@router.get(_v1("/documents"), response_model=list[DocumentStatusResponse])
async def list_documents(request: Request):
    workspace_id = workspace_from_request(request)
    async with SessionLocal() as session:
        documents = (
            await session.execute(
                select(Document).where(Document.workspace_id == workspace_id, Document.status != "deleted").order_by(Document.created_at.desc())
            )
        ).scalars().all()
    return [
        DocumentStatusResponse(
            job_id=document.id,
            status=document.status,
            stage=document.pipeline_stage,
            progress=document.progress,
            attempts=document.attempts,
            error=document.error_message,
            metadata={**(document.doc_metadata or {}), "source": document.source, "content_type": document.content_type},
        )
        for document in documents
    ]


@router.get(_v1("/documents/{job_id}/status"), response_model=DocumentStatusResponse)
@router.get("/documents/{job_id}/status", response_model=DocumentStatusResponse)
async def document_status(job_id: str, request: Request):
    workspace_id = workspace_from_request(request)
    async with SessionLocal() as session:
        document = await session.get(Document, job_id)
    if document is None or document.workspace_id != workspace_id:
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


@router.delete(_v1("/documents/{job_id}"), response_model=DeleteResponse)
@router.delete("/documents/{job_id}", response_model=DeleteResponse)
async def delete_document(job_id: str, request: Request):
    workspace_id = workspace_from_request(request)
    async with SessionLocal() as session:
        document = await session.get(Document, job_id)
        if document is None or document.workspace_id != workspace_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document was not found")
        await delete_document_vectors(job_id, workspace_id)
        await session.execute(delete(Chunk).where(Chunk.document_id == job_id))
        document.status = "deleted"
        document.pipeline_stage = "deleted"
        document.progress = 100
        await session.commit()

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await set_job_state(redis, job_id, status="deleted", stage="deleted", progress=100)
        await bump_knowledge_revision(redis, workspace_id)
    finally:
        await redis.aclose()
    return DeleteResponse(document_id=job_id, status="deleted")


@router.post(_v1("/query"), response_model=QueryResponse)
@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request):
    return await orchestrator.answer(payload.question, payload.session_id, payload.filters, payload.top_k, workspace_from_request(request))


@router.post(_v1("/query/stream"))
@router.post("/query/stream")
async def query_stream(payload: QueryRequest, request: Request):
    workspace_id = workspace_from_request(request)

    async def events():
        try:
            yield f"event: run\ndata: {json.dumps({'status': 'started'})}\n\n"
            response = await orchestrator.answer(payload.question, payload.session_id, payload.filters, payload.top_k, workspace_id)
            for item in response.trace:
                yield f"event: trace\ndata: {json.dumps(item, default=str)}\n\n"
                await asyncio.sleep(0)
            yield f"event: answer\ndata: {response.model_dump_json()}\n\n"
            yield f"event: complete\ndata: {json.dumps({'status': 'complete'})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': type(exc).__name__})}\n\n"

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

    if any(result != "ok" for result in checks.values()):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "degraded", "checks": checks})
    return {"status": "ok", "checks": checks}
