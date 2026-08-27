from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.db import engine, init_db
from app.core.telemetry import setup_telemetry
from app.core.workspace import attach_workspace_cookie, workspace_from_request
from app.retrieval.vector_store import ensure_collection


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    await ensure_collection()
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(title="Agentic Multimodal RAG Platform", version="2026.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(router)
setup_telemetry(app)


@app.middleware("http")
async def workspace_cookie_middleware(request: Request, call_next):
    workspace_from_request(request)
    response: Response = await call_next(request)
    attach_workspace_cookie(request, response)
    return response
