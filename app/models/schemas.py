from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ContentType(str, Enum):
    pdf = "pdf"
    webpage = "webpage"
    office = "office"
    image = "image"
    text = "text"
    markdown = "markdown"
    json = "json"
    csv = "csv"


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=5_000_000)
    content_type: ContentType
    metadata: dict[str, Any] = Field(default_factory=dict)
    external_id: str | None = Field(default=None, max_length=256)
    force_reindex: bool = False

    @field_validator("source")
    @classmethod
    def strip_source(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source must not be empty")
        return value


class IngestResponse(BaseModel):
    job_id: str
    status: str
    deduplicated: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12_000)
    session_id: str = Field(default="default", min_length=1, max_length=256)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=8, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question must not be empty")
        return value


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    content_type: str | None = None


class QueryResponse(BaseModel):
    status: str = "answered"
    answer: str
    citations: list[Citation]
    confidence: float
    rewritten_query: str
    retrieval_ms: float
    session_id: str
    safety_flags: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class DocumentStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: int
    attempts: int
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    document_id: str
    status: str


class IngestionMessage(BaseModel):
    job_id: str
    # ``document_id`` is explicit for interoperability; it is currently equal
    # to job_id because a document has one durable ingestion job.
    document_id: str
    workspace_id: str
    source_uri: str
    content_type: ContentType
    metadata: dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False
    document_version: int = 1
    attempt: int = 0
    enqueued_at: datetime = Field(default_factory=datetime.utcnow)
