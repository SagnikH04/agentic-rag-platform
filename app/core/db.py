from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


engine = create_async_engine(settings.database_url, pool_pre_ping=True, pool_recycle=1800)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(128), index=True, default="default")
    external_id: Mapped[str | None] = mapped_column(String(256), index=True, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    content_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    pipeline_stage: Mapped[str] = mapped_column(String(64), default="accepted")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    # ``version`` is the requested generation; retrieval uses the last fully
    # committed generation so reindexing never withdraws known-good evidence.
    active_version: Mapped[int] = mapped_column(Integer, default=0)
    index_runs: Mapped[int] = mapped_column(Integer, default=0)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "document_version", "chunk_index", name="uq_chunk_document_version_index"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(128), index=True, default="default")
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    document_version: Mapped[int] = mapped_column(Integer, default=1)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    topic: Mapped[str] = mapped_column(String(128), index=True)
    message_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    event_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QueryRun(Base):
    __tablename__ = "query_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[str] = mapped_column(String(256), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_status: Mapped[str] = mapped_column(String(32), default="running")
    citations: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    trace: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # This project predates migrations. Keep existing local volumes usable
        # while adding the generation fence required for safe replay/reindex.
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1"))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS active_version INTEGER NOT NULL DEFAULT 0"))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_runs INTEGER NOT NULL DEFAULT 0"))
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL"))
        await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_version INTEGER NOT NULL DEFAULT 1"))
        await conn.execute(text("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS uq_chunk_document_index"))
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE chunks ADD CONSTRAINT uq_chunk_document_version_index UNIQUE (document_id, document_version, chunk_index);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$;
        """))
        # Existing indexed installations predate active_version. Their current
        # generation is already the known-good one and must remain visible.
        await conn.execute(text("UPDATE documents SET active_version = version WHERE status = 'indexed' AND active_version = 0"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_workspace_document ON chunks (workspace_id, document_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_text_fts ON chunks USING GIN (to_tsvector('simple', text))"))
