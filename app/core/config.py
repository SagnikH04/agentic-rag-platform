from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "agentic-rag-platform"
    environment: str = "development"
    database_url: str
    redis_url: str
    kafka_bootstrap_servers: str
    qdrant_url: str
    otel_exporter_otlp_endpoint: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = 384
    embedding_batch_size: int = 32
    chunk_size: int = 900
    chunk_overlap: int = 140
    max_document_chars: int = 5_000_000
    max_chunks_per_document: int = 10_000
    web_fetch_max_bytes: int = 20_000_000
    allowed_web_domains: str = ""
    local_ingestion_root: str = "/data"
    retrieval_candidate_k: int = 48
    sparse_candidate_limit: int = 10_000
    max_chunks_per_document_in_response: int = 2
    session_ttl_seconds: int = 86_400
    session_max_turns: int = 8
    job_ttl_seconds: int = 604_800
    max_retries: int = 3
    retry_base_seconds: int = 5
    retry_max_seconds: int = 300
    retry_poll_seconds: float = 1.0
    outbox_poll_seconds: float = 1.0
    worker_max_poll_interval_ms: int = 1_800_000
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        return tuple(item.strip().lower() for item in self.allowed_web_domains.split(",") if item.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


settings = Settings()
