# RAG Workspace

Local-first agentic RAG workspace with FastAPI, Kafka, Redis, Postgres, Qdrant, GroqCloud, and a React/Vite UI.

## Run Locally

Install Docker Desktop first, then from this folder:

```powershell
Copy-Item .env.example .env
# Add GROQ_API_KEY to .env for GroqCloud answers
docker compose up --build
```

Open the UI at `http://localhost:5173`.

The app still boots without `GROQ_API_KEY`; ingestion and indexing work, and chat returns a clear API-key-required state. GroqCloud API calls require internet access and a valid GroqCloud API key.

## Local Architecture

```mermaid
flowchart LR
  UI[React + Vite UI] --> API[FastAPI]
  API --> P[(Postgres + transactional outbox)]
  API --> R[(Redis cache + session memory)]
  P --> K[Kafka ingestion topic]
  K --> W[Worker]
  W --> P[(Postgres)]
  W --> Q[(Qdrant)]
  API --> M[GroqCloud Chat API]
```

Kafka is the default ingestion queue. Accepted jobs are committed with an outbox
record in PostgreSQL, then a dispatcher publishes them to Kafka. A crash between
those steps is recoverable: unpublished rows remain available for retry. The
worker commits Kafka offsets only after successful indexing or after it has
durably sent the event to the retry/DLQ topic. Processing is at-least-once, with
document-version fencing and deterministic chunk IDs for idempotency. Redis
Streams is no longer a supported worker mode.

## Product Surface

- Chat: grounded answers, clickable citation chunks, confidence, and safe activity trace.
- Sources: upload files, paste text, watch status, refresh, and delete.
- Quality: visible limitations and grounding contract.

Anonymous workspaces are isolated by a signed cookie. Documents, chunks, vectors, sessions, cache keys, and deletes are scoped to the workspace.

## Grounding Contract

The server uses GroqCloud through its OpenAI-compatible Chat Completions API. The model receives one local tool: `search_knowledge_base`. That tool can only search the current workspace's indexed chunks.

Answers are normalized to one of:

- `answered`
- `insufficient_evidence`
- `refusal`

The server validates citations against retrieved chunk IDs before returning the answer.

## Cache Strategy

- Embeddings: model version plus normalized text hash.
- Retrieval: workspace, knowledge revision, question, filters, and `top_k`.
- Answers: workspace, knowledge revision, model, prompt version, bounded history hash, and question.
- Knowledge revision increments after enqueue, successful indexing, and deletion.
- Answer keys also include filters, `top_k`, retrieval mode, and embedding model.

## Retrieval and source limitations

Sparse retrieval uses PostgreSQL full-text search and dense retrieval uses
Qdrant. Their ranked candidate lists are fused with reciprocal-rank fusion; the
app does not describe its lexical heuristic as a Cross-Encoder. PDF chunks keep
page metadata, and Office chunks retain slide or sheet metadata where available.
Scanned PDFs require a separately configured OCR pipeline; image OCR failures
are explicit failures rather than fabricated placeholder text.

## Current limitations

The Groq orchestration remains a bounded retrieval tool loop; semantic
claim-verification, query decomposition, and optional Cross-Encoder reranking
need an evaluated provider/model before they can be represented as implemented.
Web URL validation rejects private resolved addresses and validates redirect
targets, but production deployments should also enforce outbound network policy
to protect against DNS rebinding. Benchmark scripts and 25K+ integration runs
have not yet been performed.

## Optional Distributed Profile

```powershell
docker compose -f docker-compose.yml -f docker-compose.distributed.yml up --build
```

This starts Kafka and an OTel collector, but the current local worker defaults to Redis Streams.

## Notes

- First Docker run is slower because images and the embedding model download.
- Keep secrets only in `.env`; never put `GROQ_API_KEY` in frontend code.
- GroqCloud is different from xAI/Grok; use keys from https://console.groq.com/keys.
- If you already have old Docker volumes from the earlier schema, run with fresh volumes during development.
