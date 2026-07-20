# Agentic Multimodal RAG Platform

Production-oriented starter for async document ingestion, vector indexing, hybrid retrieval, agentic answer generation, and observability.

## Run

```bash
docker compose up --build
```

## APIs

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"source":"https://example.com","content_type":"webpage"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the knowledge base say?","session_id":"demo"}'
```
