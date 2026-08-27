from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import settings

COLLECTION = "documents"


client = AsyncQdrantClient(url=settings.qdrant_url)


async def ensure_collection():
    collections = await client.get_collections()
    if COLLECTION not in [item.name for item in collections.collections]:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
        )
    for field in ("workspace_id", "document_id", "content_type"):
        try:
            await client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            continue


async def upsert_chunks(points: list[dict]):
    if not points:
        return
    await ensure_collection()
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ],
    )


async def delete_document_vectors(document_id: str, workspace_id: str | None = None) -> None:
    await ensure_collection()
    must = [FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    if workspace_id:
        must.append(FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id)))
    await client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(must=must)),
        wait=True,
    )


async def search_vectors(vector: list[float], top_k: int, filters: dict | None = None, workspace_id: str | None = None):
    await ensure_collection()
    conditions = []
    if workspace_id:
        conditions.append(FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id)))
    if filters:
        conditions.extend(
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
            if isinstance(value, (str, int, float, bool))
        )
    query_filter = Filter(must=conditions) if conditions else None

    return await client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
