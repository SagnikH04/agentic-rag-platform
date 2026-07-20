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
    for field in ("document_id", "content_type"):
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


async def delete_document_vectors(document_id: str) -> None:
    await ensure_collection()
    await client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        ),
        wait=True,
    )


async def search_vectors(vector: list[float], top_k: int, filters: dict | None = None):
    await ensure_collection()
    query_filter = None
    if filters:
        conditions = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
            if isinstance(value, (str, int, float, bool))
        ]
        if conditions:
            query_filter = Filter(must=conditions)

    return await client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
