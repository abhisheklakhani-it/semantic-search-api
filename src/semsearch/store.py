"""Qdrant access: one collection with a dense vector and a sparse BM25 vector per chunk."""

import uuid
from collections import Counter
from datetime import date, datetime, timezone

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from semsearch.loaders import Chunk

DENSE = "dense"
SPARSE = "bm25"
UPSERT_BATCH = 256


def point_id(source: str, chunk_index: int) -> str:
    """Same source + position always gives the same ID, so re-ingesting a file overwrites its old chunks."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}\x00{chunk_index}"))


def to_rfc3339(day: date) -> str:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def source_filter(source: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])


def build_filter(
    sources: list[str] | None = None, date_from: date | None = None, date_to: date | None = None
) -> models.Filter | None:
    """Metadata filter: any of the given sources, and a document date inside [date_from, date_to]."""
    must: list[models.Condition] = []
    if sources:
        must.append(models.FieldCondition(key="source", match=models.MatchAny(any=sources)))
    if date_from or date_to:
        must.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(
                    gte=to_rfc3339(date_from) if date_from else None,
                    lte=to_rfc3339(date_to) if date_to else None,
                ),
            )
        )
    return models.Filter(must=must) if must else None


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, local: bool = False):
        self.client = client
        self.collection = collection
        self.local = local  # in-memory mode ignores payload indexes, so skip creating them

    async def ensure_collection(self, dense_dim: int) -> None:
        if await self.client.collection_exists(self.collection):
            return
        try:
            await self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
                # Modifier.IDF: Qdrant computes BM25's inverse document frequency itself, from the whole collection.
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 409:  # another API worker created it at the same moment
                return
            raise
        if not self.local:
            # Indexes make the metadata filters fast on large collections.
            await self.client.create_payload_index(self.collection, "source", models.PayloadSchemaType.KEYWORD)
            await self.client.create_payload_index(self.collection, "date", models.PayloadSchemaType.DATETIME)

    async def replace_source(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[models.SparseVector],
        doc_date: date,
    ) -> int:
        """Store the chunks of one source, replacing whatever that source had before. Returns chunks removed."""
        source = chunks[0].source
        ingested_at = datetime.now(timezone.utc).isoformat()
        points = [
            models.PointStruct(
                id=point_id(c.source, c.chunk_index),
                vector={DENSE: d, SPARSE: s},
                payload={
                    "text": c.text,
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "date": to_rfc3339(doc_date),
                    "ingested_at": ingested_at,
                },
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), UPSERT_BATCH):
            await self.client.upsert(self.collection, points[start : start + UPSERT_BATCH])

        # Upserting first and trimming afterwards means searches never see the source disappear.
        # If the new version is shorter, delete the leftover chunks from the old one.
        leftovers = models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                models.FieldCondition(key="chunk_index", range=models.Range(gte=len(chunks))),
            ]
        )
        removed = (await self.client.count(self.collection, count_filter=leftovers, exact=True)).count
        if removed:
            await self.client.delete(self.collection, points_selector=models.FilterSelector(filter=leftovers))
        return removed

    async def delete_source(self, source: str) -> int:
        n = await self.count(source)
        if n:
            await self.client.delete(
                self.collection, points_selector=models.FilterSelector(filter=source_filter(source))
            )
        return n

    async def count(self, source: str | None = None) -> int:
        flt = source_filter(source) if source else None
        return (await self.client.count(self.collection, count_filter=flt, exact=True)).count

    async def list_sources(self) -> dict[str, int]:
        """Chunk count per source (reads only the source field of every point)."""
        counts: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = await self.client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=["source"], with_vectors=False
            )
            counts.update(p.payload["source"] for p in points)
            if offset is None:
                return dict(sorted(counts.items()))

    async def search(
        self,
        dense: list[float],
        sparse: models.SparseVector,
        mode: str,
        limit: int,
        flt: models.Filter | None = None,
        prefetch_limit: int | None = None,
    ) -> list[models.ScoredPoint]:
        """Retrieve the top `limit` chunks. mode: 'dense', 'bm25' or 'hybrid' (both, merged with RRF).

        For hybrid, each of the two searches returns `prefetch_limit` candidates. Keep it constant across
        requests: RRF scores depend on it, so changing it with the page would reorder earlier pages.
        """
        prefetch_limit = max(prefetch_limit or limit, limit)
        if mode == "dense":
            kwargs = {"query": dense, "using": DENSE}
        elif mode == "bm25":
            kwargs = {"query": sparse, "using": SPARSE}
        elif mode == "hybrid":
            # Qdrant runs both searches, then Reciprocal Rank Fusion merges the two ranked lists:
            # score = sum over lists of 1 / (60 + rank). Ranks are used, so the different score scales don't matter.
            kwargs = {
                "prefetch": [
                    models.Prefetch(query=dense, using=DENSE, filter=flt, limit=prefetch_limit),
                    models.Prefetch(query=sparse, using=SPARSE, filter=flt, limit=prefetch_limit),
                ],
                "query": models.FusionQuery(fusion=models.Fusion.RRF),
            }
        else:
            raise ValueError(f"Unknown search mode {mode!r}")
        response = await self.client.query_points(
            self.collection, query_filter=flt, limit=limit, with_payload=True, **kwargs
        )
        return response.points
