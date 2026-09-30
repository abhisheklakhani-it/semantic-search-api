"""The search service: ties the models (encoder) to the database (store).

Model inference is CPU-bound and would block the event loop, so it runs in a worker thread
(asyncio.to_thread) while the server keeps answering other requests.
"""

import asyncio
import datetime as dt
import time

from semsearch.encoders import Encoder
from semsearch.loaders import Chunk
from semsearch.schemas import SearchHit, SearchResponse, Timings
from semsearch.store import VectorStore, build_filter


class SearchService:
    def __init__(self, encoder: Encoder, store: VectorStore, rerank_candidates: int = 30, max_depth: int = 250):
        self.encoder = encoder
        self.store = store
        self.rerank_candidates = rerank_candidates
        self.max_depth = max_depth  # deepest result a client can page to (max offset + max k)

    async def start(self) -> None:
        await self.store.ensure_collection(self.encoder.dense_dim)

    async def ingest(self, chunks: list[Chunk], doc_date: dt.date | None = None) -> int:
        """Embed and store the chunks of one source. Returns how many old leftover chunks were removed."""
        dense, sparse = await asyncio.to_thread(self.encoder.embed_documents, [c.text for c in chunks])
        return await self.store.replace_source(chunks, dense, sparse, doc_date or dt.date.today())

    async def search(
        self,
        query: str,
        k: int = 10,
        offset: int = 0,
        mode: str = "hybrid",
        rerank: bool = True,
        sources: list[str] | None = None,
        date_from: dt.date | None = None,
        date_to: dt.date | None = None,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        dense, sparse = await asyncio.to_thread(self.encoder.embed_query, query)
        t1 = time.perf_counter()

        # The reranker always re-scores the same top `rerank_candidates` chunks, whatever page is asked for.
        # Results deeper than that pool keep their retrieval order. This keeps pages consistent:
        # no result appears on two pages and none is skipped.
        pool = self.rerank_candidates if rerank else 0
        limit = max(offset + k, pool)
        flt = build_filter(sources, date_from, date_to)
        points = await self.store.search(dense, sparse, mode, limit, flt, prefetch_limit=self.max_depth)
        t2 = time.perf_counter()

        hits = [
            {"retrieval_rank": i, "retrieval_score": p.score, "score": p.score, "payload": p.payload}
            for i, p in enumerate(points, start=1)
        ]
        head, tail = hits[:pool], hits[pool:]
        if head and offset < pool:  # skip the reranker when the page lies entirely past the pool
            scores = await asyncio.to_thread(self.encoder.rerank, query, [h["payload"]["text"] for h in head])
            for hit, score in zip(head, scores, strict=True):
                hit["score"] = score
            head.sort(key=lambda h: h["score"], reverse=True)
        hits = head + tail
        t3 = time.perf_counter()

        results = [
            SearchHit(
                rank=rank,
                score=round(h["score"], 4),
                retrieval_rank=h["retrieval_rank"],
                retrieval_score=round(h["retrieval_score"], 4),
                text=h["payload"]["text"],
                source=h["payload"]["source"],
                page=h["payload"]["page"],
                chunk_index=h["payload"]["chunk_index"],
                date=h["payload"]["date"][:10],
            )
            for rank, h in enumerate(hits[offset : offset + k], start=offset + 1)
        ]
        ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
        return SearchResponse(
            query=query,
            mode=mode,
            reranked=rerank,
            k=k,
            offset=offset,
            results=results,
            timings=Timings(embed_ms=ms(t0, t1), retrieve_ms=ms(t1, t2), rerank_ms=ms(t2, t3), total_ms=ms(t0, t3)),
        )
