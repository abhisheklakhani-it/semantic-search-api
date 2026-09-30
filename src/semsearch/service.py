"""The search service: ties the models (encoder) to the database (store).

Model inference is CPU-bound and would block the event loop, so it runs in a worker thread
(asyncio.to_thread) while the server keeps answering other requests.
"""

import asyncio
import datetime as dt

from semsearch.encoders import Encoder
from semsearch.loaders import Chunk
from semsearch.store import VectorStore


class SearchService:
    def __init__(self, encoder: Encoder, store: VectorStore):
        self.encoder = encoder
        self.store = store

    async def start(self) -> None:
        await self.store.ensure_collection(self.encoder.dense_dim)

    async def ingest(self, chunks: list[Chunk], doc_date: dt.date | None = None) -> int:
        """Embed and store the chunks of one source. Returns how many old leftover chunks were removed."""
        dense, sparse = await asyncio.to_thread(self.encoder.embed_documents, [c.text for c in chunks])
        return await self.store.replace_source(chunks, dense, sparse, doc_date or dt.date.today())
