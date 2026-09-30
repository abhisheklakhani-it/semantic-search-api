"""FastAPI application: HTTP endpoints for ingesting documents and searching them."""

import datetime as dt
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from qdrant_client import AsyncQdrantClient

from semsearch.config import Settings, get_settings
from semsearch.encoders import Encoder, FastEmbedEncoder
from semsearch.loaders import Chunk, UnsupportedFileType, load_bytes, split_documents
from semsearch.schemas import (
    DeleteResult,
    IngestResult,
    SearchResponse,
    SourceInfo,
    TextIngestRequest,
    TextIngestResponse,
)
from semsearch.service import SearchService
from semsearch.store import VectorStore


def make_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_url == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    return AsyncQdrantClient(url=settings.qdrant_url)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def create_app(settings: Settings | None = None, encoder: Encoder | None = None) -> FastAPI:
    """Build the app. Tests pass a fake encoder and ':memory:' Qdrant; production loads the real models."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = make_client(settings)
        # Load the models once at startup, so the first request is not slow.
        enc = encoder or FastEmbedEncoder(
            settings.dense_model, settings.sparse_model, settings.rerank_model, settings.model_cache_dir
        )
        store = VectorStore(client, settings.collection, local=settings.qdrant_url == ":memory:")
        app.state.client = client
        app.state.service = SearchService(enc, store, settings.rerank_candidates)
        await app.state.service.start()
        yield
        await client.close()

    app = FastAPI(
        title="Semantic Search API",
        version="0.1.0",
        description="Hybrid (dense + BM25) search over your documents with Qdrant and a cross-encoder reranker.",
        lifespan=lifespan,
    )

    def service(request: Request) -> SearchService:
        return request.app.state.service

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict:
        svc = service(request)
        try:
            points = await svc.store.count()
        except Exception as exc:
            raise HTTPException(503, f"Qdrant is not reachable: {exc}") from exc
        return {"status": "ok", "qdrant": "ok", "collection": svc.store.collection, "chunks": points}

    @app.get("/search", response_model=SearchResponse, tags=["search"])
    async def search(
        request: Request,
        q: str = Query(min_length=1, max_length=1000, description="The search query"),
        k: int = Query(10, ge=1, le=settings.max_k, description="Number of results"),
        mode: Literal["hybrid", "dense", "bm25"] = Query("hybrid", description="Retrieval method"),
        rerank: bool = Query(True, description="Re-score the candidates with the cross-encoder"),
    ) -> SearchResponse:
        """Search the stored chunks: retrieve candidates (hybrid by default), then rerank them with a cross-encoder."""
        return await service(request).search(q, k, mode, rerank)

    @app.post("/ingest", response_model=IngestResult, status_code=201, tags=["documents"])
    async def ingest_file(
        request: Request,
        file: UploadFile = File(description=".pdf, .txt or .md"),
        date: dt.date | None = Form(None, description="Document date (YYYY-MM-DD) for filtering; defaults to today"),
    ) -> IngestResult:
        """Upload a file: it is loaded, chunked, embedded and stored. Uploading the same file name again replaces it."""
        start = time.perf_counter()
        data = await file.read()
        if not data:
            raise HTTPException(400, "The file is empty")
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {settings.max_upload_mb:g} MB")
        try:
            docs = load_bytes(data, file.filename or "upload.txt")
        except UnsupportedFileType as exc:
            raise HTTPException(415, str(exc)) from exc
        except Exception as exc:  # corrupt PDF, wrong encoding, ...
            raise HTTPException(400, f"Could not read the file: {exc}") from exc

        chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(422, "No text found in the file (a scanned PDF without a text layer?)")
        removed = await service(request).ingest(chunks, date)
        is_pdf = chunks[0].page is not None
        return IngestResult(
            source=chunks[0].source,
            chunks=len(chunks),
            pages=len(docs) if is_pdf else None,
            removed_old_chunks=removed,
            took_ms=elapsed_ms(start),
        )

    @app.post("/ingest/text", response_model=TextIngestResponse, status_code=201, tags=["documents"])
    async def ingest_text(request: Request, body: TextIngestRequest) -> TextIngestResponse:
        """Ingest raw text documents as JSON (up to 1,000 per request); each one replaces an earlier source with the same name."""
        from langchain_core.documents import Document

        start = time.perf_counter()
        total = 0
        for doc in body.documents:
            chunks: list[Chunk] = split_documents(
                [Document(page_content=doc.text, metadata={"source": doc.source})],
                settings.chunk_size,
                settings.chunk_overlap,
            )
            if chunks:
                await service(request).ingest(chunks, doc.date)
                total += len(chunks)
        return TextIngestResponse(documents=len(body.documents), chunks=total, took_ms=elapsed_ms(start))

    @app.get("/sources", response_model=list[SourceInfo], tags=["documents"])
    async def list_sources(request: Request) -> list[SourceInfo]:
        counts = await service(request).store.list_sources()
        return [SourceInfo(source=s, chunks=n) for s, n in counts.items()]

    @app.delete("/sources/{source}", response_model=DeleteResult, tags=["documents"])
    async def delete_source(request: Request, source: str) -> DeleteResult:
        deleted = await service(request).store.delete_source(source)
        if not deleted:
            raise HTTPException(404, f"No source named {source!r}")
        return DeleteResult(source=source, deleted_chunks=deleted)

    return app


app = create_app()
