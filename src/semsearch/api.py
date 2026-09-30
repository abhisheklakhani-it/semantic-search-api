"""FastAPI application: HTTP endpoints for ingesting documents and searching them."""

import datetime as dt
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from semsearch.config import Settings, get_settings
from semsearch.encoders import Encoder, FastEmbedEncoder
from semsearch.loaders import UnsupportedFileType, load_bytes, split_documents
from semsearch.logging_setup import setup_logging
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

log = logging.getLogger("semsearch")


def make_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_url == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    return AsyncQdrantClient(url=settings.qdrant_url)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def create_app(settings: Settings | None = None, encoder: Encoder | None = None) -> FastAPI:
    """Build the app. Tests pass a fake encoder and ':memory:' Qdrant; production loads the real models."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = make_client(settings)
        # Load the models once at startup, so the first request is not slow.
        enc = encoder or FastEmbedEncoder(
            settings.dense_model,
            settings.sparse_model,
            settings.rerank_model,
            settings.model_cache_dir,
            settings.embed_batch_size,
        )
        store = VectorStore(client, settings.collection, local=settings.qdrant_url == ":memory:")
        app.state.client = client
        app.state.service = SearchService(enc, store, settings.rerank_candidates, settings.max_offset + settings.max_k)
        await app.state.service.start()
        log.info("startup complete", extra={"qdrant": settings.qdrant_url, "collection": settings.collection})
        yield
        await client.close()

    app = FastAPI(
        title="Semantic Search API",
        version="0.1.0",
        description="Hybrid (dense + BM25) search over your documents with Qdrant and a cross-encoder reranker.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """One JSON log line per request, tagged with a request ID that is also returned to the client."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
            raise
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed_ms(start),
            },
        )
        return response

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
        offset: int = Query(0, ge=0, le=settings.max_offset, description="Skip this many results (pagination)"),
        mode: Literal["hybrid", "dense", "bm25"] = Query("hybrid", description="Retrieval method"),
        rerank: bool = Query(True, description="Re-score the top candidates with the cross-encoder"),
        source: list[str] | None = Query(None, description="Only these sources (repeat the parameter for several)"),
        date_from: dt.date | None = Query(None, description="Only documents dated on or after this day"),
        date_to: dt.date | None = Query(None, description="Only documents dated on or before this day"),
    ) -> SearchResponse:
        """Search the stored chunks: retrieve candidates (hybrid by default), then rerank them with a cross-encoder."""
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must not be after date_to")
        result = await service(request).search(q, k, offset, mode, rerank, source, date_from, date_to)
        log.info(
            "search",
            extra={
                "request_id": request.state.request_id,
                "query_chars": len(q),  # log the size, not the text: queries can contain personal data
                "mode": mode,
                "rerank": rerank,
                "k": k,
                "offset": offset,
                "filtered": bool(source or date_from or date_to),
                "hits": len(result.results),
                **result.timings.model_dump(),
            },
        )
        return result

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
        result = IngestResult(
            source=chunks[0].source,
            chunks=len(chunks),
            pages=len(docs) if is_pdf else None,
            removed_old_chunks=removed,
            took_ms=elapsed_ms(start),
        )
        log.info("ingest", extra={"request_id": request.state.request_id, **result.model_dump()})
        return result

    @app.post("/ingest/text", response_model=TextIngestResponse, status_code=201, tags=["documents"])
    async def ingest_text(request: Request, body: TextIngestRequest) -> TextIngestResponse:
        """Ingest raw text documents as JSON (up to 1,000 per request); each one replaces an earlier source with the same name."""
        start = time.perf_counter()
        batch = []
        for doc in body.documents:
            chunks = split_documents(
                [Document(page_content=doc.text, metadata={"source": doc.source})],
                settings.chunk_size,
                settings.chunk_overlap,
            )
            if chunks:
                batch.append((chunks, doc.date))
        total = await service(request).ingest_many(batch)
        result = TextIngestResponse(documents=len(body.documents), chunks=total, took_ms=elapsed_ms(start))
        log.info("ingest_text", extra={"request_id": request.state.request_id, **result.model_dump()})
        return result

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
