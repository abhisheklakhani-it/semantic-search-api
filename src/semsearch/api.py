"""FastAPI application: HTTP endpoints for ingesting documents and searching them."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from qdrant_client import AsyncQdrantClient

from semsearch.config import Settings, get_settings


def make_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_url == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    return AsyncQdrantClient(url=settings.qdrant_url)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = make_client(settings)
        yield
        await app.state.client.close()

    app = FastAPI(title="Semantic Search API", version="0.1.0", lifespan=lifespan)

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        try:
            await app.state.client.get_collections()
        except Exception as exc:
            raise HTTPException(503, f"Qdrant is not reachable: {exc}") from exc
        return {"status": "ok", "qdrant": "ok"}

    return app


app = create_app()
