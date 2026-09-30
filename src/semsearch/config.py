"""All settings in one place, read from environment variables (prefix SEMSEARCH_) or a .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEMSEARCH_", env_file=".env", extra="ignore")

    # Qdrant: a server URL, or ":memory:" for an in-process database (tests, quick demos).
    qdrant_url: str = "http://localhost:6333"
    collection: str = "documents"

    # Models (all run on CPU through ONNX Runtime via fastembed, no PyTorch needed).
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # BGE was trained with this instruction in front of search queries (not documents); see its model card.
    # fastembed does not add it, so we do. Set it to "" for a dense model that uses no instruction.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    model_cache_dir: str | None = None
    # Texts per model call. Every text in a batch is padded to the longest one, and attention memory grows with
    # batch x length^2: fastembed's default of 256 ran a 4 GB container out of memory. Measured on an M2 CPU,
    # batch 1 was also the fastest (38 vs 17 chunks/s at batch 16), because no time is spent on padding.
    embed_batch_size: int = Field(1, gt=0)
    # ONNX Runtime threads per model (None = all cores). On an M2 (4 performance + 4 efficiency cores),
    # 4 threads embedded 2x faster than 8: with 8, the fast cores wait for the slow ones.
    model_threads: int | None = Field(None, gt=0)

    # Chunking
    chunk_size: int = Field(800, gt=0, description="Maximum characters per chunk")
    chunk_overlap: int = Field(100, ge=0)

    # Search
    rerank_candidates: int = Field(20, gt=0, description="Default number of retrieved chunks the cross-encoder re-scores")
    max_rerank_candidates: int = Field(100, gt=0)
    max_k: int = Field(50, gt=0, description="Largest page size a client may ask for")
    max_offset: int = Field(200, ge=0, description="Deepest pagination offset")

    # Uploads and logging
    max_upload_mb: float = Field(20, gt=0)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
