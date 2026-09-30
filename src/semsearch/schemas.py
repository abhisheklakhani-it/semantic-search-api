"""Request and response models: FastAPI validates input with them and documents them in /docs."""

import datetime as dt

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    source: str
    chunks: int = Field(description="Chunks stored for this source")
    pages: int | None = Field(None, description="Pages read (PDF only)")
    removed_old_chunks: int = Field(0, description="Leftover chunks deleted from an earlier, longer version")
    took_ms: float


class TextDocument(BaseModel):
    source: str = Field(min_length=1, max_length=512, examples=["meeting-notes-2026-09-29"])
    text: str = Field(min_length=1)
    date: dt.date | None = Field(None, description="Document date for filtering; defaults to today")


class TextIngestRequest(BaseModel):
    documents: list[TextDocument] = Field(min_length=1, max_length=1000)


class TextIngestResponse(BaseModel):
    documents: int
    chunks: int
    took_ms: float


class SourceInfo(BaseModel):
    source: str
    chunks: int


class DeleteResult(BaseModel):
    source: str
    deleted_chunks: int
