"""Turn uploaded files into chunks: LangChain loaders read the file, a text splitter cuts it into chunks."""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None  # 1-based page number for PDFs, None for plain text
    chunk_index: int  # position of the chunk inside its source


def load_file(path: Path, source: str | None = None) -> list[Document]:
    """Load a .pdf (one Document per page) or a .txt/.md file (one Document)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise UnsupportedFileType(f"Unsupported file type {suffix!r}; use one of {sorted(SUPPORTED_EXTENSIONS)}")
    for doc in docs:
        doc.metadata["source"] = source or path.name
    return docs


def load_bytes(data: bytes, filename: str) -> list[Document]:
    """Load an uploaded file. LangChain loaders read from disk, so write the bytes to a temporary file first."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"Unsupported file type {suffix!r}; use one of {sorted(SUPPORTED_EXTENSIONS)}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        return load_file(path, source=Path(filename).name)


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split into overlapping chunks, preferring paragraph, then line, then sentence, then word boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for doc in docs:
        page = doc.metadata.get("page")
        for piece in splitter.split_text(doc.page_content):
            text = re.sub(r"\s+", " ", piece).strip()  # PDFs break lines mid-sentence; store clean text
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        source=doc.metadata["source"],
                        page=page + 1 if isinstance(page, int) else None,  # PyPDFLoader counts from 0
                        chunk_index=len(chunks),
                    )
                )
    return chunks
