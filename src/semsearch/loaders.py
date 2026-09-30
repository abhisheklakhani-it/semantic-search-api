"""Turn uploaded files into chunks: read the file into LangChain Documents, then split them into chunks.

The loaders used to be langchain-community's PyPDFLoader and TextLoader. That package was sunset in
2026, so the two loaders are implemented here directly with pypdf (the library PyPDFLoader wrapped).
They return the same LangChain Documents, so the text splitter and the rest of the code are unchanged.
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None  # 1-based page number for PDFs, None for plain text
    chunk_index: int  # position of the chunk inside its source


def load_bytes(data: bytes, filename: str) -> list[Document]:
    """Load an uploaded file: a .pdf gives one Document per page, a .txt/.md file gives one Document."""
    source = Path(filename).name  # drop any directories a client put in the name
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return [
            Document(page_content=page.extract_text() or "", metadata={"source": source, "page": number})
            for number, page in enumerate(reader.pages)  # 0-based, like PyPDFLoader
        ]
    if suffix in {".txt", ".md"}:
        return [Document(page_content=data.decode("utf-8"), metadata={"source": source})]
    raise UnsupportedFileType(f"Unsupported file type {suffix!r}; use one of {sorted(SUPPORTED_EXTENSIONS)}")


def load_file(path: Path) -> list[Document]:
    return load_bytes(path.read_bytes(), path.name)


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
