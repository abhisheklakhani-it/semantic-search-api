"""Shared test setup: in-memory Qdrant and a fake encoder, so tests need no Docker, network or model files."""

import math
import re
import zlib

import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from semsearch.api import create_app
from semsearch.config import Settings

DIM = 64


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bucket(token: str, size: int) -> int:
    return zlib.crc32(token.encode()) % size  # stable across runs, unlike hash()


class FakeEncoder:
    """Deterministic stand-in for the real models, built from word hashing.

    dense: a normalized bag of words hashed into 64 buckets (texts sharing words point the same way)
    sparse: one entry per distinct word, value = its count (like BM25's term frequencies)
    rerank: share of the query's words that appear in the text
    """

    dense_dim = DIM

    def __init__(self):
        self.rerank_calls = 0
        self.reranked_texts = 0

    def _dense(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in tokens(text):
            vec[bucket(tok, DIM)] += 1.0
        vec[0] += 1e-3  # never a zero vector (cosine is undefined for it)
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def _sparse(self, text: str, query: bool = False) -> models.SparseVector:
        counts: dict[int, float] = {}
        for tok in tokens(text):
            idx = bucket(tok, 2**31)
            counts[idx] = 1.0 if query else counts.get(idx, 0.0) + 1.0
        if not counts:
            counts = {0: 1.0}
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts):
        return [self._dense(t) for t in texts], [self._sparse(t) for t in texts]

    def embed_query(self, text):
        return self._dense(text), self._sparse(text, query=True)

    def rerank(self, query, texts):
        self.rerank_calls += 1
        self.reranked_texts += len(texts)
        q = set(tokens(query))
        return [len(q & set(tokens(t))) / max(len(q), 1) for t in texts]


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, qdrant_url=":memory:", **overrides)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def make_client(encoder):
    """Factory: make_client(chunk_size=200, ...) returns a TestClient with a fresh, empty in-memory Qdrant."""
    clients = []

    def _make(**overrides) -> TestClient:
        client = TestClient(create_app(make_settings(**overrides), encoder=encoder))
        client.__enter__()  # runs the startup code (creates the collection)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def upload(client: TestClient, name: str, content: bytes | str, date: str | None = None):
    data = content.encode() if isinstance(content, str) else content
    form = {"date": date} if date else {}
    return client.post("/ingest", files={"file": (name, data)}, data=form)
