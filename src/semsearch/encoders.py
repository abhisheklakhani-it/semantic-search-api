"""The three models behind search: dense embeddings, sparse BM25 vectors and a cross-encoder reranker.

All three run through fastembed (ONNX Runtime on CPU), so the Docker image needs no PyTorch.
"""

from pathlib import Path
from typing import Protocol

from qdrant_client import models


class Encoder(Protocol):
    """What the search service needs from the models. Tests plug in a fast fake with the same methods."""

    dense_dim: int

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]: ...

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]: ...

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


def _cached_snapshot(repo_id: str, cache_dir: str | None) -> str | None:
    """Path of an already-downloaded model, or None.

    Works around a fastembed bug: Qdrant/bm25 declares a model file named "mock.file" that never exists,
    so fastembed's offline cache check always fails and the container (HF_HUB_OFFLINE=1) cannot start.
    Passing the snapshot folder directly skips that check.
    """
    if not cache_dir:
        return None
    from huggingface_hub import try_to_load_from_cache

    # Ask the Hugging Face cache for a file the model really has; its folder is the snapshot.
    config = try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir)
    return str(Path(config).parent) if isinstance(config, str) else None


class FastEmbedEncoder:
    def __init__(
        self,
        dense_model: str,
        sparse_model: str,
        rerank_model: str,
        cache_dir: str | None = None,
        batch_size: int = 16,
        threads: int | None = None,
    ):
        self.batch_size = batch_size
        # Imported here so that tests using the fake encoder never load ONNX Runtime.
        from fastembed import SparseTextEmbedding, TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._dense = TextEmbedding(dense_model, cache_dir=cache_dir, threads=threads)
        self._sparse = SparseTextEmbedding(
            sparse_model, cache_dir=cache_dir, specific_model_path=_cached_snapshot(sparse_model, cache_dir)
        )
        self._reranker = TextCrossEncoder(rerank_model, cache_dir=cache_dir, threads=threads)
        self.dense_dim = len(next(iter(self._dense.embed(["dimension probe"]))))
        # ONNX Runtime does one-time setup on the first call; pay it at startup, not on a user's first search.
        self.embed_query("warm up")
        self.rerank("warm up", ["warm up"])

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        dense = [vector.tolist() for vector in self._dense.embed(texts, batch_size=self.batch_size)]
        # BM25 document vectors hold term-frequency weights; Qdrant multiplies in the IDF at query time.
        sparse = [
            models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self._sparse.embed(texts, batch_size=self.batch_size)
        ]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        dense = next(iter(self._dense.query_embed(text))).tolist()
        s = next(iter(self._sparse.query_embed(text)))
        return dense, models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each (query, text) pair jointly; a higher score means more relevant."""
        return [float(score) for score in self._reranker.rerank(query, texts, batch_size=self.batch_size)]
