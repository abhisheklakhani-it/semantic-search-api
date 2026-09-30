# ---- Stage 1: build the Python environment and download the models ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Bake the three models into the image: the container starts fast and works offline.
    && python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models'); \
SparseTextEmbedding('Qdrant/bm25', cache_dir='/opt/models'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/opt/models')" \
    && chmod -R a+rX /opt/models \
    # Remove what the runtime never needs: installers, bundled test suites, bytecode caches.
    && pip uninstall -y pip setuptools wheel \
    && find /opt/venv -depth \( -name tests -o -name __pycache__ \) -type d -exec rm -rf {} +

# ---- Stage 2: the runtime image, with only what the app needs to run ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    SEMSEARCH_MODEL_CACHE_DIR=/opt/models \
    HF_HUB_OFFLINE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

WORKDIR /app
COPY src ./src

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "semsearch.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
