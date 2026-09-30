# ---- Stage 1: build the Python environment ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Remove what the runtime never needs: installers, bundled test suites, bytecode caches.
    && pip uninstall -y pip setuptools wheel \
    && find /opt/venv -depth \( -name tests -o -name __pycache__ \) -type d -exec rm -rf {} +

# ---- Stage 2: the runtime image, with only what the app needs to run ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "semsearch.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
