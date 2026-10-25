# ShopSense search API: FastAPI + uvicorn on $PORT, CPU only.
#
# Needs data/catalog.parquet and data/dense_bge-small-en-v1.5.npz in the build context:
#   PYTHONPATH=src python scripts/build_index.py --catalog-only   # if catalog.parquet is missing
#   docker build -t shopsense .
#   docker run --rm -p 8080:8080 shopsense
#
# LLM pipelines stay unavailable unless the GOOGLE_CLOUD_* / GEMINI_* / *_PRICE_* settings
# (see .env.example) are passed as environment variables and the runtime has
# Application Default Credentials (on Cloud Run: the service account). .env is never copied in.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf \
    FASTEMBED_CACHE_PATH=/app/models/fastembed \
    SHOPSENSE_WEB_DIR=/app/web \
    PORT=8080

WORKDIR /app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"

# Dependencies (cached unless pyproject.toml changes), then the package itself.
COPY pyproject.toml README.md ./
RUN mkdir -p src/shopsense && touch src/shopsense/__init__.py \
    && pip install . \
    && pip uninstall -y shopsense
COPY src ./src
RUN pip install --no-deps .

# Bake both models into the image, then forbid downloads at runtime.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); \
from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')"
ENV HF_HUB_OFFLINE=1

COPY web ./web
COPY data/catalog.parquet data/dense_bge-small-en-v1.5.npz ./data/

RUN useradd --create-home --uid 1000 app && chown -R app:app /app/data /app/models
USER app

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn shopsense.api:app --host 0.0.0.0 --port ${PORT}"]
