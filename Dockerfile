FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (separate layer — changes rarely)
RUN pip install --upgrade pip setuptools wheel

# ── The key fix: install heavy packages first in their own layer ──
# torch alone is 420MB. By installing it separately before the rest
# of requirements.txt, Docker caches this layer independently.
# It only re-downloads when this line changes, not on every code change.
RUN pip install --no-cache-dir \
    torch==2.10.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers separately (also heavy)
RUN pip install --no-cache-dir sentence-transformers

# Install the rest of your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the image at build time
# Restarts won't re-download the model
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/models_cache

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application code LAST (code changes don't bust package cache)
COPY . .

RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--run-now"]