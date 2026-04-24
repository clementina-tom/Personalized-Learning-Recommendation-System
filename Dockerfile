# ── PLRS API Dockerfile ───────────────────────────────────────────────────────
# Multi-stage build: builder installs deps, runtime is lean.
#
# Build:  docker build -t plrs-api .
# Run:    docker run -p 8000:8000 --env-file .env plrs-api

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for psycopg2 compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix we can copy
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --prefix=/install \
    fastapi \
    uvicorn[standard] \
    pydantic \
    networkx \
    numpy \
    pandas \
    torch \
    sqlalchemy \
    psycopg2-binary \
    redis \
    anthropic \
    openai \
    huggingface_hub \
    scikit-learn

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps (psycopg2 needs libpq at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY plrs/ ./plrs/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY pyproject.toml .

# Install the plrs package itself (editable-style without pip -e)
RUN pip install --no-deps -e .

# Create non-root user for security
RUN useradd -m -u 1000 plrs && chown -R plrs:plrs /app
USER plrs

# Healthcheck — hits /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default: run init then start server
# Override CMD in docker-compose for different startup modes
CMD ["sh", "-c", "python scripts/docker_init.py && uvicorn plrs.api.app:app --host 0.0.0.0 --port 8000 --workers 2"]
