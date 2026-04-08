FROM ghcr.io/astral-sh/uv:python3.10-slim AS builder

WORKDIR /app
COPY server/requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /repo

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the full package
COPY . .

# Ensure paths match expected structure
ENV PYTHONPATH="/repo:/repo/omnisupport_sim"

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
