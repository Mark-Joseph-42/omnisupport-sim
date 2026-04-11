# Use the official uv image for the binary
FROM ghcr.io/astral-sh/uv:latest AS uv_source

# Use standard python for the final image
FROM python:3.10-slim

# Copy uv binary from the source
COPY --from=uv_source /uv /uvx /bin/

WORKDIR /repo

# Install dependencies using uv
# We copy requirements first to leverage Docker layer caching
COPY server/requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the full package
COPY . .

# Ensure paths match expected structure
ENV PYTHONPATH="/repo"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Hardened health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Start the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
