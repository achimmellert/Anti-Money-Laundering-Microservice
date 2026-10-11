FROM python:3.11-slim

# Install uv from official binary image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# System libraries for C/C++ extensions & OpenMP (XGBoost, SHAP, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install locked dependencies into system Python using uv.lock (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.lock.txt && \
    rm /tmp/requirements.lock.txt

# Copy application source code and install aml package in editable mode
COPY . .
RUN uv pip install --system --no-cache --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "aml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
