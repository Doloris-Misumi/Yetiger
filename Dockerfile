FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV YESTIGER_HOST=0.0.0.0
ENV PORT=7860
ENV YESTIGER_CORS_ORIGIN=*
ENV YESTIGER_RUN_DIR=/app/webapp_runs
ENV HF_HOME=/app/.hf
ENV TRANSFORMERS_CACHE=/app/.hf
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_OFFLINE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-webapp.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1 torchaudio==2.3.1 \
    && grep -v -E '^(torch|torchaudio)>=' requirements-webapp.txt > /tmp/requirements-no-torch.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

COPY . .

RUN mkdir -p \
    /app/.hf \
    /app/webapp_runs/uploads \
    /app/webapp_runs/jobs \
    /app/webapp_runs/feature_cache \
    /app/webapp_runs/custom_actions

EXPOSE 7860

CMD ["python", "webapp/server.py", "--host", "0.0.0.0", "--port", "7860"]
