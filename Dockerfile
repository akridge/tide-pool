# One image for both the build pipeline and the search server.
# CPU torch by default; pass TORCH_INDEX=https://download.pytorch.org/whl/cu124 for CUDA
# (docker-compose.gpu.yml does this).
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url "$TORCH_INDEX" \
 && pip install -r requirements.txt

COPY pipeline pipeline
COPY server server
COPY web web

EXPOSE 8000
CMD ["python", "server/app.py", "--host", "0.0.0.0", "--port", "8000"]
