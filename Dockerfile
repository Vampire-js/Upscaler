# CPU-only inference image for the upscaler API.
# For a GPU deployment, swap the base image and skip the CPU-only torch install.

FROM python:3.12-slim

# System deps needed by Pillow's runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. Pin CPU-only torch to keep the image small (~200 MB
# instead of the ~2 GB CUDA build).
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application code + the model weights.
COPY pipeline/    ./pipeline/
COPY main.py serve.py ./
COPY checkpoints/latest.pt ./checkpoints/latest.pt

ENV CKPT_PATH=/app/checkpoints/latest.pt \
    MAX_SIDE=1024 \
    MAX_FILE_BYTES=5242880 \
    ALLOWED_ORIGINS=*

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
