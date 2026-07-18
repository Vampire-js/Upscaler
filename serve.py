"""
HTTP API for the upscaler.

Endpoints:
    GET  /            - health check
    POST /upscale     - accept image, return upscaled PNG

Run:
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from main import load_model, upscale


# --- config (env-overridable) ---
CKPT_PATH = Path(os.getenv("CKPT_PATH", "checkpoints/latest.pt")).resolve()
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", 5 * 1024 * 1024))     # 5 MB
MAX_SIDE = int(os.getenv("MAX_SIDE", 1024))                            # px
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- global model state (loaded at startup) ---
_model = None
_arch = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _arch
    print(f"loading checkpoint {CKPT_PATH} on {DEVICE}")
    _model, _arch = load_model(CKPT_PATH, DEVICE)
    yield
    # nothing to clean up


app = FastAPI(title="upscaler", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "arch": _arch,
        "checkpoint": str(CKPT_PATH),
    }


@app.post("/upscale")
async def upscale_endpoint(file: UploadFile = File(...)):
    # --- validation ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            413, f"file too large: {len(data)} bytes (max {MAX_FILE_BYTES})"
        )

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    w, h = img.size
    if max(w, h) > MAX_SIDE:
        raise HTTPException(
            413, f"image too large: {w}x{h} (max longer side {MAX_SIDE})"
        )

    # --- to numpy [0,1] RGB, trim to multiple of scale ---
    scale = _arch["scale"]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    arr = arr[: (h // scale) * scale, : (w // scale) * scale]

    # --- run model ---
    sr = upscale(_model, arr, DEVICE)

    # --- encode PNG ---
    out = Image.fromarray((np.clip(sr, 0, 1) * 255 + 0.5).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
