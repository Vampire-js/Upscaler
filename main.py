"""
Inference: run the trained upscaler on an image and save the result.

Usage:
    python main.py                                  # uses a default sample image
    python main.py path/to/image.png                # upscale a specific image
    python main.py path/to/image.png --degrade      # also degrade first (LR->SR)

Outputs (in debug/infer/):
    <name>__lr.png     (only with --degrade)
    <name>__sr.png
    <name>__grid.png   (LR nearest-up | SR [| HR when --degrade])
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pipeline.downScaler.downscale import DEFAULT, degrade, load_image, save_image
from pipeline.model.upscaler import Upscaler


ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "latest.pt"
IMG_DIR = ROOT / "dataset" / "images"
OUT_DIR = ROOT / "debug" / "infer"


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    # (H, W, 3) float32 [0,1] -> (1, 3, H, W)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _to_hwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()


def _nearest_up(arr: np.ndarray, factor: int) -> np.ndarray:
    return arr.repeat(factor, axis=0).repeat(factor, axis=1)


def load_model(device: str) -> Upscaler:
    if not CKPT.exists():
        raise FileNotFoundError(f"no checkpoint at {CKPT} -- train first")
    state = torch.load(CKPT, map_location=device)
    # Must match training-time architecture.
    model = Upscaler(scale=DEFAULT.scale, channels=64, num_res_blocks=8).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {CKPT.relative_to(ROOT)} "
          f"(epoch {state.get('epoch', '?')}, step {state.get('step', '?')})")
    return model


@torch.no_grad()
def upscale(model: Upscaler, lr: np.ndarray, device: str) -> np.ndarray:
    x = _to_tensor(lr).to(device)
    y = model(x)
    return _to_hwc(y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default=None,
                   help="input image path; defaults to a sample from the dataset")
    p.add_argument("--degrade", action="store_true",
                   help="degrade input first (treat given image as HR)")
    p.add_argument("--crop", type=int, default=None,
                   help="center-crop the input to this size before anything else")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image) if args.image else next(IMG_DIR.glob("*.png"))
    print(f"input : {img_path}")
    print(f"device: {device}")

    img = load_image(img_path)
    print(f"loaded shape={img.shape} min={img.min():.3f} max={img.max():.3f}")

    if args.crop is not None:
        s = args.crop
        h, w = img.shape[:2]
        if h < s or w < s:
            raise ValueError(f"image {h}x{w} smaller than crop {s}")
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        print(f"center-cropped to {img.shape}")

    # Trim so H and W are multiples of scale -- otherwise downsample/upsample
    # round-trip drops a few pixels and shapes stop matching.
    s = DEFAULT.scale
    h, w = img.shape[:2]
    img = img[: (h // s) * s, : (w // s) * s]

    model = load_model(device)
    stem = img_path.stem

    if args.degrade:
        hr = img
        lr = degrade(hr, DEFAULT)
        sr = upscale(model, lr, device)

        save_image(lr, OUT_DIR / f"{stem}__lr.png")
        save_image(sr, OUT_DIR / f"{stem}__sr.png")
        grid = np.concatenate([_nearest_up(lr, DEFAULT.scale), sr, hr], axis=1)
        save_image(grid, OUT_DIR / f"{stem}__grid.png")

        l1 = float(np.mean(np.abs(sr - hr)))
        print(f"L1(SR, HR) = {l1:.4f}")
        print(f"wrote {OUT_DIR.relative_to(ROOT)}/{stem}__(lr|sr|grid).png")
    else:
        sr = upscale(model, img, device)
        save_image(sr, OUT_DIR / f"{stem}__sr.png")
        grid = np.concatenate([_nearest_up(img, DEFAULT.scale), sr], axis=1)
        save_image(grid, OUT_DIR / f"{stem}__grid.png")
        print(f"in : {img.shape}")
        print(f"out: {sr.shape}")
        print(f"wrote {OUT_DIR.relative_to(ROOT)}/{stem}__(sr|grid).png")


if __name__ == "__main__":
    main()
