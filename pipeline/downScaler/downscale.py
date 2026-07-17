"""
Degradation pipeline: HR image -> random LR image.

Conventions
-----------
    numpy arrays, shape (H, W, 3), dtype float32, values in [0, 1], RGB.

Public API
----------
    load_image(path)              -> hr array
    random_crop(arr, size)        -> square crop
    degrade(hr, scale=4)          -> lr array
    save_image(arr, path)         -> writes PNG
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------------------------- config
@dataclass(frozen=True)
class DegradeConfig:
    scale: int = 4
    blur_sigma: tuple[float, float] = (0.0, 0.6)
    noise_sigma: tuple[float, float] = (0.0, 0.008)  # on [0, 1] scale
    clean_prob: float = 0.5                          # fraction with no blur/noise
    interpolations: tuple[int, ...] = (
        Image.BICUBIC,
        Image.BILINEAR,
        Image.BOX,   # ~ area
    )


DEFAULT = DegradeConfig()


# ------------------------------------------------------------------------ io
def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, path: str | Path) -> None:
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


# ---------------------------------------------------------- internal helpers
def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


# ------------------------------------------------------------------- pieces
def random_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {h}x{w} smaller than crop size {size}")
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return arr[y:y + size, x:x + size]


# ------------------------------------------------------------------ degrade
def degrade(hr: np.ndarray, cfg: DegradeConfig = DEFAULT) -> np.ndarray:
    """Randomly degrade an HR patch into an LR patch."""
    clean = random.random() < cfg.clean_prob
    sigma_b = 0.0 if clean else random.uniform(*cfg.blur_sigma)
    sigma_n = 0.0 if clean else random.uniform(*cfg.noise_sigma)
    interp = random.choice(cfg.interpolations)

    # blur + downsample stay in PIL (single round-trip through numpy at the end)
    img = _to_pil(hr)
    if sigma_b > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma_b))

    h, w = hr.shape[:2]
    img = img.resize((w // cfg.scale, h // cfg.scale), interp)
    lr = _from_pil(img)

    if sigma_n > 0:
        lr = lr + np.random.normal(0.0, sigma_n, lr.shape).astype(np.float32)

    return np.clip(lr, 0.0, 1.0)


# ---------------------------------------------------------- sanity check cli
def _side_by_side(hr: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """HR | nearest-upsampled LR, so LR pixels are shown at their true size."""
    up = _to_pil(lr).resize((hr.shape[1], hr.shape[0]), Image.NEAREST)
    return np.concatenate([hr, _from_pil(up)], axis=1)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    img_dir = here.parents[1] / "dataset" / "images"
    debug_dir = here.parents[1] / "debug"
    debug_dir.mkdir(exist_ok=True)

    sample = next(img_dir.glob("*.png"))
    hr_full = load_image(sample)
    print(f"loaded {sample.name}  shape={hr_full.shape} "
          f"min={hr_full.min():.3f} max={hr_full.max():.3f}")

    hr_patch = 192
    for i in range(4):
        hr = random_crop(hr_full, hr_patch)
        lr = degrade(hr)
        out = debug_dir / f"sample_{i}.png"
        save_image(_side_by_side(hr, lr), out)
        print(f"  wrote {out.name}  hr={hr.shape} lr={lr.shape}")
