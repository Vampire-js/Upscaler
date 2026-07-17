"""
SRDataset: yields (lr, hr) tensor pairs for super-resolution training.

For each __getitem__ call:
    1. Load an HR image from disk.
    2. Random-crop an HR patch of `hr_size`.
    3. Run it through the degrader to produce a matching LR patch.
    4. Convert both to torch tensors, shape (3, H, W), float32 in [0, 1].

The randomization is fresh every epoch, so 69 source images produce
effectively unlimited training samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.downScaler.downscale import (
    DEFAULT,
    DegradeConfig,
    degrade,
    load_image,
    random_crop,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    # (H, W, 3) float32 in [0, 1]  ->  (3, H, W) float32 in [0, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class SRDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        hr_size: int = 192,
        cfg: DegradeConfig = DEFAULT,
        samples_per_epoch: int | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.hr_size = hr_size
        self.cfg = cfg

        self.paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found in {self.image_dir}")

        # Skip images too small to crop from.
        self.paths = [p for p in self.paths if self._is_big_enough(p)]
        if not self.paths:
            raise RuntimeError(
                f"no images >= {hr_size}px on both sides in {self.image_dir}"
            )

        # Virtual epoch length: how many samples one epoch yields.
        # Default: cycle through images ~10x per epoch so each epoch is
        # long enough to be a meaningful training unit.
        self.samples_per_epoch = samples_per_epoch or len(self.paths) * 10

    def _is_big_enough(self, path: Path) -> bool:
        # Cheap header-only check via PIL would need an import; load_image is fine
        # for a one-time filter at __init__.
        try:
            arr = load_image(path)
        except Exception:
            return False
        return arr.shape[0] >= self.hr_size and arr.shape[1] >= self.hr_size

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # idx is only used to pick an image; the crop + degradation are random.
        path = self.paths[idx % len(self.paths)]
        hr_full = load_image(path)
        hr = random_crop(hr_full, self.hr_size)
        lr = degrade(hr, self.cfg)
        return _to_tensor(lr), _to_tensor(hr)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    img_dir = here.parent / "dataset" / "images"

    ds = SRDataset(img_dir, hr_size=192)
    print(f"images kept: {len(ds.paths)}")
    print(f"samples per epoch: {len(ds)}")

    lr, hr = ds[0]
    print(f"lr: shape={tuple(lr.shape)} dtype={lr.dtype} "
          f"min={lr.min():.3f} max={lr.max():.3f}")
    print(f"hr: shape={tuple(hr.shape)} dtype={hr.dtype} "
          f"min={hr.min():.3f} max={hr.max():.3f}")
