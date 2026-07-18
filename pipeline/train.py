"""
Training loop for the upscaler.

Loss:      L1(SR, HR)
Optimizer: Adam
Logging:   prints per-N-step averages, saves sample grids, logs to MLflow.
Output:    checkpoints/latest.pt + debug/train/step_XXXXX.png
           mlruns/ (MLflow tracking directory)
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.dataset import SRDataset
from pipeline.downScaler.downscale import DEFAULT, save_image
from pipeline.model.upscaler import Upscaler


ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "dataset" / "images"
CKPT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "debug" / "train"
MLRUNS_DIR = ROOT / "mlruns"
EXPERIMENT_NAME = "upscaler"


# --------------------------------------------------------------- helpers
def _tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [0, 1] -> (H, W, 3) numpy in [0, 1]."""
    return t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def _nearest_up(arr: np.ndarray, factor: int) -> np.ndarray:
    return arr.repeat(factor, axis=0).repeat(factor, axis=1)


def save_sample_grid(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    path: Path,
    scale: int,
) -> None:
    """Write [LR (nearest-upsampled) | SR | HR]."""
    lr_np = _nearest_up(_tensor_to_hwc(lr), scale)
    sr_np = _tensor_to_hwc(sr)
    hr_np = _tensor_to_hwc(hr)
    save_image(np.concatenate([lr_np, sr_np, hr_np], axis=1), path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def evaluate(model: nn.Module, eval_batch: tuple[torch.Tensor, torch.Tensor],
             device: str) -> tuple[float, float]:
    """Return (L1, PSNR) on a fixed held-out batch."""
    model.eval()
    lr, hr = eval_batch
    lr = lr.to(device)
    hr = hr.to(device)
    sr = model(lr).clamp(0, 1)
    l1 = torch.mean(torch.abs(sr - hr)).item()
    mse = torch.mean((sr - hr) ** 2).item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    model.train()
    return l1, psnr


def build_eval_batch(dataset: SRDataset, n: int, seed: int = 12345
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic fixed batch used for eval every time."""
    import random
    rng_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        lrs, hrs = [], []
        for i in range(n):
            lr, hr = dataset[i]
            lrs.append(lr)
            hrs.append(hr)
        return torch.stack(lrs), torch.stack(hrs)
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


# --------------------------------------------------------------- training
def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    CKPT_DIR.mkdir(exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- MLflow setup
    # SQLite for tracking metadata + local folder for artifacts.
    MLRUNS_DIR.mkdir(exist_ok=True)
    tracking_db = MLRUNS_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---- data
    dataset = SRDataset(
        IMG_DIR,
        hr_size=args.hr_size,
        cfg=DEFAULT,
        samples_per_epoch=args.samples_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"dataset: {len(dataset.paths)} images, "
          f"{len(dataset)} samples/epoch, batch={args.batch_size}")

    # Fixed eval batch (deterministic, same across runs).
    eval_batch = build_eval_batch(dataset, n=args.eval_size)

    # ---- model
    model = Upscaler(
        scale=DEFAULT.scale,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params, scale={DEFAULT.scale}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    # ---- resume
    ckpt_path = CKPT_DIR / "latest.pt"
    start_epoch = 0
    global_step = 0
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        start_epoch = state["epoch"]
        global_step = state["step"]
        print(f"resumed from epoch {start_epoch} step {global_step}")

    # ---- fixed sample for periodic visualization
    torch.manual_seed(0)
    vis_lr, vis_hr = dataset[0]
    vis_lr = vis_lr.unsqueeze(0).to(device)
    vis_hr = vis_hr.unsqueeze(0).to(device)

    # ---- MLflow run
    run_name = args.run_name or f"run_{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        # params
        params = {
            **vars(args),
            "device": device,
            "num_images": len(dataset.paths),
            "samples_per_epoch": len(dataset),
            "num_params": n_params,
            "scale": DEFAULT.scale,
            "git_commit": _git_commit(),
        }
        # Flatten the degrade config
        for k, v in asdict(DEFAULT).items():
            params[f"degrade.{k}"] = v
        mlflow.log_params({k: str(v) for k, v in params.items()})

        # ---- loop
        model.train()
        running_loss = 0.0
        running_n = 0
        t_last = time.time()

        for epoch in range(start_epoch, args.epochs):
            for lr, hr in loader:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)

                sr = model(lr)
                loss = loss_fn(sr, hr)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                running_loss += loss.item() * lr.size(0)
                running_n += lr.size(0)
                global_step += 1

                if global_step % args.log_every == 0:
                    dt = time.time() - t_last
                    avg = running_loss / running_n
                    ips = running_n / max(dt, 1e-9)
                    print(f"epoch {epoch}  step {global_step:6d}  "
                          f"L1={avg:.4f}  {ips:5.1f} img/s")
                    mlflow.log_metrics(
                        {"train/l1": avg, "train/img_per_s": ips},
                        step=global_step,
                    )
                    running_loss = 0.0
                    running_n = 0
                    t_last = time.time()

                if global_step % args.eval_every == 0:
                    val_l1, val_psnr = evaluate(model, eval_batch, device)
                    print(f"    eval  L1={val_l1:.4f}  PSNR={val_psnr:.2f} dB")
                    mlflow.log_metrics(
                        {"eval/l1": val_l1, "eval/psnr": val_psnr},
                        step=global_step,
                    )

                if global_step % args.sample_every == 0:
                    model.eval()
                    with torch.no_grad():
                        vis_sr = model(vis_lr)
                    out = SAMPLE_DIR / f"step_{global_step:06d}.png"
                    save_sample_grid(vis_lr[0], vis_sr[0], vis_hr[0], out, DEFAULT.scale)
                    mlflow.log_artifact(str(out), artifact_path="samples")
                    print(f"  wrote {out.relative_to(ROOT)}")
                    model.train()

            # end of epoch: checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": epoch + 1,
                    "step": global_step,
                    "arch": {
                        "scale": DEFAULT.scale,
                        "channels": args.channels,
                        "num_res_blocks": args.res_blocks,
                    },
                    "run_name": run_name,
                },
                ckpt_path,
            )
            print(f"  checkpoint -> {ckpt_path.relative_to(ROOT)}")

        # final eval + artifacts
        val_l1, val_psnr = evaluate(model, eval_batch, device)
        mlflow.log_metrics(
            {"final/l1": val_l1, "final/psnr": val_psnr},
            step=global_step,
        )
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")
        print(f"final: L1={val_l1:.4f}  PSNR={val_psnr:.2f} dB")

    print("done.")


# --------------------------------------------------------------- cli
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hr-size", type=int, default=192)
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--res-blocks", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-size", type=int, default=16,
                   help="number of held-out patches for eval")
    p.add_argument("--run-name", type=str, default=None,
                   help="MLflow run name; auto-generated if omitted")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
