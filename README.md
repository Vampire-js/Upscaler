# Upscaler V1

A simple 4× image super-resolution model, trained from scratch on screenshots.
Small CNN + sub-pixel upsampling, pixel-L1 loss, self-generated (LR, HR) pairs.

Built as a learning project (works only on screenshots as I didn't use real camera images in the dataset)
## Architecture

![architecture](architecture.png)

- HR patches are cropped from disk and passed through a randomised **degradation
  pipeline** (Gaussian blur + downsample + noise) to produce matching LR patches.
- The **CNN body** (head conv → residual blocks → tail conv) operates entirely
  in low-resolution space.
- The original **LR is carried around the CNN** and concatenated with the
  learned features (the "skip" arrow in the diagram) before upsampling. This
  lets the network focus on learning *corrections*, not the identity mapping.
- **Sub-pixel convolution** (`PixelShuffle`) does the actual upscaling in
  stages of 2×.
- Loss is **L1(SR, HR)**. Perceptual / embedding loss is the planned next step.

## Results so far

Trained on ~2k screenshots at 4× upscaling, pixel-L1 loss only.
Each grid is **LR (nearest-upsampled) | SR | HR**.

![sample 1](samples/diagram__grid.png)

Training-time held-out sample after ~12k steps:

![training sample](samples/train_step_012000.png)

## Layout

```
pipeline/
    downScaler/downscale.py   # HR -> random LR degradation pipeline
    model/upscaler.py         # CNN + PixelShuffle upscaler
    dataset.py                # SRDataset yielding (lr, hr) tensor pairs
    train.py                  # training loop (L1 + Adam)
main.py                       # inference / evaluation script
dataset/images/               # (not versioned) put HR images here
```

## Notes

- Scale is fixed at 4× and enforced to be a power of 2 (2×, 4×, 8× possible).
- Model is ~1M params at defaults (`channels=64`, `num_res_blocks=8`).
