# Stage 1 B1: RGB + Depth

## Scope

B1 keeps the Stage 0 MambaIRv2 backbone, reconstruction head, data split,
crop, augmentation, optimizer, scheduler, batch size, loss, training length,
and evaluation settings unchanged. The only additions are:

1. a 3x3 single-channel depth encoder;
2. concatenation of RGB and depth shallow features;
3. a 1x1 projection back to the original 174-channel feature width.

The resulting model has 23,113,179 parameters, which is 62,466 more than the
23,050,713-parameter B0 baseline.

## Data contract

For every HR basename, the dataset requires one LR RGB file and one aligned LR
depth file. For DIV2K x4, for example:

```text
HR:    0001.png
LR:    0001x4.png
Depth: 0001x4.png
```

Depth may be stored as grayscale, repeated-gray RGB, or a scalar integer PNG.
It is converted to one channel and normalized with full-image P2/P98 statistics
before the synchronized crop and augmentation.

## Full training

Run from the repository root on the Linux training server:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B1_RGBDepth_MambaIRv2_x4.yml \
  --launcher none
```

Resume automatically after an interruption:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B1_RGBDepth_MambaIRv2_x4.yml \
  --launcher none \
  --auto_resume
```

An optional 25% screening run can override the full schedule without editing
the canonical configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B1_RGBDepth_MambaIRv2_x4.yml \
  --launcher none \
  --force_yml \
    name=S1_B1_RGBDepth_MambaIRv2_x4_screen25 \
    train:total_iter=125000 \
    train:scheduler:milestones=[62500,100000,112500,118750]
```

Screening output is directional evidence only and must not be reported as the
final B1 result.

## Five-dataset evaluation

After the full run produces
`experiments/S1_B1_RGBDepth_MambaIRv2_x4/models/net_g_500000.pth`:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
  -opt options/test/mambairv2/test_S1_B1_RGBDepth_MambaIRv2_x4.yml \
  --launcher none
```

The test configuration evaluates Set5, Set14, B100, Urban100, and Manga109
with Y-channel PSNR/SSIM and a four-pixel border crop, matching B0.
