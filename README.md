# Diffusion-Augmented Topology-Preserving UDA (UDA-Lab)

This is a **self-contained, fully reproducible** PyTorch research codebase for:
**“Diffusion-Augmented Topology-Preserving Unsupervised Domain Adaptation for Segmentation.”**

All code, configs, and scripts live under `uda_lab/` and do **not** modify the
existing repository.

## Reproduce in one command

Single GPU:
```
bash uda_lab/scripts/run_all_benchmarks.sh
```

Multi-GPU (DDP):
```
bash uda_lab/scripts/run_all_benchmarks.sh --ddp --gpus 2
```

Ablations:
```
bash uda_lab/scripts/run_all_ablations.sh
```

If you are using HuggingFace datasets without local files, set:
```
export HF_DATASETS=drive,stare,chase,kvasir,retina
```

## Setup

1) Create a clean environment:
```
uv venv .venv --python python3.12
uv pip install -r uda_lab/requirements.txt --python .venv/bin/python
```

2) Export HF token (already in `.env`):
```
set -a && source .env && set +a
```

## Dataset preparation

Examples (see `uda_lab/scripts/download_datasets.py` for full instructions):
```
python uda_lab/scripts/download_datasets.py --dataset drive
python uda_lab/scripts/prepare_splits.py --images /path/to/images --masks /path/to/masks --out_dir data/drive
```

HuggingFace examples (drive/stare/chase IDs are prefilled from ICPR setup):
```
python uda_lab/scripts/train.py --config uda_lab/configs/experiments/main_full_method.yaml \
  --dataset drive_hf --model unet --seed 0 \
  --opts dataset.hf_image_key=image \
         dataset.hf_mask_key=label
python uda_lab/scripts/train.py --config uda_lab/configs/experiments/main_full_method.yaml \
  --dataset kvasir --model unet --seed 0
```

## Diffusion augmentation

```
python uda_lab/scripts/generate_diffusion_aug.py \
  --input_dir data/gta5/images/train \
  --output_dir outputs/diffusion_aug/gta5 \
  --style_name cityscapes \
  --model_path runwayml/stable-diffusion-v1-5 \
  --prompt "city street, overcast, realistic" \
  --seed 0
```

## Training + adaptation

Source-only:
```
python uda_lab/scripts/train.py --config uda_lab/configs/experiments/main_full_method.yaml \
  --dataset gta5_cityscapes --model deeplabv3p --seed 0 --log_tb
```

UDA:
```
python uda_lab/scripts/adapt.py --config uda_lab/configs/experiments/main_full_method.yaml \
  --dataset gta5_cityscapes --model deeplabv3p --seed 0 --log_tb \
  --opts uda.use_diffusion_aug=true \
         uda.diffusion_manifest=outputs/diffusion_aug/gta5/cityscapes/manifest.json
```

## Evaluation

```
python uda_lab/scripts/evaluate.py --config uda_lab/configs/experiments/main_full_method.yaml \
  --dataset gta5_cityscapes --model deeplabv3p \
  --ckpt outputs/runs/main_full_method/gta5_cityscapes/deeplabv3p/0/best.ckpt \
  --out_dir outputs/runs/main_full_method/gta5_cityscapes/deeplabv3p/0 \
  --save_preds --use_target
```

## Aggregate results + figures/tables

```
python uda_lab/scripts/aggregate_results.py --root outputs/runs --out_dir outputs/aggregate
python uda_lab/scripts/make_figures.py --aggregate_csv outputs/aggregate/results.csv --out_dir outputs/figures
python uda_lab/scripts/make_tables.py --aggregate_csv outputs/aggregate/results.csv --out_dir outputs/tables
```

## Unit tests

```
pytest uda_lab/tests
```

## Output format

All runs are saved to:
```
outputs/runs/<exp_name>/<dataset>/<model>/<seed>/
```

Each run directory includes:
- `config.yaml` (resolved config)
- `metrics.json` (epoch + final metrics)
- `best.ckpt` (best validation checkpoint)
- `predictions/` (PNG masks)
- `figures/` (training curves, qualitative grids)
- `logs/` (TensorBoard/W&B)

## Project structure

```
uda_lab/
  configs/  scripts/  src/  tests/  outputs/
```

See `uda_lab/CONTRIBUTIONS.md`, `uda_lab/METHOD_MATH.md`, and
`uda_lab/REPRODUCIBILITY.md` for method details and mathematical formalization.
