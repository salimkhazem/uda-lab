#!/usr/bin/env bash
set -e

DDP=0
GPUS=1
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ddp) DDP=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) shift ;;
  esac
done

DATASETS=("drive" "stare" "chase" "kvasir" "retina" "deepglobe" "spacenet" "ssdd" "gta5_cityscapes")
MODELS=("unet" "deeplabv3p" "segformer" "swin_unet")

contains_dataset() {
  local target="$1"
  local list="$2"
  IFS=',' read -ra items <<< "$list"
  for item in "${items[@]}"; do
    if [[ "$item" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

has_dataset() {
  if [[ -n "$HF_DATASETS" ]]; then
    if contains_dataset "$1" "$HF_DATASETS"; then
      return 0
    fi
  fi
  case "$1" in
    drive) [[ -d "data/drive/images/train" && -d "data/drive/masks/train" ]] ;;
    stare) [[ -d "data/stare/images/train" && -d "data/stare/masks/train" ]] ;;
    chase) [[ -d "data/chase/images/train" && -d "data/chase/masks/train" ]] ;;
    kvasir) [[ -d "data/kvasir/images/train" && -d "data/kvasir/masks/train" ]] ;;
    retina) [[ -d "data/retina/images/train" && -d "data/retina/masks/train" ]] ;;
    deepglobe) [[ -d "data/deepglobe/images/train" && -d "data/deepglobe/masks/train" ]] ;;
    spacenet) [[ -d "data/spacenet/images/train" && -d "data/spacenet/masks/train" ]] ;;
    ssdd) [[ -d "data/ssdd/images/train" && -d "data/ssdd/masks/train" ]] ;;
    gta5_cityscapes) [[ -d "data/gta5/images/train" && -d "data/gta5/labels/train" && -d "data/cityscapes/images/train" && -d "data/cityscapes/labels/train" ]] ;;
    *) return 1 ;;
  esac
}

ABLATIONS=(
  "ablation_no_diffusion"
  "ablation_no_topoloss"
  "ablation_no_pseudofilter"
  "ablation_no_adv"
)

for DATASET in "${DATASETS[@]}"; do
  if ! has_dataset "$DATASET"; then
    echo "Skipping $DATASET: dataset not found."
    continue
  fi
  DATASET_ARG="$DATASET"
  if [[ -n "$HF_DATASETS" ]] && contains_dataset "$DATASET" "$HF_DATASETS"; then
    case "$DATASET" in
      drive) DATASET_ARG="drive_hf" ;;
      stare) DATASET_ARG="stare_hf" ;;
      chase) DATASET_ARG="chase_hf" ;;
      *) DATASET_ARG="$DATASET" ;;
    esac
  fi
  for MODEL in "${MODELS[@]}"; do
    for EXP in "${ABLATIONS[@]}"; do
      if [[ "$DDP" -eq 1 ]]; then
        torchrun --nproc_per_node "$GPUS" scripts/adapt.py \
          --config "configs/experiments/${EXP}.yaml" \
          --dataset "$DATASET_ARG" --model "$MODEL" --seed "$SEED"
      else
        python scripts/adapt.py \
          --config "configs/experiments/${EXP}.yaml" \
          --dataset "$DATASET_ARG" --model "$MODEL" --seed "$SEED"
      fi
    done

    for TOPW in 0.0 0.05 0.1 0.2; do
      python scripts/adapt.py \
        --config configs/experiments/ablation_vary_topoweight.yaml \
        --dataset "$DATASET_ARG" --model "$MODEL" --seed "$SEED" \
        --opts uda.topo_weight="$TOPW" exp.name="ablation_topo_${TOPW}"
    done

    for RATIO in 0.0 0.25 0.5 1.0; do
      python scripts/adapt.py \
        --config configs/experiments/ablation_vary_gen_amount.yaml \
        --dataset "$DATASET_ARG" --model "$MODEL" --seed "$SEED" \
        --opts uda.diffusion_ratio="$RATIO" exp.name="ablation_gen_${RATIO}"
    done
  done
done
