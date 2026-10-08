#!/usr/bin/env bash
# VPTQ paper (arXiv:2409.17066v2), Appendix B, Table 8.
# Reported effective bits refer to Transformer blocks, not the entire checkpoint.
# Run in Linux/WSL with the repository's vptq-algo environment activated.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/quantize_2bit.sh [PRESET] [--dry-run]

Default: llama31-8b-instruct-2bit
  Model: /data/LLMs/Llama-3.1-8B-Instruct
  Hessian: /data/Hessians/Llama-3.1-8B-Instruct-6144-8k
  Inverse Hessian: /data/Hessians/Llama-3.1-8B-Instruct-6144-8k/InvHessians-Llama-31-8B-Instruct-6144-8k
  Quantization parameters transferred from Table 8's llama3-8b-2.08 row.

Presets from Table 8:
  llama2-7b-2.02    llama2-7b-2.26
  llama2-13b-2.02   llama2-13b-2.18
  llama2-70b-2.07   llama2-70b-2.11
  llama3-8b-2.08    llama3-8b-2.24
  llama3-70b-2.02   llama3-70b-2.07

Environment:
  HESSIAN_PATH       Directory of matching {layer}_{qkv,o,up,down}.pt files.
  INV_HESSIAN_PATH   Directory of matching precomputed invH/perm/zero_idx files.
  NUM_GPUS           Default 1. Multi-GPU runs use physical GPUs 0..NUM_GPUS-1.
  CUDA_VISIBLE_DEVICES  Single-GPU selection; must be 0..NUM_GPUS-1 for multi-GPU.
  MODEL_NAME         Optional model ID/local path, with the same architecture/size.
                     Hessians must come from exactly the selected model.
  OUTPUT_DIR         Default outputs/paper-2bit/PRESET (runner adds a timestamp).
  PYTHON_BIN         Default python; executable path, without extra arguments.
  KITER / KTOL       Default 100 / 1e-5 (repository settings, not Table 8).
  SEQ_LEN / SEED     Default 8192 / 0 for the Llama-3.1 preset (2048 / 0 otherwise).
                     Does not recollect calibration statistics.

--dry-run prints the command without requiring Python, CUDA or Hessian files.
See scripts/quantize_2bit.md for parameter meanings and reproduction limits.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

PRESET=llama31-8b-instruct-2bit
DRY_RUN=0
PRESET_SET=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1 ;;
        --*) die "Unknown option: $arg" ;;
        *)
            (( PRESET_SET == 0 )) || die 'Specify only one preset.'
            PRESET="$arg"
            PRESET_SET=1
            ;;
    esac
done

# Columns: outlier %, outlier v, outlier K, main v, main K, residual K, groups.
# -1 disables outlier/residual quantization where the paper shows a dash.
case "$PRESET" in
    # Target model uses the original LLaMA-3 8B 2.08-bit row's settings.
    llama31-8b-instruct-2bit) PARAMS=(1 4 4096 12 4096 4096 1) ;;
    llama2-7b-2.02)  PARAMS=(0 -1 -1 6 4096 -1 1) ;;
    llama2-7b-2.26)  PARAMS=(1 4 8192 12 4096 4096 4) ;;
    llama2-13b-2.02) PARAMS=(0 -1 -1 6 4096 -1 1) ;;
    llama2-13b-2.18) PARAMS=(2 4 8192 12 4096 4096 4) ;;
    llama2-70b-2.07) PARAMS=(1 4 8192 12 4096 4096 4) ;;
    llama2-70b-2.11) PARAMS=(1 4 8192 12 4096 4096 8) ;;
    llama3-8b-2.08)  PARAMS=(1 4 4096 12 4096 4096 1) ;;
    llama3-8b-2.24)  PARAMS=(1 4 8192 6 4096 -1 16) ;;
    llama3-70b-2.02) PARAMS=(0 -1 -1 12 4096 4096 1) ;;
    llama3-70b-2.07) PARAMS=(1 4 4096 6 4096 -1 16) ;;
    *) die "Unknown preset: $PRESET. Use --help to list presets." ;;
esac

case "$PRESET" in
    llama31-8b-instruct-2bit) DEFAULT_MODEL=/data/LLMs/Llama-3.1-8B-Instruct; LAYERS=32; MODEL_TAG=llama31-8b-instruct ;;
    llama2-7b-*)  DEFAULT_MODEL=meta-llama/Llama-2-7b-hf; LAYERS=32; MODEL_TAG=llama2-7b ;;
    llama2-13b-*) DEFAULT_MODEL=meta-llama/Llama-2-13b-hf; LAYERS=40; MODEL_TAG=llama2-13b ;;
    llama2-70b-*) DEFAULT_MODEL=meta-llama/Llama-2-70b-hf; LAYERS=80; MODEL_TAG=llama2-70b ;;
    llama3-8b-*)  DEFAULT_MODEL=meta-llama/Meta-Llama-3-8B; LAYERS=32; MODEL_TAG=llama3-8b ;;
    llama3-70b-*) DEFAULT_MODEL=meta-llama/Meta-Llama-3-70B; LAYERS=80; MODEL_TAG=llama3-70b ;;
esac

DEFAULT_HESSIAN_PATH="data/hessians/$MODEL_TAG"
DEFAULT_INV_HESSIAN_PATH="data/inv-hessians/$MODEL_TAG"
DEFAULT_SEQ_LEN=2048
PAPER_BITS="${PRESET##*-}"
if [[ "$PRESET" == llama31-8b-instruct-2bit ]]; then
    # The supplied Hessian path repeated the same absolute directory twice;
    # use one copy of that directory.
    DEFAULT_HESSIAN_PATH=/data/Hessians/Llama-3.1-8B-Instruct-6144-8k/Hessians-Llama-31-8B-Instruct-6144-8k
    DEFAULT_INV_HESSIAN_PATH=/data/Hessians/Llama-3.1-8B-Instruct-6144-8k/InvHessians-Llama-31-8B-Instruct-6144-8k
    DEFAULT_SEQ_LEN=8192
    PAPER_BITS=2.08
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_DIR"

MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
HESSIAN_PATH="${HESSIAN_PATH:-$DEFAULT_HESSIAN_PATH}"
INV_HESSIAN_PATH="${INV_HESSIAN_PATH:-$DEFAULT_INV_HESSIAN_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper-2bit/$PRESET}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
KITER="${KITER:-100}"
KTOL="${KTOL:-1e-5}"
SEQ_LEN="${SEQ_LEN:-$DEFAULT_SEQ_LEN}"
SEED="${SEED:-0}"

[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || die 'NUM_GPUS must be a positive integer without leading zeros.'
(( NUM_GPUS <= LAYERS )) || die "NUM_GPUS must not exceed $LAYERS for this preset."
[[ "$KITER" =~ ^[1-9][0-9]*$ ]] || die 'KITER must be a positive integer without leading zeros.'
[[ "$SEQ_LEN" =~ ^[1-9][0-9]*$ ]] || die 'SEQ_LEN must be a positive integer without leading zeros.'
[[ "$SEED" =~ ^[0-9]+$ ]] || die 'SEED must be a non-negative integer.'

# models/llama.py sets worker CUDA_VISIBLE_DEVICES to str(gpu_idx).
# Preserve a single-GPU selection; require matching physical IDs for multi-GPU.
GPU_LIST=0
for (( gpu=1; gpu<NUM_GPUS; gpu++ )); do GPU_LIST+=",$gpu"; done
if (( NUM_GPUS > 1 )) && [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$GPU_LIST" ]]; then
    die "This runner requires CUDA_VISIBLE_DEVICES=$GPU_LIST for NUM_GPUS=$NUM_GPUS."
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_LIST}"

COMMAND=(
    "$PYTHON_BIN" -u run_vptq.py
    --model_name "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --vector_lens "${PARAMS[1]}" "${PARAMS[3]}"
    --num_centroids "${PARAMS[2]}" "${PARAMS[4]}"
    --num_res_centroids -1 "${PARAMS[5]}"
    --npercent "${PARAMS[0]}"
    --group_num "${PARAMS[6]}"
    --group_size -1
    --blocksize 128
    --quant_step 1
    --percdamp 0.01
    --kmeans_mode hessian
    --kiter "$KITER"
    --ktol "$KTOL"
    --num_gpus "$NUM_GPUS"
    --seq_len "$SEQ_LEN"
    --seed "$SEED"
    --enable_perm
    --enable_norm
    --norm_dim 0
    --hessian_path "$HESSIAN_PATH"
    --inv_hessian_path "$INV_HESSIAN_PATH"
    --save_packed_model
    --new_eval
)
# Disk-based layer transfer avoids large multiprocessing queues.
# The current single-GPU runner does not reload --save_qlinear files.
if (( NUM_GPUS > 1 )); then COMMAND+=(--save_qlinear); fi

printf 'Preset: %s; source Table 8 row effective bits: %s\n' "$PRESET" "$PAPER_BITS"
printf 'Table 8: N%%=%s, v0=%s, k0=%s, v1=%s, k1=%s, k2=%s, groups=%s\n' "${PARAMS[@]}"
printf 'CUDA_VISIBLE_DEVICES=%s\nCommand:' "$CUDA_VISIBLE_DEVICES"
printf ' %q' "${COMMAND[@]}"
printf '\n'
if (( DRY_RUN )); then exit 0; fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
[[ -d "$HESSIAN_PATH" ]] || die "Set HESSIAN_PATH to this model's Hessian directory: $HESSIAN_PATH"
[[ -d "$INV_HESSIAN_PATH" ]] || die "Set INV_HESSIAN_PATH to the matching precomputed directory: $INV_HESSIAN_PATH"
for (( layer=0; layer<LAYERS; layer++ )); do
    for kind in qkv o up down; do
        for directory in "$HESSIAN_PATH" "$INV_HESSIAN_PATH"; do
            [[ -s "$directory/${layer}_${kind}.pt" ]] || die "Missing/empty file: $directory/${layer}_${kind}.pt"
        done
    done
done

"$PYTHON_BIN" - "$NUM_GPUS" "$KTOL" <<'PY'
import math
import sys

import cuml
import cupy
import flash_attn
import sentence_transformers
import torch
import transformers

requested = int(sys.argv[1])
tolerance = float(sys.argv[2])
if not math.isfinite(tolerance) or tolerance <= 0:
    raise SystemExit("KTOL must be positive and finite.")
if not torch.cuda.is_available() or torch.cuda.device_count() < requested:
    raise SystemExit(f"Need {requested} visible CUDA GPUs; found {torch.cuda.device_count()}.")
print(f"Preflight passed: torch={torch.__version__}, visible GPUs={torch.cuda.device_count()}")
PY

printf 'Output: %s/<timestamp>/packed_model/\n' "$OUTPUT_DIR"
exec "${COMMAND[@]}"
