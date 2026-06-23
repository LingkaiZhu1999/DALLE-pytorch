#!/bin/bash -l
#SBATCH --job-name=dalle-laion400m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Submit a smoke test:
#   sbatch scripts/train_laion400m.sh
#
# Use a specific site Python module if needed:
#   sbatch --export=ALL,PYTHON_MODULE=scicomp-python-env/2025.2 \
#     scripts/train_laion400m.sh
#
# Submit a full local-mirror epoch (command-line options override #SBATCH):
#   sbatch --time=5-00:00:00 \
#     --export=ALL,DATASET_SIZE=268836185,WANDB_MODE=offline \
#     scripts/train_laion400m.sh
#
# If your site requires a partition or account, add them to sbatch:
#   sbatch --partition=<partition> --account=<account> \
#     scripts/train_laion400m.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/laion400m-pilot}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
# One sample per GPU keeps the 1,280-token full-attention pilot conservative.
# Accumulating 32 microbatches preserves an effective global batch of 64.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
# Safe default: validate one short run before spending a full allocation.
DATASET_SIZE="${DATASET_SIZE:-4096}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
PYTHON_MODULE="${PYTHON_MODULE:-scicomp-python-env}"

# Use the site-managed scientific Python environment. This module prepends the
# environment's bin directory to PATH, so python/deepspeed come from there
# without conda/mamba activation or a repo-local venv.
module load "$PYTHON_MODULE"

if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
  echo "GPUS_PER_NODE must be an integer; got: $GPUS_PER_NODE" >&2
  exit 2
fi

if (( GLOBAL_BATCH_SIZE % GPUS_PER_NODE != 0 )); then
  echo "GLOBAL_BATCH_SIZE ($GLOBAL_BATCH_SIZE) must be divisible by GPUS_PER_NODE ($GPUS_PER_NODE)" >&2
  exit 2
fi

command -v deepspeed >/dev/null || {
  echo "deepspeed is not available after loading module: $PYTHON_MODULE" >&2
  exit 127
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_DIR/.matplotlib}"

echo "job_id=${SLURM_JOB_ID:-interactive} host=$(hostname) gpus=$GPUS_PER_NODE dataset_size=$DATASET_SIZE"
echo "repo=$REPO_ROOT output=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
cd "$OUTPUT_DIR"

deepspeed --num_gpus "$GPUS_PER_NODE" "$REPO_ROOT/train_dalle.py" \
  --image_text_folder "$DATA_ROOT" \
  --wds jpg,txt \
  --wds_dataset_size "$DATASET_SIZE" \
  --wds_num_workers "$NUM_WORKERS" \
  --wds_shuffle_buffer 10000 \
  --truncate_captions \
  --epochs 1 \
  --batch_size "$GLOBAL_BATCH_SIZE" \
  --ga_steps "$GRAD_ACCUMULATION_STEPS" \
  --learning_rate 3e-4 \
  --dim 1024 \
  --depth 24 \
  --heads 16 \
  --dim_head 64 \
  --attn_types full \
  --reversible \
  --fp16 \
  --stable_softmax \
  --zero_stage 2 \
  --save_every_n_steps 1000 \
  --keep_n_checkpoints 3 \
  --dalle_output_file_name dalle-laion400m-pilot \
  --wandb_name dalle-laion400m \
  --wandb_mode "$WANDB_MODE" \
  --deepspeed
