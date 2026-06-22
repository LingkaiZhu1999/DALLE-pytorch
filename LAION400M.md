# Training a DALL-E 1-style model on the local LAION-400M mirror

The local mirror is already in the format expected by `train_dalle.py`:

- shards: `/scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data/*.tar`
- image field: `jpg`
- caption field: `txt`
- successful image-caption pairs reported by the mirror: `268,836,185`
- image resolution: 256 x 256

The trainer uses the released OpenAI discrete VAE when `--vae_path` and
`--taming` are omitted. This gives 1,024 image tokens per image and is the
closest path in this repository to the original DALL-E tokenizer.

## Environment

This repository is from an older PyTorch ecosystem. Use a dedicated environment
and test it on a compute node before requesting a long allocation. At minimum it
needs the package itself, WebDataset, W&B, and DeepSpeed:

```bash
python -m pip install -e .
python -m pip install wandb deepspeed
```

The OpenAI VAE weights may need to be downloaded on the first run. Prime the
cache on a node with network access, or arrange the cache before submitting a
network-isolated job.

## Pilot run (recommended first)

The checked-in launcher is a roughly 300M-parameter, 24-layer pilot and is ready
for Slurm. Its safe default is a 4,096-sample smoke test on one node with two
`h200-141g-ellis` GPUs:

```bash
sbatch scripts/train_laion400m.sh
```

Monitor it with:

```bash
squeue -j <job-id>
tail -f slurm-dalle-laion400m-<job-id>.out
```

If the Python environment is not active in batch jobs, pass its path explicitly:

```bash
sbatch --export=ALL,ENV_PATH=/path/to/mamba/env scripts/train_laion400m.sh
```

After loading, loss, checkpointing, resume, and generation pass, submit a full
local-mirror epoch:

```bash
sbatch --time=5-00:00:00 \
  --export=ALL,DATASET_SIZE=268836185,WANDB_MODE=offline \
  scripts/train_laion400m.sh
```

Add site-specific `--partition` or `--account` options to `sbatch` when required.
Command-line resource options override the defaults embedded in the script. At
the default global microbatch of 2 with 32-step gradient accumulation, one local
mirror epoch is about 134.4 million forward/backward microbatches and 4.20
million optimizer steps.

`--batch_size` is the global microbatch across all ranks. It must be divisible
by the number of ranks. `--ga_steps` increases the effective optimization batch:

```text
effective batch = batch_size * ga_steps
```

The data loader gives each distributed rank disjoint shards, splits those among
its loader workers, shuffles samples with a bounded buffer, and forces every
rank to take the same number of batches.

Dense `full` attention uses PyTorch scaled-dot-product attention. With the
launcher's FP16 tensors and 64-dimensional heads, PyTorch dispatches to its
FlashAttention CUDA backend on supported GPUs such as H200. Sparse and axial
attention variants retain their specialized implementations. PyTorch SDPA
applies the standard attention scaling internally and requires PyTorch 2.0 or
newer.

## Paper-scale architecture

The commonly cited original DALL-E configuration is approximately 12B
parameters: width 3968, 64 transformer layers, and 62 attention heads. The
corresponding model flags in this implementation are:

```text
--dim 3968 --depth 64 --heads 62 --dim_head 64 --reversible
```

That is not a scale-up of the pilot that will fit on one ordinary GPU node. It
requires a carefully benchmarked multi-node plan, ZeRO-3 or equivalent
sharding, activation-memory work, a tuned sparse-attention stack, and a storage
staging strategy. This repository's 2021 DeepSpeed integration should be treated
as a starting point rather than a turnkey reproduction environment at 12B.

For a credible reproduction, preserve the 256 text tokens, 1,024 image tokens,
8,192-entry visual vocabulary, image-loss weight 7, and an alternating sparse
attention pattern. First demonstrate matching data throughput and stable loss on
the pilot; only then lock the paper-scale system design.

## Resume

DeepSpeed writes checkpoints under a `*-ds-cp` directory. Resume with the base
`.pt` name, for example:

```bash
deepspeed --num_gpus 8 train_dalle.py \
  --dalle_path runs/laion400m-pilot/dalle-laion400m-pilot.pt \
  --image_text_folder /scratch/shareddata/dldata/laion400M/img2dataset/laion400m-data \
  --wds jpg,txt --wds_dataset_size 268836185 \
  --batch_size 64 --fp16 --zero_stage 2 --deepspeed
```

Pass the same model architecture flags used for the original run. Model
hyperparameters are restored from the checkpoint, but data and optimizer launch
settings still come from the command line.
