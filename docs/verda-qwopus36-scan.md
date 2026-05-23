# Verda Qwopus3.6 BF16 scan runbook

This runbook covers the one-time BF16 safetensors scan for:

- Model: `Jackrong/Qwopus3.6-27B-v2`
- Architecture: Qwen3.5/Qwen3.6 text stack (`qwen3_5`), 64 layers
- Scan: full single-block `(i, j)` sweep on `math_16 + eq_16`
- Runner: `scripts/verda_run_qwopus36_bf16_scan.sh`

The GPU run scans and analyzes only. Export shortlisted checkpoints later on CPU/disk.

## GPU choice

Recommended presets:

| GPU | Preset | Initial batches | Notes |
|---|---|---:|---|
| RTX Pro 6000 96GB | `96gb` | Math 16 / EQ 8 | Default target |
| H200 141GB | `h200` | Math 32 / EQ 16 | Best paid speed/margin upgrade |
| B200 180GB+ | `b200` | Math 32 / EQ 16 | Usually overkill unless price is close |
| H100 80GB | `h100` | Math 8 / EQ 4 | Likely fits only with text-only loader; less margin |

The runner forces the text-only Transformers loader with `LEVELGEN_TEXT_LOADER=causal`.

## Fresh instance setup

```bash
mkdir -p /workspace
cd /workspace

export REPO_REF=<branch-or-commit-containing-these-scripts>
git clone https://github.com/Whamp/RYS.git
cd RYS
git fetch --all --tags
git checkout "$REPO_REF"
```

Push the scanner branch/commit first, then set `REPO_REF`. The runner refuses to start without a pinned repo revision unless `ALLOW_UNPINNED_REPO=1` is set. Check out the ref before running the script so the script itself exists locally.

## Run

For the default 96GB class GPU:

```bash
export HF_TOKEN=<your-huggingface-token-if-needed>
export REPO_REF=<branch-or-commit-containing-these-scripts>
export GPU_PRESET=96gb
./scripts/verda_run_qwopus36_bf16_scan.sh
```

For H200:

```bash
export HF_TOKEN=<your-huggingface-token-if-needed>
export REPO_REF=<branch-or-commit-containing-these-scripts>
export GPU_PRESET=h200
./scripts/verda_run_qwopus36_bf16_scan.sh
```

Useful overrides:

```bash
export WORKDIR=/workspace/rys-qwopus36-scan
export MODEL_DIR=$WORKDIR/models/Qwopus3.6-27B-v2
export RESULTS_DIR=$WORKDIR/results/qwopus36-bf16
export MATH_BATCH_SIZE=16
export EQ_BATCH_SIZE=8
export PYTHON_VERSION=3.12
export INSTALL_FAST_KERNELS=1
export ALLOW_SLOW_LINEAR_ATTENTION=0
```

## What the runner does

1. Installs `uv` if missing.
2. Runs `uv sync`.
3. Installs and verifies Qwen linear-attention fast-path deps: `causal-conv1d` and `flash-linear-attention`.
4. Downloads `Jackrong/Qwopus3.6-27B-v2` to `$MODEL_DIR`.
5. Initializes/resumes the default 64-layer queue.
6. Runs `scripts/run_transformers_math_eq_combined_worker.py`:
   - text-only `AutoModelForCausalLM`
   - masked padding only
   - separate Math/EQ batches
   - preflight baseline + sampled worst-span check
   - incremental results writes
7. Runs `scripts/analyze_results.py`.
8. Writes metadata and bundles results.

## Outputs

Default locations:

```text
/workspace/rys-qwopus36-scan/results/qwopus36-bf16/
  queue.json
  combined_results.pkl
  math_results.pkl
  eq_results.pkl
  analysis/
  logs/
  run_metadata_<RUN_ID>.json

/workspace/rys-qwopus36-scan/qwopus36-bf16-results_<RUN_ID>.tar.zst
```

If `zstd` is unavailable, the bundle falls back to `.tar.gz`.

## Resume

The queue and result pickles are resume-safe.

If the run stops, rerun the same command with the same `WORKDIR`/`RESULTS_DIR`:

```bash
cd /workspace/RYS
export REPO_REF=<same-branch-or-commit-as-before>
git fetch --all --tags
git checkout "$REPO_REF"
export GPU_PRESET=96gb
./scripts/verda_run_qwopus36_bf16_scan.sh
```

`init_queue.py --skip-existing` will skip completed configs and rebuild the remaining queue.

## Expected safety checks

The scanner should print:

- `Loader: AutoModelForCausalLM`
- `Padding mode: masked`
- text layer count `64`
- preflight baseline scores
- preflight worst-span sampled scores

Stop and investigate if:

- fast-path dependency verification fails, unless you intentionally set `ALLOW_SLOW_LINEAR_ATTENTION=1`
- loader is not `AutoModelForCausalLM`
- padding mode is not `masked`
- preflight crashes on the worst-span config
- outputs are empty/garbage for baseline
- adaptive batching falls all the way to batch 1 and remains unstable

## Pull results before shutting down

From local machine:

```bash
scp user@verda-host:/workspace/rys-qwopus36-scan/qwopus36-bf16-results_*.tar.* .
```

Also consider copying the full results directory if you want raw logs before compression:

```bash
rsync -avP user@verda-host:/workspace/rys-qwopus36-scan/results/qwopus36-bf16/ ./qwopus36-bf16/
```

## Export later

After inspecting top balanced configs, export on any machine with enough disk:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id Jackrong/Qwopus3.6-27B-v2 \
  --output exports/qwopus36-block-START-END \
  --blocks "START,END"
```

Export is CPU/disk I/O work, not GPU-intensive. It preserves the full model config and non-text tensors, including the vision tower.
