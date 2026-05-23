#!/usr/bin/env bash
set -euo pipefail

# One-command Verda runner for the Qwopus3.6/Qwen3.5 BF16 safetensors scan.
# Intended usage on a fresh GPU VM:
#   git clone https://github.com/Whamp/RYS.git RYS && cd RYS
#   REPO_REF=will/qwopus36-bf16-transformers-scan GPU_PRESET=96gb HF_TOKEN=... ./scripts/verda_run_qwopus36_bf16_scan.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

WORKDIR=${WORKDIR:-/workspace/rys-qwopus36-scan}
REPO_DIR=${REPO_DIR:-$DEFAULT_REPO_DIR}
REPO_URL=${REPO_URL:-https://github.com/Whamp/RYS.git}
REPO_REF=${REPO_REF:-}
ALLOW_UNPINNED_REPO=${ALLOW_UNPINNED_REPO:-0}
MODEL_REPO=${MODEL_REPO:-Jackrong/Qwopus3.6-27B-v2}
MODEL_DIR=${MODEL_DIR:-${WORKDIR}/models/Qwopus3.6-27B-v2}
RESULTS_DIR=${RESULTS_DIR:-${WORKDIR}/results/qwopus36-bf16}
LOG_DIR=${LOG_DIR:-${RESULTS_DIR}/logs}
NUM_LAYERS=${NUM_LAYERS:-64}
GPU_PRESET=${GPU_PRESET:-96gb} # h100 | 96gb | h200 | b200
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
INSTALL_FAST_KERNELS=${INSTALL_FAST_KERNELS:-1}
ALLOW_SLOW_LINEAR_ATTENTION=${ALLOW_SLOW_LINEAR_ATTENTION:-0}
SKIP_MODEL_DOWNLOAD=${SKIP_MODEL_DOWNLOAD:-0}
SKIP_UV_SYNC=${SKIP_UV_SYNC:-0}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}

case "${GPU_PRESET}" in
  h100)
    DEFAULT_MATH_BATCH=8
    DEFAULT_EQ_BATCH=4
    ;;
  96gb|rtxpro6000|rtx-pro-6000)
    DEFAULT_MATH_BATCH=16
    DEFAULT_EQ_BATCH=8
    ;;
  h200)
    DEFAULT_MATH_BATCH=32
    DEFAULT_EQ_BATCH=16
    ;;
  b200)
    DEFAULT_MATH_BATCH=32
    DEFAULT_EQ_BATCH=16
    ;;
  *)
    echo "Unknown GPU_PRESET=${GPU_PRESET}. Use h100, 96gb, h200, or b200." >&2
    exit 2
    ;;
esac

MATH_BATCH_SIZE=${MATH_BATCH_SIZE:-$DEFAULT_MATH_BATCH}
EQ_BATCH_SIZE=${EQ_BATCH_SIZE:-$DEFAULT_EQ_BATCH}
MATH_MAX_NEW=${MATH_MAX_NEW:-64}
EQ_MAX_NEW=${EQ_MAX_NEW:-64}
PREFLIGHT_SAMPLES=${PREFLIGHT_SAMPLES:-4}
PREFLIGHT_MAX_NEW=${PREFLIGHT_MAX_NEW:-64}

QUEUE_FILE=${QUEUE_FILE:-${RESULTS_DIR}/queue.json}
COMBINED_RESULTS_FILE=${COMBINED_RESULTS_FILE:-${RESULTS_DIR}/combined_results.pkl}
MATH_RESULTS_FILE=${MATH_RESULTS_FILE:-${RESULTS_DIR}/math_results.pkl}
EQ_RESULTS_FILE=${EQ_RESULTS_FILE:-${RESULTS_DIR}/eq_results.pkl}
ANALYSIS_DIR=${ANALYSIS_DIR:-${RESULTS_DIR}/analysis}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_LOG=${RUN_LOG:-${LOG_DIR}/scan_${RUN_ID}.log}
METADATA_FILE=${METADATA_FILE:-${RESULTS_DIR}/run_metadata_${RUN_ID}.json}
BUNDLE_FILE=${BUNDLE_FILE:-${WORKDIR}/qwopus36-bf16-results_${RUN_ID}.tar.zst}

mkdir -p "${WORKDIR}" "${RESULTS_DIR}" "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

install_uv_if_needed() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

prepare_repo() {
  if [ -z "${REPO_REF}" ] && [ "${ALLOW_UNPINNED_REPO}" != "1" ]; then
    echo "REPO_REF is required so the rented GPU runs a known repo revision." >&2
    echo "Set REPO_REF=<branch-or-commit>, or set ALLOW_UNPINNED_REPO=1 to override." >&2
    exit 2
  fi

  if [ -f "${REPO_DIR}/pyproject.toml" ]; then
    log "Using repo at ${REPO_DIR}"
  else
    log "Cloning repo ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi
  cd "${REPO_DIR}"
  if [ -n "${REPO_REF}" ]; then
    log "Checking out ${REPO_REF}"
    git fetch --all --tags
    git checkout "${REPO_REF}"
  else
    log "ALLOW_UNPINNED_REPO=1; using current checkout without REPO_REF."
  fi
}

install_deps() {
  install_uv_if_needed
  log "Ensuring Python ${PYTHON_VERSION} is available for CUDA extension wheel compatibility..."
  uv python install "${PYTHON_VERSION}"
  export UV_PYTHON="${PYTHON_VERSION}"

  if [ "${SKIP_UV_SYNC}" != "1" ]; then
    log "Installing Python dependencies with uv sync..."
    uv sync
  fi

  if [ "${INSTALL_FAST_KERNELS}" = "1" ]; then
    log "Installing required Qwen3.5/Qwen3.6 linear-attention fast-path deps..."
    uv pip install --upgrade --no-build-isolation causal-conv1d flash-linear-attention
  fi

  verify_fast_path_deps
}

verify_fast_path_deps() {
  log "Verifying Qwen linear-attention fast path availability..."
  set +e
  uv run python - <<'PY'
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

causal_ok = bool(is_causal_conv1d_available())
fla_ok = bool(is_flash_linear_attention_available())
print(f"is_causal_conv1d_available={causal_ok}")
print(f"is_flash_linear_attention_available={fla_ok}")
if causal_ok:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: F401
    print("causal_conv1d imports ok")
if fla_ok:
    from fla.modules import FusedRMSNormGated  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401
    print("flash-linear-attention imports ok")
raise SystemExit(0 if (causal_ok and fla_ok) else 1)
PY
  status=$?
  set -e
  if [ "$status" -ne 0 ]; then
    if [ "${ALLOW_SLOW_LINEAR_ATTENTION}" = "1" ]; then
      log "Fast-path deps unavailable; ALLOW_SLOW_LINEAR_ATTENTION=1 so continuing with slow Transformers fallbacks."
    else
      echo "Qwen linear-attention fast-path deps are unavailable." >&2
      echo "This would force slow Transformers fallbacks for most layers." >&2
      echo "Fix the install, or set ALLOW_SLOW_LINEAR_ATTENTION=1 to override intentionally." >&2
      exit 3
    fi
  fi
}

download_model() {
  if [ "${SKIP_MODEL_DOWNLOAD}" = "1" ] || [ -f "${MODEL_DIR}/config.json" ]; then
    log "Using existing model dir ${MODEL_DIR}"
    return
  fi
  mkdir -p "${MODEL_DIR}"
  log "Downloading ${MODEL_REPO} -> ${MODEL_DIR}"
  if [ -n "${HF_TOKEN:-}" ]; then
    uv run hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}" --token "${HF_TOKEN}"
  else
    uv run hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
  fi
}

write_metadata() {
  log "Writing run metadata -> ${METADATA_FILE}"
  local git_commit="unknown"
  local git_status="unknown"
  if git -C "${REPO_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)
    git_status=$(git -C "${REPO_DIR}" status --short 2>/dev/null | sed ':a;N;$!ba;s/\n/\\n/g')
  fi
  local nvidia_smi="unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia_smi=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed ':a;N;$!ba;s/\n/\\n/g' || echo unavailable)
  fi
  uv run python - <<PY
import json, os, platform, subprocess
payload = {
    "run_id": os.environ.get("RUN_ID"),
    "model_repo": os.environ.get("MODEL_REPO"),
    "model_dir": os.environ.get("MODEL_DIR"),
    "results_dir": os.environ.get("RESULTS_DIR"),
    "gpu_preset": os.environ.get("GPU_PRESET"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "math_batch_size": int(os.environ.get("MATH_BATCH_SIZE", "0")),
    "eq_batch_size": int(os.environ.get("EQ_BATCH_SIZE", "0")),
    "math_max_new": int(os.environ.get("MATH_MAX_NEW", "0")),
    "eq_max_new": int(os.environ.get("EQ_MAX_NEW", "0")),
    "queue_file": os.environ.get("QUEUE_FILE"),
    "combined_results_file": os.environ.get("COMBINED_RESULTS_FILE"),
    "math_results_file": os.environ.get("MATH_RESULTS_FILE"),
    "eq_results_file": os.environ.get("EQ_RESULTS_FILE"),
    "analysis_dir": os.environ.get("ANALYSIS_DIR"),
    "run_log": os.environ.get("RUN_LOG"),
    "repo_dir": os.environ.get("REPO_DIR"),
    "repo_url": os.environ.get("REPO_URL"),
    "repo_ref": os.environ.get("REPO_REF"),
    "git_commit": os.environ.get("GIT_COMMIT_CAPTURE", "unknown"),
    "git_status_short": os.environ.get("GIT_STATUS_CAPTURE", "unknown"),
    "nvidia_smi": os.environ.get("NVIDIA_SMI_CAPTURE", "unavailable"),
    "python": platform.python_version(),
    "platform": platform.platform(),
}
try:
    payload["uv_pip_freeze"] = subprocess.check_output(["uv", "pip", "freeze"], text=True, stderr=subprocess.STDOUT).splitlines()
except Exception as exc:
    payload["uv_pip_freeze_error"] = str(exc)
path = os.environ["METADATA_FILE"]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

init_queue() {
  log "Initializing/resuming default full queue (${NUM_LAYERS} layers)."
  uv run python scripts/init_queue.py \
    --num-layers "${NUM_LAYERS}" \
    --queue-file "${QUEUE_FILE}" \
    --results-file "${COMBINED_RESULTS_FILE}" \
    --skip-existing "${COMBINED_RESULTS_FILE}" "${MATH_RESULTS_FILE}" "${EQ_RESULTS_FILE}"
}

run_scan() {
  log "GPU inventory:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi

  log "Starting scan. Log: ${RUN_LOG}"
  export CUDA_VISIBLE_DEVICES
  export LEVELGEN_TEXT_LOADER=causal
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

  PREFLIGHT_ARGS=()
  if [ "${SKIP_PREFLIGHT}" = "1" ]; then
    PREFLIGHT_ARGS+=(--skip-preflight)
  fi

  set -o pipefail
  uv run python scripts/run_transformers_math_eq_combined_worker.py \
    --queue-file "${QUEUE_FILE}" \
    --combined-results-file "${COMBINED_RESULTS_FILE}" \
    --math-results-file "${MATH_RESULTS_FILE}" \
    --eq-results-file "${EQ_RESULTS_FILE}" \
    --model-path "${MODEL_DIR}" \
    --math-dataset-path datasets/math_16.json \
    --eq-dataset-path datasets/eq_16.json \
    --math-batch-size "${MATH_BATCH_SIZE}" \
    --eq-batch-size "${EQ_BATCH_SIZE}" \
    --math-max-new "${MATH_MAX_NEW}" \
    --eq-max-new "${EQ_MAX_NEW}" \
    --padding-mode masked \
    --adaptive-batch-retry \
    --min-batch-size 1 \
    --max-retries-per-phase 8 \
    --attention-impl eager \
    --device-map cuda:0 \
    --trust-remote-code \
    --local-files-only \
    --force-causal-loader \
    --preflight-samples "${PREFLIGHT_SAMPLES}" \
    --preflight-max-new "${PREFLIGHT_MAX_NEW}" \
    "${PREFLIGHT_ARGS[@]}" \
    2>&1 | tee "${RUN_LOG}"
}

analyze_results() {
  log "Analyzing results -> ${ANALYSIS_DIR}"
  uv run python scripts/analyze_results.py \
    --math-scores "${MATH_RESULTS_FILE}" \
    --eq-scores "${EQ_RESULTS_FILE}" \
    --out-dir "${ANALYSIS_DIR}" \
    --num-layers "${NUM_LAYERS}"
}

bundle_results() {
  log "Bundling results -> ${BUNDLE_FILE}"
  mkdir -p "$(dirname "${BUNDLE_FILE}")"
  if command -v zstd >/dev/null 2>&1; then
    tar --zstd -cf "${BUNDLE_FILE}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")"
  else
    local fallback="${BUNDLE_FILE%.zst}.gz"
    log "zstd not found; writing gzip bundle -> ${fallback}"
    tar -czf "${fallback}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")"
    BUNDLE_FILE="${fallback}"
  fi
  log "Bundle ready: ${BUNDLE_FILE}"
}

main() {
  prepare_repo
  install_deps
  download_model
  export RUN_ID MODEL_REPO MODEL_DIR RESULTS_DIR GPU_PRESET CUDA_VISIBLE_DEVICES
  export MATH_BATCH_SIZE EQ_BATCH_SIZE MATH_MAX_NEW EQ_MAX_NEW
  export QUEUE_FILE COMBINED_RESULTS_FILE MATH_RESULTS_FILE EQ_RESULTS_FILE ANALYSIS_DIR RUN_LOG
  export REPO_DIR REPO_URL REPO_REF METADATA_FILE
  export GIT_COMMIT_CAPTURE="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  export GIT_STATUS_CAPTURE="$(git -C "${REPO_DIR}" status --short 2>/dev/null || true)"
  export NVIDIA_SMI_CAPTURE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo unavailable)"
  write_metadata
  init_queue
  run_scan
  analyze_results
  write_metadata
  bundle_results
  log "Done. Results: ${RESULTS_DIR}"
}

main "$@"
