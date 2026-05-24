#!/usr/bin/env bash
set -euo pipefail

# One-command Verda runner for AutoRound W4A16 quantization of the Balanced RYS model.
# Intended usage on a fresh GPU VM:
#   git clone https://github.com/Whamp/RYS.git RYS && cd RYS
#   REPO_REF=will/qwopus36-autoround-balanced-verda HF_TOKEN=... ./scripts/verda_run_qwopus36_autoround_balanced.sh
#
# For spot instances, create the VM with keep_detached volume policy so setup/model cache survive reclaim:
#   --is-spot --os-volume-on-spot-discontinue keep_detached
# AutoRound itself is not assumed to be resumable mid-run; if spot is reclaimed, rerun this script.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

WORKDIR=${WORKDIR:-/workspace/rys-qwopus36-autoround-balanced}
REPO_DIR=${REPO_DIR:-$DEFAULT_REPO_DIR}
REPO_URL=${REPO_URL:-https://github.com/Whamp/RYS.git}
REPO_REF=${REPO_REF:-}
ALLOW_UNPINNED_REPO=${ALLOW_UNPINNED_REPO:-0}

MODEL_REPO=${MODEL_REPO:-hampsonw/Qwopus3.6-27B-v2-RYS-Balanced}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKDIR}/outputs}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16}
LOG_DIR=${LOG_DIR:-${WORKDIR}/logs}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_LOG=${RUN_LOG:-${LOG_DIR}/autoround_balanced_${RUN_ID}.log}
METADATA_FILE=${METADATA_FILE:-${LOG_DIR}/autoround_balanced_metadata_${RUN_ID}.json}
BUNDLE_FILE=${BUNDLE_FILE:-${WORKDIR}/Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16_${RUN_ID}.tar.zst}

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SKIP_UV_SYNC=${SKIP_UV_SYNC:-1}
INSTALL_FAST_KERNELS=${INSTALL_FAST_KERNELS:-0}
ALLOW_SLOW_LINEAR_ATTENTION=${ALLOW_SLOW_LINEAR_ATTENTION:-1}
ALLOW_SMALL_GPU=${ALLOW_SMALL_GPU:-0}
MIN_GPU_MEMORY_MIB=${MIN_GPU_MEMORY_MIB:-90000}

# Pin AutoRound because Qwen3.5/Qwen3.6 hybrid-attention support is moving quickly.
AUTOROUND_REF=${AUTOROUND_REF:-09bc743716e816bf59597409b394dc6d08113ba6}
AUTOROUND_SPEC=${AUTOROUND_SPEC:-auto-round @ git+https://github.com/intel/auto-round.git@${AUTOROUND_REF}}

AUTOROUND_SCHEME=${AUTOROUND_SCHEME:-W4A16}
AUTOROUND_FORMAT=${AUTOROUND_FORMAT:-auto_round}
AUTOROUND_IGNORE_LAYERS=${AUTOROUND_IGNORE_LAYERS:-visual,vision,lm_head,mtp.fc,linear_attn}
ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-0}
EXTRA_AUTOROUND_ARGS=${EXTRA_AUTOROUND_ARGS:-}

AUTO_UPLOAD=${AUTO_UPLOAD:-1}
HF_UPLOAD_REPO=${HF_UPLOAD_REPO:-hampsonw/Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16}
HF_UPLOAD_PRIVATE=${HF_UPLOAD_PRIVATE:-0}

mkdir -p "${WORKDIR}" "${OUTPUT_ROOT}" "${LOG_DIR}"

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
  log "Ensuring Python ${PYTHON_VERSION} is available..."
  uv python install "${PYTHON_VERSION}"
  export UV_PYTHON="${PYTHON_VERSION}"

  if [ "${SKIP_UV_SYNC}" != "1" ]; then
    echo "This AutoRound runner must not run repo uv sync on rented GPUs." >&2
    echo "Repo sync can install a CUDA-specific PyTorch stack unrelated to the Verda image." >&2
    echo "Leave SKIP_UV_SYNC=1 unless you are intentionally debugging locally." >&2
    exit 2
  fi

  log "Creating isolated AutoRound virtualenv without syncing the repo project..."
  uv venv --python "${PYTHON_VERSION}" .venv

  log "Installing pinned AutoRound and its runtime dependencies into .venv: ${AUTOROUND_SPEC}"
  uv pip install --python .venv/bin/python --upgrade "${AUTOROUND_SPEC}"

  if [ "${INSTALL_FAST_KERNELS}" = "1" ]; then
    log "Installing Qwen3.5/Qwen3.6 linear-attention fast-path deps..."
    uv pip install --python .venv/bin/python --upgrade --no-build-isolation causal-conv1d flash-linear-attention
  fi

  verify_torch_cuda
  verify_fast_path_deps
}

verify_torch_cuda() {
  log "Verifying PyTorch can see CUDA..."
  .venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"device={torch.cuda.get_device_name(0)}")
PY
}

verify_fast_path_deps() {
  if [ "${INSTALL_FAST_KERNELS}" != "1" ]; then
    log "INSTALL_FAST_KERNELS=0; skipping fast-path dependency verification."
    return
  fi

  log "Verifying Qwen linear-attention fast path availability..."
  set +e
  .venv/bin/python - <<'PY'
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
      log "Fast-path deps unavailable; ALLOW_SLOW_LINEAR_ATTENTION=1 so continuing."
    else
      echo "Qwen linear-attention fast-path deps are unavailable." >&2
      echo "Fix the install, or set ALLOW_SLOW_LINEAR_ATTENTION=1 to override intentionally." >&2
      exit 3
    fi
  fi
}

verify_gpu() {
  log "GPU inventory:"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; cannot verify GPU." >&2
    exit 4
  fi
  nvidia-smi

  local gpu_memory_mib
  gpu_memory_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if [ "${gpu_memory_mib}" -lt "${MIN_GPU_MEMORY_MIB}" ] && [ "${ALLOW_SMALL_GPU}" != "1" ]; then
    echo "GPU memory ${gpu_memory_mib} MiB is below MIN_GPU_MEMORY_MIB=${MIN_GPU_MEMORY_MIB}." >&2
    echo "Use an RTX PRO 6000 96GB-class GPU, or set ALLOW_SMALL_GPU=1 intentionally." >&2
    exit 4
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
  nvidia_smi=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed ':a;N;$!ba;s/\n/\\n/g' || echo unavailable)

  GIT_COMMIT_CAPTURE="${git_commit}" GIT_STATUS_CAPTURE="${git_status}" NVIDIA_SMI_CAPTURE="${nvidia_smi}" .venv/bin/python - <<'PY'
import json
import os
import platform
import subprocess

payload = {
    "run_id": os.environ.get("RUN_ID"),
    "model_repo": os.environ.get("MODEL_REPO"),
    "output_dir": os.environ.get("OUTPUT_DIR"),
    "autoround_ref": os.environ.get("AUTOROUND_REF"),
    "autoround_spec": os.environ.get("AUTOROUND_SPEC"),
    "autoround_scheme": os.environ.get("AUTOROUND_SCHEME"),
    "autoround_format": os.environ.get("AUTOROUND_FORMAT"),
    "autoround_ignore_layers": os.environ.get("AUTOROUND_IGNORE_LAYERS"),
    "enable_torch_compile": os.environ.get("ENABLE_TORCH_COMPILE"),
    "extra_autoround_args": os.environ.get("EXTRA_AUTOROUND_ARGS"),
    "auto_upload": os.environ.get("AUTO_UPLOAD"),
    "hf_upload_repo": os.environ.get("HF_UPLOAD_REPO"),
    "hf_upload_private": os.environ.get("HF_UPLOAD_PRIVATE"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
    payload["uv_pip_freeze"] = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", ".venv/bin/python"], text=True, stderr=subprocess.STDOUT
    ).splitlines()
except Exception as exc:
    payload["uv_pip_freeze_error"] = str(exc)

path = os.environ["METADATA_FILE"]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

run_autoround() {
  log "Starting AutoRound Balanced W4A16 quantization."
  log "Model: ${MODEL_REPO}"
  log "Output base: ${OUTPUT_DIR}"
  log "Ignore layers: ${AUTOROUND_IGNORE_LAYERS}"

  local compile_args=()
  if [ "${ENABLE_TORCH_COMPILE}" = "1" ]; then
    compile_args+=(--enable_torch_compile)
  fi

  # shellcheck disable=SC2206
  local extra_args=( ${EXTRA_AUTOROUND_ARGS} )

  .venv/bin/auto-round \
    --model "${MODEL_REPO}" \
    --scheme "${AUTOROUND_SCHEME}" \
    --format "${AUTOROUND_FORMAT}" \
    --ignore_layers "${AUTOROUND_IGNORE_LAYERS}" \
    --output_dir "${OUTPUT_DIR}" \
    "${compile_args[@]}" \
    "${extra_args[@]}"
}

resolve_model_output_dir() {
  if [ -f "${OUTPUT_DIR}/config.json" ]; then
    printf '%s\n' "${OUTPUT_DIR}"
    return
  fi

  if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "AutoRound output dir ${OUTPUT_DIR} does not exist." >&2
    return 1
  fi

  local candidates=()
  while IFS= read -r config_path; do
    candidates+=("$(dirname "${config_path}")")
  done < <(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 3 -type f -name config.json | sort)

  if [ "${#candidates[@]}" -eq 1 ]; then
    printf '%s\n' "${candidates[0]}"
    return
  fi

  echo "Could not uniquely resolve AutoRound model output directory under ${OUTPUT_DIR}." >&2
  printf 'Candidates:\n' >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

upload_output() {
  if [ "${AUTO_UPLOAD}" != "1" ]; then
    log "AUTO_UPLOAD=${AUTO_UPLOAD}; skipping Hugging Face upload."
    return
  fi

  local token="${HF_WRITE_TOKEN:-${HF_TOKEN:-}}"
  if [ -z "${token}" ]; then
    echo "AUTO_UPLOAD=1 but neither HF_WRITE_TOKEN nor HF_TOKEN is set." >&2
    echo "Set a write-capable token before starting the run, or set AUTO_UPLOAD=0." >&2
    exit 5
  fi

  local upload_dir
  upload_dir=$(resolve_model_output_dir)
  log "Uploading ${upload_dir} -> https://huggingface.co/${HF_UPLOAD_REPO}"

  HF_UPLOAD_TOKEN="${token}" HF_UPLOAD_DIR="${upload_dir}" .venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_UPLOAD_REPO"]
folder = os.environ["HF_UPLOAD_DIR"]
token = os.environ["HF_UPLOAD_TOKEN"]
private = os.environ.get("HF_UPLOAD_PRIVATE", "0") == "1"

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder,
    token=token,
    commit_message="Upload Qwopus3.6 RYS Balanced AutoRound W4A16",
)
print(f"Uploaded {folder} to https://huggingface.co/{repo_id}")
PY
}

bundle_output() {
  if [ ! -d "${OUTPUT_DIR}" ]; then
    log "Output dir ${OUTPUT_DIR} not found; skipping bundle. AutoRound may have created a derived subdirectory inside it."
  fi
  log "Bundling logs/output under ${WORKDIR} -> ${BUNDLE_FILE}"
  if command -v zstd >/dev/null 2>&1; then
    tar --exclude='models--*' -I 'zstd -19 -T0' -cf "${BUNDLE_FILE}" -C "${WORKDIR}" .
  else
    local fallback=${BUNDLE_FILE%.tar.zst}.tar.gz
    log "zstd unavailable; writing ${fallback}"
    tar -czf "${fallback}" -C "${WORKDIR}" .
  fi
}

main() {
  export RUN_ID MODEL_REPO OUTPUT_DIR AUTOROUND_REF AUTOROUND_SPEC AUTOROUND_SCHEME AUTOROUND_FORMAT
  export AUTOROUND_IGNORE_LAYERS ENABLE_TORCH_COMPILE EXTRA_AUTOROUND_ARGS CUDA_VISIBLE_DEVICES
  export AUTO_UPLOAD HF_UPLOAD_REPO HF_UPLOAD_PRIVATE
  export REPO_DIR REPO_URL REPO_REF METADATA_FILE

  prepare_repo
  install_deps
  verify_gpu
  write_metadata
  log "Logging AutoRound run to ${RUN_LOG}"
  run_autoround 2>&1 | tee "${RUN_LOG}"
  write_metadata
  upload_output
  bundle_output
  log "Done. Output root: ${OUTPUT_DIR}"
}

main "$@"
