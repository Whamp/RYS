# Verda Qwopus3.6 RYS Balanced AutoRound runbook

This runbook covers the one-time AutoRound W4A16 quantization of:

- Source model: `hampsonw/Qwopus3.6-27B-v2-RYS-Balanced`
- Output intent: `Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16`
- Runner: `scripts/verda_run_qwopus36_autoround_balanced.sh`

The runner keeps AutoRound's W4A16 quality defaults and only applies the sensitive-layer policy. It installs AutoRound without letting it replace the repo-pinned PyTorch/CUDA stack, then verifies `torch.cuda.is_available()` before starting the paid run.


```text
ignore_layers = visual,vision,lm_head,mtp.fc,linear_attn
```

That preserves the vision stack, language-model head, MTP `fc`, and Qwen hybrid linear-attention projections in the original precision while AutoRound tunes the normal text linear layers.

## Spot instance note

The current gamble target is:

```text
1RTXPRO6000.30V.CC
1x RTX PRO 6000 CC 96GB
FIN-03 spot
```

AutoRound is not assumed to be resumable mid-run. If the spot instance is reclaimed, the process dies and the quantization should be rerun from the beginning. Qwen fast-kernel installation is disabled by default for this AutoRound run to avoid source-building CUDA extensions on the rented VM; set `INSTALL_FAST_KERNELS=1` only if intentionally testing that path.

Use Verda's keep-detached volume policy so setup, downloads, logs, and partial files survive reclaim:

```bash
--is-spot \
--os-volume-on-spot-discontinue keep_detached
```

If creating extra storage, also use:

```bash
--storage-on-spot-discontinue keep_detached
```

## Fresh instance setup

On the VM:

```bash
mkdir -p /workspace
cd /workspace

git clone https://github.com/Whamp/RYS.git
cd RYS
export REPO_REF=will/qwopus36-autoround-balanced-verda
git fetch --all --tags
git checkout "$REPO_REF"
```

## Run

Use a write-capable Hugging Face token before starting the long run. The runner uploads automatically after quantization so the paid VM does not sit idle waiting for a manual upload command.

```bash
export HF_WRITE_TOKEN=<huggingface-read-write-token>
export REPO_REF=will/qwopus36-autoround-balanced-verda
./scripts/verda_run_qwopus36_autoround_balanced.sh
```

If you intentionally want to quantize without uploading:

```bash
export AUTO_UPLOAD=0
```

The runner quantizes with the equivalent of:

```bash
auto-round \
  --model hampsonw/Qwopus3.6-27B-v2-RYS-Balanced \
  --scheme W4A16 \
  --format auto_round \
  --ignore_layers "visual,vision,lm_head,mtp.fc,linear_attn" \
  --output_dir ./Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16
```

with setup and safety checks around it.

## Defaults worth knowing

AutoRound's default W4A16 recipe currently uses:

```text
iters: 200
seqlen: 2048
nsamples: 128
batch_size: 8
group_size: 128
sym: true
dataset: NeelNanda/pile-10k
```

We intentionally do not override these in the runner.

## Upload

By default the runner creates/uploads to:

```text
https://huggingface.co/hampsonw/Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16
```

Override if needed:

```bash
export HF_UPLOAD_REPO=hampsonw/another-repo-name
export HF_UPLOAD_PRIVATE=1  # optional; default is public
```

Do not paste tokens into logs or commit them. Rotate the write token after the upload is complete.

## Outputs

Default locations:

```text
/workspace/rys-qwopus36-autoround-balanced/
  logs/
    autoround_balanced_<RUN_ID>.log
    autoround_balanced_metadata_<RUN_ID>.json
  outputs/
    Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16/
```

The runner also attempts to bundle the workdir:

```text
/workspace/rys-qwopus36-autoround-balanced/Qwopus3.6-27B-v2-RYS-Balanced-AutoRound-W4A16_<RUN_ID>.tar.zst
```

## If spot is reclaimed

1. Create or start another compatible VM.
2. Reattach/use the kept OS volume if available.
3. Rerun the same script.

Assume the quantization computation restarts. The value of the kept volume is avoiding setup/model-cache loss, not preserving AutoRound block progress.
