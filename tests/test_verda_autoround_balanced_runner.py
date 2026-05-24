from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path("scripts/verda_run_qwopus36_autoround_balanced.sh")


def test_autoround_balanced_runner_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_autoround_balanced_runner_preserves_requested_autoround_contract():
    text = SCRIPT.read_text()

    assert "MODEL_REPO=${MODEL_REPO:-hampsonw/Qwopus3.6-27B-v2-RYS-Balanced}" in text
    assert "AUTOROUND_SCHEME=${AUTOROUND_SCHEME:-W4A16}" in text
    assert "AUTOROUND_FORMAT=${AUTOROUND_FORMAT:-auto_round}" in text
    assert (
        "AUTOROUND_IGNORE_LAYERS=${AUTOROUND_IGNORE_LAYERS:-visual,vision,lm_head,mtp.fc,linear_attn}"
        in text
    )

    assert 'uv run auto-round' in text
    assert '--model "${MODEL_REPO}"' in text
    assert '--scheme "${AUTOROUND_SCHEME}"' in text
    assert '--format "${AUTOROUND_FORMAT}"' in text
    assert '--ignore_layers "${AUTOROUND_IGNORE_LAYERS}"' in text
    assert '--output_dir "${OUTPUT_DIR}"' in text

    # AutoRound quality defaults should remain in charge unless explicitly overridden.
    assert "--seqlen" not in text
    assert "--nsamples" not in text
    assert "--batch_size" not in text
    assert "--group_size" not in text
    assert "--asym" not in text
    assert "--dataset" not in text
