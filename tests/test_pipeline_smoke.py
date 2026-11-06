"""End-to-end smoke test on synthetic data (needs torch, torchaudio, soundfile)."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("soundfile")


def test_smoke(tmp_path):
    root = os.path.dirname(os.path.dirname(__file__))
    data = str(tmp_path / "synthetic")
    subprocess.check_call([sys.executable, "-m", "scripts.make_synthetic_data", "--out", data], cwd=root)
    out = str(tmp_path / "run")
    subprocess.check_call([sys.executable, "-m", "src.train", "--config", "configs/smoke.yaml", "--set",
                           f"data.root={data}", f"out_dir={out}"], cwd=root)
    assert os.path.exists(os.path.join(out, "results.json"))
    assert os.path.exists(os.path.join(out, "svd.jsonl"))
