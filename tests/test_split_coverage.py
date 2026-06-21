"""Verify that the standard configs produce splits with all classes present.

Runs scripts/verify_split_coverage.py as a subprocess. Catches the v0
failure mode (entire class missing from one of train/val/test) before
training expensive checkpoints. Lives in tests/ so it runs in CI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verify_split_coverage.py"


@pytest.mark.parametrize("config", [
    "configs/synth_rppg_cinc.yaml",
    "configs/mitbih_baseline.yaml",
])
def test_split_coverage_passes(config: str) -> None:
    cfg_path = REPO / config
    if not cfg_path.exists():
        pytest.skip(f"{config} not in repo")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg_path)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, (
        f"verify_split_coverage failed on {config}:\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
