from __future__ import annotations

import subprocess
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from axion.checkpoints import newest_common_checkpoint

_SPEC = spec_from_file_location(
    "axion_run_week_training",
    Path(__file__).resolve().parents[1] / "scripts" / "run_week_training.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RUN_WEEK = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_WEEK)
terminate_process = _RUN_WEEK.terminate_process


def test_newest_common_checkpoint_falls_back_from_partial_save() -> None:
    filenames = [
        ["stage0-step00002000.pt", "stage0-step00004000.pt"],
        ["stage1-step00002000.pt"],
        ["stage2-step00002000.pt"],
    ]

    assert newest_common_checkpoint(filenames) == 2000


def test_newest_common_checkpoint_rejects_disjoint_stages() -> None:
    with pytest.raises(RuntimeError, match="no common distributed checkpoint"):
        newest_common_checkpoint(
            [
                ["stage0-step00004000.pt"],
                ["stage1-step00002000.pt"],
                ["stage2-step00002000.pt"],
            ]
        )


def test_terminate_process_stops_a_running_child() -> None:
    proc = subprocess.Popen(["sleep", "30"], text=True)
    try:
        terminate_process(proc, timeout_seconds=1.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
