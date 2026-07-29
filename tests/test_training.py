from __future__ import annotations

from dataclasses import replace
import json

from axion.config import load_config
from axion.metrics import read_metrics, truncate_metrics_after_step
from axion.train_pipeline import learning_rate


def test_learning_rate_warms_up_and_decays_to_floor() -> None:
    cfg = load_config("configs/axion_1m.yaml")
    cfg = replace(cfg, train=replace(cfg.train, warmup_steps=2))
    assert learning_rate(0, 10, cfg) == cfg.train.lr / 2
    assert learning_rate(1, 10, cfg) == cfg.train.lr
    assert learning_rate(2, 10, cfg) == cfg.train.lr
    assert learning_rate(10, 10, cfg) == cfg.train.min_lr


def test_resume_truncates_metrics_after_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"step": 0, "optimizer_step": 1, "loss": 4.0},
        {"step": 1, "optimizer_step": 2, "loss": 3.0},
        {"step": 2, "optimizer_step": 3, "loss": 2.0},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    truncate_metrics_after_step(run_dir, 2)

    assert [row["optimizer_step"] for row in read_metrics(run_dir)] == [1, 2]
