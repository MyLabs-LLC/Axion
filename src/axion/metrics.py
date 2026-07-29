from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


def new_run_dir(root: str = "runs", name: str | None = None) -> Path:
    run_dir = Path(root) / (name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def truncate_metrics_after_step(run_dir: Path, optimizer_step: int) -> None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_step = int(row.get("optimizer_step", int(row.get("step", -1)) + 1))
        if row_step <= optimizer_step:
            retained.append(line)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    temporary.replace(path)


class JsonlLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "metrics.jsonl"
        self.file = self.path.open("a", encoding="utf-8")
        self.losses: list[float] = []

    def log(self, **row: Any) -> None:
        loss = row.get("loss")
        if loss is not None:
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError("Big Rip: non-finite loss")
            if len(self.losses) >= 20:
                median = sorted(self.losses[-20:])[10]
                if median > 0 and value > 3.0 * median:
                    raise RuntimeError(f"Big Rip: loss {value:.4f} exceeded 3x trailing median {median:.4f}")
            self.losses.append(value)
        self.file.write(json.dumps(row, sort_keys=True) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.jsonl"
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
