#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import mlflow
import yaml

from axion.config import load_config
from axion.metrics import read_metrics
from axion.params import estimate_parameter_count, estimate_stage_parameter_counts


RESEARCH_NOTES = [
    "DiLoCo: rare synchronization after many local AdamW steps, outer Nesterov.",
    "OpenDiLoCo: globally distributed training can keep high utilization with low-frequency sync.",
    "DeMo: communicate compressed momentum/high-frequency components instead of dense gradients.",
    "NoLoCo: explore partial/random peer averaging to avoid global collectives.",
    "HALoS: hierarchy matters; local groups plus global slow sync are natural for WAN GPUs.",
]


TRIAL_DELTAS_25M: list[dict[str, Any]] = [
    {"name": "baseline", "change": {}},
    {"name": "lr_half", "change": {"train.lr": 7.5e-4, "train.min_lr": 7.5e-5}},
    {"name": "lr_high", "change": {"train.lr": 2.0e-3, "train.min_lr": 2.0e-4}},
    {"name": "more_microbatches", "change": {"train.micro_batch": 1, "train.n_microbatches": 16}},
    {"name": "fewer_microbatches", "change": {"train.micro_batch": 4, "train.n_microbatches": 4}},
    {"name": "clip_half", "change": {"train.grad_clip": 0.5}},
    {"name": "clip_two", "change": {"train.grad_clip": 2.0}},
    {"name": "wd_low", "change": {"train.weight_decay": 0.05}},
    {"name": "wd_high", "change": {"train.weight_decay": 0.2}},
    {"name": "beta2_high", "change": {"train.betas": [0.9, 0.98]}},
    {"name": "rope_long", "change": {"model.rope_theta": 50000}},
    {"name": "ffn_narrow", "change": {"model.d_ff": 1536}},
    {"name": "ffn_wide", "change": {"model.d_ff": 2560}},
]


TRIAL_DELTAS_1B: list[dict[str, Any]] = [
    {"name": "baseline", "change": {}},
    {"name": "lr_low", "change": {"train.lr": 3.0e-4, "train.min_lr": 3.0e-5}},
    {"name": "lr_high", "change": {"train.lr": 8.0e-4, "train.min_lr": 8.0e-5}},
    {"name": "more_microbatches", "change": {"train.micro_batch": 1, "train.n_microbatches": 4}},
    {"name": "clip_half", "change": {"train.grad_clip": 0.5}},
    {"name": "clip_two", "change": {"train.grad_clip": 2.0}},
    {"name": "wd_low", "change": {"train.weight_decay": 0.05}},
    {"name": "wd_high", "change": {"train.weight_decay": 0.2}},
    {"name": "beta2_high", "change": {"train.betas": [0.9, 0.98]}},
    {"name": "rope_long", "change": {"model.rope_theta": 50000}},
]


TRIAL_PLANS: dict[str, list[dict[str, Any]]] = {
    "25m": TRIAL_DELTAS_25M,
    "1b": TRIAL_DELTAS_1B,
}


def set_nested(raw: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    target = raw
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def make_trial_config(
    base: dict[str, Any],
    trial_index: int,
    trial_deltas: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    delta = trial_deltas[trial_index % len(trial_deltas)]
    generation = trial_index // len(trial_deltas)
    raw = deepcopy(base)
    for key, value in delta["change"].items():
        set_nested(raw, key, value)
    raw["train"]["seed"] = int(base["train"]["seed"]) + generation
    return delta["name"], delta["change"], raw


def run_command(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        log_path.write_text(output + f"\n[timeout_s={timeout} elapsed_s={elapsed:.3f}]\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 124, output, "")
    elapsed = time.perf_counter() - started
    log_path.write_text(proc.stdout + f"\n[exit_code={proc.returncode} elapsed_s={elapsed:.3f}]\n", encoding="utf-8")
    return proc


def cleanup_training_processes() -> None:
    hosts = os.environ.get("AXION_HOSTS", "oblivion oblivion1 oblivion2").split()
    for host in hosts:
        subprocess.run(
            ["ssh", host, "pgrep -f 'python -u -m axion.train_pipeline' | xargs -r kill -9"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def sync_trial_config(config_path: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    hosts = os.environ.get("AXION_HOSTS", "oblivion oblivion1 oblivion2").split()
    root = Path(os.environ.get("AXION_ROOT", Path.cwd()))
    output: list[str] = []
    return_code = 0
    for host in hosts[1:]:
        remote_path = root / config_path
        commands = [
            ["ssh", host, f"mkdir -p '{remote_path.parent}'"],
            ["rsync", "-az", str(config_path), f"{host}:{remote_path}"],
        ]
        for command in commands:
            proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output.append(f"$ {' '.join(command)}")
            output.append(proc.stdout)
            if proc.returncode != 0:
                return_code = proc.returncode
                break
        if return_code != 0:
            break
    log_path.write_text("\n".join(output) + f"\n[exit_code={return_code}]\n", encoding="utf-8")
    return subprocess.CompletedProcess(["sync_trial_config", str(config_path)], return_code, "\n".join(output), "")


def extract_run_dir(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("runs/"):
            return line
    return None


def write_note(path: Path, trial_index: int, name: str, change: dict[str, Any], status: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# Trial {trial_index:04d}: {name}",
        "",
        f"Status: {status}",
        "",
        "## One Change",
        json.dumps(change or {"baseline": True}, indent=2, sort_keys=True),
        "",
        "## Research Bias",
        *[f"- {note}" for note in RESEARCH_NOTES],
        "",
        "## Result",
        json.dumps(summary, indent=2, sort_keys=True),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_architecture_capture(
    path: Path,
    *,
    trial_index: int,
    name: str,
    change: dict[str, Any],
    raw: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    parameter_count = estimate_parameter_count(cfg.model)
    stage_plan = estimate_stage_parameter_counts(cfg.model)
    tokens_per_step = cfg.train.micro_batch * cfg.train.n_microbatches * cfg.model.seq_len
    activation_shape = [cfg.train.micro_batch, cfg.model.seq_len, cfg.model.d_model]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trial": trial_index,
        "name": name,
        "change": change or {"baseline": True},
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "parameter_count": parameter_count,
        "stage_plan": stage_plan,
        "tokens_per_step": tokens_per_step,
        "activation_shape": activation_shape,
        "activation_dtype": raw["train"]["precision"],
        "activation_bytes_per_tensor_bf16": activation_shape[0] * activation_shape[1] * activation_shape[2] * 2,
        "model": raw["model"],
        "train": raw["train"],
        "data": raw.get("data", {}),
    }
    json_path = path / "architecture.json"
    md_path = path / "architecture.md"
    write_json(json_path, payload)
    md_lines = [
        f"# Trial {trial_index:04d}: {name} architecture",
        "",
        f"- parameter_count: {parameter_count}",
        f"- config_sha256: {payload['config_sha256']}",
        f"- tokens_per_step: {tokens_per_step}",
        f"- activation_shape: {activation_shape}",
        f"- activation_bytes_per_tensor_bf16: {payload['activation_bytes_per_tensor_bf16']}",
        "",
        "## Model",
        "```yaml",
        yaml.safe_dump(raw["model"], sort_keys=False).rstrip(),
        "```",
        "",
        "## Stage Plan",
    ]
    for stage in stage_plan:
        md_lines.append(
            f"- stage {stage['stage']}: layers={stage['layers']}, parameter_count={stage['parameter_count']}"
        )
    md_lines.extend(
        [
            "",
            "## One Change",
            "```json",
            json.dumps(change or {"baseline": True}, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return payload


def write_run_capture(
    path: Path,
    *,
    status: str,
    summary: dict[str, Any],
    architecture: dict[str, Any],
) -> Path:
    capture_path = path / "run_capture.json"
    write_json(
        capture_path,
        {
            "schema_version": 1,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "architecture": architecture,
            "summary": summary,
            "artifacts": {
                "architecture_json": str(path / "architecture.json"),
                "architecture_md": str(path / "architecture.md"),
                "config": str(path / "config.yaml"),
                "note": str(path / "note.md"),
                "launch_log": str(path / "launch.log"),
                "merge_log": str(path / "merge.log"),
                "eval_json": str(path / "eval.json"),
                "eval_log": str(path / "eval.log"),
            },
        },
    )
    return capture_path


def log_mlflow_params(prefix: str, raw: dict[str, Any]) -> None:
    for section, values in raw.items():
        if isinstance(values, dict):
            for key, value in values.items():
                mlflow.log_param(f"{prefix}{section}.{key}", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/axion_25m.yaml")
    parser.add_argument("--trial-plan", choices=sorted(TRIAL_PLANS), default="25m")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="axion-25m-subhalo")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--local-cpu", action="store_true")
    parser.add_argument("--keep-relics", action="store_true")
    parser.add_argument("--launch-timeout", type=int, default=180)
    args = parser.parse_args()

    base_path = Path(args.base_config)
    base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    trial_deltas = TRIAL_PLANS[args.trial_plan]
    root = Path(args.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    (root / "research_notes.json").write_text(json.dumps(RESEARCH_NOTES, indent=2) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    best_val_ce = float("inf")

    for trial in range(args.start_index, args.start_index + args.trials):
        name, change, raw = make_trial_config(base_raw, trial, trial_deltas)
        trial_dir = root / f"trial_{trial:04d}_{name}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        architecture = write_architecture_capture(
            trial_dir,
            trial_index=trial,
            name=name,
            change=change,
            raw=raw,
            config_path=config_path,
        )

        status = "failed"
        summary: dict[str, Any] = {
            "trial": trial,
            "name": name,
            "change": change,
            "parameter_count": architecture["parameter_count"],
            "architecture_capture": str(trial_dir / "architecture.json"),
            "run_capture": str(trial_dir / "run_capture.json"),
        }
        with mlflow.start_run(run_name=f"{trial:04d}-{name}"):
            mlflow.log_param("trial", trial)
            mlflow.log_param("change_name", name)
            mlflow.log_param("steps", args.steps)
            mlflow.log_param("val_batches", args.val_batches)
            mlflow.log_param("parameter_count", architecture["parameter_count"])
            mlflow.log_param("config_sha256", architecture["config_sha256"])
            mlflow.log_param("tokens_per_step", architecture["tokens_per_step"])
            mlflow.log_text(json.dumps(change or {"baseline": True}, indent=2), "change.json")
            mlflow.log_text("\n".join(RESEARCH_NOTES), "research_notes.txt")
            mlflow.log_artifact(str(trial_dir / "architecture.json"))
            mlflow.log_artifact(str(trial_dir / "architecture.md"))
            log_mlflow_params("", raw)

            if args.sync:
                sync_log = trial_dir / "sync.log"
                sync_proc = run_command(["bash", "scripts/sync_subhalo.sh"], sync_log)
                mlflow.log_artifact(str(sync_log))
                if sync_proc.returncode != 0:
                    summary["error"] = "sync failed"
                    summary["exit_code"] = sync_proc.returncode
                    write_note(trial_dir / "note.md", trial, name, change, status, summary)
                    mlflow.log_artifact(str(write_run_capture(trial_dir, status=status, summary=summary, architecture=architecture)))
                    mlflow.log_artifact(str(trial_dir / "note.md"))
                    continue
                config_sync_log = trial_dir / "config_sync.log"
                config_sync_proc = sync_trial_config(config_path, config_sync_log)
                mlflow.log_artifact(str(config_sync_log))
                if config_sync_proc.returncode != 0:
                    summary["error"] = "trial config sync failed"
                    summary["exit_code"] = config_sync_proc.returncode
                    write_note(trial_dir / "note.md", trial, name, change, status, summary)
                    mlflow.log_artifact(str(write_run_capture(trial_dir, status=status, summary=summary, architecture=architecture)))
                    mlflow.log_artifact(str(trial_dir / "note.md"))
                    continue

            env = os.environ.copy()
            env["AXION_STEPS"] = str(args.steps)
            if args.local_cpu:
                env["AXION_CPU"] = "1"
            launch_log = trial_dir / "launch.log"
            proc = run_command(["bash", "scripts/launch_m1.sh", str(config_path)], launch_log, env=env, timeout=args.launch_timeout)
            if proc.returncode == 124:
                cleanup_training_processes()
            mlflow.log_artifact(str(config_path))
            mlflow.log_artifact(str(launch_log))
            summary["launch_exit_code"] = proc.returncode
            run_dir = extract_run_dir(proc.stdout)
            summary["run_dir"] = run_dir
            if proc.returncode != 0 or run_dir is None:
                summary["error"] = "training failed or did not report a run dir"
                write_note(trial_dir / "note.md", trial, name, change, status, summary)
                mlflow.log_artifact(str(write_run_capture(trial_dir, status=status, summary=summary, architecture=architecture)))
                mlflow.log_artifact(str(trial_dir / "note.md"))
                continue

            metrics = read_metrics(run_dir)
            if metrics:
                last = metrics[-1]
                summary["train_loss"] = float(last.get("loss", math.nan))
                summary["tokens_per_s"] = float(last.get("tokens_per_s", 0.0))
                summary["bytes_tx"] = int(last.get("bytes_tx", 0))
                summary["bytes_rx"] = int(last.get("bytes_rx", 0))
                for key in ("train_loss", "tokens_per_s", "bytes_tx", "bytes_rx"):
                    mlflow.log_metric(key, summary[key])
                mlflow.log_artifact(str(Path(run_dir) / "metrics.jsonl"))

            relic_dir = Path("relics") / Path(run_dir).name
            merge_log = trial_dir / "merge.log"
            merge_proc = run_command(
                ["uv", "run", "scripts/merge_ckpt.py", str(relic_dir), "--config", str(config_path)],
                merge_log,
            )
            mlflow.log_artifact(str(merge_log))
            if merge_proc.returncode != 0:
                summary["error"] = "merge failed"
                write_note(trial_dir / "note.md", trial, name, change, status, summary)
                mlflow.log_artifact(str(write_run_capture(trial_dir, status=status, summary=summary, architecture=architecture)))
                mlflow.log_artifact(str(trial_dir / "note.md"))
                continue

            eval_path = trial_dir / "eval.json"
            eval_log = trial_dir / "eval.log"
            eval_proc = run_command(
                [
                    "uv",
                    "run",
                    "scripts/evaluate_relic.py",
                    str(relic_dir / "merged.pt"),
                    "--config",
                    str(config_path),
                    "--batches",
                    str(args.val_batches),
                    "--out",
                    str(eval_path),
                ],
                eval_log,
            )
            mlflow.log_artifact(str(eval_log))
            if eval_proc.returncode == 0 and eval_path.exists():
                eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))
                summary.update(eval_metrics)
                mlflow.log_artifact(str(eval_path))
                mlflow.log_metric("val_ce", float(eval_metrics["val_ce"]))
                mlflow.log_metric("val_ppl", float(eval_metrics["val_ppl"]))
                if float(eval_metrics["val_ce"]) < best_val_ce:
                    best_val_ce = float(eval_metrics["val_ce"])
                    best_path = root / "best_merged.pt"
                    shutil.copy2(relic_dir / "merged.pt", best_path)
                    (root / "best.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    mlflow.log_artifact(str(root / "best.json"))
                    mlflow.log_param("best_checkpoint_path", str(best_path))
                status = "completed"
            else:
                summary["error"] = "evaluation failed"

            write_note(trial_dir / "note.md", trial, name, change, status, summary)
            mlflow.log_artifact(str(write_run_capture(trial_dir, status=status, summary=summary, architecture=architecture)))
            mlflow.log_artifact(str(trial_dir / "note.md"))
            if not args.keep_relics and relic_dir.exists():
                shutil.rmtree(relic_dir)


if __name__ == "__main__":
    main()
