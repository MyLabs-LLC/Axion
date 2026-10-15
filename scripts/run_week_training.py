#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import yaml

from axion.config import AxionConfig, load_config
from axion.checkpoints import newest_common_checkpoint
from axion.data import TokenSource
from axion.params import estimate_parameter_count, estimate_stage_parameter_counts


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        return proc.wait()


def terminate_process(proc: subprocess.Popen[str], timeout_seconds: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def sync_data(hosts: list[str], root: Path, log_path: Path) -> int:
    commands: list[list[str]] = []
    for host in hosts[1:]:
        commands.append(["ssh", host, f"mkdir -p '{root}/data'"])
        commands.append(["rsync", "-a", "--partial", "data/", f"{host}:{root}/data/"])
    status = 0
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                status = proc.returncode
                break
        log.write(f"[exit_code={status}]\n")
    return status


def write_architecture(
    run_dir: Path,
    config_path: Path,
    steps: int,
    token_target: int,
    run_kind: str,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    tokens_per_step = cfg.train.micro_batch * cfg.train.n_microbatches * cfg.model.seq_len
    payload: dict[str, Any] = {
        "schema_version": 2,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_kind": run_kind,
        "config": str(config_path),
        "parameter_count": estimate_parameter_count(cfg.model),
        "stage_plan": estimate_stage_parameter_counts(cfg.model),
        "tokens_per_step": tokens_per_step,
        "planned_steps": steps,
        "planned_train_tokens": steps * tokens_per_step,
        "train_token_target": token_target,
        "corpus_token_target": cfg.data.expected_tokens or token_target,
        "model": vars(cfg.model),
        "train": vars(cfg.train),
        "data": vars(cfg.data),
        "macho_nodes": [
            {"macho": "macho0", "host": "oblivion", "role": "stage0", "gpu": "RTX 3090"},
            {"macho": "macho1", "host": "oblivion1", "role": "stage1", "gpu": "RTX 3090"},
            {"macho": "macho2", "host": "oblivion2", "role": "stage2", "gpu": "RTX 3090"},
        ],
        "network_assumption": "Three-stage GPipe over slow links; fused CUDA compute stays local to each RTX 3090.",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "architecture.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "architecture.md").write_text(
        "\n".join(
            [
                f"# Axion {run_kind} Run Architecture",
                "",
                f"- parameter_count: {payload['parameter_count']}",
                f"- planned_steps: {steps}",
                f"- planned_train_tokens: {payload['planned_train_tokens']}",
                f"- corpus_token_target: {payload['corpus_token_target']}",
                f"- tokens_per_step: {tokens_per_step}",
                "",
                "## Stage Plan",
                *[
                    f"- stage {stage['stage']}: layers={stage['layers']}, parameter_count={stage['parameter_count']}"
                    for stage in payload["stage_plan"]
                ],
                "",
                "## Model",
                "```yaml",
                yaml.safe_dump(payload["model"], sort_keys=False).rstrip(),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def log_new_metrics(metrics_path: Path, offset: int) -> int:
    if not metrics_path.exists():
        return offset
    with metrics_path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row.get("step", 0))
            for key, value in row.items():
                if key in {
                    "loss",
                    "val_loss",
                    "val_ppl",
                    "lr",
                    "grad_norm",
                    "tokens_per_s",
                    "elapsed_ms",
                    "bytes_tx",
                    "bytes_rx",
                    "send_ms",
                    "recv_ms",
                    "bubble_frac",
                    "gpu_mem_allocated_mb",
                    "gpu_mem_reserved_mb",
                    "gpu_mem_peak_mb",
                    "tokens_seen",
                }:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        mlflow.log_metric(key, numeric, step=step)
        return handle.tell()


def corpus_ready(cfg: AxionConfig) -> bool:
    try:
        TokenSource(cfg.data, cfg.model)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return True


def resume_checkpoint_step(hosts: list[str], root: Path, run_name: str) -> int | None:
    relic_dir = root / "relics" / run_name
    filenames: list[list[str]] = []
    filenames.append([path.name for path in relic_dir.glob("stage0-step*.pt")])
    for rank, host in enumerate(hosts[1:], start=1):
        command = (
            f"if [ -d {shlex.quote(str(relic_dir))} ]; then "
            f"find {shlex.quote(str(relic_dir))} -maxdepth 1 -type f "
            f"-name {shlex.quote(f'stage{rank}-step*.pt')} -printf '%f\\n'; fi"
        )
        result = subprocess.run(["ssh", host, command], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"could not inspect stage {rank} checkpoints on {host}: {result.stderr.strip()}")
        filenames.append([line for line in result.stdout.splitlines() if line])

    return newest_common_checkpoint(filenames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/axion_1b_week.yaml")
    parser.add_argument("--experiment-name", default="axion-macho-training")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-kind", default="macho_training")
    parser.add_argument("--milestone", default="long_run")
    parser.add_argument("--dataset-label", default="FineWeb-Edu")
    parser.add_argument("--token-target", type=int, default=586_000_000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--monitor-seconds", type=int, default=60)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--data-resume-tokens", type=int, default=None)
    parser.add_argument("--sync-data", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--resumed-from", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    cfg = load_config(args.config)
    tokens_per_step = cfg.train.micro_batch * cfg.train.n_microbatches * cfg.model.seq_len
    train_token_target = cfg.data.train_tokens or args.token_target
    corpus_token_target = cfg.data.expected_tokens or train_token_target
    steps = args.steps or math.ceil(train_token_target / tokens_per_step)
    run_name = args.run_name or f"{args.run_kind}-{time.strftime('%Y%m%d-%H%M%S')}"
    capture_dir = Path("experiments") / run_name
    capture_dir.mkdir(parents=True, exist_ok=True)
    architecture = write_architecture(capture_dir, Path(args.config), steps, train_token_target, args.run_kind)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "axion.run_kind": args.run_kind,
                "axion.milestone": args.milestone,
                "axion.nodes": "oblivion,oblivion1,oblivion2",
                "axion.network_assumption": "internet_like_weak_interaction",
                "axion.dataset": args.dataset_label,
                "axion.resumed_from": args.resumed_from or "none",
            }
        )
        mlflow.log_param("parameter_count", architecture["parameter_count"])
        mlflow.log_param("planned_steps", steps)
        mlflow.log_param("planned_train_tokens", architecture["planned_train_tokens"])
        mlflow.log_param("corpus_tokens", corpus_token_target)
        mlflow.log_param("tokens_per_step", tokens_per_step)
        mlflow.log_param("checkpoint_every", args.checkpoint_every)
        mlflow.log_param("eval_every", args.eval_every)
        mlflow.log_artifact(str(capture_dir / "architecture.json"))
        mlflow.log_artifact(str(capture_dir / "architecture.md"))
        mlflow.log_artifact(args.config)

        data_path = Path(cfg.data.path)
        meta_path = args.meta or cfg.data.meta_path or f"{data_path}.meta.json"
        if args.prepare_data and not corpus_ready(cfg):
            mlflow.set_tag("axion.phase", "prepare_data")
            data_log = capture_dir / "prepare_data.log"
            data_tokens = cfg.data.expected_tokens or args.token_target
            train_tokens = cfg.data.train_tokens or args.token_target
            prepare_command = [
                "uv",
                "run",
                "scripts/prepare_data.py",
                "--tokens",
                str(data_tokens),
                "--train-tokens",
                str(train_tokens),
                "--val-tokens",
                str(cfg.data.val_tokens),
                "--out",
                str(data_path),
                "--dataset-prefix",
                cfg.data.dataset_prefix,
                "--meta",
                str(meta_path),
            ]
            tokenizer = args.tokenizer or cfg.data.tokenizer_path
            if tokenizer:
                prepare_command.extend(["--tokenizer", tokenizer])
            if args.data_resume_tokens is not None or Path(f"{data_path}.progress.json").exists():
                prepare_command.append("--resume")
            if args.data_resume_tokens is not None:
                prepare_command.extend(["--resume-tokens", str(args.data_resume_tokens)])
            status = run_command(prepare_command, data_log)
            mlflow.log_artifact(str(data_log))
            if status != 0:
                raise RuntimeError(f"data preparation failed with exit code {status}")

        if not corpus_ready(cfg):
            raise RuntimeError(f"FineWeb-Edu corpus is incomplete or invalid: {data_path}")

        if args.sync_data:
            mlflow.set_tag("axion.phase", "sync_code")
            sync_code_log = capture_dir / "sync_code.log"
            status = run_command(["bash", "scripts/sync_subhalo.sh"], sync_code_log)
            mlflow.log_artifact(str(sync_code_log))
            if status != 0:
                raise RuntimeError(f"code sync failed with exit code {status}")
            mlflow.set_tag("axion.phase", "sync_data")
            sync_log = capture_dir / "sync_data.log"
            status = sync_data(os.environ.get("AXION_HOSTS", "oblivion oblivion1 oblivion2").split(), root, sync_log)
            mlflow.log_artifact(str(sync_log))
            if status != 0:
                raise RuntimeError(f"data sync failed with exit code {status}")

        hosts = os.environ.get("AXION_HOSTS", "oblivion oblivion1 oblivion2").split()
        resume_step = resume_checkpoint_step(hosts, root, run_name) if args.resume_training else None
        mlflow.set_tag("axion.resume_status", f"step_{resume_step}" if resume_step is not None else "fresh")

        mlflow.set_tag("axion.phase", "train")
        launch_log = capture_dir / "launch.log"
        env = os.environ.copy()
        env["AXION_STEPS"] = str(steps)
        env["AXION_RUN_NAME"] = run_name
        env["AXION_CHECKPOINT_EVERY"] = str(args.checkpoint_every)
        env["AXION_KEEP_CHECKPOINTS"] = str(args.keep_checkpoints)
        env["AXION_EVAL_EVERY"] = str(args.eval_every)
        env["AXION_VAL_BATCHES"] = str(args.val_batches)
        env["AXION_RESUME"] = "1" if resume_step is not None else "0"
        if resume_step is not None:
            env["AXION_RESUME_STEP"] = str(resume_step)
        env["AXION_COLLECT_RELICS"] = "1"
        env.setdefault("AXION_MASTER_PORT", "29610")
        with launch_log.open("a", encoding="utf-8") as launch_output:
            proc = subprocess.Popen(
                ["bash", "scripts/launch_m1.sh", args.config],
                stdout=launch_output,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            metrics_path = Path("runs") / run_name / "metrics.jsonl"
            offset = 0
            try:
                while proc.poll() is None:
                    offset = log_new_metrics(metrics_path, offset)
                    mlflow.log_metric("heartbeat", time.time())
                    time.sleep(args.monitor_seconds)
                log_new_metrics(metrics_path, offset)
            finally:
                terminate_process(proc)
                launch_output.flush()
        mlflow.log_artifact(str(launch_log))
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path))
        if proc.returncode == 75:
            mlflow.set_tag("axion.phase", "already_running")
            return
        if proc.returncode != 0:
            raise RuntimeError(f"training failed with exit code {proc.returncode}")

        relic_dir = Path("relics") / run_name
        merge_log = capture_dir / "merge.log"
        merge_status = run_command(
            ["uv", "run", "scripts/merge_ckpt.py", str(relic_dir), "--config", args.config],
            merge_log,
        )
        mlflow.log_artifact(str(merge_log))
        if merge_status != 0:
            raise RuntimeError(f"checkpoint merge failed with exit code {merge_status}")
        score_path = capture_dir / "final_score.json"
        eval_log = capture_dir / "evaluate.log"
        eval_status = run_command(
            [
                "uv",
                "run",
                "scripts/evaluate_relic.py",
                str(relic_dir / "merged.pt"),
                "--config",
                args.config,
                "--batches",
                str(args.val_batches),
                "--out",
                str(score_path),
            ],
            eval_log,
        )
        mlflow.log_artifact(str(eval_log))
        if eval_status != 0:
            raise RuntimeError(f"final evaluation failed with exit code {eval_status}")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        mlflow.log_metric("final_val_ce", float(score["val_ce"]), step=steps)
        mlflow.log_metric("final_val_ppl", float(score["val_ppl"]), step=steps)
        mlflow.log_artifact(str(score_path))
        mlflow.set_tag("axion.phase", "complete")


if __name__ == "__main__":
    main()
