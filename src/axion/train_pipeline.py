from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from axion.config import AxionConfig, load_config
from axion.dark_photon import TransferStats, recv, send
from axion.data import TokenSource, batch_for_step
from axion.metrics import JsonlLogger, new_run_dir, truncate_metrics_after_step
from axion.stages import StageModule
from axion.utils import default_device, set_seed


ACT_BASE = 1000
GRAD_BASE = 2000
EVAL_ACT_BASE = 3000


def _backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed() -> tuple[
    int,
    int,
    torch.device,
    dist.ProcessGroup,
    tuple[dist.ProcessGroup, dist.ProcessGroup],
]:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(_backend())
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 3:
        raise RuntimeError(f"M1 pipeline expects exactly 3 ranks, got {world}")
    backend = _backend()
    stage_01_group = dist.new_group(ranks=[0, 1], backend=backend)
    stage_12_group = dist.new_group(ranks=[1, 2], backend=backend)
    control_group = dist.new_group(backend="gloo") if backend == "nccl" else dist.group.WORLD
    return rank, world, default_device(), control_group, (stage_01_group, stage_12_group)


def learning_rate(step: int, total_steps: int, cfg: AxionConfig) -> float:
    if step < cfg.train.warmup_steps:
        return cfg.train.lr * (step + 1) / max(1, cfg.train.warmup_steps)
    decay_steps = max(1, total_steps - cfg.train.warmup_steps)
    progress = min(1.0, (step - cfg.train.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.train.min_lr + (cfg.train.lr - cfg.train.min_lr) * cosine


def global_clip_grad_norm(module: StageModule, max_norm: float, device: torch.device) -> float:
    local_squared = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            local_squared += parameter.grad.detach().float().square().sum()
    dist.all_reduce(local_squared, op=dist.ReduceOp.SUM)
    total_norm = local_squared.sqrt()
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return float(total_norm.detach().cpu())


def pipeline_train_step(
    module: StageModule,
    source: TokenSource,
    cfg: AxionConfig,
    step: int,
    rank: int,
    device: torch.device,
    stats: TransferStats,
    stage_groups: tuple[dist.ProcessGroup, dist.ProcessGroup],
) -> float:
    act_shape = (cfg.train.micro_batch, cfg.model.seq_len, cfg.model.d_model)
    use_bf16 = device.type == "cuda" and cfg.train.precision == "bf16"
    act_dtype = torch.bfloat16 if use_bf16 else torch.float32
    saved_inputs: list[torch.Tensor] = []
    saved_outputs: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    loss_accum = 0.0

    for micro in range(cfg.train.n_microbatches):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if rank == 0:
                x, _ = batch_for_step(source, cfg.model, cfg.train, step, micro, device)
                out = module(x).to(act_dtype)
                saved_outputs.append(out)
                send(out.detach(), dst=1, tag=ACT_BASE + micro, stats=stats, group=stage_groups[0])
            elif rank == 1:
                inp = recv(
                    act_shape,
                    act_dtype,
                    device,
                    src=0,
                    tag=ACT_BASE + micro,
                    stats=stats,
                    group=stage_groups[0],
                ).requires_grad_()
                out = module(inp).to(act_dtype)
                saved_inputs.append(inp)
                saved_outputs.append(out)
                send(out.detach(), dst=2, tag=ACT_BASE + micro, stats=stats, group=stage_groups[1])
            else:
                inp = recv(
                    act_shape,
                    act_dtype,
                    device,
                    src=1,
                    tag=ACT_BASE + micro,
                    stats=stats,
                    group=stage_groups[1],
                ).requires_grad_()
                _, labels = batch_for_step(source, cfg.model, cfg.train, step, micro, device)
                logits = module(inp)
                loss = module.loss(logits.float(), labels)
                saved_inputs.append(inp)
                losses.append(loss)
                loss_accum += float(loss.detach())

    for micro in reversed(range(cfg.train.n_microbatches)):
        if rank == 2:
            (losses[micro] / cfg.train.n_microbatches).backward()
            grad = saved_inputs[micro].grad
            assert grad is not None
            send(grad.detach(), dst=1, tag=GRAD_BASE + micro, stats=stats, group=stage_groups[1])
        elif rank == 1:
            grad = recv(
                act_shape,
                act_dtype,
                device,
                src=2,
                tag=GRAD_BASE + micro,
                stats=stats,
                group=stage_groups[1],
            )
            saved_outputs[micro].backward(grad / cfg.train.n_microbatches)
            input_grad = saved_inputs[micro].grad
            assert input_grad is not None
            send(input_grad.detach(), dst=0, tag=GRAD_BASE + micro, stats=stats, group=stage_groups[0])
        else:
            grad = recv(
                act_shape,
                act_dtype,
                device,
                src=1,
                tag=GRAD_BASE + micro,
                stats=stats,
                group=stage_groups[0],
            )
            saved_outputs[micro].backward(grad / cfg.train.n_microbatches)
    return loss_accum


@torch.no_grad()
def pipeline_evaluate(
    module: StageModule,
    source: TokenSource,
    cfg: AxionConfig,
    rank: int,
    device: torch.device,
    batches: int,
    stage_groups: tuple[dist.ProcessGroup, dist.ProcessGroup],
) -> float:
    act_shape = (cfg.train.micro_batch, cfg.model.seq_len, cfg.model.d_model)
    use_bf16 = device.type == "cuda" and cfg.train.precision == "bf16"
    act_dtype = torch.bfloat16 if use_bf16 else torch.float32
    stats = TransferStats()
    loss_accum = 0.0
    module.eval()
    for batch in range(batches):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if rank == 0:
                x, _ = batch_for_step(source, cfg.model, cfg.train, batch, 0, device, split="val")
                send(
                    module(x).to(act_dtype),
                    dst=1,
                    tag=EVAL_ACT_BASE + batch,
                    stats=stats,
                    group=stage_groups[0],
                )
            elif rank == 1:
                inp = recv(
                    act_shape,
                    act_dtype,
                    device,
                    src=0,
                    tag=EVAL_ACT_BASE + batch,
                    stats=stats,
                    group=stage_groups[0],
                )
                send(
                    module(inp).to(act_dtype),
                    dst=2,
                    tag=EVAL_ACT_BASE + batch,
                    stats=stats,
                    group=stage_groups[1],
                )
            else:
                inp = recv(
                    act_shape,
                    act_dtype,
                    device,
                    src=1,
                    tag=EVAL_ACT_BASE + batch,
                    stats=stats,
                    group=stage_groups[1],
                )
                _, labels = batch_for_step(source, cfg.model, cfg.train, batch, 0, device, split="val")
                loss_accum += float(module.loss(module(inp).float(), labels))
    module.train()
    loss_tensor = torch.tensor(loss_accum, dtype=torch.float32, device=device)
    dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
    return float(loss_tensor.detach().cpu()) / batches


def save_checkpoint(
    relic_dir: Path,
    rank: int,
    step: int,
    module: StageModule,
    optimizer: torch.optim.Optimizer,
    keep: int,
) -> None:
    path = relic_dir / f"stage{rank}-step{step:08d}.pt"
    temporary = path.with_suffix(".tmp")
    payload = {
        "stage": rank,
        "step": step,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    checkpoints = sorted(relic_dir.glob(f"stage{rank}-step*.pt"))
    for old in checkpoints[:-keep]:
        old.unlink()


def load_checkpoint(
    relic_dir: Path,
    rank: int,
    module: StageModule,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    step: int | None = None,
) -> int:
    if step is None:
        checkpoints = sorted(relic_dir.glob(f"stage{rank}-step*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no resumable stage {rank} checkpoint in {relic_dir}")
        checkpoint = checkpoints[-1]
    else:
        checkpoint = relic_dir / f"stage{rank}-step{step:08d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"stage {rank} checkpoint step {step} not found in {relic_dir}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("stage") != rank:
        raise ValueError(f"checkpoint stage mismatch in {checkpoint}")
    if step is not None and int(payload.get("step", -1)) != step:
        raise ValueError(f"checkpoint step mismatch in {checkpoint}")
    module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu())
    return int(payload["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/axion_1m.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-step", type=int, default=None)
    args = parser.parse_args()

    if args.resume_step is not None and not args.resume:
        raise ValueError("--resume-step requires --resume")

    cfg = load_config(args.config)
    steps = args.steps or cfg.train.smoke_steps
    rank, world, device, control_group, stage_groups = init_distributed()
    set_seed(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    module = StageModule(cfg.model, rank).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
        fused=device.type == "cuda",
    )
    source = TokenSource(cfg.data, cfg.model)

    if args.resume:
        if not args.run_name:
            raise ValueError("--resume requires --run-name")
        run_dir = Path("runs") / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = [args.run_name]
    else:
        run_dir = new_run_dir(name=args.run_name) if rank == 0 else None
        run_name = [run_dir.name if run_dir else ""]
        dist.broadcast_object_list(run_name, src=0)
        if rank != 0:
            run_dir = Path("runs") / run_name[0]
            run_dir.mkdir(parents=True, exist_ok=True)
    assert run_dir is not None

    relic_dir = Path("relics") / run_name[0]
    relic_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if args.resume_step is None:
            local_checkpoint = any(relic_dir.glob(f"stage{rank}-step*.pt"))
        else:
            local_checkpoint = (relic_dir / f"stage{rank}-step{args.resume_step:08d}.pt").exists()
        availability = [torch.zeros((), dtype=torch.int64, device=device) for _ in range(world)]
        dist.all_gather(availability, torch.tensor(int(local_checkpoint), dtype=torch.int64, device=device))
        checkpoint_states = [bool(int(value.detach().cpu())) for value in availability]
        if all(checkpoint_states):
            start_step = load_checkpoint(relic_dir, rank, module, optimizer, device, args.resume_step)
        elif any(checkpoint_states):
            raise RuntimeError(f"distributed resume checkpoint set is incomplete: {checkpoint_states}")
        else:
            can_start_fresh = torch.tensor(1, dtype=torch.int64, device=device)
            if rank == 0:
                metrics_path = run_dir / "metrics.jsonl"
                can_start_fresh.fill_(int(not metrics_path.exists() or metrics_path.stat().st_size == 0))
            dist.broadcast(can_start_fresh, src=0)
            if not bool(int(can_start_fresh.detach().cpu())):
                raise RuntimeError("resume requested without checkpoints for a run that already has metrics")
            start_step = 0
            torch.save(
                {"stage": rank, "step": 0, "model": module.state_dict()},
                relic_dir / f"stage{rank}-relic-0-PQ.pt",
            )
            if rank == 0:
                print(f"no checkpoints found for {run_name[0]}; starting a fresh run")
        gathered = [torch.zeros((), dtype=torch.int64, device=device) for _ in range(world)]
        dist.all_gather(gathered, torch.tensor(start_step, dtype=torch.int64, device=device))
        resume_steps = {int(value.detach().cpu()) for value in gathered}
        if len(resume_steps) != 1:
            raise RuntimeError(f"stage checkpoints disagree on resume step: {sorted(resume_steps)}")
    else:
        start_step = 0
        torch.save({"stage": rank, "step": 0, "model": module.state_dict()}, relic_dir / f"stage{rank}-relic-0-PQ.pt")

    if rank == 0 and args.resume and start_step > 0:
        truncate_metrics_after_step(run_dir, start_step)
    logger = JsonlLogger(run_dir) if rank == 0 else None
    stats = TransferStats()
    tokens_per_step = cfg.train.micro_batch * cfg.train.n_microbatches * cfg.model.seq_len
    bubble_fraction = (world - 1) / (cfg.train.n_microbatches + world - 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for step in range(start_step, steps):
            stats.reset()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            started = time.perf_counter()
            loss_accum = pipeline_train_step(module, source, cfg, step, rank, device, stats, stage_groups)
            # Keep a rank from entering NCCL collectives while another rank still has P2P work queued.
            dist.barrier(group=control_group)
            grad_norm = global_clip_grad_norm(module, cfg.train.grad_clip, device)
            optimizer.step()

            loss_tensor = torch.tensor(loss_accum, dtype=torch.float32, device=device)
            dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            val_loss = None
            if source.val_tokens > 0 and args.eval_every > 0 and (
                (step + 1) % args.eval_every == 0 or step + 1 == steps
            ):
                val_loss = pipeline_evaluate(module, source, cfg, rank, device, args.val_batches, stage_groups)
            elapsed = time.perf_counter() - started

            if rank == 0 and logger is not None:
                row: dict[str, float | int | str] = {
                    "step": step,
                    "optimizer_step": step + 1,
                    "tokens_seen": (step + 1) * tokens_per_step,
                    "loss": float(loss_tensor.detach().cpu()) / cfg.train.n_microbatches,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_per_s": tokens_per_step / elapsed,
                    "elapsed_ms": elapsed * 1000.0,
                    "bytes_tx": stats.bytes_tx,
                    "bytes_rx": stats.bytes_rx,
                    "send_ms": stats.send_ms,
                    "recv_ms": stats.recv_ms,
                    "bubble_frac": bubble_fraction,
                    "source": str(source.path),
                }
                if device.type == "cuda":
                    row["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
                    row["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
                    row["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if val_loss is not None:
                    row["val_loss"] = val_loss
                    row["val_ppl"] = math.exp(min(20.0, val_loss))
                logger.log(**row)

            if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
                save_checkpoint(relic_dir, rank, step + 1, module, optimizer, args.keep_checkpoints)

        save_checkpoint(relic_dir, rank, steps, module, optimizer, args.keep_checkpoints)
        torch.save(
            {"stage": rank, "step": steps, "model": module.state_dict()},
            relic_dir / f"stage{rank}-final.pt",
        )
    finally:
        if logger is not None:
            logger.close()
        dist.destroy_process_group()
    if rank == 0:
        print(run_dir)


if __name__ == "__main__":
    main()
