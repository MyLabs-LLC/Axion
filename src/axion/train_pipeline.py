from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from axion.config import load_config
from axion.dark_photon import TransferStats, recv, send
from axion.data import TokenSource, batch_for_step
from axion.metrics import JsonlLogger, new_run_dir
from axion.stages import StageModule
from axion.utils import default_device, set_seed


ACT_BASE = 1000
GRAD_BASE = 2000


def _backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed() -> tuple[int, int, torch.device]:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(_backend())
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 3:
        raise RuntimeError(f"M1 smoke expects exactly 3 ranks, got {world}")
    return rank, world, default_device()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/axion_1m.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = args.steps or cfg.train.smoke_steps
    rank, _, device = init_distributed()
    set_seed(cfg.train.seed)
    module = StageModule(cfg.model, rank).to(device)
    if torch.cuda.is_available() and cfg.train.precision == "bf16":
        module = module.to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )
    source = TokenSource(cfg.data, cfg.model)
    run_dir = new_run_dir() if rank == 0 else None
    run_name = [run_dir.name if run_dir else ""]
    dist.broadcast_object_list(run_name, src=0)
    if rank != 0:
        run_dir = Path("runs") / run_name[0]
        run_dir.mkdir(parents=True, exist_ok=True)
    relic_dir = Path("relics") / run_name[0]
    relic_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": rank, "model": module.state_dict()}, relic_dir / f"stage{rank}-relic-0-PQ.pt")
    logger = JsonlLogger(run_dir) if rank == 0 else None
    stats = TransferStats()
    act_shape = (cfg.train.micro_batch, cfg.model.seq_len, cfg.model.d_model)
    act_dtype = torch.bfloat16 if torch.cuda.is_available() and cfg.train.precision == "bf16" else torch.float32
    tokens_per_step = cfg.train.micro_batch * cfg.train.n_microbatches * cfg.model.seq_len
    try:
        for step in range(steps):
            stats.reset()
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss_accum = 0.0
            for micro in range(cfg.train.n_microbatches):
                if rank == 0:
                    x, _ = batch_for_step(source, cfg.model, cfg.train, step, micro, device)
                    out = module(x)
                    send(out.detach(), dst=1, tag=ACT_BASE + micro, stats=stats)
                    grad = recv(act_shape, act_dtype, device, src=1, tag=GRAD_BASE + micro, stats=stats)
                    out.backward(grad / cfg.train.n_microbatches)
                elif rank == 1:
                    inp = recv(act_shape, act_dtype, device, src=0, tag=ACT_BASE + micro, stats=stats).requires_grad_()
                    out = module(inp)
                    send(out.detach(), dst=2, tag=ACT_BASE + micro, stats=stats)
                    grad = recv(act_shape, act_dtype, device, src=2, tag=GRAD_BASE + micro, stats=stats)
                    out.backward(grad / cfg.train.n_microbatches)
                    assert inp.grad is not None
                    send(inp.grad.detach(), dst=0, tag=GRAD_BASE + micro, stats=stats)
                else:
                    inp = recv(act_shape, act_dtype, device, src=1, tag=ACT_BASE + micro, stats=stats).requires_grad_()
                    _, y = batch_for_step(source, cfg.model, cfg.train, step, micro, device)
                    logits = module(inp)
                    loss = module.loss(logits.float(), y)
                    (loss / cfg.train.n_microbatches).backward()
                    assert inp.grad is not None
                    send(inp.grad.detach(), dst=1, tag=GRAD_BASE + micro, stats=stats)
                    loss_accum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.train.grad_clip)
            optimizer.step()
            loss_tensor = torch.tensor([loss_accum], dtype=torch.float64, device=device)
            dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            elapsed = time.perf_counter() - started
            if rank == 0 and logger is not None:
                logger.log(
                    step=step,
                    loss=float(loss_tensor.item() / cfg.train.n_microbatches),
                    tokens_per_s=tokens_per_step / elapsed,
                    elapsed_ms=elapsed * 1000.0,
                    bytes_tx=stats.bytes_tx,
                    bytes_rx=stats.bytes_rx,
                    send_ms=stats.send_ms,
                    recv_ms=stats.recv_ms,
                    bubble_frac=2 / 18,
                    source="synthetic" if source.synthetic else str(source.path),
                )
        torch.save({"stage": rank, "model": module.state_dict()}, relic_dir / f"stage{rank}-final.pt")
    finally:
        if logger is not None:
            logger.close()
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(run_dir)


if __name__ == "__main__":
    main()
