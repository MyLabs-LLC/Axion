from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from axion.config import load_config
from axion.data import TokenSource, batch_for_step
from axion.metrics import JsonlLogger, new_run_dir
from axion.model import AxionModel
from axion.utils import default_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/axion_1m.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = args.steps or cfg.train.smoke_steps
    set_seed(cfg.train.seed)
    device = default_device()
    model = AxionModel(cfg.model).to(device)
    source = TokenSource(cfg.data, cfg.model)
    run_dir = new_run_dir()
    relic_dir = Path("relics") / run_dir.name
    relic_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": json.dumps(args.config)}, relic_dir / "relic-0-PQ.pt")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )
    logger = JsonlLogger(run_dir)
    tokens_per_step = cfg.train.micro_batch * cfg.train.n_microbatches * cfg.model.seq_len
    try:
        for step in range(steps):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro in range(cfg.train.n_microbatches):
                x, y = batch_for_step(source, cfg.model, cfg.train, step, micro, device)
                _, loss = model(x, y)
                assert loss is not None
                (loss / cfg.train.n_microbatches).backward()
                total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            elapsed = time.perf_counter() - started
            logger.log(
                step=step,
                loss=total_loss / cfg.train.n_microbatches,
                tokens_per_s=tokens_per_step / elapsed,
                elapsed_ms=elapsed * 1000.0,
                source="synthetic" if source.synthetic else str(source.path),
            )
        torch.save({"model": model.state_dict()}, relic_dir / "final.pt")
    finally:
        logger.close()
    print(run_dir)


if __name__ == "__main__":
    main()
