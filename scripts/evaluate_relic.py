#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from axion.config import load_config
from axion.data import TokenSource, batch_for_step
from axion.model import AxionModel
from axion.utils import default_device, set_seed


@torch.inference_mode()
def evaluate(relic: Path, config: Path, batches: int) -> dict[str, float | int | str]:
    cfg = load_config(config)
    set_seed(cfg.train.seed)
    device = default_device()
    model = AxionModel(cfg.model).to(device)
    payload = torch.load(relic, map_location=device, weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"{relic} does not contain a model state dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    source = TokenSource(cfg.data, cfg.model)
    total_loss = 0.0
    total_tokens = 0
    split = "val" if source.val_tokens > 0 else "train"
    use_bf16 = device.type == "cuda" and cfg.train.precision == "bf16"
    for batch in range(batches):
        x, y = batch_for_step(
            source,
            cfg.model,
            cfg.train,
            step=batch,
            microbatch=0,
            device=device,
            split=split,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        total_loss += float(loss.detach()) * y.numel()
        total_tokens += y.numel()
    val_ce = total_loss / max(1, total_tokens)
    return {
        "relic": str(relic),
        "config": str(config),
        "split": split,
        "val_batches": batches,
        "val_tokens": total_tokens,
        "val_ce": val_ce,
        "val_ppl": math.exp(min(20.0, val_ce)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("relic")
    parser.add_argument("--config", default="configs/axion_1m.yaml")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics = evaluate(Path(args.relic), Path(args.config), args.batches)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
