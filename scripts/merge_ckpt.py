#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from axion.config import load_config
from axion.model import AxionModel
from axion.stages import layer_range


def _load_stage(relic_dir: Path, stage: int, kind: str) -> dict[str, torch.Tensor]:
    path = relic_dir / f"stage{stage}-{kind}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing stage shard: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != stage:
        raise ValueError(f"{path} declares stage {payload.get('stage')}, expected {stage}")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dict")
    return state


def _copy_stage_layers(
    merged: dict[str, torch.Tensor],
    stage_state: dict[str, torch.Tensor],
    *,
    cfg_layers: int,
    stage_rank: int,
) -> None:
    stage_layers = tuple(layer_range(cfg_layers, stage_rank, 3))
    for key, value in stage_state.items():
        target = key
        if key.startswith("layers."):
            _, local_layer, rest = key.split(".", 2)
            target = f"layers.{stage_layers[int(local_layer)]}.{rest}"
        if key.startswith(("embed.", "norm.", "lm_head.")) or key.startswith("layers."):
            merged[target] = value.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("relic_dir")
    parser.add_argument("--config", default="configs/axion_1m.yaml")
    parser.add_argument("--kind", default="final", choices=("final", "relic-0-PQ"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    relic_dir = Path(args.relic_dir)
    cfg = load_config(args.config)
    model = AxionModel(cfg.model)
    merged = model.state_dict()
    for stage in range(3):
        _copy_stage_layers(merged, _load_stage(relic_dir, stage, args.kind), cfg_layers=cfg.model.n_layers, stage_rank=stage)
    missing, unexpected = model.load_state_dict(merged, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"bad merged state: missing={missing}, unexpected={unexpected}")

    out = Path(args.out) if args.out else relic_dir / "merged.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": args.config,
            "source_relic_dir": str(relic_dir),
            "source_kind": args.kind,
        },
        out,
    )
    print(out)


if __name__ == "__main__":
    main()
