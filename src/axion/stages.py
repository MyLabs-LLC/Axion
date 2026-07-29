from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from axion.config import ModelConfig
from axion.model import RMSNorm, TransformerBlock, misalignment_init, run_transformer_block


def layer_range(n_layers: int, stage_rank: int, n_stages: int = 3) -> range:
    if not 0 <= stage_rank < n_stages:
        raise ValueError(f"stage_rank must be in [0, {n_stages}), got {stage_rank}")
    base = n_layers // n_stages
    extra = n_layers % n_stages
    start = stage_rank * base + min(stage_rank, extra)
    stop = start + base + (1 if stage_rank < extra else 0)
    return range(start, stop)


class StageModule(nn.Module):
    def __init__(self, cfg: ModelConfig, stage_rank: int, n_stages: int = 3) -> None:
        super().__init__()
        if n_stages != 3:
            raise ValueError("The smoke pipeline currently expects exactly 3 stages")
        self.cfg = cfg
        self.stage_rank = stage_rank
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model) if stage_rank == 0 else None
        self.layer_ids = tuple(layer_range(cfg.n_layers, stage_rank, n_stages))
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in self.layer_ids])
        self.norm = RMSNorm(cfg.d_model) if stage_rank == 2 else None
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) if stage_rank == 2 else None
        self.apply(lambda module: misalignment_init(module, cfg.n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stage_rank == 0:
            x = self.embed(x)
        for layer in self.layers:
            x = run_transformer_block(
                layer,
                x,
                checkpoint_activations=self.cfg.activation_checkpointing and self.training,
            )
        if self.stage_rank == 2:
            x = self.lm_head(self.norm(x))
        return x

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.stage_rank != 2:
            raise RuntimeError("Only the final stage can compute loss")
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())
