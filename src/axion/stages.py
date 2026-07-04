from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from axion.config import ModelConfig
from axion.model import RMSNorm, TransformerBlock, misalignment_init


class StageModule(nn.Module):
    def __init__(self, cfg: ModelConfig, stage_rank: int, n_stages: int = 3) -> None:
        super().__init__()
        if n_stages != 3 or cfg.n_layers != 6:
            raise ValueError("The smoke pipeline currently supports 6 layers over 3 stages")
        self.cfg = cfg
        self.stage_rank = stage_rank
        layer_start = stage_rank * 2
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model) if stage_rank == 0 else None
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(layer_start, layer_start + 2)])
        self.norm = RMSNorm(cfg.d_model) if stage_rank == 2 else None
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) if stage_rank == 2 else None
        self.apply(lambda module: misalignment_init(module, cfg.n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stage_rank == 0:
            x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        if self.stage_rank == 2:
            x = self.lm_head(self.norm(x))
        return x

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.stage_rank != 2:
            raise RuntimeError("Only the final stage can compute loss")
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())
