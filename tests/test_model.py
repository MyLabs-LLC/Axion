from __future__ import annotations

import math

import torch

from axion.config import load_config
from axion.model import AxionModel, parameter_count


def test_axion_1m_shapes_and_loss() -> None:
    cfg = load_config("configs/axion_1m.yaml")
    torch.manual_seed(cfg.train.seed)
    model = AxionModel(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    y = torch.randint(0, cfg.model.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, cfg.model.vocab_size)
    assert loss is not None
    assert abs(float(loss.detach()) - math.log(cfg.model.vocab_size)) < 0.15
    assert 2_200_000 <= parameter_count(model) <= 2_300_000
