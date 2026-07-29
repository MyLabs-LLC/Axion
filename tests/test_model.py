from __future__ import annotations

import math
from dataclasses import replace

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


def test_activation_checkpointing_preserves_loss_and_gradients() -> None:
    cfg = load_config("configs/axion_1m.yaml")
    plain = AxionModel(cfg.model)
    checkpointed = AxionModel(replace(cfg.model, activation_checkpointing=True))
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    y = torch.randint(0, cfg.model.vocab_size, (1, 8))

    _, plain_loss = plain(x, y)
    _, checkpointed_loss = checkpointed(x, y)
    assert plain_loss is not None and checkpointed_loss is not None
    plain_loss.backward()
    checkpointed_loss.backward()

    assert torch.equal(plain_loss, checkpointed_loss)
    for plain_parameter, checkpointed_parameter in zip(
        plain.parameters(), checkpointed.parameters(), strict=True
    ):
        assert plain_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        assert torch.equal(plain_parameter.grad, checkpointed_parameter.grad)
