from __future__ import annotations

from dataclasses import replace

import torch

from axion.config import load_config
from axion.data import TokenSource, batch_for_step


def test_synthetic_batches_are_deterministic() -> None:
    cfg = load_config("configs/axion_1m.yaml")
    data = replace(cfg.data, path="data/does-not-exist-test.bin", allow_synthetic=True)
    source = TokenSource(data, cfg.model)
    x1, y1 = batch_for_step(source, cfg.model, cfg.train, 0, 0, torch.device("cpu"))
    x2, y2 = batch_for_step(source, cfg.model, cfg.train, 0, 0, torch.device("cpu"))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])


def test_missing_data_requires_explicit_synthetic() -> None:
    cfg = load_config("configs/axion_1m.yaml")
    data = replace(cfg.data, path="data/does-not-exist-test.bin", allow_synthetic=False)
    try:
        TokenSource(data, cfg.model)
    except FileNotFoundError as exc:
        assert "FineWeb-Edu tokens" in str(exc)
    else:
        raise AssertionError("TokenSource should fail when the FineWeb-Edu token file is missing")
