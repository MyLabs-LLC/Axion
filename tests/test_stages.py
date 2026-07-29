from __future__ import annotations

from axion.config import load_config
from axion.stages import StageModule, layer_range


def test_layer_range_balances_remainder() -> None:
    assert list(layer_range(6, 0, 3)) == [0, 1]
    assert list(layer_range(6, 1, 3)) == [2, 3]
    assert list(layer_range(6, 2, 3)) == [4, 5]
    assert list(layer_range(8, 0, 3)) == [0, 1, 2]
    assert list(layer_range(8, 1, 3)) == [3, 4, 5]
    assert list(layer_range(8, 2, 3)) == [6, 7]


def test_25m_stage_modules_construct() -> None:
    cfg = load_config("configs/axion_25m.yaml")
    stages = [StageModule(cfg.model, rank) for rank in range(3)]
    assert [stage.layer_ids for stage in stages] == [(0, 1, 2), (3, 4, 5), (6, 7)]
