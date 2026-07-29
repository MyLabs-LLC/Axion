from __future__ import annotations

from axion.config import load_config
from axion.params import estimate_parameter_count, estimate_stage_parameter_counts


def test_1b_config_is_1b_class() -> None:
    cfg = load_config("configs/axion_1b.yaml")
    assert estimate_parameter_count(cfg.model) == 1_275_168_768
    assert cfg.model.seq_len == 32_768
    assert cfg.model.rope_theta == 500_000
    assert cfg.model.activation_checkpointing is True
    stages = estimate_stage_parameter_counts(cfg.model)
    assert [stage["layers"] for stage in stages] == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
    ]
    assert [stage["parameter_count"] for stage in stages] == [436_240_384, 402_685_952, 436_242_432]
    assert sum(int(stage["parameter_count"]) for stage in stages) == estimate_parameter_count(cfg.model)


def test_250m_config_has_expected_size_and_stage_balance() -> None:
    cfg = load_config("configs/axion_250m.yaml")
    assert estimate_parameter_count(cfg.model) == 234_914_816
    stages = estimate_stage_parameter_counts(cfg.model)
    assert [stage["parameter_count"] for stage in stages] == [92_286_976, 62_924_800, 79_703_040]
    assert sum(int(stage["parameter_count"]) for stage in stages) == estimate_parameter_count(cfg.model)
