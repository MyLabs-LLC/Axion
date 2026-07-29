from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    seq_len: int
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class TrainConfig:
    micro_batch: int
    n_microbatches: int
    optimizer: str
    betas: tuple[float, float]
    lr: float
    min_lr: float
    warmup_steps: int
    lr_schedule: str
    weight_decay: float
    grad_clip: float
    precision: str
    total_tokens: int
    seed: int
    smoke_steps: int = 2


@dataclass(frozen=True)
class DataConfig:
    path: str = "data/fineweb_edu_100M.bin"
    dataset_prefix: str = "sample/10BT/"
    synthetic_tokens: int = 2_000_000
    allow_synthetic: bool = False
    expected_tokens: int | None = None
    train_tokens: int | None = None
    val_tokens: int = 0
    meta_path: str | None = None
    tokenizer_path: str | None = None


@dataclass(frozen=True)
class AxionConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig


def load_config(path: str | Path) -> AxionConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    train = dict(raw["train"])
    train["betas"] = tuple(train["betas"])
    return AxionConfig(
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**train),
        data=DataConfig(**raw.get("data", {})),
    )
