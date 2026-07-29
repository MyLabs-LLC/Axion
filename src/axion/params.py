from __future__ import annotations

from axion.config import ModelConfig
from axion.stages import layer_range


def estimate_layer_parameter_count(cfg: ModelConfig) -> int:
    return (4 * cfg.d_model * cfg.d_model) + (2 * cfg.d_model * cfg.d_ff) + (2 * cfg.d_model)


def estimate_parameter_count(cfg: ModelConfig) -> int:
    embedding = cfg.vocab_size * cfg.d_model
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    final_norm = cfg.d_model
    return embedding + lm_head + (cfg.n_layers * estimate_layer_parameter_count(cfg)) + final_norm


def estimate_stage_parameter_counts(cfg: ModelConfig, n_stages: int = 3) -> list[dict[str, int | list[int]]]:
    per_layer = estimate_layer_parameter_count(cfg)
    stages: list[dict[str, int | list[int]]] = []
    for stage_rank in range(n_stages):
        layers = list(layer_range(cfg.n_layers, stage_rank, n_stages))
        params = len(layers) * per_layer
        if stage_rank == 0:
            params += cfg.vocab_size * cfg.d_model
        if stage_rank == n_stages - 1:
            params += cfg.d_model
            if not cfg.tie_embeddings:
                params += cfg.vocab_size * cfg.d_model
        stages.append({"stage": stage_rank, "layers": layers, "parameter_count": params})
    return stages
