from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from axion.config import DataConfig, ModelConfig, TrainConfig


class TokenSource:
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        self.path = Path(data_cfg.path)
        self.vocab_size = model_cfg.vocab_size
        if self.path.exists():
            self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
            self.n_tokens = int(self.tokens.shape[0])
            self.synthetic = False
        else:
            if not data_cfg.allow_synthetic:
                raise FileNotFoundError(
                    f"token data not found at {self.path}. Run scripts/prepare_data.py "
                    "to build FineWeb-Edu tokens, or set data.allow_synthetic=true only for tests."
                )
            self.tokens = None
            self.n_tokens = data_cfg.synthetic_tokens
            self.synthetic = True

    def read(self, offset: int, length: int) -> np.ndarray:
        if self.tokens is not None:
            end = offset + length
            if end <= self.n_tokens:
                return np.asarray(self.tokens[offset:end], dtype=np.int64)
            first = np.asarray(self.tokens[offset:], dtype=np.int64)
            second = np.asarray(self.tokens[: end % self.n_tokens], dtype=np.int64)
            return np.concatenate([first, second])
        positions = np.arange(offset, offset + length, dtype=np.uint64)
        vals = (positions * np.uint64(1103515245) + np.uint64(12345)) % np.uint64(self.vocab_size)
        return vals.astype(np.int64)


def sample_offset(step: int, microbatch: int, train: TrainConfig, model: ModelConfig, n_tokens: int) -> int:
    span = model.seq_len + 1
    max_start = max(1, n_tokens - span)
    stride = train.micro_batch * span
    return ((step * train.n_microbatches + microbatch) * stride + train.seed) % max_start


def batch_for_step(
    source: TokenSource,
    model: ModelConfig,
    train: TrainConfig,
    step: int,
    microbatch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    labels = []
    base = sample_offset(step, microbatch, train, model, source.n_tokens)
    for item in range(train.micro_batch):
        offset = (base + item * (model.seq_len + 1)) % max(1, source.n_tokens - model.seq_len - 1)
        chunk = source.read(offset, model.seq_len + 1)
        rows.append(chunk[:-1])
        labels.append(chunk[1:])
    x = torch.as_tensor(np.stack(rows), dtype=torch.long, device=device)
    y = torch.as_tensor(np.stack(labels), dtype=torch.long, device=device)
    return x, y
