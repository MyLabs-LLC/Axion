from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from axion.config import DataConfig, ModelConfig, TrainConfig


def normalize_dataset_prefix(prefix: str) -> str:
    stripped = prefix.strip()
    normalized = stripped + ("" if stripped.endswith("/") else "/")
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError(f"invalid dataset prefix: {prefix!r}")
    return normalized


class TokenSource:
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        self.path = Path(data_cfg.path)
        self.vocab_size = model_cfg.vocab_size
        if self.path.exists():
            self._validate(data_cfg, model_cfg)
            self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
            self.total_tokens = int(self.tokens.shape[0])
            self.val_tokens = data_cfg.val_tokens
            self.n_tokens = data_cfg.train_tokens or (self.total_tokens - self.val_tokens)
            if self.n_tokens <= model_cfg.seq_len or self.n_tokens + self.val_tokens != self.total_tokens:
                raise ValueError(
                    f"invalid train/validation split: train={self.n_tokens:,}, "
                    f"val={self.val_tokens:,}, total={self.total_tokens:,}"
                )
            self.synthetic = False
        else:
            if not data_cfg.allow_synthetic:
                raise FileNotFoundError(
                    f"token data not found at {self.path}. Run scripts/prepare_data.py "
                    "to build FineWeb-Edu tokens, or set data.allow_synthetic=true only for tests."
                )
            self.tokens = None
            self.total_tokens = data_cfg.synthetic_tokens
            self.n_tokens = data_cfg.synthetic_tokens
            self.val_tokens = 0
            self.synthetic = True

    def _validate(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        actual_bytes = self.path.stat().st_size
        if actual_bytes % np.dtype(np.uint16).itemsize:
            raise ValueError(f"token corpus has an invalid byte length: {self.path} ({actual_bytes:,} bytes)")
        actual_tokens = actual_bytes // np.dtype(np.uint16).itemsize
        if data_cfg.expected_tokens is not None and actual_tokens != data_cfg.expected_tokens:
            raise ValueError(
                f"token corpus size mismatch for {self.path}: expected {data_cfg.expected_tokens:,} "
                f"tokens, found {actual_tokens:,}"
            )
        if data_cfg.meta_path is None:
            return
        meta_path = Path(data_cfg.meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"missing completion metadata for token corpus: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = data_cfg.expected_tokens or actual_tokens
        required = {
            "source": "HuggingFaceFW/fineweb-edu",
            "config": normalize_dataset_prefix(data_cfg.dataset_prefix).strip("/").replace("/", "-"),
            "dtype": "uint16",
            "tokens": expected,
            "tokenizer_vocab_size": model_cfg.vocab_size,
        }
        mismatches = {
            key: (required_value, meta.get(key))
            for key, required_value in required.items()
            if meta.get(key) != required_value
        }
        if mismatches:
            raise ValueError(f"token corpus metadata mismatch in {meta_path}: {mismatches}")
        if data_cfg.tokenizer_path:
            tokenizer_path = Path(data_cfg.tokenizer_path)
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"missing tokenizer declared by corpus config: {tokenizer_path}")
            digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            if meta.get("tokenizer_sha256") != digest:
                raise ValueError(f"tokenizer hash does not match {meta_path}: {tokenizer_path}")

    def split_tokens(self, split: str) -> int:
        if split == "train":
            return self.n_tokens
        if split == "val" and self.val_tokens > 0:
            return self.val_tokens
        raise ValueError(f"token split {split!r} is not available")

    def read(self, offset: int, length: int, split: str = "train") -> np.ndarray:
        split_tokens = self.split_tokens(split) if not self.synthetic else self.n_tokens
        split_start = self.n_tokens if split == "val" else 0
        offset %= split_tokens
        if self.tokens is not None:
            end = offset + length
            if end <= split_tokens:
                return np.asarray(self.tokens[split_start + offset : split_start + end], dtype=np.int64)
            first = np.asarray(self.tokens[split_start + offset : split_start + split_tokens], dtype=np.int64)
            second = np.asarray(self.tokens[split_start : split_start + (end % split_tokens)], dtype=np.int64)
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
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    labels = []
    split_tokens = source.split_tokens(split) if not source.synthetic else source.n_tokens
    base = sample_offset(step, microbatch, train, model, split_tokens)
    for item in range(train.micro_batch):
        offset = (base + item * (model.seq_len + 1)) % max(1, split_tokens - model.seq_len - 1)
        chunk = source.read(offset, model.seq_len + 1, split=split)
        rows.append(chunk[:-1])
        labels.append(chunk[1:])
    x = torch.as_tensor(np.stack(rows), dtype=torch.long, device=device)
    y = torch.as_tensor(np.stack(labels), dtype=torch.long, device=device)
    return x, y
