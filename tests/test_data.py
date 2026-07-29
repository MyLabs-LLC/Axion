from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
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


def test_corpus_size_contract_rejects_preallocated_partial_file(tmp_path) -> None:
    cfg = load_config("configs/axion_1m.yaml")
    path = tmp_path / "partial.bin"
    np.arange(10, dtype=np.uint16).tofile(path)
    data = replace(cfg.data, path=str(path), expected_tokens=11)
    with pytest.raises(ValueError, match="size mismatch"):
        TokenSource(data, replace(cfg.model, seq_len=4))


def test_validation_split_reads_only_heldout_tail(tmp_path) -> None:
    cfg = load_config("configs/axion_1m.yaml")
    path = tmp_path / "split.bin"
    np.arange(20, dtype=np.uint16).tofile(path)
    data = replace(cfg.data, path=str(path), expected_tokens=20, train_tokens=12, val_tokens=8)
    source = TokenSource(data, replace(cfg.model, seq_len=4))
    assert source.read(0, 4, split="val").tolist() == [12, 13, 14, 15]
    assert source.read(6, 4, split="val").tolist() == [18, 19, 12, 13]


def test_corpus_metadata_rejects_wrong_dataset_prefix(tmp_path) -> None:
    cfg = load_config("configs/axion_1m.yaml")
    path = tmp_path / "tokens.bin"
    np.arange(20, dtype=np.uint16).tofile(path)
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "source": "HuggingFaceFW/fineweb-edu",
                "config": "sample-10BT",
                "dtype": "uint16",
                "tokens": 20,
                "tokenizer_vocab_size": cfg.model.vocab_size,
            }
        ),
        encoding="utf-8",
    )
    data = replace(
        cfg.data,
        path=str(path),
        expected_tokens=20,
        train_tokens=12,
        val_tokens=8,
        meta_path=str(meta),
        dataset_prefix="sample/100BT/",
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        TokenSource(data, replace(cfg.model, seq_len=4))
