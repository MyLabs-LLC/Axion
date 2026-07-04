#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from huggingface_hub import hf_hub_download, list_repo_files


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_PREFIX = "sample/10BT/"


def fineweb_parquet_files() -> list[str]:
    files = list_repo_files(DATASET_REPO, repo_type="dataset")
    shards = [name for name in files if name.startswith(DATASET_PREFIX) and name.endswith(".parquet")]
    if not shards:
        raise RuntimeError(f"no FineWeb-Edu parquet shards found under {DATASET_PREFIX}")
    return sorted(shards)


def iter_fineweb_texts() -> Iterator[str]:
    for filename in fineweb_parquet_files():
        local = hf_hub_download(DATASET_REPO, filename=filename, repo_type="dataset")
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
            for text in batch.column("text").to_pylist():
                if isinstance(text, str) and text.strip():
                    yield text.replace("\0", " ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=102_000_000)
    parser.add_argument("--out", default="data/fineweb_edu_100M.bin")
    parser.add_argument("--tokenizer", default="data/tokenizer/axion4k.model")
    parser.add_argument("--meta", default="data/meta.json")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer at {tokenizer_path}; run scripts/train_tokenizer.py first")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    if tokenizer.vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 token storage requires vocab_size <= 65536")

    tokens = np.memmap(out, dtype=np.uint16, mode="w+", shape=(args.tokens,))
    pos = 0
    docs = 0
    buffer: list[int] = []
    for text in iter_fineweb_texts():
        buffer.extend(tokenizer.encode(text, out_type=int))
        buffer.append(tokenizer.eos_id())
        docs += 1
        while len(buffer) >= args.chunk_size and pos < args.tokens:
            n = min(args.chunk_size, args.tokens - pos)
            tokens[pos : pos + n] = np.asarray(buffer[:n], dtype=np.uint16)
            del buffer[:n]
            pos += n
            print(f"tokens: {pos:,}/{args.tokens:,}", flush=True)
        if pos >= args.tokens:
            break
    if pos < args.tokens and buffer:
        n = min(len(buffer), args.tokens - pos)
        tokens[pos : pos + n] = np.asarray(buffer[:n], dtype=np.uint16)
        pos += n
    tokens.flush()
    if pos < args.tokens:
        raise RuntimeError(f"FineWeb-Edu stream ended at {pos:,} tokens, expected {args.tokens:,}")

    meta = {
        "source": DATASET_REPO,
        "config": "sample-10BT",
        "split": "train",
        "tokens": args.tokens,
        "dtype": "uint16",
        "docs_seen": docs,
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({args.tokens:,} FineWeb-Edu tokens)")


if __name__ == "__main__":
    main()
