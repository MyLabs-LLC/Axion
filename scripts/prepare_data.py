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

from axion.data import normalize_dataset_prefix


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_PREFIX = "sample/10BT/"


def fineweb_parquet_files(prefix: str = DATASET_PREFIX) -> list[str]:
    prefix = normalize_dataset_prefix(prefix)
    files = list_repo_files(DATASET_REPO, repo_type="dataset")
    shards = [name for name in files if name.startswith(prefix) and name.endswith(".parquet")]
    if not shards:
        raise RuntimeError(f"no FineWeb-Edu parquet shards found under {prefix}")
    return sorted(shards)


def iter_fineweb_texts(
    prefix: str = DATASET_PREFIX,
    shard_cache_dir: str | Path | None = None,
) -> Iterator[str]:
    for filename in fineweb_parquet_files(prefix):
        local = Path(
            hf_hub_download(
                DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=shard_cache_dir,
            )
        )
        try:
            with pq.ParquetFile(local) as parquet:
                for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
                    for text in batch.column("text").to_pylist():
                        if isinstance(text, str) and text.strip():
                            yield text.replace("\0", " ")
        finally:
            if shard_cache_dir is not None:
                local.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=102_000_000)
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=0)
    parser.add_argument("--out", default="data/fineweb_edu_100M.bin")
    parser.add_argument("--dataset-prefix", default=DATASET_PREFIX)
    parser.add_argument("--shard-cache-dir", default="data/.fineweb_shards")
    parser.add_argument("--tokenizer", default="data/tokenizer/axion4k.model")
    parser.add_argument("--meta", default="data/meta.json")
    parser.add_argument("--progress", default=None)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--progress-every-chunks", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-tokens", type=int, default=None)
    args = parser.parse_args()
    dataset_prefix = normalize_dataset_prefix(args.dataset_prefix)

    train_tokens = args.train_tokens if args.train_tokens is not None else args.tokens - args.val_tokens
    if train_tokens <= 0 or args.val_tokens < 0 or train_tokens + args.val_tokens != args.tokens:
        raise ValueError("train_tokens + val_tokens must equal tokens, with a non-empty training split")
    if args.chunk_size <= 0 or args.progress_every_chunks <= 0:
        raise ValueError("chunk and progress intervals must be positive")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta)
    progress_path = Path(args.progress) if args.progress else Path(f"{out}.progress.json")
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer at {tokenizer_path}; run scripts/train_tokenizer.py first")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    if tokenizer.vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 token storage requires vocab_size <= 65536")
    tokenizer_hash = file_sha256(tokenizer_path)

    resume_at = 0
    if args.resume:
        if not out.exists():
            raise FileNotFoundError(f"cannot resume missing token file: {out}")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            resume_at = int(progress["tokens_written"])
            if progress.get("tokenizer_sha256") != tokenizer_hash:
                raise ValueError(f"resume tokenizer does not match {progress_path}")
            progress_prefix = progress.get("dataset_prefix", DATASET_PREFIX)
            if normalize_dataset_prefix(str(progress_prefix)) != dataset_prefix:
                raise ValueError(f"resume dataset prefix does not match {progress_path}")
        elif args.resume_tokens is not None:
            resume_at = args.resume_tokens
        else:
            raise FileNotFoundError(
                f"resume requires {progress_path} or an explicitly verified --resume-tokens boundary"
            )
        if resume_at < 0 or resume_at > args.tokens or resume_at % args.chunk_size:
            raise ValueError("resume boundary must be a chunk-aligned token count within the target corpus")
        if out.stat().st_size < resume_at * np.dtype(np.uint16).itemsize:
            raise ValueError(f"token file is shorter than the requested resume boundary: {out}")
        with out.open("r+b") as handle:
            handle.truncate(args.tokens * np.dtype(np.uint16).itemsize)
        tokens = np.memmap(out, dtype=np.uint16, mode="r+", shape=(args.tokens,))
    else:
        tokens = np.memmap(out, dtype=np.uint16, mode="w+", shape=(args.tokens,))

    def progress_payload(tokens_written: int, docs_seen: int, completed: bool = False) -> dict[str, object]:
        return {
            "completed": completed,
            "docs_seen": docs_seen,
            "dtype": "uint16",
            "out": str(out),
            "target_tokens": args.tokens,
            "dataset_prefix": dataset_prefix,
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_hash,
            "tokens_written": tokens_written,
        }

    pos = 0
    docs = 0
    chunks_since_flush = 0
    resume_validated = resume_at == 0

    def consume(values: list[int]) -> None:
        nonlocal pos, chunks_since_flush, resume_validated
        array = np.asarray(values, dtype=np.uint16)
        end = pos + len(array)
        if pos < resume_at:
            if end > resume_at:
                raise ValueError("resume boundary crosses a preparation chunk")
            if not np.array_equal(np.asarray(tokens[pos:end]), array):
                raise ValueError(f"existing corpus diverges from FineWeb-Edu at token {pos:,}")
            action = "validated"
        else:
            tokens[pos:end] = array
            chunks_since_flush += 1
            action = "tokens"
        pos = end
        if not resume_validated and pos == resume_at:
            resume_validated = True
            write_json(progress_path, progress_payload(pos, docs))
            print(f"resume prefix validated: {pos:,} tokens", flush=True)
        if chunks_since_flush >= args.progress_every_chunks:
            tokens.flush()
            write_json(progress_path, progress_payload(pos, docs))
            chunks_since_flush = 0
        print(f"{action}: {pos:,}/{args.tokens:,}", flush=True)

    buffer: list[int] = []
    for text in iter_fineweb_texts(dataset_prefix, args.shard_cache_dir):
        buffer.extend(tokenizer.encode(text, out_type=int))
        buffer.append(tokenizer.eos_id())
        docs += 1
        while len(buffer) >= args.chunk_size and pos < args.tokens:
            n = min(args.chunk_size, args.tokens - pos)
            consume(buffer[:n])
            del buffer[:n]
        if pos >= args.tokens:
            break
    if pos < args.tokens and buffer:
        n = min(len(buffer), args.tokens - pos)
        consume(buffer[:n])
    tokens.flush()
    if pos < args.tokens:
        write_json(progress_path, progress_payload(pos, docs))
        raise RuntimeError(f"FineWeb-Edu stream ended at {pos:,} tokens, expected {args.tokens:,}")

    meta = {
        "source": DATASET_REPO,
        "config": dataset_prefix.strip("/").replace("/", "-"),
        "dataset_prefix": dataset_prefix,
        "split": "train",
        "tokens": args.tokens,
        "train_tokens": train_tokens,
        "val_tokens": args.val_tokens,
        "dtype": "uint16",
        "docs_seen": docs,
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size(),
        "tokenizer_sha256": tokenizer_hash,
    }
    write_json(meta_path, meta)
    write_json(progress_path, progress_payload(pos, docs, completed=True))
    print(
        f"wrote {out} ({train_tokens:,} train + {args.val_tokens:,} validation FineWeb-Edu tokens)",
        flush=True,
    )


if __name__ == "__main__":
    main()
