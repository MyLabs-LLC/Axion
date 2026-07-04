#!/usr/bin/env python
from __future__ import annotations

import argparse
import tempfile
from collections.abc import Iterator
from pathlib import Path

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


def write_sample(path: Path, max_bytes: int) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for text in iter_fineweb_texts():
            chunk = text.strip() + "\n"
            encoded = chunk.encode("utf-8")
            if written + len(encoded) > max_bytes:
                break
            handle.write(chunk)
            written += len(encoded)
    if written == 0:
        raise RuntimeError("FineWeb-Edu stream yielded no tokenizer sample text")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/tokenizer")
    parser.add_argument("--prefix", default="axion4k")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--sample-bytes", type=int, default=500_000_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="axion-tokenizer-") as tmp:
        sample_path = Path(tmp) / "fineweb_edu_sample.txt"
        written = write_sample(sample_path, args.sample_bytes)
        spm.SentencePieceTrainer.train(
            input=str(sample_path),
            model_prefix=str(out_dir / args.prefix),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            byte_fallback=True,
            bos_id=-1,
            eos_id=0,
            unk_id=1,
            pad_id=-1,
            input_sentence_size=5_000_000,
            shuffle_input_sentence=True,
        )
    print(f"wrote {out_dir / (args.prefix + '.model')} from {written} FineWeb-Edu bytes")


if __name__ == "__main__":
    main()
