#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from axion.config import AxionConfig as TrainingConfig
from axion.config import load_config
from axion.stages import layer_range


MODEL_CARD = """\
---
language:
- en
pipeline_tag: text-generation
library_name: transformers
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- axion
- base-model
- custom-code
- vllm
---

# Axion 1B FineWeb-Edu — Step 36K

This is an early base-model checkpoint from the Axion three-stage pipeline. It
has **1,275,168,768 parameters** and was pretrained on FineWeb-Edu with the
Axion 16K SentencePiece tokenizer.

## Checkpoint

- Optimizer step: **36,000**
- Training tokens seen: **2,359,296,000**
- Validation cross-entropy: **2.786487**
- Validation perplexity: **16.22393**
- Context length: **32,768 tokens**
- Weight format: **BF16 SafeTensors**

This checkpoint is not instruction-tuned or chat-tuned. It may generate
incorrect, repetitive, biased, or otherwise unreliable text.

## Transformers

Custom model code is included because Axion is not a stock Llama architecture.
Review the repository code before enabling `trust_remote_code`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "YOUR_NAMESPACE/YOUR_REPOSITORY"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="auto",
)

inputs = tokenizer("The future of science is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## vLLM

Axion uses vLLM's Transformers modeling backend:

```bash
vllm serve YOUR_NAMESPACE/YOUR_REPOSITORY \
  --trust-remote-code \
  --model-impl transformers \
  --dtype bfloat16 \
  --max-model-len 32768
```

For a 12GB GPU, begin with `--max-model-len 8192` to leave sufficient room for
the KV cache. A 24GB GPU can use the full context more comfortably.

## Upload

From the Axion repository:

```bash
hf repos create YOUR_NAMESPACE/YOUR_REPOSITORY --private --exist-ok
hf upload-large-folder YOUR_NAMESPACE/YOUR_REPOSITORY {export_dir}
```

No model license is asserted by this export. The repository owner should select
and document a license before making the model public.
"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_name(stage: int, step: int) -> str:
    return f"stage{stage}-step{step:08d}.pt"


def collect_checkpoint(
    *,
    source_dir: Path,
    run_name: str,
    step: int,
    hosts: list[str],
    root: Path,
) -> None:
    if len(hosts) != 3:
        raise ValueError("--hosts must identify exactly three stage hosts")
    source_dir.mkdir(parents=True, exist_ok=True)
    for stage, host in enumerate(hosts):
        destination = source_dir / checkpoint_name(stage, step)
        if destination.exists():
            continue
        remote = root / "relics" / run_name / destination.name
        subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                f"{host}:{remote}",
                str(destination),
            ],
            check=True,
        )


def load_stage(path: Path, stage: int, step: int) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint shard: {path}")
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(payload.get("stage", -1)) != stage:
        raise ValueError(f"{path} declares stage {payload.get('stage')}, expected {stage}")
    if int(payload.get("step", -1)) != step:
        raise ValueError(f"{path} declares step {payload.get('step')}, expected {step}")
    state = payload.get("model")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(f"{path} does not contain a tensor model state dict")
    return state


def target_name(
    source_name: str,
    *,
    stage: int,
    training_config: TrainingConfig,
) -> str:
    if source_name == "embed.weight":
        return "model.embed_tokens.weight"
    if source_name == "norm.weight":
        return "model.norm.weight"
    if source_name == "lm_head.weight":
        return "lm_head.weight"
    if not source_name.startswith("layers."):
        raise ValueError(f"unsupported stage tensor: {source_name}")
    _, local_layer, suffix = source_name.split(".", 2)
    layer_ids = tuple(layer_range(training_config.model.n_layers, stage, 3))
    try:
        global_layer = layer_ids[int(local_layer)]
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"invalid local layer {local_layer} for stage {stage}"
        ) from error
    suffixes = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn.qkv.weight": "self_attn.qkv_proj.weight",
        "attn.proj.weight": "self_attn.o_proj.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.fc.weight": "mlp.up_proj.weight",
        "mlp.proj.weight": "mlp.down_proj.weight",
    }
    try:
        mapped_suffix = suffixes[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported layer tensor: {source_name}") from error
    return f"model.layers.{global_layer}.{mapped_suffix}"


def expected_weight_names(config: TrainingConfig) -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    layer_suffixes = (
        "input_layernorm.weight",
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for layer in range(config.model.n_layers):
        names.update(f"model.layers.{layer}.{suffix}" for suffix in layer_suffixes)
    return names


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported export dtype: {name}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_config(config: TrainingConfig, dtype_name: str) -> dict[str, Any]:
    model = config.model
    return {
        "architectures": ["AxionForCausalLM"],
        "attention_bias": False,
        "auto_map": {
            "AutoConfig": "configuration_axion.AxionConfig",
            "AutoModel": "modeling_axion.AxionModel",
            "AutoModelForCausalLM": "modeling_axion.AxionForCausalLM",
        },
        "bos_token_id": None,
        "dtype": dtype_name,
        "eos_token_id": 0,
        "head_dim": model.d_model // model.n_heads,
        "hidden_act": "gelu",
        "hidden_size": model.d_model,
        "intermediate_size": model.d_ff,
        "is_decoder": True,
        "max_position_embeddings": model.seq_len,
        "mlp_bias": False,
        "model_type": "axion",
        "num_attention_heads": model.n_heads,
        "num_hidden_layers": model.n_layers,
        "num_key_value_heads": model.n_heads,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_theta": model.rope_theta,
        "tie_word_embeddings": model.tie_embeddings,
        "torch_dtype": dtype_name,
        "use_cache": True,
        "vocab_size": model.vocab_size,
    }


def tokenizer_metadata(
    tokenizer_path: Path,
    output_dir: Path,
    max_length: int,
) -> dict[str, Any]:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    actual = {
        "vocab_size": processor.vocab_size(),
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "unk_id": processor.unk_id(),
        "pad_id": processor.pad_id(),
    }
    expected = {
        "vocab_size": 16_384,
        "bos_id": -1,
        "eos_id": 0,
        "unk_id": 1,
        "pad_id": -1,
    }
    if actual != expected:
        raise ValueError(f"unexpected tokenizer contract: {actual}, expected {expected}")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.model")
    source_vocab = tokenizer_path.with_suffix(".vocab")
    if source_vocab.exists():
        shutil.copy2(source_vocab, output_dir / "tokenizer.vocab")
    write_json(
        output_dir / "tokenizer_config.json",
        {
            "add_bos_token": False,
            "add_eos_token": False,
            "bos_token": None,
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": True,
            "model_max_length": max_length,
            "pad_token": "</s>",
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
        },
    )
    write_json(
        output_dir / "special_tokens_map.json",
        {
            "bos_token": None,
            "eos_token": "</s>",
            "pad_token": "</s>",
            "unk_token": "<unk>",
        },
    )
    return actual


def build_export(
    *,
    source_dir: Path,
    output_dir: Path,
    config_path: Path,
    tokenizer_path: Path,
    step: int,
    dtype_name: str,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError(
            "safetensors is required; run with `uv run --with safetensors`"
        ) from error

    config = load_config(config_path)
    export_dtype = dtype_from_name(dtype_name)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    weight_map: dict[str, str] = {}
    total_size = 0
    total_parameters = 0
    try:
        code_dir = Path(__file__).resolve().parents[1] / "src" / "axion" / "hf"
        for filename in ("configuration_axion.py", "modeling_axion.py"):
            shutil.copy2(code_dir / filename, temporary / filename)
        (temporary / "__init__.py").write_text("", encoding="utf-8")

        for stage in range(3):
            source = source_dir / checkpoint_name(stage, step)
            source_hashes[source.name] = file_sha256(source)
            state = load_stage(source, stage, step)
            converted: dict[str, torch.Tensor] = {}
            for name, tensor in state.items():
                destination_name = target_name(
                    name,
                    stage=stage,
                    training_config=config,
                )
                if destination_name in weight_map:
                    raise ValueError(f"duplicate exported tensor: {destination_name}")
                value = tensor.detach().to(dtype=export_dtype).contiguous()
                converted[destination_name] = value
                total_size += value.numel() * value.element_size()
                total_parameters += value.numel()
            shard_name = f"model-{stage + 1:05d}-of-00003.safetensors"
            shard_path = temporary / shard_name
            save_file(converted, shard_path, metadata={"format": "pt"})
            output_hashes[shard_name] = file_sha256(shard_path)
            weight_map.update({name: shard_name for name in converted})
            del state, converted
            gc.collect()

        missing = expected_weight_names(config) - weight_map.keys()
        unexpected = weight_map.keys() - expected_weight_names(config)
        if missing or unexpected:
            raise RuntimeError(
                f"bad export tensor set: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        write_json(
            temporary / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
        )
        write_json(temporary / "config.json", export_config(config, dtype_name))
        write_json(
            temporary / "generation_config.json",
            {
                "bos_token_id": None,
                "do_sample": True,
                "eos_token_id": 0,
                "pad_token_id": 0,
                "temperature": 0.8,
                "top_p": 0.95,
            },
        )
        tokenizer = tokenizer_metadata(
            tokenizer_path,
            temporary,
            config.model.seq_len,
        )
        card = MODEL_CARD.format(export_dir=output_dir)
        (temporary / "README.md").write_text(card, encoding="utf-8")
        manifest = {
            "architecture": "AxionForCausalLM",
            "checkpoint_step": step,
            "dtype": dtype_name,
            "parameter_count": total_parameters,
            "source_checkpoint_sha256": source_hashes,
            "safetensors_sha256": output_hashes,
            "tokenizer": tokenizer,
            "training_tokens_seen": step * config.train.micro_batch
            * config.train.n_microbatches
            * config.model.seq_len,
        }
        write_json(temporary / "axion_export_manifest.json", manifest)
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a distributed Axion checkpoint for Hugging Face."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", default="configs/axion_1b.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/axion16k.model")
    parser.add_argument("--hosts", nargs=3, default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if args.step <= 0:
        raise ValueError("--step must be positive")
    root = Path(args.root or os.environ.get("AXION_ROOT", Path.cwd())).resolve()
    hosts = args.hosts or os.environ.get(
        "AXION_HOSTS",
        "oblivion oblivion1 oblivion2",
    ).split()
    source_dir = Path(
        args.source_dir
        or f"hf_export/sources/{args.run_name}-step{args.step}"
    )
    output_dir = Path(
        args.out or f"hf_export/{args.run_name}-step{args.step}"
    )
    collect_checkpoint(
        source_dir=source_dir,
        run_name=args.run_name,
        step=args.step,
        hosts=hosts,
        root=root,
    )
    build_export(
        source_dir=source_dir,
        output_dir=output_dir,
        config_path=Path(args.config),
        tokenizer_path=Path(args.tokenizer),
        step=args.step,
        dtype_name=args.dtype,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
