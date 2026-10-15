from __future__ import annotations

import json
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from axion.config import load_config
from axion.model import AxionModel as TrainingModel

_SPEC = spec_from_file_location(
    "axion_export_hf",
    Path(__file__).resolve().parents[1] / "scripts" / "export_hf.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_EXPORT = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXPORT)
expected_weight_names = _EXPORT.expected_weight_names
read_validation_metrics = _EXPORT.read_validation_metrics
target_name = _EXPORT.target_name


def test_export_reads_validation_metrics_for_requested_step(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {"optimizer_step": 71_000, "val_loss": 2.7, "val_ppl": 14.8},
        {"optimizer_step": 72_000, "loss": 2.6},
        {"optimizer_step": 72_000, "val_loss": 2.67, "val_ppl": 14.5},
    ]
    metrics.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert read_validation_metrics(metrics, 72_000) == (2.67, 14.5)
    assert read_validation_metrics(metrics, 73_000) is None


def test_export_maps_all_stage_layers_once() -> None:
    config = load_config("configs/axion_1b.yaml")
    mapped = set()
    for stage in range(3):
        for local_layer in range(8):
            mapped.add(
                target_name(
                    f"layers.{local_layer}.attn.qkv.weight",
                    stage=stage,
                    training_config=config,
                )
            )
    assert mapped == {
        f"model.layers.{layer}.self_attn.qkv_proj.weight"
        for layer in range(24)
    }
    assert len(expected_weight_names(config)) == 147


def test_hf_model_preserves_axion_logits_and_cache() -> None:
    pytest.importorskip("transformers")
    from axion.hf.configuration_axion import AxionConfig
    from axion.hf.modeling_axion import AxionForCausalLM

    training_config = load_config("configs/axion_1m.yaml")
    tiny_model_config = replace(
        training_config.model,
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        seq_len=128,
        activation_checkpointing=False,
    )
    torch.manual_seed(7)
    training_model = TrainingModel(tiny_model_config).eval()
    hf_config = AxionConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        rope_theta=tiny_model_config.rope_theta,
        eos_token_id=0,
        pad_token_id=0,
    )
    hf_model = AxionForCausalLM(hf_config).eval()
    mapped_state = {
        "model.embed_tokens.weight": training_model.embed.weight,
        "model.norm.weight": training_model.norm.weight,
        "lm_head.weight": training_model.lm_head.weight,
    }
    suffixes = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn.qkv.weight": "self_attn.qkv_proj.weight",
        "attn.proj.weight": "self_attn.o_proj.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.fc.weight": "mlp.up_proj.weight",
        "mlp.proj.weight": "mlp.down_proj.weight",
    }
    for name, value in training_model.state_dict().items():
        if not name.startswith("layers."):
            continue
        _, layer, suffix = name.split(".", 2)
        mapped_state[f"model.layers.{layer}.{suffixes[suffix]}"] = value
    hf_model.load_state_dict(mapped_state, strict=True)

    token_ids = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        expected_logits, _ = training_model(token_ids)
        actual = hf_model(token_ids)
        prefix = hf_model(token_ids[:, :3], use_cache=True)
        incremental = hf_model(
            token_ids[:, 3:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(actual.logits, expected_logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        incremental.logits[:, -1],
        actual.logits[:, -1],
        atol=2e-6,
        rtol=2e-6,
    )
