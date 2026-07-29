from __future__ import annotations

import math
import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_axion import AxionConfig


_MASK_ARGUMENTS = inspect.signature(create_causal_mask).parameters


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


class AxionRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.variance_epsilon)


class AxionRotaryEmbedding(nn.Module):
    def __init__(self, config: AxionConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                / self.head_dim
            )
        )
        inv_freq = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            frequencies = (inv_freq.float() @ positions).transpose(1, 2)
        return frequencies.cos().to(hidden_states.dtype), frequencies.sin().to(hidden_states.dtype)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        output = torch.empty_like(tensor)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output

    return rotate(query), rotate(key)


class AxionAttention(nn.Module):
    is_causal = True

    def __init__(self, config: AxionConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query_states, key_states, value_states = qkv.unbind(dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            *position_embeddings,
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(
                self.config._attn_implementation,
                eager_attention_forward,
            )
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.config.hidden_size,
        )
        return self.o_proj(attention_output), attention_weights


class AxionMLP(nn.Module):
    def __init__(self, config: AxionConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(hidden_states)))


class AxionDecoderLayer(nn.Module):
    def __init__(self, config: AxionConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = AxionAttention(config, layer_idx)
        self.mlp = AxionMLP(config)
        self.input_layernorm = AxionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = AxionRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states, attention_weights = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, attention_weights


class AxionPreTrainedModel(PreTrainedModel):
    config_class = AxionConfig
    base_model_prefix = "model"
    _no_split_modules = ["AxionDecoderLayer"]
    _supports_attention_backend = True
    _supports_sdpa = True
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class AxionModel(AxionPreTrainedModel):
    _supports_attention_backend = True

    def __init__(self, config: AxionConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [AxionDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = AxionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = AxionRotaryEmbedding(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        cache_position = kwargs.pop("cache_position", None)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        past_length = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": self.config,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if "cache_position" in _MASK_ARGUMENTS:
            mask_kwargs["input_embeds"] = inputs_embeds
            mask_kwargs["cache_position"] = cache_position
        else:
            mask_kwargs["inputs_embeds"] = inputs_embeds
        causal_mask = create_causal_mask(**mask_kwargs)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        all_attentions: list[torch.Tensor] | None = [] if output_attentions else None
        for decoder_layer in self.layers:
            hidden_states, attention_weights = decoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
                **kwargs,
            )
            if all_attentions is not None and attention_weights is not None:
                all_attentions.append(attention_weights)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )


class AxionForCausalLM(AxionPreTrainedModel, GenerationMixin):
    def __init__(self, config: AxionConfig) -> None:
        super().__init__(config)
        self.model = AxionModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        cache_position = kwargs.pop("cache_position", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
        )
