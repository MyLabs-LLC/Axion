from __future__ import annotations

from transformers import PretrainedConfig


class AxionConfig(PretrainedConfig):
    model_type = "axion"

    def __init__(
        self,
        vocab_size: int = 16_384,
        hidden_size: int = 2_048,
        intermediate_size: int = 8_192,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        max_position_embeddings: int = 32_768,
        rope_theta: float = 500_000.0,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        **kwargs: object,
    ) -> None:
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if hidden_act != "gelu":
            raise ValueError("Axion checkpoints require hidden_act='gelu'")
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        # Transformers 5 no longer preserves arbitrary config kwargs as
        # attributes, while the generation path reads this value directly.
        self.use_cache = use_cache
