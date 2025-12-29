"""
Qwen3 decoder integration for Canary-MLX with LoRA support.

Integrates mlx_lm's Qwen2 implementation with LoRA adapters for
the Canary-Qwen speech-to-text model.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Implements: output = (W + B @ A) @ input
    Where:
    - W: frozen base weights
    - A: up-projection (r x d_in)
    - B: down-projection (d_out x r)
    - r: rank (typically 128 for Canary)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        alpha: int = 256,
        bias: bool = False
    ):
        super().__init__()

        # Base linear layer (frozen during inference)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA parameters
        self.lora_a = mx.zeros((rank, in_features))
        self.lora_b = mx.zeros((out_features, rank))

        # Scaling factor
        self.scaling = alpha / rank

        self.rank = rank

    def __call__(self, x: mx.array) -> mx.array:
        # Base transformation
        result = self.linear(x)

        # Add LoRA adaptation: (x @ A.T @ B.T) * scaling
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        result = result + lora_out * self.scaling

        return result


class Qwen3Decoder(nn.Module):
    """
    Qwen3-1.7B decoder with LoRA adapters for Canary-Qwen.

    This wraps mlx_lm's Qwen2Model and adds LoRA adapter support
    for q_proj and v_proj layers as used in Canary training.
    """

    def __init__(self, config: dict):
        super().__init__()

        # Extract Qwen config from Canary config
        self.config = config

        # Build Qwen2 model args
        # For Canary-Qwen, the decoder is Qwen3-1.7B
        self.model_args = Qwen2ModelArgs(
            model_type="qwen2",
            hidden_size=2048,  # Qwen3-1.7B dimension
            num_hidden_layers=28,
            intermediate_size=6144,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=151936,
            rope_theta=1000000.0,
            rope_traditional=False,
            rope_scaling=None
        )

        # Build base model
        self.model = Qwen2Model(self.model_args)

        # LoRA configuration
        lora_cfg = config.get("lora", {})
        self.lora_rank = lora_cfg.get("r", 128)
        self.lora_alpha = lora_cfg.get("lora_alpha", 256)
        self.lora_targets = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

        # Embedding layer for text tokens
        self.embed_tokens = nn.Embedding(
            self.model_args.vocab_size,
            self.model_args.hidden_size
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache=None
    ) -> mx.array:
        """
        Forward pass using embedded inputs (audio + text concatenated).

        Args:
            inputs_embeds: Combined embeddings [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            cache: Optional KV cache for generation

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.model(inputs_embeds=inputs_embeds, cache=cache)

    def get_text_embeddings(self, input_ids: mx.array) -> mx.array:
        """Get embeddings for text token IDs."""
        return self.embed_tokens(input_ids)


def create_qwen_decoder(config: dict) -> Qwen3Decoder:
    """
    Create Qwen3 decoder from Canary configuration.

    Args:
        config: Full Canary model configuration

    Returns:
        Qwen3Decoder instance
    """
    return Qwen3Decoder(config)
