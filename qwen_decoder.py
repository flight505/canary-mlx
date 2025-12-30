"""
Qwen3 decoder integration for Canary-MLX with LoRA support.

Uses mlx_lm's built-in LoRA infrastructure for proper weight handling.
Integrates mlx_lm's Qwen2 implementation with LoRA adapters for
the Canary-Qwen speech-to-text model.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers


def add_qk_norm_to_qwen(model: Qwen2Model, head_dim: int = 128, num_layers: int = 28):
    """
    Add QK-normalization to Qwen2 model (Qwen3 feature).

    Args:
        model: Qwen2Model instance
        head_dim: Dimension per attention head (default: 128 for Qwen3-1.7B)
        num_layers: Number of layers to add QK-norm to (default: 28, layers 0-27)
    """
    for i in range(num_layers):
        layer = model.model.layers[i]
        # Add RMSNorm for Q and K projections
        layer.self_attn.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        layer.self_attn.k_norm = nn.RMSNorm(head_dim, eps=1e-6)


class Qwen3Decoder(nn.Module):
    """
    Qwen3-1.7B decoder with LoRA adapters for Canary-Qwen.

    Uses mlx_lm's LoRA infrastructure to properly handle LoRA weights
    for q_proj and v_proj layers as used in Canary training.
    """

    def __init__(self, config: dict):
        super().__init__()

        # Extract Qwen config from Canary config
        self.config = config

        # Build Qwen2 model args
        # For Canary-Qwen: decoder has 28 layers with LoRA (NOT 32 - that was confused with encoder)
        self.model_args = Qwen2ModelArgs(
            model_type="qwen2",
            hidden_size=2048,  # Qwen3-1.7B dimension
            num_hidden_layers=28,  # Canary decoder has 28 layers
            intermediate_size=6144,
            num_attention_heads=16,
            num_key_value_heads=8,  # GQA with 8 KV heads (not 16)
            vocab_size=151936,
            rope_theta=1000000.0,
            rope_traditional=False,
            rope_scaling=None,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True  # lm_head shares weights with embed_tokens
        )

        # Build base model (already includes embed_tokens)
        self.model = Qwen2Model(self.model_args)

        # Add QK-normalization (Qwen3 feature)
        # Head dim = hidden_size / num_heads = 2048 / 16 = 128
        head_dim = self.model_args.hidden_size // self.model_args.num_attention_heads
        add_qk_norm_to_qwen(self.model, head_dim=head_dim)

        # LoRA configuration
        lora_cfg = config.get("lora", {})
        self.lora_rank = lora_cfg.get("r", 128)
        self.lora_alpha = lora_cfg.get("lora_alpha", 256)
        self.lora_scale = self.lora_alpha / self.lora_rank
        self.lora_dropout = lora_cfg.get("lora_dropout", 0.01)

        # Flag to track if LoRA has been applied
        self._lora_applied = False

    @property
    def layers(self):
        """Expose layers property for LoRA conversion."""
        return self.model.layers

    def apply_lora_to_layers(self):
        """
        Convert q_proj and v_proj layers to LoRA-enabled versions.

        This applies LoRA to layers 0-27 (28 layers total) where LoRA
        weights exist in the Canary model.
        """
        if self._lora_applied:
            return

        print("Converting layers to LoRA...")

        # Manually convert q_proj and v_proj in layers 0-27
        for layer_idx in range(28):  # Layers 0-27 have LoRA
            layer = self.model.layers[layer_idx]

            # Convert q_proj to LoRA
            q_proj = layer.self_attn.q_proj
            layer.self_attn.q_proj = LoRALinear.from_base(
                q_proj,
                r=self.lora_rank,
                dropout=self.lora_dropout,
                scale=self.lora_scale
            )

            # Convert v_proj to LoRA
            v_proj = layer.self_attn.v_proj
            layer.self_attn.v_proj = LoRALinear.from_base(
                v_proj,
                r=self.lora_rank,
                dropout=self.lora_dropout,
                scale=self.lora_scale
            )

        self._lora_applied = True
        print(f"Applied LoRA (rank={self.lora_rank}, scale={self.lora_scale}) to 28 layers")

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
        # mlx_lm's Qwen2Model uses 'input_embeddings' parameter
        # Pass None as input_ids since we're providing embeddings directly
        # Note: self.model is actually mlx_lm's Model wrapper (imported as Qwen2Model),
        # which already includes the lm_head logic, so this returns logits, not hidden states
        return self.model(None, cache=cache, input_embeddings=inputs_embeds)

    def get_text_embeddings(self, input_ids: mx.array) -> mx.array:
        """Get embeddings for text token IDs."""
        return self.model.model.embed_tokens(input_ids)


def create_qwen_decoder(config: dict) -> Qwen3Decoder:
    """
    Create Qwen3 decoder from Canary configuration.

    Args:
        config: Full Canary model configuration

    Returns:
        Qwen3Decoder instance
    """
    return Qwen3Decoder(config)
