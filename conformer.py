"""
FastConformer encoder implementation for Canary-MLX.

This implementation is adapted from parakeet-mlx (https://github.com/senstella/parakeet-mlx)
which is licensed under Apache 2.0. Modifications have been made to support Canary-Qwen
architecture.

Original Copyright: senstella/parakeet-mlx contributors
Modified for: Canary-Qwen MLX implementation
License: Apache 2.0

Key modifications:
- Adapted for Canary's specific encoder configuration
- Simplified for speech-to-text use case
- Integrated with Qwen decoder pipeline
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ConformerConfig:
    """Configuration for FastConformer encoder."""
    feat_in: int = 128  # Input feature dimension (mel bins)
    n_layers: int = 17  # Number of conformer layers
    d_model: int = 1024  # Model dimension
    n_heads: int = 8  # Number of attention heads
    ff_expansion_factor: int = 4  # Feed-forward expansion
    conv_kernel_size: int = 9  # Convolution kernel size
    subsampling_factor: int = 8  # Temporal downsampling factor
    dropout: float = 0.1


class ConformerFeedForward(nn.Module):
    """Feed-forward module in Conformer block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class ConformerConvolution(nn.Module):
    """Depthwise separable convolution module."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) // 2

        # Pointwise expansion
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=d_model
        )

        self.batch_norm = nn.BatchNorm(d_model)
        self.activation = nn.SiLU()

        # Pointwise compression
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, time, channels]
        # Conv1d expects [batch, time, channels] in MLX

        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=-1)  # Gated activation

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN + Attention + Conv + FFN."""

    def __init__(self, config: ConformerConfig):
        super().__init__()

        # Feed-forward modules (before and after)
        d_ff = config.d_model * config.ff_expansion_factor
        self.ffn1 = ConformerFeedForward(config.d_model, d_ff, config.dropout)
        self.ffn2 = ConformerFeedForward(config.d_model, d_ff, config.dropout)

        # Multi-head self-attention
        self.attention = nn.MultiHeadAttention(
            config.d_model,
            config.n_heads,
            bias=True
        )

        # Convolution module
        self.conv = ConformerConvolution(
            config.d_model,
            config.conv_kernel_size,
            config.dropout
        )

        # Layer norms
        self.norm_ffn1 = nn.LayerNorm(config.d_model)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_conv = nn.LayerNorm(config.d_model)
        self.norm_ffn2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Feed-forward 1 (half-step residual)
        residual = x
        x = self.norm_ffn1(x)
        x = residual + 0.5 * self.ffn1(x)

        # Multi-head attention
        residual = x
        x = self.norm_attn(x)
        x = self.attention(x, x, x, mask)
        x = residual + self.dropout(x)

        # Convolution
        residual = x
        x = self.norm_conv(x)
        x = self.conv(x)
        x = residual + x

        # Feed-forward 2 (half-step residual)
        residual = x
        x = self.norm_ffn2(x)
        x = residual + 0.5 * self.ffn2(x)

        return x


class ConformerSubsampling(nn.Module):
    """Subsampling module with strided convolutions."""

    def __init__(self, feat_in: int, d_model: int, subsampling_factor: int = 8):
        super().__init__()

        # Canary uses 8x subsampling typically with 3 conv layers (2x2x2=8)
        # Each layer has stride 2
        channels = [feat_in, 256, 256, d_model]

        self.conv_layers = []
        for i in range(3):
            self.conv_layers.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            if i < 2:  # Activation for first two layers
                self.conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*self.conv_layers)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, time, feat_in]
        return self.conv_layers(x)


class FastConformerEncoder(nn.Module):
    """
    FastConformer encoder for audio processing.

    Architecture based on NVIDIA NeMo's FastConformer with optimizations
    for efficient inference on Apple Silicon.
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        # Subsampling front-end
        self.subsample = ConformerSubsampling(
            config.feat_in,
            config.d_model,
            config.subsampling_factor
        )

        # Conformer blocks
        self.layers = [
            ConformerBlock(config) for _ in range(config.n_layers)
        ]

        # Output normalization
        self.norm_out = nn.LayerNorm(config.d_model)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input features [batch, time, feat_in]
            mask: Optional attention mask

        Returns:
            Encoded features [batch, time/8, d_model]
        """
        # Subsampling
        x = self.subsample(x)

        # Conformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output normalization
        x = self.norm_out(x)

        return x


def create_conformer_from_config(config_dict: dict) -> FastConformerEncoder:
    """
    Create FastConformer encoder from configuration dictionary.

    Args:
        config_dict: Configuration from model config.json

    Returns:
        FastConformerEncoder instance
    """
    encoder_cfg = config_dict.get("perception", {}).get("encoder", {})

    config = ConformerConfig(
        feat_in=encoder_cfg.get("feat_in", 128),
        n_layers=encoder_cfg.get("n_layers", 17),
        d_model=encoder_cfg.get("d_model", 1024),
        n_heads=encoder_cfg.get("n_heads", 8),
        ff_expansion_factor=encoder_cfg.get("ff_expansion_factor", 4),
        conv_kernel_size=encoder_cfg.get("conv_kernel_size", 9),
        dropout=encoder_cfg.get("dropout", 0.1)
    )

    return FastConformerEncoder(config)
