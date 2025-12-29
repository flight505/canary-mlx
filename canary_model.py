"""
Canary-Qwen Model Architecture for MLX.

This implements a Speech-Augmented Language Model (SALM) combining:
- FastConformer encoder (audio processing)
- Qwen3-1.7B decoder (language model)
- Linear projection layer (audio -> text embedding space)
- LoRA adapters (fine-tuning)

Architecture based on NVIDIA's Canary-Qwen-2.5B:
https://huggingface.co/nvidia/canary-qwen-2.5b
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from conformer import FastConformerEncoder, create_conformer_from_config


class ModalityAdapter(nn.Module):
    """Projects audio encoder output to LLM embedding space."""

    def __init__(self, encoder_dim: int, llm_dim: int):
        super().__init__()
        self.projection = nn.Linear(encoder_dim, llm_dim, bias=False)

    def __call__(self, audio_features: mx.array) -> mx.array:
        """
        Args:
            audio_features: [batch, audio_seq_len, encoder_dim]

        Returns:
            Projected features: [batch, audio_seq_len, llm_dim]
        """
        return self.projection(audio_features)


class CanaryModel(nn.Module):
    """
    Canary-Qwen hybrid model for speech recognition.

    Combines:
    1. Audio encoder (FastConformer)
    2. Modality adapter (projection)
    3. Text decoder (Qwen3-1.7B)
    """

    def __init__(self, config_path: str, weights_path: str):
        super().__init__()

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Extract dimensions
        enc_cfg = self.config.get("encoder", {})
        self.encoder_dim = enc_cfg.get("d_model", 1024)

        # For Qwen3-1.7B, hidden_size is typically 1536
        llm_cfg = self.config.get("llm", {})
        self.llm_dim = llm_cfg.get("hidden_size", 1536)

        # Build components
        print("Building encoder...")
        self.encoder = create_conformer_from_config(self.config)

        print("Building adapter...")
        self.adapter = ModalityAdapter(self.encoder_dim, self.llm_dim)

        print("Building decoder...")
        # For now, we use a placeholder. In practice, you'd load Qwen3-1.7B
        # from mlx-community/Qwen3-1.7B-bf16 or similar
        # This requires integrating with mlx_lm's Qwen implementation
        self.decoder = self._build_decoder(llm_cfg)

        # Load weights
        print(f"Loading weights from {weights_path}...")
        self.load_weights(weights_path)

        # Audio locator tag (used in prompts)
        self.audio_locator_tag = "<|audioplaceholder|>"

    def _build_decoder(self, llm_cfg: dict):
        """
        Build Qwen3 decoder.

        TODO: Integrate with mlx_lm's Qwen implementation:
        from mlx_lm.models.qwen2 import Model, ModelArgs

        For now, this is a placeholder.
        """
        # Placeholder - in production, load actual Qwen3 model
        class PlaceholderDecoder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def __call__(self, x):
                return x

        return PlaceholderDecoder(self.llm_dim)

    def load_weights(self, weights_path: str):
        """Load converted MLX weights."""
        weights = mx.load(weights_path)
        self.load_weights_dict(weights)

    def load_weights_dict(self, weights: dict):
        """Load weights from dictionary with key mapping."""
        # Create a mapping of our module structure to weight keys
        # This may require custom logic depending on exact key names
        self.update(weights)

    def __call__(
        self,
        audio_features: mx.array,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass combining audio and text.

        Args:
            audio_features: [batch, audio_time, mel_dim=80]
            input_ids: [batch, text_seq_len] - tokenized text
            attention_mask: Optional attention mask

        Returns:
            logits: [batch, total_seq_len, vocab_size]
        """
        # 1. Encode audio
        audio_encoded = self.encoder(audio_features)

        # 2. Project to LLM space
        audio_projected = self.adapter(audio_encoded)

        # 3. Get text embeddings (requires embedding layer from decoder)
        # This is a simplified version - actual implementation needs
        # proper integration with Qwen3's embedding layer
        # text_embedded = self.decoder.embed_tokens(input_ids)

        # 4. Concatenate audio and text along sequence dimension
        # SALM architecture: [audio_features, text_features]
        # combined = mx.concatenate([audio_projected, text_embedded], axis=1)

        # 5. Pass through decoder
        # logits = self.decoder(combined, attention_mask)

        # Placeholder return
        return audio_projected


def load_model(model_dir: str) -> CanaryModel:
    """
    Convenience function to load a converted Canary model.

    Args:
        model_dir: Directory containing config.json and model.safetensors

    Returns:
        CanaryModel instance
    """
    config_path = Path(model_dir) / "config.json"
    weights_path = Path(model_dir) / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    return CanaryModel(str(config_path), str(weights_path))
