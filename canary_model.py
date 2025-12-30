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
from qwen_decoder import Qwen3Decoder, create_qwen_decoder


class ModalityAdapter(nn.Module):
    """Projects audio encoder output to LLM embedding space."""

    def __init__(self, encoder_dim: int, llm_dim: int):
        super().__init__()
        self.projection = nn.Linear(encoder_dim, llm_dim, bias=True)

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

        # For Qwen3-1.7B, hidden_size is 2048
        llm_cfg = self.config.get("llm", {})
        self.llm_dim = llm_cfg.get("hidden_size", 2048)

        # Build components
        print("Building encoder...")
        self.encoder = create_conformer_from_config(self.config)

        print("Building adapter...")
        self.adapter = ModalityAdapter(self.encoder_dim, self.llm_dim)

        print("Building decoder...")
        self.decoder = create_qwen_decoder(self.config)

        # Apply LoRA to decoder layers BEFORE loading weights
        print("Preparing LoRA layers...")
        self.decoder.apply_lora_to_layers()

        # Load weights
        print(f"Loading weights from {weights_path}...")
        self.load_weights_from_file(weights_path)

        # Audio locator tag (used in prompts)
        self.audio_locator_tag = self.config.get("audio_locator_tag", "<|audioplaceholder|>")

    def load_weights_from_file(self, weights_path: str):
        """Load converted MLX weights from file."""
        weights = mx.load(weights_path)
        self.load_weights_dict(weights)

    def load_weights_dict(self, weights: dict):
        """
        Load weights from dictionary with LoRA-aware key mapping.

        Handles the complex NeMo structure including:
        - base_layer.weight (frozen base weights)
        - lora_A.default.weight (LoRA up-projection)
        - lora_B.default.weight (LoRA down-projection)
        """
        mapped_weights = {}
        skipped = []

        for key, value in weights.items():
            new_key = None

            # Map encoder weights: perception.encoder.* -> encoder.*
            if key.startswith("perception.encoder."):
                new_key = key.replace("perception.encoder.", "encoder.")

            # Map projection/adapter: perception.proj.* -> adapter.projection.*
            elif key.startswith("perception.proj."):
                new_key = "adapter." + key.replace("perception.proj.", "projection.")

            # Map embedding: embed_tokens.* -> decoder.model.model.embed_tokens.*
            elif key.startswith("embed_tokens."):
                new_key = "decoder.model.model." + key

            # Map decoder weights with LoRA handling
            elif key.startswith("base_model.model.model."):
                # Remove base_model.model prefix -> decoder.model.model.*
                decoder_key = key.replace("base_model.model.model.", "")

                # Handle LoRA structure
                # Note: convert.py already transposes LoRA weights to MLX format
                # Just rename the keys here
                if ".lora_A.default.weight" in key:
                    new_key = "decoder.model.model." + decoder_key.replace(".lora_A.default.weight", ".lora_a")
                elif ".lora_B.default.weight" in key:
                    new_key = "decoder.model.model." + decoder_key.replace(".lora_B.default.weight", ".lora_b")
                elif ".base_layer.weight" in key:
                    # base_layer.weight -> linear.weight (LoRALinear stores base in .linear)
                    new_key = "decoder.model.model." + decoder_key.replace(".base_layer.weight", ".linear.weight")
                elif ".base_layer.bias" in key:
                    # base_layer.bias -> linear.bias
                    new_key = "decoder.model.model." + decoder_key.replace(".base_layer.bias", ".linear.bias")
                else:
                    # Regular weight (no LoRA)
                    new_key = "decoder.model.model." + decoder_key

            # Skip unrecognized keys
            else:
                skipped.append(key)
                continue

            if new_key:
                # Note: Weights are already in MLX format from convert.py
                # No additional transposition needed here
                mapped_weights[new_key] = value

        # Report skipped keys
        if skipped:
            print(f"  Skipped {len(skipped)} unmapped keys (first 5):")
            for k in skipped[:5]:
                print(f"    - {k}")

        # Update model with mapped weights using load_weights (list of tuples)
        # Use strict=False to allow missing bias parameters (Canary doesn't use bias for attention projections)
        print(f"Loading {len(mapped_weights)} weight tensors...")
        self.load_weights(list(mapped_weights.items()), strict=False)

    def __call__(
        self,
        audio_features: mx.array,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache=None
    ) -> mx.array:
        """
        Forward pass combining audio and text.

        Args:
            audio_features: [batch, audio_time, mel_dim] - raw audio features
            input_ids: [batch, text_seq_len] - tokenized text
            attention_mask: Optional attention mask
            cache: Optional KV cache for generation

        Returns:
            logits: [batch, total_seq_len, vocab_size]
        """
        # 1. Encode audio -> [batch, audio_seq_len, encoder_dim]
        # parakeet Conformer returns (encoded, lengths)
        audio_encoded, _ = self.encoder(audio_features, lengths=None)

        # 2. Project to LLM space -> [batch, audio_seq_len, llm_dim]
        audio_projected = self.adapter(audio_encoded)

        # 3. Get text embeddings -> [batch, text_seq_len, llm_dim]
        text_embedded = self.decoder.get_text_embeddings(input_ids)

        # 4. Concatenate along sequence dimension
        # SALM architecture: [audio_features, text_features]
        inputs_embeds = mx.concatenate([audio_projected, text_embedded], axis=1)

        # 5. Pass through decoder
        logits = self.decoder(inputs_embeds, attention_mask, cache)

        return logits


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
