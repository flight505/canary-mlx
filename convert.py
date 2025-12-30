"""
Convert NVIDIA Canary-Qwen model from NeMo format to MLX format.

Based on research from parakeet-mlx by senstella:
https://github.com/senstella/parakeet-mlx
https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import torch
from safetensors.torch import load_file, save_file


def convert_canary(model_path: str, output_dir: str):
    """Convert Canary weights from NeMo to MLX format."""
    print(f"Loading weights from {model_path}")

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    mlx_state = {}
    stats = {"encoder": 0, "adapter": 0, "decoder": 0, "skipped": 0}

    print("Converting weights...")
    for key, val in state_dict.items():
        original_key = key

        if key.startswith("model."):
            key = key[6:]

        if "num_batches_tracked" in key or "preprocessor" in key:
            stats["skipped"] += 1
            continue

        # --- ENCODER (FastConformer) ---
        if "encoder" in key:
            # Permute convolutions: (Out, In, Len) -> (Out, Len, In)
            # Based on parakeet-mlx conversion approach
            if "conv" in key and val.ndim == 4:
                val = val.permute(0, 2, 3, 1)
            elif "conv" in key and val.ndim == 3:
                val = val.permute(0, 2, 1)

            mlx_state[key] = val.contiguous()
            stats["encoder"] += 1

        # --- ADAPTER/PROJECTION (Audio -> LLM) ---
        elif "adapter" in key or "projection" in key or "modality_adapter" in key or ("perception" in key and "proj" in key):
            # PyTorch and MLX use same convention (out, in) - no transpose needed
            mlx_state[key] = val.contiguous()
            stats["adapter"] += 1

        # --- EMBEDDINGS ---
        elif "embed_tokens" in key:
            # Embedding weights don't need transposition
            mlx_state[key] = val.contiguous()
            stats["decoder"] += 1

        # --- DECODER (Qwen3-1.7B) ---
        elif "llm" in key or "decoder" in key:
            # Strip NeMo prefixes to match standard Qwen structure
            new_key = key.replace("llm.model.", "model.").replace("llm.", "")

            # Handle LoRA weights if present
            # NeMo LoRA format: lora_A (rank, in_features), lora_B (out_features, rank)
            # MLX expects: lora_a (in_features, rank), lora_b (rank, out_features)
            # ONLY LoRA weights need transposition - base decoder weights are already in PyTorch format
            if "lora" in key.lower() and "weight" in key and val.ndim == 2:
                val = val.T
            # Note: Base decoder weights are already in correct format (out_features, in_features)
            # Do NOT transpose them!

            mlx_state[new_key] = val.contiguous()
            stats["decoder"] += 1

        else:
            print(f"  Unhandled key: {original_key}")
            mlx_state[key] = val.contiguous()

    print(f"\nConversion statistics:")
    print(f"  Encoder keys: {stats['encoder']}")
    print(f"  Adapter keys: {stats['adapter']}")
    print(f"  Decoder keys: {stats['decoder']}")
    print(f"  Skipped keys: {stats['skipped']}")

    # Save weights
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.safetensors")
    save_file(mlx_state, output_path)
    print(f"\nWeights saved to {output_path}")

    # --- CONFIGURATION HANDLING ---
    src_dir = Path(model_path).parent
    src_config_path = src_dir / "config.json"

    if src_config_path.exists():
        with open(src_config_path, 'r') as f:
            full_config = json.load(f)

        # Save full config for reference
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(full_config, f, indent=2)
        print(f"Config saved to {output_dir}/config.json")

        # Extract LLM config if nested
        if "llm" in full_config:
            llm_config = full_config["llm"]
            with open(os.path.join(output_dir, "llm_config.json"), 'w') as f:
                json.dump(llm_config, f, indent=2)
            print(f"LLM config saved to {output_dir}/llm_config.json")

    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "special_tokens_map.json", "added_tokens.json"]

    copied = 0
    for filename in os.listdir(src_dir):
        if any(token_file in filename for token_file in tokenizer_files):
            shutil.copy(src_dir / filename, output_dir)
            copied += 1

    if copied > 0:
        print(f"Copied {copied} tokenizer files")

    print("\n✓ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA Canary-Qwen from NeMo to MLX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights (.safetensors or .ckpt/.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mlx-canary",
        help="Output directory for converted model (default: mlx-canary)"
    )

    args = parser.parse_args()
    convert_canary(args.model, args.output)


if __name__ == "__main__":
    main()
