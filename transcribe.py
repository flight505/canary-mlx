"""
Inference script for Canary-Qwen speech recognition.

Handles:
- Audio preprocessing using parakeet-mlx (NeMo-compatible)
- Prompt formatting with audio locator tags
- Greedy decoding generation
"""

import argparse
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from parakeet_mlx.audio import PreprocessArgs, get_logmel, load_audio
from transformers import AutoTokenizer

from canary_model import load_model


def create_canary_preprocessor_config() -> PreprocessArgs:
    """
    Create preprocessor configuration for Canary-Qwen model.

    Matches NeMo Canary preprocessing parameters using parakeet-mlx's
    PreprocessArgs class.

    Note: hop_length and win_length are computed from window_stride and window_size:
    - hop_length = int(window_stride * sample_rate) = 160 samples (10ms @ 16kHz)
    - win_length = int(window_size * sample_rate) = 400 samples (25ms @ 16kHz)
    """
    return PreprocessArgs(
        sample_rate=16000,
        window_size=0.025,  # 25ms window (= 400 samples @ 16kHz)
        window_stride=0.01,  # 10ms stride (= 160 samples @ 16kHz)
        window="hann",  # Window function
        features=128,  # Canary uses 128 mel bins (not default 80)
        n_fft=512,
        normalize="per_feature",  # Per mel-bin normalization
        preemph=0.97,  # NeMo preemphasis
        dither=1e-05,  # Dithering value
        pad_to=0,  # No padding
        pad_value=0.0,
        mag_power=2.0,  # Power for magnitude calculation
    )


def extract_mel_features(
    audio_path: str,
    preprocessor_config: PreprocessArgs
) -> mx.array:
    """
    Extract log-mel spectrogram features using parakeet-mlx preprocessing.

    This ensures NeMo-compatible preprocessing that matches what the
    FastConformer encoder was trained with.

    Args:
        audio_path: Path to audio file
        preprocessor_config: Preprocessor configuration

    Returns:
        Log-mel features [time, n_mels]
    """
    # Load audio using parakeet-mlx (handles resampling)
    audio = load_audio(audio_path, preprocessor_config.sample_rate)

    # Extract log-mel features using parakeet-mlx
    # This applies NeMo-compatible preprocessing:
    # - Preemphasis filter (0.97)
    # - Mel spectrogram
    # - Log compression
    # - Normalization
    mel = get_logmel(audio, preprocessor_config)

    return mel


def format_prompt(
    prompt_text: str,
    audio_locator: str,
    language: str = "en",
    task: str = "transcribe",
    timestamps: bool = False
) -> str:
    """
    Format prompt with Canary control tokens and audio locator.

    Canary expects prompts with language and task specification:
    "<|en|><|transcribe|><|notimestamps|> Transcribe the following: <|audioplaceholder|>"

    Args:
        prompt_text: Base prompt (e.g., "Transcribe the following:")
        audio_locator: Audio locator tag (default: "<|audioplaceholder|>")
        language: Language code (default: "en")
        task: Task type - "transcribe" or "translate" (default: "transcribe")
        timestamps: Whether to include timestamps (default: False)

    Returns:
        Formatted prompt string with Canary control tokens
    """
    # Canary control tokens
    lang_token = f"<|{language}|>"
    task_token = f"<|{task}|>"
    time_token = "<|timestamps|>" if timestamps else "<|notimestamps|>"

    # Format: <lang><task><timestamps> prompt text <audio>
    return f"{lang_token}{task_token}{time_token} {prompt_text} {audio_locator}"


def greedy_generate(
    model,
    tokenizer,
    audio_features: mx.array,
    prompt_ids: mx.array,
    max_tokens: int = 128,
    eos_token_id: Optional[int] = None
) -> mx.array:
    """
    Generate tokens using greedy decoding.

    Args:
        model: CanaryModel instance
        tokenizer: Tokenizer
        audio_features: Processed audio [1, time, mel_dim]
        prompt_ids: Prompt token IDs [1, prompt_len]
        max_tokens: Maximum tokens to generate
        eos_token_id: End-of-sequence token ID

    Returns:
        Generated token IDs [1, generated_len]
    """
    # Start with prompt
    generated_ids = prompt_ids

    # Get EOS token if not provided
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    for _ in range(max_tokens):
        # Forward pass
        logits = model(audio_features, generated_ids)

        # Get next token (greedy - take argmax of last position)
        next_token_logits = logits[:, -1, :]  # [1, vocab_size]
        next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)  # [1, 1]

        # Append to generated sequence
        generated_ids = mx.concatenate([generated_ids, next_token], axis=1)

        # Check for EOS
        if int(next_token[0, 0]) == eos_token_id:
            break

    # Return only the newly generated tokens (exclude prompt)
    return generated_ids[:, prompt_ids.shape[1]:]


def transcribe(
    model,
    tokenizer,
    audio_path: str,
    prompt: str = "Transcribe the following:",
    max_tokens: int = 128
) -> str:
    """
    Transcribe audio file using parakeet-mlx preprocessing.

    Args:
        model: CanaryModel instance
        tokenizer: Tokenizer for text encoding/decoding
        audio_path: Path to audio file
        prompt: Base prompt text
        max_tokens: Maximum tokens to generate

    Returns:
        Transcribed text
    """
    print(f"Loading audio: {audio_path}")

    # Create Canary preprocessor config
    preprocessor_config = create_canary_preprocessor_config()

    # Load and preprocess audio using parakeet-mlx
    audio_features = extract_mel_features(audio_path, preprocessor_config)

    # parakeet's get_logmel returns shape [1, 1, time, n_mels]
    # We need [1, time, n_mels] for the model
    if audio_features.ndim == 4:
        audio_features = audio_features.squeeze(1)  # Remove extra dimension

    print(f"  Feature shape: {audio_features.shape}")
    print(f"  Using parakeet-mlx preprocessing (NeMo-compatible)")
    print(f"    - Sample rate: {preprocessor_config.sample_rate} Hz")
    print(f"    - Mel bins: {preprocessor_config.features}")
    print(f"    - Preemphasis: {preprocessor_config.preemph}")
    print(f"    - FFT size: {preprocessor_config.n_fft}")
    print(f"    - Window: {int(preprocessor_config.window_size * preprocessor_config.sample_rate)} samples ({preprocessor_config.window_size*1000:.0f}ms)")
    print(f"    - Hop: {int(preprocessor_config.window_stride * preprocessor_config.sample_rate)} samples ({preprocessor_config.window_stride*1000:.0f}ms)")

    # Format prompt with audio locator tag
    full_prompt = format_prompt(prompt, model.audio_locator_tag)
    print(f"\nPrompt: {full_prompt}")

    # Tokenize prompt
    tokens = tokenizer.encode(full_prompt)
    prompt_ids = mx.array([tokens])  # Add batch dim

    print(f"Generating (max {max_tokens} tokens)...")

    # Generate transcription
    generated_ids = greedy_generate(
        model,
        tokenizer,
        audio_features,
        prompt_ids,
        max_tokens=max_tokens
    )

    # Decode generated tokens
    transcription = tokenizer.decode(generated_ids[0].tolist())

    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Canary-Qwen model with parakeet-mlx preprocessing"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="mlx-canary",
        help="Directory containing converted model"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the following:",
        help="Prompt text for transcription"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Check paths
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return

    if not Path(args.model_dir).exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        print("\nRun conversion first:")
        print("  python convert.py --model <path> --output mlx-canary")
        return

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("\nMake sure tokenizer files were copied during conversion")
        return

    # Load model
    print("Loading model...")
    try:
        model = load_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Transcribe
    result = transcribe(
        model,
        tokenizer,
        args.audio,
        args.prompt,
        args.max_tokens
    )

    print("\n" + "="*60)
    print("TRANSCRIPTION")
    print("="*60)
    print(result)
    print("="*60)


if __name__ == "__main__":
    main()
