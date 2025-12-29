"""
Inference script for Canary-Qwen speech recognition.

Handles:
- Audio preprocessing (16kHz, mono, log-mel spectrograms)
- Prompt formatting with audio locator tags
- Greedy decoding generation
"""

import argparse
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoTokenizer

from canary_model import load_model


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file (.wav, .flac, etc.)
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Audio waveform as numpy array, mono, at target sample rate
    """
    # Load with soundfile
    waveform, sample_rate = sf.read(file_path)

    # Convert to torch for resampling
    waveform = torch.from_numpy(waveform).float()

    # Handle stereo -> mono
    if len(waveform.shape) > 1:
        waveform = waveform.mean(dim=-1)

    # Ensure 2D shape [1, samples] for torchaudio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def extract_mel_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 128  # Canary uses 128 mel bins, not 80
) -> mx.array:
    """
    Extract log-mel spectrogram features.

    NeMo Canary uses:
    - Sample rate: 16kHz
    - FFT size: 512
    - Window: 25ms (400 samples @ 16kHz)
    - Hop: 10ms (160 samples @ 16kHz)
    - Mel bins: 128

    Args:
        waveform: Audio waveform [samples]
        sample_rate: Sample rate (default: 16000)
        n_fft: FFT size
        win_length: Window length in samples
        hop_length: Hop length in samples
        n_mels: Number of mel filterbanks

    Returns:
        Log-mel features [time, n_mels]
    """
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform).float()

    # Add channel dimension if needed
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    mel_spec = mel_transform(waveform)  # [1, n_mels, time]

    # Log compression
    log_mel = torch.log(mel_spec + 1e-5)

    # Normalize (per-utterance mean/std normalization)
    mean = log_mel.mean()
    std = log_mel.std()
    log_mel = (log_mel - mean) / (std + 1e-8)

    # Transpose to [time, n_mels] and remove batch dim
    log_mel = log_mel.squeeze(0).T  # [time, n_mels]

    # Convert to MLX array
    return mx.array(log_mel.numpy())


def format_prompt(prompt_text: str, audio_locator: str) -> str:
    """
    Format prompt with audio locator tag.

    Canary expects prompts like:
    "Transcribe the following: <|audioplaceholder|>"

    Args:
        prompt_text: Base prompt (e.g., "Transcribe the following:")
        audio_locator: Audio locator tag

    Returns:
        Formatted prompt string
    """
    return f"{prompt_text} {audio_locator}"


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
    Transcribe audio file.

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
    waveform = load_audio(audio_path)

    print(f"  Duration: {len(waveform) / 16000:.2f}s")
    if len(waveform) / 16000 > 40:
        print("  WARNING: Audio longer than 40s may produce degraded results")

    print("Extracting features...")
    audio_features = extract_mel_features(waveform)
    audio_features = mx.expand_dims(audio_features, 0)  # Add batch dim

    print(f"  Feature shape: {audio_features.shape}")

    # Format prompt with audio locator tag
    full_prompt = format_prompt(prompt, model.audio_locator_tag)
    print(f"Prompt: {full_prompt}")

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
        description="Transcribe audio using Canary-Qwen model"
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
