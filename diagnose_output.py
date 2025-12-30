"""
Diagnostic script to analyze the gibberish output from Canary model.

Investigates:
1. Token distribution patterns
2. Vocabulary biases
3. Logits values and temperature effects
4. Model initialization state
"""

import argparse
from pathlib import Path
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from canary_model import load_model
from transcribe import load_audio, extract_mel_features, format_prompt


def analyze_logits_distribution(logits: mx.array, tokenizer, top_k: int = 20):
    """Analyze the distribution of logits to identify patterns."""
    print("\n" + "="*60)
    print("LOGITS DISTRIBUTION ANALYSIS")
    print("="*60)

    # Get last token's logits
    last_logits = logits[0, -1, :]  # [vocab_size]

    # Statistics
    print(f"\nLogits statistics:")
    print(f"  Mean: {float(mx.mean(last_logits)):.4f}")
    print(f"  Std:  {float(mx.std(last_logits)):.4f}")
    print(f"  Min:  {float(mx.min(last_logits)):.4f}")
    print(f"  Max:  {float(mx.max(last_logits)):.4f}")

    # Check for NaN/Inf
    has_nan = bool(mx.any(mx.isnan(last_logits)))
    has_inf = bool(mx.any(mx.isinf(last_logits)))
    print(f"  NaN values: {has_nan}")
    print(f"  Inf values: {has_inf}")

    # Top-k tokens by logit value
    print(f"\nTop {top_k} tokens by logit value:")
    logits_np = np.array(last_logits)
    top_indices = np.argsort(logits_np)[-top_k:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        token_id = int(idx)
        logit_val = logits_np[idx]
        token_str = tokenizer.decode([token_id])
        print(f"  {rank:2d}. ID {token_id:6d} | logit: {logit_val:8.4f} | \"{token_str}\"")

    # Apply softmax to see probabilities
    print(f"\nTop {top_k} tokens by probability (after softmax):")
    probs = mx.softmax(last_logits, axis=-1)
    probs_np = np.array(probs)
    top_prob_indices = np.argsort(probs_np)[-top_k:][::-1]

    for rank, idx in enumerate(top_prob_indices, 1):
        token_id = int(idx)
        prob = probs_np[idx]
        token_str = tokenizer.decode([token_id])
        print(f"  {rank:2d}. ID {token_id:6d} | prob: {prob:8.6f} ({prob*100:5.2f}%) | \"{token_str}\"")

    # Entropy of distribution
    entropy = -float(mx.sum(probs * mx.log(probs + 1e-10)))
    max_entropy = np.log(tokenizer.vocab_size)
    print(f"\nEntropy: {entropy:.4f} / {max_entropy:.4f} (max)")
    print(f"Normalized entropy: {entropy/max_entropy:.4f} (1.0 = uniform, 0.0 = deterministic)")


def analyze_vocabulary_bias(logits: mx.array, tokenizer):
    """Check if there's bias towards specific language tokens."""
    print("\n" + "="*60)
    print("VOCABULARY BIAS ANALYSIS")
    print("="*60)

    # Define language ranges (approximate, based on Unicode)
    # This is a rough heuristic - proper language detection would need better methods
    language_ranges = {
        'English': (0, 50000),      # ASCII and common English tokens
        'Chinese': (70000, 120000), # CJK characters
        'Arabic': (120000, 130000), # Arabic script
        'Thai': (130000, 140000),   # Thai script
        'Cyrillic': (140000, 150000) # Cyrillic script
    }

    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits, axis=-1)
    probs_np = np.array(probs)

    print("\nProbability mass by language region (approximate):")
    for lang, (start, end) in language_ranges.items():
        end = min(end, tokenizer.vocab_size)
        mass = np.sum(probs_np[start:end])
        print(f"  {lang:12s}: {mass:.6f} ({mass*100:5.2f}%)")


def test_temperature_effect(logits: mx.array, tokenizer, temperatures=[0.1, 0.5, 1.0, 2.0]):
    """Test different temperature settings."""
    print("\n" + "="*60)
    print("TEMPERATURE EFFECT ANALYSIS")
    print("="*60)

    last_logits = logits[0, -1, :]

    for temp in temperatures:
        print(f"\nTemperature = {temp}")
        scaled_logits = last_logits / temp
        probs = mx.softmax(scaled_logits, axis=-1)

        # Top-3 tokens
        probs_np = np.array(probs)
        top_indices = np.argsort(probs_np)[-3:][::-1]

        print(f"  Top 3 tokens:")
        for idx in top_indices:
            token_id = int(idx)
            prob = probs_np[idx]
            token_str = tokenizer.decode([token_id])
            print(f"    ID {token_id:6d} | prob: {prob:.6f} | \"{token_str}\"")


def analyze_weight_initialization(model):
    """Check if weights appear randomly initialized or properly loaded."""
    print("\n" + "="*60)
    print("WEIGHT INITIALIZATION CHECK")
    print("="*60)

    # Check decoder embedding weights
    embed_weights = model.decoder.model.model.embed_tokens.weight
    print(f"\nEmbedding weights:")
    print(f"  Shape: {embed_weights.shape}")
    print(f"  Mean: {float(mx.mean(embed_weights)):.6f}")
    print(f"  Std:  {float(mx.std(embed_weights)):.6f}")
    print(f"  Min:  {float(mx.min(embed_weights)):.6f}")
    print(f"  Max:  {float(mx.max(embed_weights)):.6f}")

    # Check if weights are suspiciously uniform (random init)
    # Well-trained embeddings typically have mean ~0 and std around 0.01-0.1
    mean_val = abs(float(mx.mean(embed_weights)))
    std_val = float(mx.std(embed_weights))

    if mean_val < 0.001 and std_val > 0.5:
        print("  ⚠️  WARNING: Weights look like random initialization (mean~0, high std)")
    elif std_val < 0.001:
        print("  ⚠️  WARNING: Weights have very low variance (possibly zeros or constants)")
    else:
        print("  ✓  Weights appear to be trained (reasonable mean/std)")

    # Check adapter projection
    adapter_weight = model.adapter.projection.weight
    print(f"\nAdapter projection weights:")
    print(f"  Shape: {adapter_weight.shape}")
    print(f"  Mean: {float(mx.mean(adapter_weight)):.6f}")
    print(f"  Std:  {float(mx.std(adapter_weight)):.6f}")


def generate_single_token(model, tokenizer, audio_features, prompt_ids, analyze: bool = True):
    """Generate a single token and analyze the decision."""
    print("\n" + "="*60)
    print("SINGLE TOKEN GENERATION")
    print("="*60)

    # Forward pass
    logits = model(audio_features, prompt_ids)

    print(f"\nInput shapes:")
    print(f"  Audio features: {audio_features.shape}")
    print(f"  Prompt IDs: {prompt_ids.shape}")
    print(f"  Output logits: {logits.shape}")

    # Analyze logits if requested
    if analyze:
        analyze_logits_distribution(logits, tokenizer)
        analyze_vocabulary_bias(logits, tokenizer)
        test_temperature_effect(logits, tokenizer)

    # Generate token (greedy)
    next_token_logits = logits[:, -1, :]
    next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)

    print(f"\nGenerated token:")
    token_id = int(next_token[0, 0])
    token_str = tokenizer.decode([token_id])
    print(f"  ID: {token_id}")
    print(f"  Text: \"{token_str}\"")

    return next_token


def main():
    parser = argparse.ArgumentParser(description="Diagnose Canary model output")
    parser.add_argument("--model-dir", type=str, default="mlx-canary",
                       help="Model directory")
    parser.add_argument("--audio", type=str, required=True,
                       help="Audio file to test")
    parser.add_argument("--prompt", type=str, default="Transcribe the following:",
                       help="Prompt text")
    args = parser.parse_args()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = load_model(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print(f"\nTokenizer info:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model expects: 151936")

    if tokenizer.vocab_size != 151936:
        print(f"  ⚠️  MISMATCH: Tokenizer vocab ({tokenizer.vocab_size}) != Model vocab (151936)")

    # Check weight initialization
    analyze_weight_initialization(model)

    # Load and process audio
    print(f"\nLoading audio: {args.audio}")
    waveform = load_audio(args.audio)
    audio_features = extract_mel_features(waveform)
    audio_features = mx.expand_dims(audio_features, 0)

    # Format prompt
    full_prompt = format_prompt(args.prompt, model.audio_locator_tag)
    print(f"Prompt: \"{full_prompt}\"")

    tokens = tokenizer.encode(full_prompt)
    prompt_ids = mx.array([tokens])

    print(f"Prompt tokens: {tokens}")

    # Generate and analyze first token
    generate_single_token(model, tokenizer, audio_features, prompt_ids, analyze=True)


if __name__ == "__main__":
    main()
