"""
Fix tokenizer vocabulary size mismatch for Canary-Qwen model.

The model expects 151,936 tokens but Qwen3-1.7B tokenizer only has 151,669.
This script adds the missing 267 Canary-specific tokens.
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def extend_tokenizer(output_dir: str):
    """
    Extend Qwen3-1.7B tokenizer with Canary-specific tokens.

    Args:
        output_dir: Directory to save the extended tokenizer
    """
    print("Loading Qwen3-1.7B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', trust_remote_code=True)

    print(f"Original tokenizer:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Total length: {len(tokenizer)}")

    # Calculate missing tokens
    current_size = len(tokenizer)
    target_size = 151936
    tokens_to_add = target_size - current_size

    print(f"\nNeed to add: {tokens_to_add} tokens")

    # Define Canary-specific special tokens
    # Based on NVIDIA Canary documentation and ASR conventions
    canary_tokens = [
        '<|audioplaceholder|>',      # Audio input placeholder
        '<|startoftranscript|>',     # Transcript start
        '<|endoftranscript|>',       # Transcript end
        '<|transcribe|>',            # Task: transcription
        '<|translate|>',             # Task: translation
        '<|pnc|>',                   # With punctuation/capitalization
        '<|nopnc|>',                 # Without punctuation/capitalization
        '<|notimestamps|>',          # Without timestamps
        '<|timestamps|>',            # With timestamps
    ]

    # Language control tokens (Canary supports 25+ languages)
    # ISO 639-1 codes for supported languages
    languages = [
        'en',  # English
        'de',  # German
        'fr',  # French
        'es',  # Spanish
        'it',  # Italian
        'pt',  # Portuguese
        'ru',  # Russian
        'zh',  # Chinese
        'ja',  # Japanese
        'ko',  # Korean
        'ar',  # Arabic
        'hi',  # Hindi
        'nl',  # Dutch
        'pl',  # Polish
        'tr',  # Turkish
        'vi',  # Vietnamese
        'th',  # Thai
        'id',  # Indonesian
        'uk',  # Ukrainian
        'cs',  # Czech
        'sv',  # Swedish
        'da',  # Danish
        'fi',  # Finnish
        'no',  # Norwegian
        'ro',  # Romanian
    ]

    for lang in languages:
        canary_tokens.append(f'<|{lang}|>')

    # Add reserved tokens to reach exactly 267
    reserved_count = tokens_to_add - len(canary_tokens)
    for i in range(reserved_count):
        canary_tokens.append(f'<|reserved_{i}|>')

    print(f"\nAdding {len(canary_tokens)} special tokens:")
    print(f"  Audio/task tokens: 9")
    print(f"  Language tokens: {len(languages)}")
    print(f"  Reserved tokens: {reserved_count}")
    print(f"\nFirst 20 tokens: {canary_tokens[:20]}")

    # Add tokens to tokenizer
    print("\nExtending tokenizer...")
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': canary_tokens
    })

    print(f"Added {num_added} tokens")
    print(f"New tokenizer size: {len(tokenizer)}")

    if len(tokenizer) != target_size:
        print(f"\n⚠️  WARNING: Expected {target_size}, got {len(tokenizer)}")
    else:
        print(f"\n✓ Tokenizer size matches model requirements: {target_size}")

    # Save extended tokenizer
    print(f"\nSaving tokenizer to {output_dir}...")
    tokenizer.save_pretrained(output_dir)

    # Verify saved tokenizer
    print("\nVerifying saved tokenizer...")
    test_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    print(f"  Loaded vocab size: {test_tokenizer.vocab_size}")
    print(f"  Loaded total length: {len(test_tokenizer)}")

    # Test encoding/decoding
    test_text = "Hello world"
    tokens = test_tokenizer.encode(test_text)
    decoded = test_tokenizer.decode(tokens)
    print(f"\nTest encoding/decoding:")
    print(f"  Input: \"{test_text}\"")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: \"{decoded}\"")

    # Test Canary-specific tokens
    canary_prompt = "Transcribe the following: <|audioplaceholder|>"
    tokens = test_tokenizer.encode(canary_prompt)
    decoded = test_tokenizer.decode(tokens)
    print(f"\nCanary prompt test:")
    print(f"  Input: \"{canary_prompt}\"")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: \"{decoded}\"")

    print("\n✓ Tokenizer extension complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extend Qwen3 tokenizer with Canary-specific tokens"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mlx-canary",
        help="Output directory for extended tokenizer (default: mlx-canary)"
    )

    args = parser.parse_args()
    extend_tokenizer(args.output)


if __name__ == "__main__":
    main()
