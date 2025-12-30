"""
Extend Qwen3-1.7B tokenizer with Canary-specific special tokens.

This script replicates NVIDIA NeMo's tokenizer extension process:
1. Load base Qwen3-1.7B tokenizer (151,669 tokens)
2. Add 267 Canary-specific tokens (audio, task, language controls)
3. Reach target vocabulary size of 151,936 tokens

Based on NeMo's salm.py and build_canary_2_special_tokenizer.py

Usage:
    python extend_tokenizer.py --output mlx-canary
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer


def create_canary_tokens() -> list:
    """
    Create the 267 Canary-specific special tokens.

    Returns:
        List of 267 special token strings
    """
    # Audio and task control tokens (9 tokens)
    control_tokens = [
        '<|audioplaceholder|>',      # Audio input marker (CRITICAL!)
        '<|startoftranscript|>',     # Transcript start
        '<|endoftranscript|>',       # Transcript end
        '<|transcribe|>',            # Task: Transcription
        '<|translate|>',             # Task: Translation
        '<|pnc|>',                   # With punctuation/capitalization
        '<|nopnc|>',                 # Without punctuation
        '<|notimestamps|>',          # No timestamps in output
        '<|timestamps|>',            # Include timestamps
    ]

    # Language codes (25 tokens for main languages)
    # Based on Canary-1B-v2 which supports 25 languages
    language_codes = [
        'en', 'de', 'fr', 'es', 'it',      # Western European
        'pt', 'ru', 'zh', 'ja', 'ko',      # Major world languages
        'ar', 'hi', 'nl', 'pl', 'tr',      # Middle East, South Asia, Eastern Europe
        'vi', 'th', 'id', 'uk', 'cs',      # Southeast Asia, Eastern Europe
        'sv', 'da', 'fi', 'no', 'ro',      # Nordic, Romanian
    ]
    lang_tokens = [f'<|{lang}|>' for lang in language_codes]

    # Reserved tokens to fill remaining space (233 tokens)
    # 267 total - 9 control - 25 language = 233 reserved
    reserved_tokens = [f'<|reserved_{i}|>' for i in range(233)]

    # Combine all tokens
    canary_tokens = control_tokens + lang_tokens + reserved_tokens

    assert len(canary_tokens) == 267, f"Expected 267 tokens, got {len(canary_tokens)}"

    return canary_tokens


def extend_tokenizer(output_dir: str = "mlx-canary"):
    """
    Extend Qwen3-1.7B tokenizer with Canary special tokens.

    Args:
        output_dir: Directory to save extended tokenizer
    """
    print("="*80)
    print("CANARY-QWEN TOKENIZER EXTENSION")
    print("="*80)

    # Load base Qwen3-1.7B tokenizer
    print("\n1. Loading base Qwen3-1.7B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True
    )

    print(f"   Base tokenizer loaded:")
    print(f"     vocab_size property: {tokenizer.vocab_size}")
    print(f"     len(tokenizer): {len(tokenizer)}")
    print(f"     Gap to fill: {tokenizer.vocab_size - len(tokenizer)} tokens")

    # Create Canary tokens
    print("\n2. Creating 267 Canary-specific tokens...")
    canary_tokens = create_canary_tokens()

    print(f"   Token breakdown:")
    print(f"     Audio/task control: 9 tokens")
    print(f"     Language codes: 25 tokens")
    print(f"     Reserved: 233 tokens")
    print(f"     Total: {len(canary_tokens)} tokens")

    print(f"\n   Sample tokens:")
    print(f"     Control: {canary_tokens[:3]}")
    print(f"     Languages: {canary_tokens[9:14]}")
    print(f"     Reserved: {canary_tokens[-2:]}")

    # Add special tokens to tokenizer
    print("\n3. Adding tokens to tokenizer...")
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': canary_tokens
    })

    print(f"   Tokens added: {num_added}")
    print(f"   New vocabulary size: {len(tokenizer)}")

    # Verify final size
    if len(tokenizer) != tokenizer.vocab_size:
        print(f"\n   ⚠ WARNING: Size mismatch!")
        print(f"     Expected: {tokenizer.vocab_size}")
        print(f"     Actual: {len(tokenizer)}")
        print(f"     Difference: {tokenizer.vocab_size - len(tokenizer)}")
    else:
        print(f"   ✓ Vocabulary size matches config: {len(tokenizer)}")

    # Verify critical token
    print("\n4. Verifying critical tokens...")
    audio_token_id = tokenizer.convert_tokens_to_ids('<|audioplaceholder|>')
    print(f"   <|audioplaceholder|> token ID: {audio_token_id}")

    # Test encoding/decoding
    test_prompt = "<|en|><|transcribe|><|notimestamps|> Transcribe the following: <|audioplaceholder|>"
    encoded = tokenizer.encode(test_prompt)
    decoded = tokenizer.decode(encoded)

    print(f"\n   Test prompt encoding:")
    print(f"     Input: {test_prompt}")
    print(f"     Token IDs: {encoded}")
    print(f"     Decoded: {decoded}")
    print(f"     Tokens: {len(encoded)}")

    # Save extended tokenizer
    print(f"\n5. Saving extended tokenizer to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_dir)

    print(f"   ✓ Tokenizer saved!")
    print(f"\n   Files saved:")
    for file in sorted(output_path.glob("*")):
        if file.is_file():
            print(f"     - {file.name}")

    # Final summary
    print("\n" + "="*80)
    print("TOKENIZER EXTENSION COMPLETE")
    print("="*80)
    print(f"\nFinal vocabulary size: {len(tokenizer)} tokens")
    print(f"Audio placeholder token ID: {audio_token_id}")
    print(f"\nNext steps:")
    print(f"  1. Test transcription with updated tokenizer:")
    print(f"     python transcribe.py --model-dir {output_dir} --audio test_audio/macos_speech.wav")
    print(f"  2. Run WER evaluation:")
    print(f"     python evaluation.py --model-dir {output_dir} --num-samples 10")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Extend Qwen3-1.7B tokenizer with Canary special tokens"
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
