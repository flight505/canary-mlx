# Analysis: Gibberish Output from Canary Model

## Executive Summary

The Canary model is generating mixed-language gibberish tokens because of a **critical vocabulary size mismatch** between the tokenizer and the model's embedding layer, combined with a **severe Chinese token bias** in the output distribution.

## Key Findings

### 1. Vocabulary Size Mismatch (CRITICAL)

**Problem:**
- Model embedding layer expects: **151,936 tokens**
- Tokenizer provides: **151,643 tokens** (base vocab) or **151,669 tokens** (with added tokens)
- Gap: **267-293 tokens missing**

**Evidence:**
```
Tokenizer info:
  Vocab size: 151643
  Model expects: 151936
  ⚠️  MISMATCH: Tokenizer vocab (151643) != Model vocab (151936)
```

**Impact:**
When the model generates token IDs in the range 151,643-151,936, the tokenizer cannot decode them properly, resulting in:
- Out-of-bounds token IDs
- Undefined behavior during decoding
- Potential fallback to random/default tokens

### 2. Chinese Token Bias (72% of probability mass)

**Problem:**
The model is heavily biased toward Chinese tokens, with 72% of the probability mass concentrated in the Chinese token range.

**Evidence from diagnostic output:**
```
Probability mass by language region (approximate):
  English     : 0.115495 (11.55%)
  Chinese     : 0.721682 (72.17%)  ← SEVERE BIAS
  Arabic      : 0.045520 ( 4.55%)
  Thai        : 0.027761 ( 2.78%)
  Cyrillic    : 0.025192 ( 2.52%)
```

**Top predicted token:**
```
Generated token:
  ID: 104301
  Text: "这就是"  (Chinese: "this is")
```

### 3. Why Mixed Language Tokens?

The gibberish pattern shows tokens from multiple languages:
```
"至上——ognition规模化روuffixceпоч养护ogleIBivorหาย..."
```

This is caused by:

1. **High entropy distribution** (normalized entropy: 0.5343)
   - Not confident enough for deterministic decoding
   - Probability spread across many tokens

2. **Language region overlap**
   - Top 20 tokens include: English ("able", "ely"), Chinese ("这就是", "马拉"), Vietnamese ("ầm"), Japanese ("ロ"), Thai, etc.
   - Model is "confused" about which language to generate

3. **Tokenizer decoding artifacts**
   - When token IDs exceed vocab size (151,643), decoding may produce garbage
   - Byte-level fallback creates partial UTF-8 sequences

### 4. Is This Random Initialization?

**No** - the weights appear properly trained:

```
Embedding weights:
  Shape: (151936, 2048)
  Mean: -0.000112
  Std:  0.034424
  Min:  -0.353516
  Max:  0.292969
  ✓  Weights appear to be trained (reasonable mean/std)
```

Well-trained embeddings typically have:
- Mean ≈ 0 ✓
- Std ≈ 0.01-0.1 ✓
- Reasonable min/max range ✓

The model is **trained**, but producing incorrect outputs due to configuration issues.

## Root Cause Analysis

### Primary Cause: Extended Tokenizer vs Standard Tokenizer

The model was likely trained with **Qwen3-1.7B's extended tokenizer** (151,936 tokens), which includes:
- Base vocab: 151,643 tokens
- Added special tokens: 26 tokens (IDs 151643-151668)
- **Additional extended tokens: 267-293 tokens (IDs 151669-151935)**

We are using the **standard tokenizer** which only has 151,669 tokens total.

### Secondary Cause: Incorrect Tokenizer Source

We downloaded the tokenizer from `Qwen/Qwen2-1.5B` instead of `Qwen/Qwen3-1.7B`:
- Qwen2 tokenizers have different configurations
- Qwen3-1.7B has extended special tokens for vision, tool calling, reasoning
- The model card states: "The tokenizer was inherited from `Qwen/Qwen3-1.7B`"

### Contributing Factor: Chinese Token Bias

Even with the correct tokenizer, the model shows extreme bias toward Chinese tokens (72%). This suggests:

1. **Training data distribution issue**: Model may have been fine-tuned on predominantly Chinese audio data
2. **LoRA adapter misconfiguration**: LoRA adapters may be reinforcing Chinese tokens
3. **Prompt formatting issue**: Missing language control tokens that Canary models typically use

## Comparison: Expected vs Actual Behavior

### Expected (Canary models)
- Use language control prompts: `<|transcribe|> en` for English
- Balanced language distribution based on audio content
- Proper handling of multilingual input

### Actual (our implementation)
- No language control in prompt
- Extreme Chinese bias regardless of audio content
- Vocabulary mismatch causing decoding errors

## Solutions

### Solution 1: Use Correct Tokenizer (IMMEDIATE FIX)

```python
# Instead of:
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

# Use:
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
```

**Verify vocab size matches:**
```python
assert len(tokenizer) == 151669  # Base + added tokens
# But model expects 151936, so we may need extended version
```

### Solution 2: Check for Extended Tokenizer Files

The model may require additional tokenizer files that extend the vocabulary to 151,936 tokens. Check the original NVIDIA model repository for:
- Extended vocab.json
- Additional special tokens
- Custom tokenizer class

### Solution 3: Add Language Control to Prompts

Canary models typically use language control prompts:

```python
# Current prompt:
"Transcribe the following: <|audioplaceholder|>"

# Should be:
"<|transcribe|> en Transcribe the following: <|audioplaceholder|>"
# or
"<|startoftranscript|><|en|><|transcribe|><|notimestamps|> <|audioplaceholder|>"
```

### Solution 4: Verify LoRA Weights

Check if LoRA weights are being loaded correctly:
- LoRA may be introducing bias if trained on Chinese-heavy data
- Verify LoRA scaling factor (currently 2.0)
- Check if base weights vs LoRA weights have proper balance

## Testing Recommendations

1. **Test with correct tokenizer first**
   - Download Qwen3-1.7B tokenizer
   - Verify vocab size matches 151,936
   - Test single token generation

2. **Test with language control**
   - Add explicit language tokens to prompt
   - Test English, Chinese, and multilingual audio

3. **Test temperature/sampling**
   - Current: greedy decoding
   - Try: temperature=0.1 (more deterministic)
   - Try: top-k/top-p sampling

4. **Verify conversion script**
   - Ensure all tokenizer files were copied
   - Check for extended vocab files
   - Verify special token mappings

## Expected Outcome After Fixes

With the correct tokenizer and language control:

```
Input: "Hello, how are you today?"
Expected: "Hello, how are you today?"
Current: "这就是ognition规模化..."  ← BROKEN
```

## References

- Model card: https://huggingface.co/nvidia/canary-qwen-2.5b
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen3 tokenizer: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/tokenizer_config.json

## Next Steps

1. Update conversion script to use Qwen3-1.7B tokenizer
2. Investigate extended vocabulary (151,669 → 151,936)
3. Add language control tokens to prompts
4. Test with English audio and verify output
5. If still biased, check LoRA adapter configuration
