# Canary-Qwen MLX Conversion Debugging Log

## Current Status (2025-12-30)

**Issue**: Model outputs gibberish for both text-only and audio input, despite all weights being verified as correct.

## Key Findings

### ✅ Verified Correct
1. **Base Qwen weights**: Identical to official Qwen3-1.7B (0.000000 difference)
2. **LoRA weights**: Match original file perfectly (0.000000 difference after transpose)
3. **Embedding table**: 151,936 tokens, all rows match official Qwen
4. **Architecture**: 28 layers, correct dimensions, GQA with 8 KV heads
5. **Weight loading**: All 1,684 weights loaded successfully
6. **Model structure**: Using mlx_lm's Model wrapper (includes lm_head logic)
7. **Output format**: Decoder correctly returns logits, not hidden states

### ❌ Issues Identified
1. **LoRA causes divergence**: With same inputs, LoRA model produces very different outputs from base Qwen
   - Layer 0 q_proj output diff: max=13.5, mean=2.3 (huge!)
   - This is expected since LoRA was trained for audio+text, not pure text

2. **Weight transposition bugs (FIXED)**:
   - LoRA weights were correctly transposed in convert.py
   - Base linear weights were correctly NOT transposed

3. **Missing output projection (INVESTIGATED)**:
   - Initially thought decoder was returning hidden states
   - Actually using mlx_lm's Model wrapper which includes lm_head
   - The import `from mlx_lm.models.qwen2 import Model as Qwen2Model` was confusing

4. **Dropout during inference (FIXED)**:
   - Model was in training mode by default
   - Fixed by calling `model.eval()`
   - However, still produces gibberish even in eval mode

## Detailed Investigation

### Weight Comparison Results
```
Decoder MLP down_proj:
  Original shape: (2048, 6144)
  Converted shape: (2048, 6144)
  Max diff: 0.000000 ✓

LoRA A (q_proj):
  Original (NeMo): (128, 2048)
  Loaded (MLX): (2048, 128)
  Max diff after transpose: 0.000000 ✓

LoRA B (q_proj):
  Original (NeMo): (2048, 128)
  Loaded (MLX): (128, 2048)
  Max diff after transpose: 0.000000 ✓
```

### Forward Pass Analysis
```
Embeddings: IDENTICAL
Layer 0 output: DIVERGES (max diff: 1.74)
Final logits: VERY DIFFERENT (max diff: 18.7)

With LoRA disabled (scale=0):
  Layer 0 q_proj diff: 0.022 (small but non-zero due to computational precision)

With LoRA enabled (scale=2.0):
  Layer 0 q_proj diff: 13.5 (huge!)
```

## Theories

### Most Likely: LoRA Incompatibility with Text-Only Input
The LoRA weights were trained on audio+text pairs. When using text-only input:
- The LoRA adaptations are designed to work WITH audio features
- Without audio, the adaptations produce nonsensical outputs
- This would explain why even with audio, transcription fails (audio preprocessing may be incorrect)

### To Investigate Next
1. **Audio preprocessing**: Verify mel-spectrogram extraction matches NeMo exactly
   - Preemphasis filter (0.97) ✓ Added
   - Per-feature normalization ✓ Added
   - Need to verify actual feature values match NeMo output

2. **Audio-text concatenation**: Check how audio and text embeddings are combined in forward pass
   - Sequence ordering
   - Position embeddings
   - Attention masks

3. **Prompt format**: Verify Canary control tokens are correct
   - `<|en|><|transcribe|><|notimestamps|>` ✓ Added
   - Token IDs match expected values

## Code Fixes Applied

### convert.py
- Fixed LoRA weight transposition
- Removed incorrect decoder weight transposition
- Added proper handling for embeddings and adapter weights

### qwen_decoder.py
- Fixed GQA: num_key_value_heads=8 (was 16)
- Changed from dummy zeros to None for input_ids when using embeddings
- Added clarifying comments about Model wrapper

### transcribe.py
- Added preemphasis filter (0.97)
- Changed to per-feature normalization
- Fixed mel bins to 128
- Added Canary control tokens to prompts

### extend_tokenizer.py
- Created script to extend Qwen3 tokenizer with 267 Canary tokens
- Vocabulary: 151,669 → 151,936 tokens

## Next Steps

1. Test with properly extracted audio features
2. Compare audio feature extraction with reference NeMo implementation
3. Verify audio-text concatenation logic
4. Consider testing with official Canary model in PyTorch/NeMo to validate our audio preprocessing

## References
- Original model: https://huggingface.co/nvidia/canary-qwen-2.5b
- mlx-audio-plus (Parakeet reference): https://github.com/DePasqualeOrg/mlx-audio-plus
- NeMo toolkit: https://github.com/NVIDIA/NeMo
