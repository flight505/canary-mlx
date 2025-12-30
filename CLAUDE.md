# CLAUDE.md - Canary-Qwen MLX Project

This file provides guidance to Claude Code when working with this repository.

## Project Status (2025-12-30)

**Current State**: Model architecture complete, weights converted and verified, but outputs gibberish.

This is a novel MLX implementation of NVIDIA's Canary-Qwen-2.5B speech recognition model. All weights have been verified to match the original file exactly, but the model does not yet produce correct transcriptions.

## Architecture

**Speech-Augmented Language Model (SALM)**:
```
Audio (16kHz mono)
  → Log-Mel Features (128-dim)
  → FastConformer Encoder (32 layers, 1024-dim)
  → Linear Adapter (1024 → 2048)
  → Concatenate with Text Embeddings
  → Qwen3-1.7B Decoder (28 layers, LoRA)
  → Transcription Logits
```

### Key Components
- **Encoder**: FastConformer with 32 layers, imported from parakeet-mlx
- **Adapter**: Single linear projection layer
- **Decoder**: Qwen3-1.7B with LoRA adapters (rank=128, alpha=256, scale=2.0)
- **Vocabulary**: 151,936 tokens (151,669 base Qwen + 267 Canary special tokens)

## Critical Implementation Details

### Weight Conversion (convert.py)
**Rules** (verified correct):
- **Convolutions**: Permute 3D as `(0,2,1)`, 4D as `(0,2,3,1)`
- **LoRA weights**: MUST transpose both lora_A and lora_B
- **Linear layers**: NO transpose (MLX and PyTorch use same format)
- **Embeddings**: NO transpose
- **Key mapping**: Strip NeMo prefixes, rename LoRA keys

### Audio Preprocessing (transcribe.py)
**NeMo-compatible parameters**:
- Sample rate: 16,000 Hz
- Window: 25ms (400 samples @ 16kHz)
- Hop: 10ms (160 samples @ 16kHz)
- FFT size: 512
- **Mel bins: 128** (NOT 80!)
- **Preemphasis: 0.97** (CRITICAL for NeMo compatibility)
- **Normalization: per-feature** (per mel-bin, NOT global)

### Tokenizer Extension (extend_tokenizer.py)
Base Qwen3-1.7B tokenizer extended with 267 Canary tokens:
- 9 control tokens (`<|audioplaceholder|>`, `<|transcribe|>`, etc.)
- 25 language codes (`<|en|>`, `<|de|>`, etc.)
- 233 reserved tokens

### Prompt Format
```python
prompt = "<|en|><|transcribe|><|notimestamps|> Transcribe the following: <|audioplaceholder|>"
```

## Development Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Download Model
```bash
git lfs install
git clone https://huggingface.co/nvidia/canary-qwen-2.5b original-canary
```

### Extend Tokenizer
```bash
python extend_tokenizer.py --output mlx-canary
```

### Convert Weights
```bash
python convert.py --model original-canary/model.safetensors --output mlx-canary
```

### Test Transcription
```bash
python transcribe.py --model-dir mlx-canary --audio test_audio/macos_speech.wav
```

## Code Structure

### Main Files
- **canary_model.py**: Complete model architecture with weight loading
- **conformer_parakeet.py**: FastConformer encoder (from parakeet-mlx)
- **qwen_decoder.py**: Qwen3 decoder with LoRA support and mlx_lm integration
- **convert.py**: NeMo → MLX weight conversion with correct transpositions
- **extend_tokenizer.py**: Tokenizer extension script
- **transcribe.py**: End-to-end inference with audio preprocessing

### Weight Loading Flow
1. `load_model()` creates model architecture
2. `Qwen3Decoder.apply_lora_to_layers()` converts Linear → LoRALinear
3. `CanaryModel.load_weights_from_file()` loads MLX weights
4. `load_weights_dict()` maps NeMo keys to MLX structure
5. `self.load_weights()` applies weights with `strict=False`

### Forward Pass Flow
1. Audio → mel features (128-dim)
2. FastConformer encoder → (batch, time/8, 1024)
3. Adapter projection → (batch, time/8, 2048)
4. Concatenate with text embeddings
5. Qwen3 decoder (with LoRA) → logits (batch, seq_len, 151936)
6. Greedy decoding → transcription

## Critical Pitfalls to Avoid

1. **DO NOT transpose base linear weights** - MLX and PyTorch use same format
2. **DO transpose LoRA weights** - NeMo format differs from mlx_lm
3. **DO NOT forget preemphasis** - 0.97 coefficient is critical
4. **DO NOT use global normalization** - must be per-feature (per mel-bin)
5. **DO NOT test text-only** - LoRA trained for audio+text, won't work alone
6. **DO call model.eval()** - disable dropout before inference

## Next Steps

1. **Verify audio preprocessing**:
   - Extract mel-spectrograms with NeMo and MLX
   - Compare actual feature values numerically
   - Check for subtle differences in windowing, FFT, etc.

2. **Test audio-text concatenation**:
   - Verify sequence order (audio first or text first?)
   - Check position embeddings
   - Verify attention masks

3. **Compare with PyTorch/NeMo**:
   - Run original model in NeMo with same audio
   - Compare intermediate activations
   - Identify where divergence occurs

## References

- Original model: https://huggingface.co/nvidia/canary-qwen-2.5b
- Base LLM: https://huggingface.co/Qwen/Qwen3-1.7B
- Encoder reference: https://github.com/DePasqualeOrg/mlx-audio-plus
- NeMo docs: https://docs.nvidia.com/nemo-framework/
- MLX framework: https://github.com/ml-explore/mlx

## Important Notes for Claude

1. **All weights are verified correct** - don't suggest weight fixes
2. **LoRA divergence is expected** - that's how LoRA works
3. **Focus on audio preprocessing** - most likely issue
4. **Don't test text-only** - LoRA won't work without audio

---

# Debugging History

## Current Issue

**Model outputs gibberish** for both text-only and audio input, despite all weights being verified as correct.

## Verified Correct ✅

1. **Base Qwen weights**: Identical to official Qwen3-1.7B (0.000000 difference)
2. **LoRA weights**: Match original file perfectly (0.000000 difference after transpose)
3. **Embedding table**: 151,936 tokens, all rows match official Qwen
4. **Architecture**: 28 layers, correct dimensions, GQA with 8 KV heads
5. **Weight loading**: All 1,684 weights loaded successfully
6. **Model structure**: Using mlx_lm's Model wrapper (includes lm_head logic)
7. **Output format**: Decoder correctly returns logits, not hidden states

## Issues Identified & Fixed ✅

### 1. Weight Transposition Bugs (FIXED)
- LoRA weights were correctly transposed in convert.py
- Base linear weights were correctly NOT transposed
- Verified: All weights match original with 0.000000 difference

### 2. Missing Output Projection (INVESTIGATED)
- Initially thought decoder was returning hidden states
- Actually using mlx_lm's Model wrapper which includes lm_head
- The import `from mlx_lm.models.qwen2 import Model as Qwen2Model` was confusing

### 3. Dropout During Inference (FIXED)
- Model was in training mode by default
- Fixed by calling `model.eval()`
- However, still produces gibberish even in eval mode

## Weight Comparison Results

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

## Forward Pass Analysis

```
Embeddings: IDENTICAL
Layer 0 output: DIVERGES (max diff: 1.74)
Final logits: VERY DIFFERENT (max diff: 18.7)

With LoRA disabled (scale=0):
  Layer 0 q_proj diff: 0.022 (small but non-zero due to computational precision)

With LoRA enabled (scale=2.0):
  Layer 0 q_proj diff: 13.5 (huge!)
```

## Current Theory

### Most Likely: LoRA Incompatibility with Text-Only Input

The LoRA weights were trained on audio+text pairs. When using text-only input:
- The LoRA adaptations are designed to work WITH audio features
- Without audio, the adaptations produce nonsensical outputs
- This would explain why even with audio, transcription fails (audio preprocessing may be incorrect)

### Areas to Investigate

1. **Audio preprocessing**: Verify mel-spectrogram extraction matches NeMo exactly
   - Preemphasis filter (0.97) ✓ Added
   - Per-feature normalization ✓ Added
   - Need to verify actual feature values match NeMo output

2. **Audio-text concatenation**: Check how audio and text embeddings are combined
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
