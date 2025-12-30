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

## Current Debugging Status

### ✅ Verified Working
1. Weight conversion correctly transposes LoRA weights
2. All 1,684 weights load successfully
3. Base Qwen weights are identical to official Qwen3-1.7B (0.000000 difference)
4. LoRA weights match original file after transposition (0.000000 difference)
5. Embedding table (151,936 tokens) matches
6. Model architecture matches specification
7. Decoder uses mlx_lm's Model wrapper (includes lm_head for logits)
8. Model correctly set to eval() mode (dropout disabled)

### ❌ Known Issues
1. **Model outputs gibberish** for both text-only and audio input
2. LoRA causes large divergence even with correct weights:
   - Layer 0 q_proj output differs by max=13.5 from base Qwen
   - This is expected since LoRA was trained for audio+text pairs
3. Text-only testing shows LoRA adaptations don't work without audio
4. Audio testing still produces gibberish (audio preprocessing may be wrong)

### Hypotheses
1. **Most likely**: Audio feature extraction doesn't match NeMo exactly
   - Need to verify actual mel-spectrogram values against NeMo output
   - Subtle differences in preprocessing could break the model

2. **Possible**: Audio-text embedding concatenation is incorrect
   - Sequence order might matter
   - Position embeddings might need adjustment
   - Attention masks might be required

3. **Unlikely**: Weight corruption (weights verified identical)

See `DEBUGGING_LOG.md` for detailed investigation history.

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
5. **Refer to DEBUGGING_LOG.md** - full investigation history
