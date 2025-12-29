# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **novel implementation** converting NVIDIA's Canary-Qwen speech recognition model from NeMo/PyTorch to MLX for Apple Silicon. No pre-existing MLX version exists.

**Architecture**: Speech-Augmented Language Model (SALM)
- **Encoder**: FastConformer (1024-dim, 17 layers) for audio processing
- **Adapter**: Linear projection (encoder_dim → llm_dim)
- **Decoder**: Qwen3-1.7B LLM with LoRA fine-tuning
- **Total params**: 2.5B (not to be confused with decoder-only size)

## Critical Architecture Notes

### Hybrid Model Flow
```
Audio (16kHz mono)
  → Log-Mel Features (80-dim)
  → FastConformer Encoder
  → Modality Adapter (projection)
  → Concatenate with Text Embeddings
  → Qwen3 Decoder
  → Transcription
```

Audio and text embeddings are **concatenated along the sequence dimension** before feeding to the decoder, not added or merged.

### Weight Conversion Rules (convert.py)
Based on parakeet-mlx research:
- **Convolutions**: Permute 3D as (0,2,1), 4D as (0,2,3,1)
- **Linear layers**: Transpose 2D weights (.T) - though MLX may handle automatically
- **Skip**: `preprocessor`, `num_batches_tracked` keys
- **LoRA weights**: Preserve structure (lora_A: down-projection, lora_B: up-projection)
- **Key mapping**: Strip `model.`, `llm.model.` prefixes to match standard Qwen structure

### Audio Preprocessing (transcribe.py)
**NeMo Canary parameters** (do not modify):
- Sample rate: 16000 Hz
- FFT size: 512
- Window: 25ms (400 samples)
- Hop: 10ms (160 samples)
- Mel bins: 80
- Log compression: log(mel + 1e-5)
- Normalization: Per-utterance mean/std

### Prompt Format
MUST use audio locator tag:
```python
prompt = f"Transcribe the following: {model.audio_locator_tag}"
# model.audio_locator_tag = "<|audioplaceholder|>"
```

## Essential Commands

### Setup
```bash
pip install -r requirements.txt
```

### Model Download
```bash
huggingface-cli download nvidia/canary-qwen-2.5b --local-dir original-canary
```

### Weight Conversion
```bash
python convert.py --model original-canary/model.safetensors --output mlx-canary
```

### Verify Conversion
```bash
python -c "import mlx.core as mx; w = mx.load('mlx-canary/model.safetensors'); print(f'Keys: {len(w)}')"
```

### Transcription (when complete)
```bash
python transcribe.py --audio test.wav --model-dir mlx-canary
```

## Implementation Status & Known Gaps

### ✅ Implemented
- Weight conversion framework with NeMo → MLX key mapping
- Audio preprocessing pipeline (matches NeMo exactly)
- Model architecture skeleton

### ⚠️ INCOMPLETE - Critical TODOs

#### 1. FastConformer Encoder (canary_model.py:20-88)
**Current**: Simplified placeholder using basic Conv1d + TransformerEncoderLayer
**Required**: Full FastConformer with:
- Subsampling convolutions (8x downsampling)
- Conformer blocks (self-attention + depthwise separable convolution + feed-forward)
- Relative positional encoding

**Recommended approach**: Integrate from [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
```bash
git clone https://github.com/senstella/parakeet-mlx.git
# Copy FastConformer implementation to this repo
```

#### 2. Qwen3 Decoder Integration (canary_model.py:148-158)
**Current**: Placeholder class
**Required**:
- Load from `mlx-community/Qwen3-1.7B-bf16`
- Use `from mlx_lm.models.qwen2 import Model, ModelArgs`
- Properly initialize embedding layer for text token lookup

#### 3. Generation Loop (transcribe.py:163-167)
**Current**: Forward pass only, no token generation
**Required**:
- Implement greedy decoding or use `mlx_lm.utils.generate()`
- KV cache for efficiency
- Proper stopping criteria (EOS token detection)

#### 4. LoRA Adapter Handling
**Architecture**: `W_final = W_base + lora_B @ lora_A`
- W_base: Frozen pre-trained weights
- lora_A: Up-projection [r, d_in]
- lora_B: Down-projection [d_out, r]

**TODO**:
- Implement LoRA layer class in canary_model.py
- Load lora_A/lora_B weights during model initialization
- Merge or apply during forward pass

## Testing Workflow

### Phase 1: Conversion Validation
```bash
# Download
huggingface-cli download nvidia/canary-qwen-2.5b --local-dir original-canary

# Convert
python convert.py --model original-canary/model.safetensors --output mlx-canary

# Inspect keys
python -c "
import mlx.core as mx
w = mx.load('mlx-canary/model.safetensors')
encoder_keys = [k for k in w.keys() if 'encoder' in k]
decoder_keys = [k for k in w.keys() if 'llm' in k or 'decoder' in k]
print(f'Encoder: {len(encoder_keys)}, Decoder: {len(decoder_keys)}')
"
```

### Phase 2: Model Loading
```python
from canary_model import load_model
model = load_model("mlx-canary")
# Check for shape mismatches or missing keys
```

### Phase 3: Audio Pipeline
```python
from transcribe import load_audio, extract_mel_features
audio = load_audio("test.wav")
features = extract_mel_features(audio)
assert features.shape[-1] == 80, "Mel dim must be 80"
```

## Common Issues & Solutions

### Shape Mismatch on Weight Loading
- **Cause**: FastConformer layer definition doesn't match weight keys
- **Solution**: Print weight keys, adjust layer structure in `FastConformerEncoder.__init__()`

### Config Extraction Fails
- **Cause**: NeMo configs are deeply nested, may not have standard structure
- **Solution**: Manually inspect `original-canary/config.json`, extract encoder/llm sections separately

### Tokenizer Missing
- **Cause**: Tokenizer files not copied during conversion
- **Solution**: Verify convert.py copied: tokenizer.json, vocab.json, merges.txt, special_tokens_map.json

### Audio Duration Warning
- Model trained on max 40s audio
- Longer inputs may degrade accuracy
- Consider chunking for long files

## Key Differences from Standard LLM Conversion

1. **Hybrid architecture**: Cannot use `mlx_lm.convert` directly
2. **Audio modality**: Requires mel spectrogram preprocessing, not just text tokenization
3. **Concatenation strategy**: Audio embeddings prepend text, not cross-attention
4. **LoRA adapters**: Must handle low-rank weight merging
5. **Special tokens**: Audio locator tag `<|audioplaceholder|>` is model-specific

## Reference Resources

- **Model Card**: https://huggingface.co/nvidia/canary-qwen-2.5b
- **NeMo SALM Docs**: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/models.html
- **Parakeet-MLX**: https://github.com/senstella/parakeet-mlx (FastConformer reference)
- **Conversion Gist**: https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed

## Development Priorities

See NOTES.md for comprehensive TODO list. Immediate priorities:
1. Integrate FastConformer from parakeet-mlx
2. Add Qwen3-1.7B decoder loading
3. Implement generation loop
4. Test end-to-end with sample audio
