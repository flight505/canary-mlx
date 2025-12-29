# Implementation Notes & Research Findings

## Research Summary (2025-12-29)

### Key Findings

1. **No Existing Implementation**
   - This is a **novel project** - no MLX version of Canary-Qwen exists yet
   - Regular Qwen3 models are available in MLX via mlx-community
   - Parakeet-MLX exists but only for Parakeet models (different decoder)

2. **Architecture Corrections**
   - Decoder is **Qwen3-1.7B**, not "Qwen 2.5B"
   - 2.5B refers to total parameter count (encoder + decoder)
   - Model uses **LoRA adapters** for fine-tuning (not in original PRD)
   - Architecture type: **SALM (Speech-Augmented Language Model)**

3. **Weight Conversion Insights**
   - Parakeet-MLX conversion **does not transpose linear layers**
   - Only permutes convolutions: 3D (0,2,1) and 4D (0,2,3,1)
   - MLX handles linear transposition automatically for standard models
   - Special handling needed for RNN layers (if present)

4. **Audio Processing**
   - 16kHz sample rate (mono)
   - 80-dim log-mel spectrograms
   - 25ms window (400 samples), 10ms hop (160 samples)
   - Per-utterance normalization (mean/std)

5. **Prompt Format**
   - Uses audio locator tag: `<|audioplaceholder|>`
   - Prompt example: `"Transcribe the following: <|audioplaceholder|>"`
   - Follows Qwen chat template format

## Issues with Original PRD

### ❌ Incorrect Assumptions
1. **Linear layer transposition**: PRD transposes all linear layers, but this may not be needed
2. **Model size**: States "Qwen 2.5B" but decoder is actually Qwen3-1.7B
3. **ModelArgs.from_dict()**: This method is poorly documented in mlx-lm
4. **Missing LoRA handling**: LoRA adapters are critical but not mentioned

### ✅ Correct Elements
1. Hybrid architecture approach
2. Audio preprocessing pipeline basics
3. Reference to parakeet-mlx for FastConformer
4. Safetensors conversion approach

## Current Implementation Status

### ✅ Completed
- [x] `convert.py` - Weight conversion with corrections
- [x] `canary_model.py` - Model architecture skeleton
- [x] `transcribe.py` - Inference pipeline with audio preprocessing
- [x] `requirements.txt` - Dependencies

### ⚠️  Incomplete/TODO

#### 1. FastConformer Implementation
**Status**: Placeholder only

**Options**:
- Option A: Integrate senstella/parakeet-mlx FastConformer
- Option B: Implement from scratch (500+ lines)
- Option C: Use simplified version for testing

**Recommendation**: Use parakeet-mlx implementation
```bash
git clone https://github.com/senstella/parakeet-mlx.git
# Copy FastConformer implementation
```

#### 2. Qwen3 Decoder Integration
**Status**: Placeholder only

**TODO**:
- Load Qwen3-1.7B from mlx-community
- Integrate with MLX-LM's generation utilities
- Handle LoRA adapter merging

**Resources**:
- Model: `mlx-community/Qwen3-1.7B-bf16`
- Code: `from mlx_lm.models.qwen2 import Model, ModelArgs`

#### 3. Generation Loop
**Status**: Not implemented

**Requirements**:
- Implement greedy decoding
- Or integrate with `mlx_lm.utils.generate()`
- Handle KV cache for efficiency
- Proper stopping criteria

#### 4. Weight Loading
**Status**: Basic structure only

**TODO**:
- Verify key mapping from NeMo to MLX structure
- Handle LoRA weight merging
- Test with actual converted weights
- Debug shape mismatches

#### 5. LoRA Adapter Support
**Status**: Not implemented

**Architecture**:
```
W_0 + ΔW = W_0 + B * A
- W_0: Original weights (frozen)
- B: Down-projection [d_out, r]
- A: Up-projection [r, d_in]
```

**TODO**:
- Implement LoRA layer class
- Load LoRA weights from checkpoint
- Merge or apply LoRA during inference

## Testing Strategy

### Phase 1: Conversion
```bash
# Download model
huggingface-cli download nvidia/canary-qwen-2.5b --local-dir original-canary

# Convert
python convert.py --model original-canary/model.safetensors --output mlx-canary

# Verify
python -c "import mlx.core as mx; w = mx.load('mlx-canary/model.safetensors'); print(list(w.keys())[:10])"
```

### Phase 2: Model Loading
```python
from canary_model import load_model
model = load_model("mlx-canary")
# Should load without errors
```

### Phase 3: Audio Processing
```python
from transcribe import load_audio, extract_mel_features
audio = load_audio("test.wav")
features = extract_mel_features(audio)
# Verify shape: [time, 80]
```

### Phase 4: End-to-End
```bash
python transcribe.py --audio test.wav --model-dir mlx-canary
```

## Known Issues

1. **FastConformer Complexity**
   - Full implementation is architecture-specific
   - Layer count, attention type, conv configs must match exactly
   - Recommend using parakeet-mlx implementation

2. **Config Structure**
   - NeMo configs are deeply nested
   - May need manual extraction of encoder/decoder configs
   - Qwen3 config might not be directly extractable

3. **Tokenizer Compatibility**
   - Qwen3 tokenizer should be compatible
   - Verify special tokens match expectations
   - Audio locator tag handling

4. **Performance**
   - No quantization yet (can add with mlx-lm)
   - No KV cache optimization
   - No streaming support

## Next Steps

### Immediate
1. Test weight conversion with actual model
2. Integrate parakeet-mlx FastConformer
3. Load Qwen3-1.7B decoder from mlx-community
4. Implement basic generation loop

### Short-term
1. End-to-end testing with sample audio
2. Compare outputs with NeMo reference
3. Debug and fix shape mismatches
4. Document findings

### Long-term
1. Add quantization support (4-bit, 8-bit)
2. Optimize inference performance
3. Add streaming transcription
4. Support batch processing
5. Add LoRA fine-tuning support

## Resources

### Documentation
- [Canary-Qwen Model Card](https://huggingface.co/nvidia/canary-qwen-2.5b)
- [NeMo SALM Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/models.html)
- [MLX Documentation](https://mlx-framework.org)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-lm)

### Reference Implementations
- [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
- [NeMo Conversion Script](https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed)
- [mlx-community/Qwen3](https://huggingface.co/mlx-community)

### Papers & Blogs
- [NVIDIA Canary Blog](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/)
- [FastConformer Paper](https://arxiv.org/html/2509.14128v2)
- [Qwen3 Release](https://qwen.readthedocs.io/)

## Contributors
- Initial PRD: Unknown
- Research & Implementation: Claude Code (2025-12-29)
