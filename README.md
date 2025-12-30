# Canary-Qwen MLX

**Status**: 🚧 Work in Progress - Model converted but not yet producing correct transcriptions

MLX implementation of NVIDIA's Canary-Qwen-2.5B speech recognition model for Apple Silicon.

## Overview

Canary-Qwen is a speech-to-text model combining:
- **Audio Encoder**: FastConformer (32 layers, 1024-dim)
- **Modality Adapter**: Linear projection (1024 → 2048)
- **Text Decoder**: Qwen3-1.7B (28 layers) with LoRA adapters

The model was trained by NVIDIA for English speech recognition.

## Project Structure

```
├── canary_model.py           # Main model architecture
├── conformer_parakeet.py     # FastConformer encoder (from parakeet-mlx)
├── qwen_decoder.py           # Qwen3 decoder with LoRA support
├── convert.py                # NeMo → MLX weight conversion
├── extend_tokenizer.py       # Extend tokenizer with Canary tokens
├── transcribe.py             # Inference script
├── mlx-canary/              # Converted model weights
└── original-canary/         # Original NeMo weights
```

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### 1. Download Original Model

Download the Canary-Qwen-2.5B model from HuggingFace:
```bash
# Clone the repository
git lfs install
git clone https://huggingface.co/nvidia/canary-qwen-2.5b original-canary
```

### 2. Extend Tokenizer

Extend the Qwen3 tokenizer with Canary-specific tokens:
```bash
python extend_tokenizer.py --output mlx-canary
```

This adds 267 special tokens (audio placeholder, task controls, language codes) to the base Qwen3-1.7B tokenizer.

### 3. Convert Weights

Convert NeMo weights to MLX format:
```bash
python convert.py --model original-canary/model.safetensors --output mlx-canary
```

### 4. Run Transcription

```bash
python transcribe.py --model-dir mlx-canary --audio test_audio/sample.wav
```

## Current Status

### ✅ Completed
- Weight conversion from NeMo to MLX format
- LoRA weight transposition and loading
- Tokenizer extension (151,669 → 151,936 tokens)
- Audio preprocessing (preemphasis, per-feature normalization)
- FastConformer encoder integration
- Qwen3 decoder with LoRA support

### ✅ Verified
- All weights match original file exactly (0.000000 difference)
- Base Qwen weights identical to official Qwen3-1.7B
- Model architecture matches specification
- All 1,684 weight tensors loaded successfully

### ❌ Known Issues
- **Model outputs gibberish** for both text and audio input
- LoRA weights cause significant divergence from base Qwen behavior
- Likely issues:
  - Audio feature extraction may not match NeMo preprocessing exactly
  - Audio-text embedding concatenation needs verification
  - LoRA trained for audio+text doesn't work for text-only

See `DEBUGGING_LOG.md` for detailed investigation.

## Model Details

### Architecture
- **Encoder**: FastConformer, 32 layers, 1024-dim, 80ms frame rate
- **Adapter**: Linear projection, 1024 → 2048
- **Decoder**: Qwen3-1.7B, 28 layers, 2048-dim
- **LoRA**: Rank 128, alpha 256, scale 2.0 on q_proj and v_proj
- **Vocabulary**: 151,936 tokens (151,669 base + 267 Canary)

### Audio Preprocessing
- Sample rate: 16kHz
- Window: 25ms (400 samples)
- Hop: 10ms (160 samples)
- FFT size: 512
- Mel bins: 128
- Preemphasis: 0.97
- Normalization: per-feature (per mel-bin)

### Special Tokens
- Audio placeholder: `<|audioplaceholder|>`
- Tasks: `<|transcribe|>`, `<|translate|>`
- Languages: `<|en|>`, `<|de|>`, `<|fr|>`, etc. (25 languages)
- Timestamps: `<|timestamps|>`, `<|notimestamps|>`

## Development

### Testing
```bash
# Test with sample audio
python transcribe.py --model-dir mlx-canary --audio test_audio/macos_speech.wav

# Test text-only (for debugging decoder)
python -c "from canary_model import load_model; model = load_model('mlx-canary'); print('Model loaded successfully')"
```

### Debug Mode
Set logging in Python scripts to see detailed weight loading and forward pass information.

## References

- Original model: [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b)
- Base LLM: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- Encoder reference: [DePasqualeOrg/mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus)
- NeMo toolkit: [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
- MLX framework: [ml-explore/mlx](https://github.com/ml-explore/mlx)

## License

This project follows the same license as the original Canary model (CC-BY-4.0).

## Contributing

This is a research project. Issues and pull requests welcome, especially for:
- Audio preprocessing verification
- Forward pass debugging
- NeMo compatibility testing
