# Canary-MLX

Convert NVIDIA's Canary speech recognition model to run natively on Apple Silicon using MLX.

## Overview

Canary is a hybrid model combining:
- **FastConformer** audio encoder (NeMo)
- **Qwen 2.5B** language model decoder

This repository provides tools to convert the model from NeMo/PyTorch format to MLX format for efficient inference on Mac.

## Features

- ✅ Weight conversion with proper transposition
- ✅ Hybrid architecture support (Audio Encoder + LLM Decoder)
- ✅ Audio preprocessing pipeline (16kHz, 80-dim Mel spectrograms)
- ✅ Inference script for transcription

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- `mlx` and `mlx-lm` for Apple Silicon acceleration
- `torch` and `torchaudio` for audio preprocessing
- `transformers` for tokenization
- `safetensors` for weight management

## Quick Start

### 1. Download the Model

```bash
huggingface-cli download nvidia/canary-qwen-2.5b --local-dir original-canary
```

### 2. Convert to MLX Format

```bash
python convert.py --model original-canary/model.safetensors --output mlx-canary
```

### 3. Run Transcription

```bash
python transcribe.py --audio test_audio.wav
```

## Project Structure

```
canary-mlx/
├── convert.py          # Weight converter (PyTorch → MLX)
├── canary_model.py     # Model architecture definition
├── transcribe.py       # Inference script
├── requirements.txt    # Python dependencies
└── PRD.md             # Detailed implementation guide
```

## Implementation Notes

- The FastConformer encoder requires precise layer matching
- Audio preprocessing follows NeMo defaults (16kHz, 80 Mel bins)
- Generation uses greedy decoding by default

## Credits

- Model: [NVIDIA Canary](https://huggingface.co/nvidia/canary-qwen-2.5b)
- Framework: [MLX](https://github.com/ml-explore/mlx) by Apple

## License

MIT
