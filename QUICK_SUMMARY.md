# Quick Summary: Gibberish Output Root Cause

## The Problem
Model outputs mixed language gibberish: "至上——ognition规模化روuffixce..."

## Root Causes Found

### 1. Vocabulary Size Mismatch (CRITICAL)
- **Model expects:** 151,936 tokens
- **Tokenizer has:** 151,643 tokens  
- **Missing:** 267 Canary-specific special tokens
- **Impact:** Token IDs 151,669-151,935 cannot be decoded → garbage output

### 2. Chinese Token Bias (CRITICAL)  
- **72% of probability mass** concentrated on Chinese tokens
- **Only 11.5%** on English tokens
- **Cause:** Missing language control tokens in prompts

### 3. Wrong Tokenizer Source
- **Downloaded:** Qwen2-1.5B tokenizer
- **Should use:** Qwen3-1.7B tokenizer + Canary extensions

## The Fix

### Step 1: Extend Tokenizer
```bash
python fix_tokenizer.py --output mlx-canary
```
This adds the 267 missing Canary special tokens.

### Step 2: Update Prompts (in transcribe.py)
```python
# Add language control tokens
prompt = f"<|en|><|transcribe|><|notimestamps|> {text} <|audioplaceholder|>"
```

### Step 3: Test
```bash
python transcribe.py --audio test_audio/sample.wav
```

## Expected Result
- ✅ No vocab mismatch errors
- ✅ English output for English audio
- ✅ Proper language control via `<|lang|>` tokens

## Files Created
- `diagnose_output.py` - Diagnostic tool to analyze model output
- `fix_tokenizer.py` - Script to extend tokenizer to 151,936 tokens
- `GIBBERISH_ROOT_CAUSE_ANALYSIS.md` - Detailed technical analysis
- `ANALYSIS_GIBBERISH_OUTPUT.md` - Initial investigation notes
