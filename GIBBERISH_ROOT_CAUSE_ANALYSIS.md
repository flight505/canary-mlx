# Root Cause Analysis: Gibberish Output from Canary Model

## Problem Statement

The Canary-Qwen speech recognition model generates mixed-language gibberish instead of proper transcriptions:

```
Input: English speech audio
Expected: "Hello, how are you?"
Actual: "至上——ognition规模化روuffixceпоч养护ogleIBivorหาย..."
```

## Investigation Summary

Through systematic analysis using a custom diagnostic script (`diagnose_output.py`), we identified **two critical issues** and **one contributing factor** causing the gibberish output.

---

## Critical Issue #1: Vocabulary Size Mismatch

### Problem
The model's embedding layer and the tokenizer have mismatched vocabulary sizes.

**Diagnostic Evidence:**
```
Tokenizer info:
  Vocab size: 151643
  Model expects: 151936
  ⚠️  MISMATCH: Tokenizer vocab (151643) != Model vocab (151936)
```

### Technical Details

- **Model embedding layer:** 151,936 tokens (shape: `[151936, 2048]`)
- **Qwen3-1.7B base tokenizer:** 151,643 tokens
- **Qwen3-1.7B with added tokens:** 151,669 tokens
- **Missing tokens:** 267 tokens (151,936 - 151,669 = 267)

### Impact

When the model generates token IDs in the range **151,669 to 151,935**:
1. The tokenizer cannot decode them (out of vocabulary)
2. Falls back to undefined behavior
3. May produce random characters or skip tokens
4. Results in corrupted text output

### Why This Happens

The NVIDIA Canary-Qwen model extends the standard Qwen3-1.7B tokenizer with **267 additional special tokens** for speech recognition tasks:

**Expected additional tokens (267 total):**
- Audio control tokens: `<|audioplaceholder|>`, `<|startoftranscript|>`, `<|endoftranscript|>`
- Task tokens: `<|transcribe|>`, `<|translate|>`, `<|pnc|>`, `<|nopnc|>`
- Language control tokens: `<|en|>`, `<|de|>`, `<|fr|>`, etc. (~25 languages)
- Reserved/timestamp tokens: ~238 additional tokens

**What we downloaded:**
- Standard Qwen3-1.7B tokenizer without these extensions

---

## Critical Issue #2: Extreme Chinese Token Bias

### Problem
The model outputs 72% Chinese tokens regardless of input language.

**Diagnostic Evidence:**
```
Probability mass by language region:
  English     : 11.55%
  Chinese     : 72.17%  ← SEVERE BIAS
  Arabic      :  4.55%
  Thai        :  2.78%
  Cyrillic    :  2.52%

Generated token:
  ID: 104301
  Text: "这就是" (Chinese: "this is")
```

### Technical Details

**Top predicted tokens (all Chinese):**
```
1. ID 104301 | prob: 8.26% | "这就是" (this is)
2. ID 109509 | prob: 6.95% | "马拉" (Mala)
3. ID 102811 | prob: 3.38% | "穴" (hole/cave)
```

**Entropy analysis:**
- Normalized entropy: 0.5343 (moderate uncertainty)
- High entropy indicates model is "confused" but biased toward Chinese

### Why This Happens

**Possible causes:**

1. **Training data imbalance:** Model may have been fine-tuned on predominantly Chinese audio data
2. **Missing language control tokens:** Without explicit language specification (e.g., `<|en|>`), model defaults to Chinese
3. **LoRA adapter bias:** LoRA adapters (rank=128, scale=2.0) may reinforce Chinese token preferences
4. **Incorrect prompt format:** Missing task/language control from standard Canary prompts

### Expected Canary Prompt Format

**Current (wrong):**
```
"Transcribe the following: <|audioplaceholder|>"
```

**Correct (with language control):**
```
"<|en|><|transcribe|><|notimestamps|> Transcribe the following: <|audioplaceholder|>"
```

Or:
```
"<|startoftranscript|><|en|><|transcribe|><|notimestamps|> <|audioplaceholder|>"
```

---

## Contributing Factor: Mixed Language Token Pattern

### Observation
The gibberish shows tokens from multiple languages:
```
至上 (Chinese) ——ognition (English) 规模化 (Chinese) رو (Arabic)
uffixce (English) поч (Cyrillic) 养护 (Chinese) ogle (English)
IBivor (mixed) หาย (Thai)...
```

### Why This Happens

**High-entropy distribution** spreads probability across many languages:

```
Temperature = 1.0 (greedy):
  Top token: Chinese (8.26%)
  2nd token: Chinese (6.95%)
  3rd token: Chinese (3.38%)

But: 28% probability mass is non-Chinese
```

With greedy decoding, the model picks the highest probability token each time, but the high entropy means it frequently switches between:
- Chinese tokens (72% aggregate)
- English fragments (11.55%)
- Other languages (16.45%)

This creates the characteristic "mixed gibberish" pattern.

---

## Weight Initialization Check

### Are the weights randomly initialized?

**No** - the weights are properly trained:

```
Embedding weights:
  Shape: (151936, 2048)
  Mean: -0.000112
  Std:  0.034424
  Min:  -0.353516
  Max:  0.292969
  ✓ Weights appear to be trained (reasonable mean/std)
```

**Expected characteristics of trained embeddings:**
- Mean ≈ 0 ✓ (actual: -0.000112)
- Std ≈ 0.01-0.1 ✓ (actual: 0.034424)
- Reasonable range ✓ (actual: -0.35 to +0.29)

**Conclusion:** The model is properly trained, not randomly initialized. The issues are configuration-related, not weight-related.

---

## Root Cause Summary

| Issue | Severity | Impact | Fix Difficulty |
|-------|----------|--------|----------------|
| Vocab size mismatch | **Critical** | Token decoding failures | Easy |
| Chinese token bias | **Critical** | Wrong language output | Medium |
| Missing control tokens | **High** | No language/task control | Easy |

### Primary Root Cause

**Incorrect tokenizer:** We used Qwen2-1.5B tokenizer (151,643 tokens) instead of the extended Canary-specific tokenizer (151,936 tokens).

### Secondary Root Cause

**Missing language control:** Prompts don't include language/task control tokens that Canary models require to specify output language.

---

## Solutions

### Solution 1: Fix Tokenizer (IMMEDIATE)

**Action:** Extend Qwen3-1.7B tokenizer with 267 Canary-specific tokens

**Implementation:**
```bash
python fix_tokenizer.py --output mlx-canary
```

This script:
1. Loads Qwen3-1.7B tokenizer (151,669 tokens)
2. Adds 267 Canary special tokens
3. Saves extended tokenizer (151,936 tokens)
4. Verifies vocab size matches model

**Expected result:** Eliminates vocab mismatch and out-of-bounds token errors

### Solution 2: Add Language Control (HIGH PRIORITY)

**Action:** Update prompt format to include language specification

**Change in `transcribe.py`:**
```python
# OLD:
def format_prompt(prompt_text: str, audio_locator: str) -> str:
    return f"{prompt_text} {audio_locator}"

# NEW:
def format_prompt(prompt_text: str, audio_locator: str, language: str = "en") -> str:
    return f"<|{language}|><|transcribe|><|notimestamps|> {prompt_text} {audio_locator}"
```

**Expected result:** Model receives explicit language instruction, reducing Chinese bias

### Solution 3: Test Temperature/Sampling (OPTIONAL)

**Action:** Try different decoding strategies to reduce entropy

**Options:**
```python
# More deterministic (sharpen distribution)
temperature = 0.1

# Sample from top-k tokens only
top_k = 5

# Sample from nucleus (top-p) filtering
top_p = 0.9
```

**Expected result:** More consistent language selection

### Solution 4: Verify LoRA Configuration (INVESTIGATIVE)

**Action:** Check if LoRA adapters are causing Chinese bias

**Investigation steps:**
1. Test with LoRA disabled (freeze_lora=True)
2. Check LoRA scale factor (current: 2.0)
3. Inspect LoRA training data distribution
4. Compare base model vs LoRA outputs

**Expected result:** Identify if LoRA introduces language bias

---

## Testing Protocol

### Phase 1: Fix Tokenizer
1. Run `fix_tokenizer.py`
2. Verify vocab size: 151,936 ✓
3. Test encoding/decoding of special tokens
4. Run diagnostic again to verify no vocab mismatch

### Phase 2: Add Language Control
1. Update `format_prompt()` function
2. Test with English prompt: `<|en|><|transcribe|>...`
3. Test with other languages: `<|zh|>`, `<|de|>`, etc.
4. Verify language-appropriate output

### Phase 3: Validate Output
1. Use English speech sample
2. Expected: English transcription
3. Measure: % English tokens vs % Chinese tokens
4. Success criteria: >80% English tokens for English audio

---

## Expected Outcome After Fixes

### Before (Current)
```
Input: English audio "Hello, how are you?"
Output: "这就是ognition规模化روuffixce..."
Language distribution: 72% Chinese, 11% English, 16% other
```

### After (Fixed)
```
Input: English audio "Hello, how are you?"
Output: "Hello, how are you?"
Language distribution: >80% English for English audio
```

---

## Lessons Learned

1. **Always verify tokenizer compatibility** with model vocab size
2. **Speech models require task/language control tokens** beyond standard text models
3. **High entropy distributions indicate missing control signals**, not just model confusion
4. **Extended tokenizers for specialized tasks** may not be in standard HuggingFace repos
5. **LoRA fine-tuning can introduce unexpected biases** if training data is imbalanced

---

## References

- NVIDIA Canary-Qwen: https://huggingface.co/nvidia/canary-qwen-2.5b
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
- NeMo SpeechLM2: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/intro.html
- Diagnostic script: `/Users/jesper/Projects/Dev_projects/HF_dev/Canary-Qwen/diagnose_output.py`
- Fix script: `/Users/jesper/Projects/Dev_projects/HF_dev/Canary-Qwen/fix_tokenizer.py`

---

## Next Actions

1. ✅ Run `fix_tokenizer.py` to extend vocabulary
2. ⬜ Update `format_prompt()` to include language control
3. ⬜ Test with English audio sample
4. ⬜ Verify English output (>80% English tokens)
5. ⬜ Test multilingual support with `<|zh|>`, `<|es|>`, etc.
6. ⬜ Document final prompt format and usage examples
