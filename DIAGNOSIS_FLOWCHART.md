# Diagnosis Flowchart: Why Gibberish?

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: English audio "Hello, how are you?"                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Audio Processing                                    │
│ ✓ Load audio → 16kHz mono                                  │
│ ✓ Extract log-mel features → [101, 128]                    │
│ ✓ Pass through FastConformer encoder → [101, 1024]         │
│ ✓ Project to LLM space → [101, 2048]                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ ✓ Audio encoding works correctly
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Prompt Tokenization                                 │
│ Input: "Transcribe the following: <|audioplaceholder|>"    │
│ Tokens: [3167, 3114, 279, 2701, 25, ???, ???, ???]        │
│                                                              │
│ ⚠️  PROBLEM 1: Missing <|audioplaceholder|> token!         │
│    Tokenizer vocab: 151,643 tokens                          │
│    Model expects:   151,936 tokens                          │
│    Missing:         267 special tokens                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ ⚠️ Prompt partially broken
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Combine Audio + Text                                │
│ Audio features:  [batch=1, seq=101, dim=2048]              │
│ Text embeddings: [batch=1, seq=12,  dim=2048]              │
│ Combined:        [batch=1, seq=113, dim=2048]              │
│ ✓ Concatenation works                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Decoder Forward Pass                                │
│ Input embeddings → Qwen3 decoder → Logits                   │
│ Output shape: [1, 113, 151936]                              │
│ ✓ Model forward pass works                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Token Generation (greedy)                           │
│ Logits for last position: [151936] values                   │
│                                                              │
│ ⚠️  PROBLEM 2: Extreme language bias!                      │
│                                                              │
│ Probability distribution:                                   │
│   Chinese tokens [70K-120K]:  72.17% ⚠️                     │
│   English tokens [0-50K]:     11.55%                        │
│   Arabic tokens  [120K-130K]:  4.55%                        │
│   Thai tokens    [130K-140K]:  2.78%                        │
│   Other tokens:               ~9%                           │
│                                                              │
│ Top predicted token:                                        │
│   ID 104301 = "这就是" (Chinese: "this is")                 │
│                                                              │
│ WHY? Missing language control: <|en|><|transcribe|>        │
└──────────────────────┬──────────────────────────────────────┘
                       │ ⚠️ Picks Chinese token
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Iterative Generation                                │
│                                                              │
│ Token 1: ID 104301 → "这就是" (Chinese)                     │
│ Token 2: ID 109509 → "马拉"   (Chinese)                     │
│ Token 3: ID ???    → Out of vocab! → "��" (garbage)        │
│ Token 4: ID 480    → "able"    (English fragment)          │
│ Token 5: ID 125139 → "ầm"      (Vietnamese)                 │
│ ...                                                          │
│                                                              │
│ High entropy (0.5343) + vocab mismatch = mixed gibberish   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Decode Tokens                                       │
│                                                              │
│ Token IDs: [104301, 109509, ???, 480, 125139, ...]         │
│                                                              │
│ ⚠️  PROBLEM 3: Tokens > 151,669 cannot be decoded!         │
│                                                              │
│ Decoded text: "这就是马拉��able ầm..."                      │
│                                                              │
│ Result: Mixed language gibberish! ❌                        │
└─────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════
                        THE FIX
══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│ FIX 1: Extend Tokenizer (267 tokens)                        │
│                                                              │
│ Add Canary special tokens:                                  │
│   • <|audioplaceholder|>, <|startoftranscript|>, ...       │
│   • <|en|>, <|zh|>, <|de|>, <|fr|>, ... (25 languages)     │
│   • <|transcribe|>, <|translate|>, <|pnc|>, ...            │
│   • Reserved tokens: <|reserved_0|> ... <|reserved_237|>   │
│                                                              │
│ Result: Tokenizer vocab = 151,936 ✓                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FIX 2: Add Language Control to Prompts                      │
│                                                              │
│ OLD: "Transcribe the following: <|audioplaceholder|>"      │
│ NEW: "<|en|><|transcribe|><|notimestamps|> Transcribe..."  │
│                                                              │
│ Result: Model knows to output English ✓                    │
└─────────────────────────────────────────────────────────────┘

                           ▼
┌─────────────────────────────────────────────────────────────┐
│ EXPECTED OUTPUT                                              │
│                                                              │
│ Input:  English audio "Hello, how are you?"                │
│ Output: "Hello, how are you?"                              │
│                                                              │
│ Language distribution:                                      │
│   English tokens: >80% ✓                                   │
│   Chinese tokens: <5%  ✓                                   │
│   Other tokens:   <15% ✓                                   │
└─────────────────────────────────────────────────────────────┘
```

## Summary of Issues

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| **Vocab Mismatch** | Out-of-vocab tokens → `��` | Missing 267 Canary tokens | `fix_tokenizer.py` |
| **Chinese Bias** | 72% Chinese tokens | No language control in prompt | Add `<\|en\|>` token |
| **Mixed Languages** | Random language switching | High entropy + no control | Language tokens + temp tuning |
| **Wrong Tokenizer** | Missing special tokens | Used Qwen2 instead of Qwen3 | Use Qwen3-1.7B base |

## Key Insight

The gibberish is NOT due to:
- ❌ Random initialization (weights are trained ✓)
- ❌ Wrong model architecture (matches Canary ✓)
- ❌ Corrupted weights (embeddings look normal ✓)

The gibberish IS due to:
- ✅ Vocabulary size mismatch (267 tokens missing)
- ✅ Missing language control tokens in prompts
- ✅ Wrong tokenizer source (Qwen2 vs Qwen3)
