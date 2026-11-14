# Benchmark and accuracy prompts (token ids)

Comma-separated token ids, fed with `bin\runz.exe <model> --tokens-file <file>`.

| File | Tokens | Use |
|---|---|---|
| `qwen-code-11k.ids.txt`, `qwen-code-34k.ids.txt` | 11,240 / 33,937 | time to first token and long-context generation, Qwen tokenizer (code scans of two local projects) |
| `gemma-code-11k.ids.txt`, `gemma-code-34k.ids.txt` | 13,921 / 35,957 | the same texts with the Gemma 4 tokenizer |
| `gemma-code160.ids.txt` | 160 | Gemma 4 accuracy: the first 160 tokens of `gemma-code-11k` |
| `gemma-raw120.ids.txt` | 120 | Gemma 4 accuracy: 120 tokens of raw text (llama.cpp documentation) |
| `qwen-prose-3584.ids.txt` | 3,584 | long-prompt accuracy, Qwen tokenizer: the start of docs/lessons-intel-arc-igpu.md (prose; tokenized by llama-tokenize b12292). For Flash Next, past QSA's dense range: `REF_ALL_FROM=3328` / `ARC_KL_FROM=3328` |

Accuracy against the f32 CPU reference (the reference dumps are 120–170 MB, so they are not kept here; make them
once with `tools\ref_forward`):

```powershell
$g = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Filter "gemma-4-12b-it-qat-q4_0.gguf" | Select-Object -First 1).FullName
$ids = (Get-Content bench\prompts\gemma-code160.ids.txt -Raw).Trim()
$env:REF_ALL = "1"; .\bin\ref_forward.exe $g $ids ref_g12_code160.bin; $env:REF_ALL = $null
.\bin\runz.exe $g --tokens-file bench\prompts\gemma-code160.ids.txt --kl-ref ref_g12_code160.bin --n 1        # prompt path
.\bin\runz.exe $g --tokens-file bench\prompts\gemma-code160.ids.txt --kl-ref ref_g12_code160.bin --kl-decode  # generation path
```

Results on 10 October 2026 (the one-pass mode of then): prompt path 0.018, generation path 0.013 (code); generation
path 0.029 (raw text). Since 2026-10-22 Gemma 4 12B runs two int8 passes: KL 0.0002 on the code prompt.
Long code prompts are a poor KL test: Flash Next on the first 3,584 tokens of `qwen-code-11k` measures 0.18, almost
all of it four positions in a run of digits and identifiers where the engine, ref_forward and llama.cpp b12292 each
predict a different token; `qwen-prose-3584` measures 0.0105 (default) / 0.0060 (precise) on the same model
(2026-11-14). `ARC_KL_SHOW=x` (runz) lists the positions above x with both sides' probability of the text's next token.
For a long prompt, `REF_ALL_FROM=N` (ref_forward) writes the reference only from position N on and `ARC_KL_FROM=N`
(runz) compares from there, so the reference of a 3.5K-token prompt does not need every position.
