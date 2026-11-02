# Tuning on Arc B-series GPUs (Xe2: B580, Pro B60)

The engine was written and tuned on a Core Ultra 5 125H (Xe-LPG iGPU, no XMX). On Xe2 cards the prompt GEMMs use
XMX (DPAS), but their settings are still the iGPU's, and only the owners of these cards can measure what is faster.
Every switch below gives the same results as the default (checked bit for bit on the laptop), except `--act256`,
which changes the numbers and needs a KL check before it could become a default.

The environment switches live in `zero/xe2/engine_xe2.h` and act only on **discrete** GPUs with XMX (one card or the
multi-GPU shards on them). On every iGPU, with XMX (Lunar Lake 140V, Arrow Lake 140T) or without (Xe-LPG), they are
ignored and every kernel is built exactly as before (checked: the same kernel builds, bit-identical logits). This
keeps the tuned iGPU path safe from settings meant for the B-series. (On the 125H iGPU, forced on with
`ARC_XMX_TRY`, 128 registers per thread made the 35B's prompt 2.7-3.5x slower: 519 -> 194 tok/s for the dense GEMMs,
148 with the MoE GEMMs too.)

**Default since 2026-11-02:** the dense prompt GEMM uses 128 registers per thread on discrete cards (B60, 27B:
450 -> 617 tok/s). `ARC_GEMM_GRF=256` goes back to the old build; `ARC_MOE_GRF=128` tries the same for the MoE
expert GEMMs (not yet measured on a B-series card).

## What the data so far says (1x Arc Pro B60, Qwen3.8 27B IQ3_S)

| | B60 | Estimate |
|---|---:|---|
| Prompt | 450 tok/s (617 with 128 registers, the default since 2026-11-02); DP4a only 406 | ~33 int8 TOPS at 617, a small share of the card's XMX peak; XMX only 1.1-1.5x DP4a, so the time goes elsewhere: unpacking IQ3_S (grid lookups), the activation scales (ACT=256 +13 %), or the kernels outside the GEMMs. `runz --bench-pp 4096 --prof` with and without `ARC_NO_XMX=1` shows which |
| Generation, plain | 21.5 tok/s | 11.3 GB read per token = 243 GB/s, about half the card's bandwidth; the ceiling for plain decoding is ~40 tok/s |
| Generation with drafts | not measured | on the laptop DFlash2 + MTP give 3.7x plain on code (4.9 -> 18.1 tok/s); on the B60 the drafts' extra compute is cheap |

These are estimates from the code and the logs, not measurements (from a DeepSeek review, checked against the code).

## 1. Prompt: `bench\gemm_sweep.ps1`

```powershell
.\bench\gemm_sweep.ps1 -Models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -Tokens 4096
```

One setting at a time from the defaults, then the best of each together. New since 2026-11-01:

| Setting | Values | What it changes |
|---|---|---|
| `GRF` (`ARC_GEMM_GRF`) | 256, 128 | registers per thread of the dense GEMM; 128 (the default here since 2026-11-02) allows more threads per core |
| `MOEGRF` (`ARC_MOE_GRF`) | 256, 128 | the same for the MoE expert GEMMs (default 256); swept for MoE models only |
| `CG` (`ARC_GEMM_CG`) | 1, 2, 4 | 16-column groups per work-group (activation rows shared by more columns) |
| `ACT` (`runz --act256`) | 32, 256 | one activation scale per 256 values: much less work after each DPAS, less accuracy; not part of "best of each together" |
| `TM` (`ARC_GEMM_TM`) | 16, 32 | tokens per sub-group: **TM 32 ran DP4a before 2026-11-01** (the XMX check failed at 32), so earlier TM 32 rows did not test XMX |

Before an `ACT=256` result can become a default, its KL must stay within this project's limit (0.015 against
f32). On the laptop's 35B-A3B (where it changes only the attention / DeltaNet inputs: the experts keep per-32
scales) its KL against the default was 0.009 / 0.00001 / 0.026 / 0.0055 at four positions, and the greedy text
changed at the 4th token: probably over the limit against f32, so it stays a switch. A dense 27B quantizes more
matrices this way.

## 2. Generation: `bench\decode_try.ps1`

```powershell
.\bench\decode_try.ps1 -Models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -DFlash <the 27B's DFlash2 drafter folder>
```

The same greedy answer (256 tokens of Python) plain, with `--mtp`, `--mtp --mtp-drafts 7`, with the DFlash2
drafter (`--draft auto`), and with `ARC_GEMV_GRF=256` / `ARC_GVROWS=1,2,4` (the single-token GEMV's register file and
rows per sub-group). The drafted runs are likely the largest gain: the B60 logs so far were all plain.

## 3. Longer chunks and resident weights

```powershell
.\bin\runz.exe C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 8192 --n 64 --chunk 8192
.\bin\runz.exe C:\hf\Qwen3.5-9B-UD-Q4_K_XL.gguf --bench-pp 4096 --n 64 --budget 20
```

`--chunk 8192`: half as many just-in-time weight decodes per prompt. `--budget 20`: models whose raw and decoded
weights fit 20 GB keep the decoded ones (no decode per chunk; 4B and 9B on a 24 GB card).

## 4. Qwen3.6 35B-A3B on several B60s

With `--gpus N` each GPU caches only its share of the experts (1/N), so on 4 cards the whole model fits the GPU
caches and nothing streams from the SSD after the first pass; the RAM tier (`--expert-ram-gb`) covers what does
not fit. See [multi-gpu.md](multi-gpu.md).

Measured on 4x B60 (2026-11-02): the prompt is **slower** split (3,232 tok/s on 1 GPU, 2,040-2,122 on 2-4), because
without peer access between the cards on Windows every sum goes through host memory (65-80 % of the time waiting);
generation gains 19-31 % (77.5 -> 92.2 / 95.3 / 101.7 tok/s). The 35B fits one 24 GB card, so for now: **1 GPU for
long prompts, 2 GPUs for generation** (most of the gain; 3-4 add little).

## Not done yet (needs an Xe2 card to develop on)

- An XMX kernel built around the 2D block loads of Xe2 (`cl_intel_subgroup_2d_block_io`).
- XMX for the 2-8 token verification passes of MTP / DFlash2 (now the GEMV-style kernels).
- Several models' worth of GPUs as replicas (one engine per GPU, requests spread over them) and pipeline
  parallelism for prompts.
