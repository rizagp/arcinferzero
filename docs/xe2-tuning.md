# Tuning on Arc B-series GPUs (Xe2: B580, Pro B60, Pro B70)

All the switches (environment variables; each is read once per process):

| Switch | Default | What it does | Section |
|---|---|---|---|
| `ARC_GEMM_GRF=256` | 128 on discrete cards | the dense prompt GEMM's register file | 1 |
| `ARC_MOE_GRF=128` | 256 | the MoE expert GEMMs' register file | 1 |
| `ARC_GEMM_TM` / `_RG` / `_WGY` / `_SPLIT` / `_CG` | 16 / 4 / 8 / 17 / 1 | the int8 prompt GEMM's tiles | 1 |
| `ARC_NO_XMX=1` | XMX where the GPU has it | DP4a only | 1 |
| `ARC_X16=1` | off | the f16 XMX GEMM for W8 weights | 6 |
| `ARC_X16_TM` / `_GRF` / `_CG` | 32 / 256 / 1 | its tiles and register file | 6 |
| `ARC_GEMV_GRF=256` | 128 | the single-token GEMV's register file | 2 |
| `ARC_GEMV_MT_GRF=256` | 128 on discrete cards | the 2-8 token GEMV's register file (MTP, DFlash2, batches) | 7 |
| `ARC_GVROWS=1/2/4` | by matrix size | rows per sub-group of the codebook GEMV | 2 |
| `ARC_GEMV_PF=1` | off | the single-token GEMV issues all its rows' reads of a step first (bit-identical; the laptop iGPU, forced: 8 % slower, 5.23 -> 4.82 tok/s on the 27B; a B60 to measure) | 7 |
| `ARC_LAUNCH_STATS=1` | off | kernel launches and host time per token | 5 |
| `ARC_TP_*` | | the multi-GPU sums and split (`ARC_TP_SUM`, `ARC_TP_RS_MIN`, `ARC_TP_RG_GU`, `ARC_TP_TIMEOUT`, `ARC_TP_ATT_QSPLIT`) | [multi-gpu.md](multi-gpu.md) |

The tools: `bench\gemm_sweep.ps1` (prompt), `bench\decode_try.ps1` (generation), `bin\mgpu_probe.exe` and
`bin\event_probe.exe` (several GPUs), `bin\replay_probe.exe` (kernel launches), `bin\clinfo.exe` (what the driver
offers).

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

## 5. Kernel launches per token: would replaying a recorded step help? (`bin\replay_probe.exe`)

vLLM's CUDA graphs (a step recorded once, replayed) gave +72 % generation on a B70. Ours launches every kernel on its
own: 933 per token for the 27B, 692 for the 35B-A3B (`ARC_LAUNCH_STATS=1 runz ... --n 32` prints them and the host
time). On the laptop the host takes 3.6 / 2.6 ms per token, hidden behind the GPU's work, but the GPU idles ~7 us
between kernels (step time against the kernels' own time: 35B 5.2 ms of 51.8, 27B 6.8 of 208.7).

`replay_probe` times 700 dependent kernels appended as the engine does against a recorded list replayed. Laptop:
4,096-float kernels 1.36 -> 1.07 ms, 65,536-float ones 1.95 -> 2.01: a replay saves ~0.4 us a kernel at most, under
1 % of a token. The idle time between kernels is the GPU's own (one kernel finishes before the next starts), so
fewer kernels (fused) is the way there, not a replay. On a B-series card:

```powershell
.\bin\replay_probe.exe --kernels 700 --floats 4096
.\bin\replay_probe.exe --kernels 700 --floats 65536
$env:ARC_LAUNCH_STATS=1; .\bin\runz.exe <35B-A3B gguf> --bench-pp 256 --n 32; Remove-Item Env:ARC_LAUNCH_STATS
```

A replay saving several us a kernel there would make it worth building (35B: up to a third of a 13 ms token).
Whatever is built for the B-series goes in `zero/xe2/` and kernels of its own, chosen on discrete GPUs only: the
iGPU kernels stay as they are.

## 6. The f16 XMX prompt GEMM (`ARC_X16=1`, 2026-11-02)

The int8 XMX GEMM (`gemm.cl -DXMX`) was only ~10 % faster than DP4a on a B60 (27B: 450 vs 406 tok/s). Every 16- or
32-value block of a W8 weight row has its own scale, so each int8 DPAS (8 tokens x 16 columns x 32 values) starts
from zero and its 8 results are scaled one by one on the vector unit: about as much work as the DP4a dot products it
replaced. At 617 tok/s the 27B's prompt does ~33 int8 TOPS, about 17 % of the B60's XMX peak.

`kernels/gemm_x16.cl` works like the f16 GEMMs of vLLM and llama.cpp's SYCL backend instead:
- each 16-column group's weights are decoded to f16 once with their scales applied (times 256, so small weights stay
  normal f16) and used for 8 to 32 tokens;
- the activations are made f16 once per GEMM (`act_half`), each row divided by its largest scale, so every value
  stays within +-127 whatever the activations;
- one f32 DPAS accumulator per 8 tokens runs over the whole row (`intel_sub_group_f16_f16_matrix_mad_k16`);
- the row's scale is multiplied back at the end.

It covers the W8 layouts (100..103). That is the 27B's IQ3_S, decoded just in time. Other weight types keep
`gemm.cl`.

The kernel is off until a B60 measures it: `ARC_X16=1` turns it on, on a discrete GPU with XMX only. At start it is
checked against the DP4a kernel on random data: every W8 layout, per-32 and per-256 activation scales, stored,
split-K, added and fused SwiGLU. It is taken only when the error is under 0.5 % of the result's size. f16 rounding
gives ~0.03 %; a wrong layout gives ~100 %.

Changes from the self audit (2026-11-02):
- Weights are decoded times 16, not 256. 256 turned any weight of 256 or more into inf, and its column into NaN.
- The f16 activations' buffer (143 MB for the 27B at 4096-token chunks) is reserved with the other buffers, so the
  memory plan counts it.
- A short chunk (padded to 16 rows) takes 16 or 8 tokens per sub-group instead of falling back to `gemm.cl`.
- The start-up check goes through the engine's own launch, with:
  - a partial row group, a column offset and a row stride wider than N;
  - a matrix whose column groups CG does not divide;
  - activation scales spread over three decades.
- The check's verdict is kept per device, and a GPU wait that gives up during it is not taken as a verdict.
- A fused SwiGLU is never cut into split-K pieces (this also applies to `gemm.cl`; no model does it yet).

`ARC_X16_TRY=1` runs the kernel anywhere, with DPAS emulated by shuffles where the GPU has none. That is how it was
checked on the 125H laptop (Qwen3.8 27B IQ3_S): the check's largest error was 3.1e-4, and the next-token top-10 was
unchanged.

`ARC_X16_CG=2` / `4` (2026-11-02): each sub-group covers 2 or 4 column groups. Their weights are decoded once per
k-step, and each 8-token read of the activations feeds 2 or 4 DPAS instead of one. With one group, every 16
columns read the whole activation block again from L2, about 16 flop per byte, which caps the kernel far below
XMX's rate (inferred). sycl-tla's BMG GEMM uses 32 x 64 per sub-group and 256 x 256 per work-group. Each
accumulator still adds its 16-blocks in the same order, so CG 1 / 2 / 4 give the same logits bit for bit (checked
with DPAS emulated, 27B).

```powershell
.\bench\gemm_sweep.ps1 -Models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf   # rows X16, X16TM, X16GRF, X16CG
$env:ARC_X16=1; .\bin\runz.exe C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 4096 --n 64; Remove-Item Env:ARC_X16
```

## 7. Smaller switches (2026-11-02)

- The 2-8 token GEMV (MTP / DFlash2 verification, batched slots) with the default register file (128) instead of 256
  registers per thread: **the default on discrete GPUs since 2026-11-03** (B60, 2026-11-02: 27B MTP 43.3 -> 47.1
  tok/s). `ARC_GEMV_MT_GRF=256` goes back; `bench\decode_try.ps1` has a row for it ("mtp MT_GRF=256"). The iGPUs keep
  256: the 125H, forced to 128 for a test (2026-11-03, 27B `--mtp`, 256 tokens, warm run), went 11.78 -> 9.57 tok/s
  with the same tokens.
- The XMX checks run once per process. A split model's GPUs are the same device, and each shard checked again: 4
  B60s took 8.7-14.9 s to load.

## Not done yet (needs an Xe2 card to develop on)

- An XMX kernel built around the 2D block loads of Xe2 (`cl_intel_subgroup_2d_block_io`).
- XMX for the 2-8 token verification passes of MTP / DFlash2 (now the GEMV-style kernels).
- Several models' worth of GPUs as replicas (one engine per GPU, requests spread over them) and pipeline
  parallelism for prompts.
