# Tuning on Arc B-series GPUs (Xe2: B580, Pro B60, Pro B70)

All the switches (environment variables; each is read once per process):

| Switch | Default | What it does | Section |
|---|---|---|---|
| `ARC_GEMM_GRF=256` | 128 on GPUs with 16-lane XMX (discrete cards, Lunar / Panther Lake) | the dense prompt GEMM's register file | 1 |
| `ARC_MOE_GRF=128` | 256 | the MoE expert GEMMs' register file | 1 |
| `ARC_MOE_CG` / `ARC_MOE_WY` | 1 / 1 | sub-groups per work-group of the MoE expert GEMM (prompt chunks of 1,280+ tokens): along a tile's columns (its activation rows shared through L1) / along the tiles, mostly one expert's (its weights shared). Bit-identical at every shape. A B60's 35B 16K prompt spent 1.55 s of 5.08 in these GEMMs (+0.26 s `decode_w8`), about 21 TOPS against ~50 for the dense GEMMs; laptop: no gain (2026-11-07) | 1 |
| `ARC_GEMM_TM` / `_RG` / `_WGY` / `_SPLIT` / `_CG` | 16 / 4 / 8 / 17 / 1 | the int8 prompt GEMM's tiles | 1 |
| `ARC_NO_XMX=1` | XMX where the GPU has it | DP4a only | 1 |
| `ARC_X16=1` | off | the f16 XMX GEMM for W8 weights; since 2026-11-08 also on iGPUs with 8-lane DPAS (the 140T: `gemm_x16.cl -DX16_SG8`) | 6, 8 |
| `ARC_140T=0` | on, on iGPUs with 8-lane DPAS and weights decoded just in time | the Arc 140T preset off: per GEMM, chunks of 512+ tokens take XMX + H8 + TM 32 + CG 2 + 128 registers, shorter chunks and resident weights the DP4a default (140T 27B: 2K 180 -> 221, 8K 185 -> 231 tok/s; 512 tokens the same). `=1` with `ARC_XMX_TRY=2`: the preset on a GPU without XMX, for tests | 8 |
| `ARC_XMX_FORCE=1` | off | on a GPU with 8-lane sub-groups (Arrow Lake H): XMX (gemm_x8.cl), still checked against DP4a bit for bit; off because a 140T measured it 1.6x slower in the 27B's GEMMs. On iGPUs with 16-lane XMX: XMX without the speed check | 8 |
| `ARC_X8_W=1` | off | 8-lane GPUs with `ARC_XMX_FORCE=1`: gemm_x8 with one sub-group for both 8-column halves of a 16-column group, the activations loaded once (ocloc `arl-h`: 28 % fewer memory messages and 17 % fewer instructions per DPAS, no spills at 256 registers and 16 tokens); bit-identical | 8 |
| `ARC_GEMM_GRF=128` on 8-lane GPUs | 256 there | the default register file for gemm_x8 too (no spills at 16 tokens without `ARC_X8_W`; spills with it or with 32 tokens, except H8). Since 2026-11-08 also the DP4a GEMM of iGPUs (it spills there: 6x slower on the 125H) | 8 |
| `ARC_GEMM_CG` on iGPUs | 1 | since 2026-11-08 also the DP4a GEMM of iGPUs without XMX switches, when set (bit-identical) | 8 |
| `ARC_GEMM_WPF=n` | off | the XMX prompt GEMM prefetches its weights n 64-value chunks ahead (4: the next superblock), within its column group; bit-identical (2026-11-08) | 8 |
| `ARC_H8_SWI=0` | on | `swiglu_quant` before an H8 down GEMM again; by default the rotation reads the up GEMM's output (silu(gate) * up in f32) itself and `swiglu_quant` is left out (2026-11-12, the same results: the 27B on the laptop +2-2.5 % a prompt; a B60's 27B 4K prompt with both 1,024.5 -> 1,080.9 tok/s, issue #4) | 8 |
| `ARC_H8_REUSE=0` | on | H8 GEMMs on the same input (q / k / v, gate / up, qkv / z) rotate it before each GEMM again, instead of once (`h8_act`; 2026-11-12, the same results: the 27B on the laptop 400 -> 256 rotations a prompt, -30 % of their time) | 8 |
| `ARC_H8_FUSE=0` | on | H8's weight rotation after `decode_w8` again, instead of in one pass with the decode (`decode.cl h8r_decode`); the same values | 8 |
| `ARC_XMX_SPEED=1` | off | the start-time speed check (a model-sized GEMM, XMX against DP4a) on discrete cards too, and under `ARC_XMX_TRY` (tests); iGPUs with XMX always run it | 8 |
| `ARC_GEMM_2D=0` / `ARC_GEMM_2D_PF=0` | on, where the driver offers them with XMX | the XMX prompt GEMM's activations by 2D block loads (one load for 8 token rows x 64 bytes), with a prefetch of the next 64 bytes; these switches turn them off | 8 |
| `ARC_H8=0` | on for Qwen dense models on discrete cards with 16-lane XMX (with `ARC_GEMM_CG=4`), and in the 140T preset | H8 off (2026-11-08: B60 27B 512 / 4K / 16K 615 / 766 / 706 -> 677 / 1,024 / 937 tok/s with it) | 8 |
| `ARC_H8=1` | off | the H8 GEMM (QUEUE 31.12): weights and activations Hadamard-rotated, int8 with one scale per column and per token; DPAS on XMX, DP4a else; iGPU too. Qwen models rotate the activations from f32 (`ARC_H8_INT=1`: from int8). `ARC_H8_SB=1`: one scale per 256 values, as the B60 ran it on 2026-11-06. `ARC_H8_D=1`: with random signs. On a B60's 27B the fastest so far (918 tok/s against 764); slower than the default on the laptop | 8 |
| `ARC_GEMM_XW=1` | on where H8 is on by default on discrete cards since 2026-11-13 (gemm_xw 32 x 64: a B60's 27B 512 / 4K / 16K prompts 691 / 1,067 / 979 -> 717 / 1,292 / 1,170 tok/s) and in the Arc 140T preset (gemm_xw8 32 x 16 at 256 registers, 2026-11-11: 27B 4K prompt 214 -> 299 tok/s); off elsewhere; `=0` off | the H8 (per-column / per-token scale) prompt GEMMs through `gemm_xw.cl` (QUEUE 33.4, 2026-11-11): a sub-group holds 32 tokens x 64 columns instead of 16 or 32 x 16, so each 64-value chunk of activations is loaded once for 64 columns and each column group's weights once for 32 tokens (twice the DPAS work per byte loaded). Bit-identical to `gemm.cl` (checked at start bit for bit; the laptop with DPAS emulated: the 4B's logits identical at every shape). `_TM=16/32`, `_NC=1/2/4` (16-column groups), `_GRF=128`; `bench\xw_sweep.ps1`. ocloc `bmg`: 32 x 64 at 256 registers no spills (128 DPAS a superblock), at 128 registers 5.6 KB spilled; the smaller shapes no spills at either | 11 |
| `ARC_GEMV_GU=1` | off | one token: the FFN's gate and up GEMVs in one launch of 2F rows instead of two of F (QUEUE 33.5, 2026-11-11), each row as in its own launch; bit-identical. The laptop: no faster (27B ~3 % slower); `decode_try.ps1` row `GEMV_GU=1` | 7 |
| `ARC_X16_TM` / `_GRF` / `_CG` | 32 / 256 / 1 | its tiles and register file | 6 |
| `ARC_GEMV_GRF=256` | 128 | the single-token GEMV's register file | 2 |
| `ARC_GV_IQ3S=1` | off | single-token IQ3_S weights through `gemv_v` (straight into registers) instead of `gemv`'s local window; bit-identical; the 125H 6 % slower (27B step 210 -> 223 ms), a B60 to measure (2026-11-08) | 2 |
| `ARC_MOE_GR_GU` / `ARC_MOE_GR_DN` | 1 / 4 | rows per sub-group of the experts' gate / up and down GEMVs for one token (`bench\decode_try.ps1` rows for MoE models) | 2 |
| `ARC_GEMV_MT_GRF=256` | 128 on GPUs with 16-lane XMX (discrete cards, Lunar / Panther Lake) | the 2-8 token GEMV's register file (MTP, DFlash2, batches) | 7 |
| `ARC_GVROWS=1/2/4` | by matrix size | rows per sub-group of the codebook GEMV | 2 |
| `ARC_GEMV_PF=1` | on on discrete cards for dense models since 2026-11-08 (B60 27B 22.33 -> 23.08 tok/s) and MoE models since 2026-11-12 (B60 35B 79.93 -> 88.91; off for them 2026-11-09 to 11-12 after a run of older code, 78.77 with it, 85.15 without); off on iGPUs; `=0` / `=1` force it | the single-token GEMV issues all its rows' reads of a step first (bit-identical; the laptop iGPU, forced: 8 % slower, 5.23 -> 4.82 tok/s on the 27B; a B60, 2026-11-06, best of 2: 27B 22.39 -> 23.11, 35B 84.08 -> 83.88) | 7 |
| `ARC_GDN_CHUNK=1` / `2` / `3` | off (`gdn_recur`) | the DeltaNet of prompt passes (32+ tokens) in chunks of 32 (QUEUE F33.1, 2026-11-09): `1` `gdn_chunk` (vector, 4 work-items per state row, any GPU), `2` its 16-float local reads, `3` on XMX: `gdn_chunk_x` (16-lane DPAS) or `gdn_chunk_x8` (8 lanes, the 140T), checked at start against `gdn_chunk`. Not bit-identical: KL (`bench\f33_sweep.ps1`) | 9 |
| `ARC_GDN_XEMU=1` / `8` / `16` | off | with `ARC_GDN_CHUNK=3`: DPAS emulated on any GPU, in its own / the 8- / 16-lane form (tests) | 9 |
| `ARC_ATTN_XMX=0` | on on discrete cards since 2026-11-10 (B60, 64K prompts: 4B +11.7 %, 9B +8.4 %, 27B +2.1 %, 35B +10.2 %), off on iGPUs; `=1` forces it | the prompt attention's integer dot products (`attn_q8`) as int8 DPAS, bit-identical, checked at start bit for bit (QUEUE F33.2, 2026-11-09; 16-lane DPAS, Qwen); `ARC_ATTN_XEMU=1`: emulated (tests) | 9 |
| `ARC_GEMV_SMALL_WG=1` | off (on on discrete cards 2026-11-10 to 11-12: a B60's 27B no gain, 23.71 without against 23.72 tok/s, gemm_ab 13.35 -> 12.72 ms; its 35B 80.35 without against 55.85 tok/s, package 18acafc); `=1` forces it | the single-token GEMV of a matrix with at most 512 sub-groups (DeltaNet's alpha / beta, 96 rows; MoE routers) in work-groups of 4 sub-groups instead of 16, spread over more Xe cores; bit-identical |10 |
| `ARC_WIDE_ROWS=0` | on (every GPU) since 2026-11-10 | the 1-8 token row kernels that quantize for the next GEMV (`swiglu_quant` on the Qwen and MoE paths, `rmsnorm_quant`, `gated_norm_quant`, `attn_gate_quant`) in one work-group per row, as before; by default their superblocks spread over many work-groups; bit-identical |10 |
| `ARC_XALL=0` | on on discrete cards since 2026-11-09, off on iGPUs; `=1` forces it | MoE models, one token: with every expert in the GPU's cache, the layers skip reading their experts' ids on the host (the same tokens; a B60's 35B 78.77 -> 88.05 tok/s, host waits 43 -> 3 a token) | 5 |
| `ARC_LAUNCH_STATS=1` | off | kernel launches and host time per token | 5 |
| `ARC_TP_*` | | the multi-GPU sums and split (`ARC_TP_SUM`, `ARC_TP_RS_MIN`, `ARC_TP_RG_GU`, `ARC_TP_TIMEOUT`, `ARC_TP_ATT_QSPLIT`) | [multi-gpu.md](multi-gpu.md) |

The tools: `bench\gemm_sweep.ps1` (prompt), `bench\decode_try.ps1` (generation), `bin\mgpu_probe.exe` and
`bin\event_probe.exe` (several GPUs), `bin\replay_probe.exe` (kernel launches), `bin\membw_probe.exe` (memory read
bandwidth of the CPU, the GPU and both; docs/speed-progress.md), `bin\clinfo.exe` (what the driver
offers), `bin\gpu_bench.exe` (the matrix engines' int8 / f16 / bf16, the vector engines and memory measured against
the GPU's figures on paper; the launcher's More > Benchmark this GPU; a page in `%USERPROFILE%\.inferzero\bench\gpu`
that compares every result file there).

**Measured with `gpu_bench` (2026-11-11), against EUs x maximum clock x the per-EU rate of the GPU's kind:**

| GPU | INT8 XMX | FP16 XMX | FP32 vector | FP16 vector | INT8 DP4a | Memory read |
|---|---|---|---|---|---|---|
| Core Ultra 5 125H (Xe-LPG, 112 EUs, 2.2 GHz) | none | none | 3.85 of 3.94 TFLOPS (98 %) | 7.69 of 7.88 (98 %) | 12.98 of 15.77 TOPS (82 %) | 76.2 GB/s |
| Arc 140T (Xe-LPG Plus, 128 EUs, 2.35 GHz; issue #5, 2026-11-11) | 76.75 of 77.0 TOPS (100 %) | 38.38 of 38.5 TFLOPS (100 %); BF16 37.56 (98 %) | 4.70 of 4.81 (98 %) | 9.39 of 9.63 (98 %) | 15.07 of 19.25 TOPS (78 %) | 81.5 GB/s |
| Arc B580 (Xe2-HPG, 160 EUs, 2.85 GHz; Reddit, 2026-11-12, driver 101.9175) | 233.21 of 233.47 TOPS (100 %) | 116.60 of 116.74 TFLOPS (100 %); BF16 116.60 (100 %) | 14.52 of 14.59 (99.5 %) | 13.72 of 29.18 (47 %: the FP32 rate, no packed FP16) | 55.24 of 58.37 TOPS (95 %) | 1,149 GB/s, **not valid** (a one-value buffer, 456 on paper; random data since 2026-11-12) |
| Pro B60, Arc B390 | | | | | | waiting for their owners' runs |

The 140T run settles its matrix figures: Intel's 77 int8 TOPS is what its XMX does (a spec database's 154 int8 TOPS and 77 FP16 TFLOPS are twice that), and the engines run at their maximum clock under load. Its 27B prompt (~12 int8 TOPS at 4K) uses about a sixth of them: the room is in the kernels, not the hardware.

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

**Measured 2026-11-12 (issue #4, package fffaf1a, one Pro B60):**

| Switch | 27B | 35B-A3B | Taken |
|---|---|---|---|
| plain | 23.72 | 79.93 | |
| `--mtp` | 49.94 (69.2 % accepted) | 110.06 (73.0 %) | (the server's default) |
| `ARC_GEMV_GU=1` | 22.79 | 80.74 | no |
| `ARC_GEMV_PF` (prefetch) | 22.46 without (on for dense) | **88.91 with** | **on for MoE too** |
| `ARC_BATCH=16 / 32 / 128` | 21.25 / 22.67 / 22.02 | 85.11 / 83.42 / 76.66 | no (35B: to measure with the prefetch on) |
| `ARC_GVROWS=1 / 2 / 4` | 20.41 / 23.09 / 22.60 | 79.62 / 80.75 / 80.87 | no |
| `ARC_GEMV_GRF=256` | 21.53 | 78.80 | no |

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
**A B60, 2026-11-06 (4,096 floats):** 700 kernels 1.790 ms appended, 0.459 ms replayed (2.56 -> 0.66 us a kernel). The
35B takes 632 launches and 2.89 ms of host time per 12.0 ms token (4.6 us a launch). So a replay could save 1.2-2.5 ms
of a token, ~10-20 %, if the host's time is on the token's critical path and a step can be replayed at all: the
experts change every token. Not measured in the engine, not built.
Whatever is built for the B-series goes in `zero/xe2/` and kernels of its own, chosen on discrete GPUs only: the
iGPU kernels stay as they are.

**Built 2026-11-08, off by default: `ARC_BATCH=n`.** A step cannot be recorded once and replayed as it is, because its
arguments (positions, lengths) change every token. So each step is recorded again, in regular in-order lists of up
to n kernels. Each list is appended to the immediate list as soon as it is full
(`zeCommandListImmediateAppendCommandListsExp`), so the GPU starts on the first list while the host records the next.
The lists start at 4 kernels after each host wait and double up to n. The commands, their order and the list are the
same as before, so the results are bit-identical.

- **The laptop:** the call works. `replay_probe` shows no gain there (as before), and the engine's generation is the
  same on the 4B.
- **The 35B's catch: it waits on the host about 40 times a token**, because every MoE layer reads its experts' ids for
  the expert cache (`ARC_LAUNCH_STATS=1` prints the waits). Between waits there are only ~15 kernels, so the lists
  stay short; on the laptop that was -5 %.
- **`ARC_XALL`** drops those reads when every expert is cached, as on a 24 GB card. The kernels then read the same
  expert addresses with no waits between layers. On the laptop the cache never holds every expert. A B60, 2026-11-09:
  the 35B 78.77 -> 88.05 tok/s, host waits 43 -> 3 a token, the same tokens: **on by default on discrete cards since
  then** (`ARC_XALL=0` off).

`bench\decode_try.ps1` has rows for both and checks that their tokens match plain.

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

## 8. 2D block loads and H8, measured (issue #4, 2026-11-06, commit 9c18778)

One Pro B60, Qwen3.8 27B IQ3_S, `runz --bench-pp 4096 --n 16 --prof`, the owner's logs:

| Prompt GEMMs | tok/s | GEMMs' GPU time (`gemm_down` / `up` / `gate`, ms) | Rotations |
|---|---|---|---|
| DP4a (`ARC_NO_XMX=1`) | 402 | 2,209 / 2,200 / 2,073 | - |
| XMX | 608 | 1,401 / 1,350 / 1,316 | - |
| XMX + H8 | 570 | 1,133 / 1,087 / 1,033 | 1,575 ms |
| XMX + 2D | 752 | 1,100 / 1,072 / 1,005 | - |
| **XMX + 2D + prefetch (the default now)** | **763** | 1,087 / 1,066 / 982 | - |
| XMX + 2D + H8 | 688 | 824 / 802 / 761 | 1,575 ms |

- **2D block loads with the prefetch are the default on discrete cards.** Generation does not use these kernels (20.5
  tok/s in every row). iGPUs do not get them: the laptop's logits are bit-identical (4B, 16 tokens, `--dump-logits`),
  and an iGPU with XMX has not measured them.
- **What guards them:** at start the kernels are compared with DP4a bit for bit on random data (K = 1024), and the log
  says "with 2D block loads" when they pass. On a B60, 2026-11-07, the 27B's logits on an 11-token prompt and 16 tokens
  were bit-identical with and without them (`fc`), and they ran over 2 and 4 cards. The check:

  ```powershell
  $env:ARC_GEMM_2D=0; .\bin\runz.exe <model.gguf> "<a prompt>" --n 16 --dump-logits off.bin; Remove-Item Env:ARC_GEMM_2D
  .\bin\runz.exe <model.gguf> "<a prompt>" --n 16 --dump-logits on.bin
  fc.exe /b off.bin on.bin
  .\bin\runz.exe <model.gguf> --bench-pp 4096 --n 16 --gpus 4
  ```

- **H8 stays off.** Its GEMMs are ~20 % faster alone and ~40 % with 2D loads, but rotating the weights (1,105 ms) and
  the activations (470 ms) costs more. With rotations costing nothing, 2D + H8 would take 4,278 ms of GPU time, about
  950 tok/s: an upper bound worked out from the table, not a measurement. Fusing the rotations into `decode_w8` and
  the quantization kernels is what could reach part of it.
- **H8's accuracy on the B60:** USAGE.md, 4 x 512: perplexity 6.6371 -> 6.6577; KL against the default path mean
  0.00120, max 0.0777, same top token 98.04 %.
- **H8 over several B60s:** [multi-gpu.md](multi-gpu.md).

**Why that H8 did not gain what urakozz's does.** Its engine (urakozz/intel-arc-inference-server, Apache-2.0, a B70, a
W4A16 checkpoint; `src/kernels/prefill/pf_int8.cl`, `src/runtime/prefill/int8.cc`, spec 5) gains 1.28x over its bf16
prompt, 2,104 tok/s. Its GEMM runs 2.29x faster than the bf16 one (gate || up: 6.08 -> 2.66 ms) at 72-81 % of the int8
DPAS peak, and pays about 50 % of that GEMM's time for its rotations. Ours on the B60 (2D + H8): the GEMM 1.33x, ~60
of ~194 T/s (31 %), the rotations 47 %. So the rotations cost about the same share: the GEMM is what falls short. The
differences its code shows:

| | urakozz | ours before 2026-11-06 | ours now (`ARC_H8=1`) |
|---|---|---|---|
| Scales | one per column, one per token, applied once at the end of K | one per 256 values | the same as urakozz |
| Column scales | computed once per weight | every decode | once per weight |
| Rotation | in registers, 64 floats x 16 lanes; words through local memory once | through local memory, a barrier per stage | in registers |
| Signs before the rotation | random, per K | none | none (`ARC_H8_D=1`: random): no gain measured on our models |
| Activations rotated from | bf16, not rounded before | int8 | f32, but the int8 rounding still runs |
| Weights rotated from | the file's int4, in one kernel | our W8 form, decoded first | the same as before |
| DPAS sums | stay in the DPAS accumulators | each DPAS starts at zero, then an add per token | stay in the accumulators (no B60 run yet) |
| Activations | 2D loads, 32 rows at once | 2D loads, 8 rows at once | the same as before |
| Weights | 2D loads with a prefetch, 32 columns a sub-group | block reads, 16 columns | the same as before |
| Sub-group tile | 32 tokens x 32 columns, a 256 x 128 work-group, a split barrier each 64 values | 16 x 16 | the same as before |
| Registers | not stated for its production kernel (its W4A8 probe: 256) | 128 | the same as before |
| Slabs | rotates and multiplies 1024 columns at a time | a whole matrix at a time | the same as before |
| Fused GEMMs | gate and up in one GEMM, the SiLU in its end; qkv and z in one | separate launches | the same as before |

Its production default since 2026-10-27 is bf16 again, by its operator's decision, with int8 opt-in.

**The iGPU, 2026-11-06, 27B IQ3_S, `--bench-pp 4096 --n 16 --prof`, best of 3 per run, alternated with the default;
KL against the default path on USAGE.md, 4 x 512:**

| Form | tok/s | GEMMs' GPU time | Rotations | 27B KL mean / max | 4B KL mean / max |
|---|---|---|---|---|---|
| Default | 139.9-140.3 | 25.7-26.1 s | - | - | - |
| H8, one scale per 256 values | 111.9 | 25.6 s | 6.64 s | 0.00100 / 0.030 | 0.00403 / 2.41 |
| H8, per column and token, a second rounding pass | 105.7 | 26.8 s | 7.87 s | 0.00198 / 0.094 | 0.00521 / 2.07 |
| H8 in registers, one pass, signs, from int8 | 121.2-122.5 | 25.0-25.3 s | 4.23-4.33 s | 0.00148 / 0.048 | 0.00243 / 0.072 |
| **H8 now: also from f32, no signs** | **115.8-117.2** | not read | 4.24-4.33 s | 0.00154 / 0.064 | 0.00267 / 0.118 |

- On DP4a the scales were not the cost: the GEMMs gained 1-4 %. DP4a runs the same dot products with or without them.
  With the rotations free, H8 could only reach ~+3 % over the default. **It stays off on the iGPU.**
- The rotation in registers: `h8_act` 1.68 -> 0.58 s, `h8_weights` 4.96 -> 3.70 s.
- The signs show no clear gain. Against the default path on USAGE.md the 4B went 0.00310 / max 0.56 without them to
  0.00243 / 0.072 with them, but against f32 below it goes the other way, so they are off (`ARC_H8_D=1`: on).
- 4B (weights resident, rotated once): 846-925 tok/s against 885-900 by default, within the run-to-run spread.
- These are against the default path, not against f32 (the default path: 27B KL 0.0003-0.0004 against f32).
- Split models (`ARC_TP_SIM`, 2 and 4 shards, a 903-token prompt): 27B greedy text equal to one GPU over 16 tokens, KL
  per row at most 0.006. 4B: the text parts at a close choice (one GPU H8, row 3, KL 0.137, the worst seen: 0.66 / 0.23
  against 0.46 / 0.45; 2 and 4 shards, row 6, a tie of 0.31 / 0.30).

**Against f32** (`tools/ref_forward`, 480 tokens of USAGE.md, one chunk; `runz --kl-ref` now also prints the logits'
relative L2, worst cosine and top-5 overlap, each row's mean taken out). "Simulated" is the closest to urakozz's method
on our models: `ref_forward REF_HAD=1024 REF_HAD_D=1`, activations rotated straight from f32, one int8 scale per token
and per weight row. It keeps our weights' quantization (IQ3_S, Q4_K), not his W4A16 file.

| Model, form | KL mean | KL max | Same top token | Relative L2 | Worst cosine | Top-5 |
|---|---|---|---|---|---|---|
| 27B default | 0.00054 | 0.015 | 475 / 480 | 2.55 % | 0.952 | 0.986 |
| 27B H8 per 256 values | 0.00109 | 0.020 | 468 | 3.55 % | 0.936 | 0.977 |
| 27B H8 from int8, signs | 0.00141 | 0.058 | 465 | 5.45 % | 0.790 | 0.976 |
| 27B H8 from int8, no signs | 0.00151 | 0.042 | 470 | 5.46 % | 0.805 | 0.975 |
| 27B simulated | 0.00080 | 0.032 | 476 | 4.59 % | 0.802 | 0.983 |
| 27B simulated, no signs | 0.00083 | 0.026 | 476 | 3.22 % | 0.939 | 0.980 |
| 4B default | 0.00100 | 0.032 | 472 / 480 | 2.46 % | 0.998 | 0.979 |
| 4B H8 per 256 values | 0.00195 | 0.025 | 469 | 3.42 % | 0.992 | 0.974 |
| 4B H8 from int8, signs | 0.00274 | 0.126 | 468 | 4.03 % | 0.989 | 0.968 |
| 4B H8 from int8, no signs | 0.00245 | 0.030 | 468 | 3.98 % | 0.995 | 0.970 |
| 4B simulated | 0.00150 | 0.011 | 471 | 3.15 % | 0.996 | 0.970 |
| 4B simulated, no signs | 0.00155 | 0.039 | 468 | 3.20 % | 0.996 | 0.978 |

| 27B H8 from f32, signs | 0.00133 | 0.127 | 476 | 5.13 % | 0.791 | 0.975 |
| **27B H8 now: from f32, no signs** | **0.00113** | 0.046 | 474 | 4.05 % | 0.828 | 0.977 |
| 4B H8 from f32, signs | 0.00191 | 0.050 | 469 | 3.55 % | 0.995 | 0.974 |
| **4B H8 now: from f32, no signs** | **0.00195** | 0.018 | 464 | 3.63 % | 0.995 | 0.977 |

- Every form is about ten times under the 0.015 limit on the mean.
- **Rotating the activations from f32 instead of their int8 form** lowers the KL: 27B 0.00151 -> 0.00113, 4B 0.00245
  -> 0.00195 (both without signs). It costs ~5 % on the iGPU (122 -> 116-117 tok/s): the quantization kernels write a
  float copy as well (0.99 -> 2.2 s), `h8_act` reads it value by value (0.58 -> 0.99 s), and it takes four more times
  the int8 activations' memory (27B at 4096 tokens: +285 MB). The int8 rounding still runs for the GEMMs H8 does not
  cover and for generation; a chunk's kernels could skip it. It closes part of the gap to the simulation. What is left is not separated yet. The simulation has only the GEMMs' error, where
  the engine also has its attention and its weights first decoded to int8; the default's own 0.00054 / 0.00100 is of
  that size.
- The signs do not earn their place in these runs: the largest row's KL is higher with them in five of six pairs
  (27B from f32 0.127 against 0.046), the means about equal.
- The per-column form has a higher KL than the per-256 one in the engine: coarser scales cost accuracy.
- Urakozz publishes no KL for this path. His 12.0 -> 12.5 % relative L2 (against bf16, the file's own error included)
  means about 3.5 % added, if the errors are independent (worked out here): the same size as ours, 3.2-4.6 %
  in the simulation, but on another model file, prompts and reference.

**The B60, 2026-11-07, commit 515f416, 27B IQ3_S, `--bench-pp 4096 --n 16 --prof`, the owner's log:**

| Form | tok/s | GEMMs' GPU time | Rotations | Quantization kernels |
|---|---|---|---|---|
| Default: 2D + prefetch | 758.7-763.9 | 4,373 ms | - | 166 ms |
| **H8 now** (per column and token, from f32, no signs, DPAS accumulators) | **916.2-918.5** | 2,766 ms | 555 ms | 328 ms |
| H8 now, `ARC_GEMM_TM=32` | 747.6-750.0 | 3,772 ms | 552 ms | 328 ms |
| H8 now, `ARC_GEMM_GRF=256` | 781.8-783.3 | 3,531 ms | 554 ms | 327 ms |

| GPUs | Default: 2D + prefetch | H8 now |
|---|---|---|
| 2 | 1,090-1,097 tok/s, step 27.3 ms | not run |
| 4 | 1,455-1,501 tok/s, step 18.4 ms | 1,546-1,570 tok/s, step 18.3 ms |

- The per-column scales did what they were built for on XMX: the GEMMs are 1.58x the default's (1.33x in the per-256
  form), about 72 T/s on gate + up, ~37 % of the int8 peak (worked out; urakozz's 72-81 % on a B70).
- Against the default path: KL mean 0.00144, max 0.0274, same top token 97.25 %, perplexity 6.6371 -> 6.6501.
- H8 takes 0.34 GiB more GPU memory. Generation is unchanged (20.5 tok/s).
- Over 4 cards H8 adds 5 %: GPU time per card would allow ~1.1 s per chunk and a chunk takes 2.6 s, so the sums limit it.
- Not run yet: H8 on the 35B or on a model with resident weights, so it stays opt-in.

**What the B60 was asked to run first** (the kernels build for Battlemage with ocloc,
and a failure of the start check now only leaves H8 out of the XMX GEMMs, with the rest of XMX kept for runs without
`ARC_H8`):

```powershell
$env:ARC_H8=1; .\bin\runz.exe <27B gguf> --bench-pp 4096 --n 16 --prof
$env:ARC_GEMM_TM=32; .\bin\runz.exe <27B gguf> --bench-pp 4096 --n 16 --prof; Remove-Item Env:ARC_GEMM_TM
$env:ARC_GEMM_GRF=256; .\bin\runz.exe <27B gguf> --bench-pp 4096 --n 16 --prof; Remove-Item Env:ARC_GEMM_GRF
.\bin\runz.exe <27B gguf> --perplexity USAGE.md --ppl-chunks 4 --ppl-kl base.kl; Remove-Item Env:ARC_H8
```

If the GEMMs reach far more than the 1.33x, the next steps are the weights in 2D loads and urakozz's wider tile.

**The B60's sweep (2026-11-08, package 4aae0c5).**
- **27B, default path:** 770 tok/s at 4K. No setting was 3 % ahead (column groups 4: 780, no split-K: 781); 256
  registers 514, 32 tokens 679.
- **27B, H8:** 924 tok/s at 4K, and **966 with `ARC_GEMM_CG=4`** (CG 2: 949). At 16K, 892 and 930.
- **But at 512 tokens H8 loses:** 493-500 tok/s against 619 for the default. The rotations do not pay at short
  prompts, so H8 as a default would need a chunk-size floor.
- **Accuracy:** H8's KL against the default is 0.00136 (max 0.024, same top token 97.8 %), perplexity 6.2865 ->
  6.2951. CG 4 gives the same numbers.
- **35B:** `ARC_MOE_WY=4` 3,221 against 3,172 (+1.5 %; WY 2 and 8 alike), bit-identical. `ARC_MOE_CG`, H8 and f16
  XMX: no gain. Chunk 8192 stays best (4096: 2,966; 2048: 2,624).

**Generation (2026-11-08).** One B60, from issue #4's logs:
- **The 27B** generates 22.4 tok/s (44.6 ms a step) and reads 11.3 GB a token, about 57 % of the measured 429
  GB/s; at full bandwidth a step would be ~26 ms (~38 tok/s). With MTP it reaches 47 tok/s.
- **The 35B** generates 84 tok/s and reads 2.4 GB a token (42 %).
- **The laptop's `--prof-gen` of the 27B:** the GEMVs take ~92 % of a step (gate / up / down alone 59 %), the small
  kernels under 10 %.

The levers, largest first:
- **Command-list replay.** `replay_probe` on a B60 measured 2.56 us per kernel on the engine's immediate list
  against 0.66 us replayed. With 632 kernels a token that is ~1.2 ms of the 35B's 12 ms (~10 %), and ~1.8 ms of the
  27B's 44.6 (~4 %). It needs the per-token values (position, context length, KV offsets) in GPU memory, or mutable
  command lists, so the list is recorded once. Not built yet.
- **The IQ3_S GEMV.** It goes through a local window with two barriers a step: `ARC_GV_IQ3S=1` (gemv_v) to measure.
  `ARC_GEMV_PF=1` gave +3 % (22.39 -> 23.11).
- **The MoE per-tile GEMVs:** their rows per sub-group (`ARC_MOE_GR_GU` / `_DN`), unmeasured on a B60.
- `bench\decode_try.ps1` now ends with a `--prof-gen` table per model, the first such profile from a B60.

**Generation (2026-11-09, package 2c132a9, `decode_try.ps1`, 256 tokens, best of 2).** One B60 (of four in the
machine):
- **27B IQ3_S:** plain 23.21 tok/s, MTP 49.31 (69 % accepted). No switch beat plain clearly: GEMV_PF=0 22.45,
  GVROWS 1/2/4 21.99-23.04, GV_IQ3S=1 23.31, BATCH 16/32/128 22.5-23.1 (same tokens), GEMV_GRF=256 21.17. "mtp
  MT_GRF=128" 51.53 runs the default setting, so MTP's run-to-run spread is about ±4 %. The profile: 56 % of 431
  GB/s; gate / up / down 59 % of the GPU time, 837 kernels and 4.4 ms of host time a token (the GPU stays busy: not
  the limit). The ceiling at full bandwidth is ~38 tok/s; ~30 is a realistic target for plain generation.
- **35B-A3B UD-IQ4_XS** (every expert cached): plain 78.77, **XALL=1 88.05** (same tokens), **GEMV_PF=0 85.15**, MTP
  113.73 (mtp7 109.54), BATCH 16/32/128 **49-51** (much slower: command lists do not suit the MoE layers' waits),
  XALL=1 BATCH=32 85.27, MOE_GR_GU / _DN 77.95-78.33. Made defaults: XALL on and GEMV_PF off for MoE models on
  discrete cards (their sum not measured yet).
- **27B prompt, `--prof`:** 512 tokens 669 tok/s, of which `h8_decode` 231 ms (31 %): the H8 weight decode costs the
  same at any prompt length. 4096 tokens 1,018 tok/s: the FFN GEMMs 47 %, `gdn_recur` 7.4 %, `h8_decode` 5.9 %,
  attention 5.0 %, `h8_act` 4.5 %.

**Towards 1,200 tok/s on the 27B (2026-11-08).** At 4K the H8 path with CG 4 takes ~4.2 s of GPU time; 1,200 tok/s
is 3.4 s.
- **Done: the weight rotation in the decode pass** (`h8r_decode`), bit-identical. The B60 spent 561 ms in
  decode_w8 + h8_weights per 4K prompt. On the laptop the fused kernel took a third of the two passes' time, so
  roughly 0.35 s might be saved on a B60 (an estimate).
- **Done: `ARC_GEMM_WPF`**, a weight prefetch switch to measure.
- **Not done: the activation rotation in the norm / SwiGLU kernels** (`h8_act`, 183 ms). The four kernels would need
  a sub-group per rotation block and a row-wide maximum, and the sums would come in another order (not
  bit-identical) for ~4 %.
- **Not done: the SwiGLU / norm quantization in the GEMM epilogue.** It needs each row's largest value over all
  columns, which are spread over many work-groups; the SwiGLU product itself is already fused into the up GEMM.
- **Not done: a wider XMX tile (32 x 64 per sub-group, as sycl-tla and urakozz).** The GEMMs run at roughly 80 TOPS
  of int8 (the B60's peak is around 197, from memory). This is the largest lever left, and a rewrite of the XMX
  kernel.

**The B60 sweep run before that (2026-11-07): `bench\b60_sweep.ps1`.** From each of the two best paths, the default (2D loads with
the prefetch, 764 tok/s) and H8 (918), it changes one setting at a time (TM, WGY, RG, CG, registers, split-K, 2D loads;
a winner must be 3 % ahead), puts the winners together (or keeps the single best change if that is faster), then measures default, H8 and both "best" sets at 512, 4K and
16K tokens (16K also in 4K chunks), checks their KL and perplexity against the default over 4 chunks of USAGE.md, and
with `-MoeModel` the 35B's `ARC_MOE_CG` / `ARC_MOE_WY` shapes and H8 there.

**iGPUs with XMX (2026-11-07): the same defaults as the discrete cards, none of it measured on one yet.**

| GPU | XMX GEMM in ocloc 2026.1 | 2D loads | Registers for the GEMM |
|---|---|---|---|
| Lunar Lake (140V), `lnl-m` | real `dpas.8x8` | yes | 128 |
| Panther Lake (B390 and others), `ptl-h` | real `dpas.8x8` in all 19 builds (W8 100-103, Q8_0, Q4_K, Q5_K, Q6_K, each with and without per-32 activations, H8R, the MoE tiles at 128 and 256 registers), no DP4a left, 16 2D loads, **no spills** (read 2026-11-07) | yes | 128 |
| Arrow Lake H (140T), `arl-h` | `gemm_x8.cl`: real `dpas.8x8 (8\|M0)` in all W8 and H8 layouts, no spills (read again 2026-11-07 with the 140T's options, 256 registers: scratch 0 in all ten). For the same 16 DPAS it has ~2,430 instructions against Panther Lake's ~1,890, each DPAS covering 8 columns instead of 16: about 2.5x the vector work per multiply. Native types stay on DP4a | no | 256 (the 16-lane form spilled with 128: its registers are 32 bytes) |

**The start-time speed check on iGPUs with XMX (2026-11-07).** After the bit-for-bit check, an iGPU with 16-lane XMX
times a GEMM of a model's size (512 tokens x 17,408 x 5,120, the 27B's gate: 89 MB of weights, past the caches), as the
engine launches it, 2D loads included. Best of 3 each after a warm-up; XMX is taken only if at least 3 % faster. DP4a is
timed as it would run without XMX: the GPU's own defaults, not the discrete cards' switches (128 registers made the 125H's
DP4a GEMM 4.7x slower in this check, 47 ms against 10). If XMX loses, those switches are dropped as well. The log line is
"XMX speed check (a 512 x 17408 x 5120 GEMM)". Discrete cards skip it (`ARC_XMX_SPEED=1` runs it there);
`ARC_XMX_FORCE=1` skips it. Tested on the 125H only with DP4a in the XMX slot (`ARC_XMX_TRY=2 ARC_XMX_SPEED=1`): DP4a
10.0-10.3 ms, the stand-in 15.6-15.7 ms, DP4a kept, the same greedy text. No iGPU with XMX has run it yet.
| Meteor Lake (125H) | the driver does not offer it | no | unchanged, 256: logits bit-identical |

- **Why a kernel of its own:** the extension says a device whose smallest sub-group is 8 lanes must call DPAS from an
  8-lane sub-group, with an `int8` operand a. `gemm.cl`'s 16-lane `short8` form is undefined there. On a 140T (issue #5,
  commit 515f416, driver 32.0.101.8864) its results differed from DP4a in every layout, and the engine stayed on DP4a. An
  earlier note here called the DPAS "emulated": that came from an ocloc test using the wrong form.
- **`gemm_x8.cl`:** 8 columns per sub-group, so a 16-column group is two sub-groups, with more loads per value. Tested on
  the 125H with DP4a in place of DPAS (`ARC_XMX_TRY=2`, `-DX8_EMU`): logits bit-identical to the DP4a kernel (4B, 27B, H8),
  the same KL against f32, the same greedy text over 2 and 4 simulated shards; a wrong kernel (`ARC_XMX_TRY=3`) is refused.
- **On real DPAS (140T, issue #5, package 911b6fa, 2026-11-07): right but slower.** The start-time check passed and the
  27B's logits were bit-identical to DP4a (`fc`), but its 2K prompt ran at 113-120 tok/s against 178-182 on DP4a: gate /
  up / down 3.54-3.70 s each against 2.25-2.30 s (1.6x). Generation is the same either way (5.9-6.0 tok/s: GEMVs). With
  H8 on top, 106-112 tok/s. The start-time speed check (a 64 x 64 x 1024 GEMM) had found XMX 17 % faster and taken it:
  that small shape stays in the caches, and with 8 columns per sub-group the real GEMMs read activations twice as often.
- **Since then XMX is off on 8-lane GPUs** (the GPU runs as with `ARC_NO_XMX=1`, which is the 180 tok/s above);
  `ARC_XMX_FORCE=1` takes it, with the bit-for-bit check. The speed check is gone.
- **Variants for the 140T's owner to measure (2026-11-07): `bench\igpu_sweep.ps1`.** `ARC_X8_W=1` (both halves in one
  sub-group), `ARC_GEMM_GRF=128`, and the tiles that already applied (`ARC_GEMM_TM` / `_CG` / `_WGY` / `_RG`), all with
  XMX forced. ocloc for `arl-h`, per DPAS:

  | Build | Instructions | Memory messages | Spills |
  |---|---|---|---|
  | as before (256 registers, 16 tokens) | 152 | 16.6 | none |
  | `ARC_X8_W=1` | 126 | 11.9 | none |
  | `ARC_X8_W=1`, 32 tokens | 116 | 9.7 | per-16 layouts (100, 103) spill |
  | 128 registers | 153 | 16.6 | none |
  | 128 registers with `ARC_X8_W` or 32 tokens | | | spill (except H8) |

  On the 125H with DP4a in place of DPAS (`ARC_XMX_TRY=2`): logits bit-identical to DP4a for X8W, X8W + 32 tokens and
  X8W + 128 registers (4B, 27B; a 512-token chunked prompt), X8W + H8 identical to H8, and a wrong kernel refused. The
  script also measures prompts of real use (512, 2048, 8192 tokens; 8192 in 1024 / 2048-token chunks) for DP4a, XMX and
  the best variant, and the GEMV's switches (`ARC_GEMV_GRF`, `ARC_GVROWS`, `ARC_GEMV_PF`) for generation.
- **The 140T's sweep (2026-11-07, package 6b8fb75, two runs): DP4a stays ahead.** 27B, 2K prompt: DP4a 172-183 tok/s;
  XMX 108-114; the best XMX variant, 128 registers, 162-168 (8-9 % behind DP4a at 512, 2K and 8K tokens alike), then
  32 tokens per sub-group 156-162. `ARC_X8_W=1` did not help (107), and with 128 registers it fell to 88. Column groups,
  work-group heights, row groups and H8: 101-120. So the XMX kernel is held back by how many threads it keeps in flight
  (128 registers: twice the threads) more than by its loads, which X8W cut. Chunk sizes (1K / 2K / 4K for an 8K prompt)
  change nothing. Generation: the GEMV switches within noise (second run 5.89-6.09 tok/s; the first run's low values
  were a stall in the first steps, its median steps 162-168 ms as the second's).
- **Generation on the 140T (2026-11-08):** the 27B makes 6.0 tok/s (~165 ms a token), reading 11.3 GB a token at
  ~85 % of its ~79 GB/s. The bandwidth bound is ~143 ms (~7 tok/s), so the GEMVs have at most ~15 % left.
  - **The lever is drafting:** MTP (the `-mtp` GGUF's own block) and DFlash2 read the weights once for several
    tokens. That took a B60 from 22 to 47 tok/s and the 125H from 4.9 to 18.1; not measured on a 140T yet.
  - **The GEMV switches now act wherever they are set** (`ARC_GEMV_GRF`, `ARC_GEMV_PF`, and `ARC_GEMV_MT_GRF=128`).
    Before, they needed the xe2 switches on, which the 140T no longer has with DP4a prompts, so they were ignored
    there.
  - `bench\decode_try.ps1` covers all of it, with `-DFlash <drafter folder>`.
- **The 140T's `igpu_grf_sweep` (2026-11-08, package 4aae0c5, two runs): XMX wins for the first time, with H8.**
  - **The best set:** XMX + `ARC_GEMM_GRF=128` + `ARC_H8=1` + `ARC_GEMM_TM=32` + `ARC_GEMM_CG=2`, 200.5-200.6 tok/s on
    the 27B's 2K prompt against DP4a's 173-182 (+10 %). KL against DP4a 0.00154 (max 0.064).
  - **Without H8**, XMX with 128 registers and CG 4 only matches DP4a (183).
  - **DP4a has no setting that gains.**
  - **f16 XMX is out on this GPU:** 42-75 tok/s.
  - **That package predates the fused H8 decode (685ff57).** The 140T's 27B weights are decoded just in time, as the
    laptop's, where fusing gave H8 +13 %.
  - **Not yet measured:** short prompts (on a B60, H8 lost at 512 tokens) and the newer package; both are asked for
    before any default changes.
- **f16 XMX at 8 lanes for the 140T (2026-11-08): `ARC_X16=1`, `gemm_x16.cl -DX16_SG8`.**
  - **Why:** the 140T's int8 XMX kernel lost to DP4a mostly in the vector work between DPAS (the per-32 rescaling). In
    f16 the weights are decoded with their scales, and one DPAS accumulator sums the whole row. Strata made the same
    choice for the A770, which is Xe1 with 8-lane XMX like the 140T.
  - **Form:** the extension's 8-lane form (a: `int8`, two halves per lane), two sub-groups per 16-column group, the
    grid of the 16-lane kernel.
  - **ocloc `arl-h`:** real `dpas.8x8` in all four W8 layouts at 8, 16 and 32 tokens per sub-group, and no spills at
    256 or 128 registers. Counted per multiply it has slightly more instructions than the int8 kernel (one f16 DPAS
    covers k = 16, the int8 one k = 32), so only a 140T can say whether it is faster.
  - **On the 125H, DPAS emulated (`ARC_X16_TRY=1`):** the start-time check passes with a largest error of 2.9e-4 of
    the result. KL against DP4a over 4 perplexity chunks: 4B 0.00087 (max 0.048), 27B 0.00069 (max 0.018); 27B
    perplexity 6.2847 -> 6.2797. Results are the same at every TM.
  - **Not done:** MoE experts (Flash Next) on it; they stay on DP4a per tile.
- **Next for the 140T (2026-11-08): `bench\igpu_grf_sweep.ps1`.** Four bases swept: XMX with 128 registers and DP4a
  at its 256, each without and with H8. Each setting (TM, WGY, RG, CG, split-K) is changed alone, the two best are
  crossed, and all gains are put together. Then the other H8 forms (`ARC_H8_SB`, `ARC_H8_D`) on the faster H8 base,
  and KL / perplexity of every best set against DP4a. `ARC_GEMM_GRF=128` and `ARC_GEMM_CG` now also act on the DP4a
  GEMM of iGPUs when asked for (bit-identical; 4B and 27B logits on the 125H, H8 too). DP4a with 128 registers is
  measured once only: ocloc for `arl-h` shows it spilling in every layout (750 scratch messages for the 27B's layout,
  none at 256), and on the 125H it ran 6x slower (27B 120 -> 19 tok/s, 4B 958 -> 160).
- The 140T before, all on DP4a: 27B pp2048 180 tok/s, 6.0 tok/s; with `ARC_H8=1` 160 tok/s, as the rotations (1.56 s)
  cost more than the GEMMs gain (~5 %).

## 9. DeltaNet in chunks and attention on XMX (QUEUE F33.1 / F33.2, 2026-11-09): `bench\f33_sweep.ps1`

The B60's 27B 4K prompt spent 290 ms in `gdn_recur` and 198 ms in `attention` (2026-11-09 profile).

- **`gdn_recur`** walks the tokens one by one, a work-item per state row: nv*128 work-items (6,144 for the 27B's 48
  heads), about 30 % of a B60's threads. **`gdn_chunk`** takes 32 tokens at a time: the chunk's tokens' deltas from
  one forward substitution over a 32 x 32 matrix, the outputs from the chunk's q.k matrix and the state at its start,
  the state moved once. Four work-items per state row: 4x the work-items. **`gdn_chunk_x`** holds the state as DPAS
  accumulators and does all of the chunk's products as f16 DPAS: k.k and q.k (an 8 x 16 tile per sub-group), state x
  k, state x q, the deltas through (I + A)^-1, the outputs and the state update, every operand split into high and
  low halves (~22 bits) after a power-of-two scale that keeps the low halves out of f16's subnormal range (a DPAS
  that flushes them would lose them): 192 DPAS per sub-group and chunk on a B60, no spills. **`gdn_chunk_x8`**: the
  same at 8 lanes for the 140T (264 DPAS). Only (I + A)^-1 (32 rows, one sub-group) stays on the vector units.
- **The vector form `gdn_chunk` is no gain on the laptop:** 4B, 4K prompt, `gdn_recur` 252-272 ms against
  `gdn_chunk` 2,409 ms (16-float reads) / 3,662 ms: its k.k / q.k products and the local-memory reads cost more than
  the extra work-items win back. It stays as `ARC_GDN_CHUNK=1` / `2` for a B60 to measure; `=3` without XMX (or
  with a failed check) runs `gdn_recur`, not it.
- **`attn_q8` on XMX:** its int8 q.k sums (8 DPAS per 16-key tile at 256 dims) and its u8 P x int8 V sums (16 DPAS,
  half of each zero: a tile has 16 keys, a DPAS takes 32) replace 2 x 256 DP4a per tile and head group; everything
  around them is attn_q8's, in the same order. DPAS sums integers exactly, as DP4a does: bit-identical. Not for the
  140T: the attention kernels run 16-lane sub-groups there, and its DPAS takes 8.
- **Checks without XMX (the laptop):** the chunked math in numpy (f32 1e-7 relative); every kernel on the CPU's
  OpenCL (oneAPI) against a double reference (`gdn_chunk` 5.5e-7, both XMX forms with DPAS emulated 4e-6, output);
  ocloc for Meteor Lake, Battlemage and Arrow Lake H: no spills. At start each XMX kernel is compared with its vector
  twin on random data (`gdn_chunk`: 1e-4; `attn_q8`: bit for bit) and not used if they differ.

## Not done yet (needs an Xe2 card to develop on)

- XMX for the 2-8 token verification passes of MTP / DFlash2 (now the GEMV-style kernels).
- Several models' worth of GPUs as replicas (one engine per GPU, requests spread over them) and pipeline
  parallelism for prompts.

## 10. Small kernels and matrices of few rows in a generation step (2026-11-10)

A Pro B60's 27B (issue #4, package 2c132a9, `decode_try.ps1`'s `--prof-gen`) against the bytes of each matrix role
in its GGUF (their sum is the 11,321 MB a token runz reports):

| Role | MB a token | ms a token | GB/s | % of 431 GB/s |
|---|---|---|---|---|
| `lm_head` (Q4_K) | 715 | 1.67 | 429 | 99 |
| `ffn_down` | 2,597 | 7.85 | 331 | 77 |
| `ffn_gate` / `ffn_up` (mostly IQ3_XXS, IQ3_S, IQ2) | 2,390 / 2,410 | 8.60 / 8.60 | 278 / 280 | 64 / 65 |
| DeltaNet qkv + z, `ssm_out`, `attn_q`, `attn_output` | 3,098 | 11.03 | 275-301 | 64-70 |
| `attn_k` / `attn_v` (1,024 rows) | 43 / 42 | 0.22 / 0.22 | 194 / 193 | 45 |
| DeltaNet alpha / beta (96 rows, Q8_0) | 25 | 0.42 | 60 | 14 |
| every GEMV | 11,321 | 38.60 | 293 | 68 |

- The GEMV reads at full bandwidth where decoding is cheap and the matrix tall (the Q4_K head): what holds the rest
  back is the codebook formats' decoding, and the matrices with few rows.
- The other kernels took 3.59 ms a token, and the GPU idled ~2.9 ms (a step 45.1 ms, its kernels 42.2).
  `swiglu_quant` took 15 us each (64 a token) and `rmsnorm_quant` 7 us (129 a token): one work-group per row, so one
  Xe core of 20 did each row.
- urakozz's engine (a B70, a W4A16 27B) reaches 78 % of its card's measured bandwidth by the same means: its GEMVs at
  89-97 % per shape, split-K for the matrices with few rows, and its small kernels spread over more sub-groups
  (its norm 16 -> 320 sub-groups, -2.2 ms a token; its alpha / beta GEMV over 128 sub-groups, -1.9 ms).

Since 2026-11-10, both bit-identical:
- **`ARC_WIDE_ROWS`** (every GPU): a 1-8 token pass's `swiglu_quant`, `gated_norm_quant` and `attn_gate_quant` run one
  sub-group per superblock, as Gemma's `swiglu_quant` already did; `rmsnorm_quant` one work-group per 16 superblocks,
  each summing the whole row's squares the same way. `=0` as before.
- **`ARC_GEMV_SMALL_WG`** (discrete cards): the single-token GEMV of a matrix with at most 512 sub-groups in
  work-groups of 64 instead of 256 (alpha / beta: 6 -> 24 work-groups). Off by default again since 2026-11-12 (see
  below); `=1` turns it on.

`decode_try.ps1` had rows for both off (`WIDE_ROWS=0`, `SMALL_WG=0`, `both off`); since 2026-11-12 it has
`WIDE_ROWS=0` and `SMALL_WG=1`, their tokens checked against plain.
**Measured on a B60 (issue #4, package 97d225b):** 27B plain 23.72 tok/s, WIDE_ROWS=0 23.19 (+2.3 % with it),
SMALL_WG=0 23.71 (no gain), both off 23.18; `swiglu_quant` 31.4 -> 8.8 ms per 32 steps, `rmsnorm_quant` 29.0 -> 25.1
(its work-group still sums the whole row: the next lever for it is a two-stage sum).
**Measured on the 35B (2026-11-12, package 18acafc):** plain 55.85 tok/s, SMALL_WG=0 80.35, both off 80.50,
GEMV_PF=1 79.12, all with the same tokens. So `ARC_GEMV_SMALL_WG` is off by default again; why the 64-item
work-groups cost ~5.5 ms a token on the MoE model only (its profile: 10.3 ms of GPU time in a 20.4 ms step) is not
known yet. Left: `attn_k` / `attn_v` (1,024 sub-groups, rows of 20 superblocks), `attn_prep` (23 us
each: its RoPE's `pow` / `cos` / `sin` in 4 lanes), and the codebook GEMVs themselves.

## 11. The H8 GEMMs with wider XMX tiles (QUEUE 33.4, 2026-11-11): `ARC_GEMM_XW=1`, `bench\xw_sweep.ps1`

**A Pro B60, 2026-11-12 (issue #4, package fffaf1a), 27B, tok/s at 512 / 4K / 16K tokens:** default 674.3 / 992.9 /
841.7; 32 x 64 585.2 / 944.6 / 717.3; 32 x 32 524.3 / 804.3 / 693.3; 16 x 64 505.5 / 750.5 / 780.4; 32 x 16 542.3 /
747.2 / 541.9 (all identical logits). The default (gemm.cl's XMX kernel with 2D block loads) won at every size, and
the wide tiles stayed off on discrete cards.

**The same B60, 2026-11-13 (package 2026.11.13.04):** default 691.1 / 1,066.5 / 978.9; 32 x 64 **716.8 / 1,291.7 /
1,169.5**; 32 x 32 739.5 / 1,267.5 / 1,147.6; 16 x 64 598.2 / 940.5 / 871.1; 32 x 16 636.9 / 955.9 / 884.1 (identical
logits). The kernels did not change between the runs (`gemm_xw.cl` and its 2D loads date from 2026-11-11). The
2026-11-12 run was on a slowed card: in that session the default's `--prof` passes fell from 486.6 to 421.3 tok/s and
the measured bandwidth was 316 GB/s (433 on 2026-11-13), and that run's default (992.9) sits below the card's other
runs of the day (1,068-1,081). So 32 x 64 is the default since 2026-11-13 wherever H8 is the default on a discrete
card; the 140T keeps gemm_xw8 32 x 16 (8-lane XMX, no 2D loads).

`gemm.cl`'s XMX kernel gives a sub-group 16 or 32 tokens x 16 columns: each 64-value chunk of a column group's
weights (1 KB) serves 16 / 32 tokens, and the activation rows are loaded again by every column group. `gemm_xw.cl`
gives it 32 tokens x 64 columns (16 DPAS accumulators, 128 registers): per chunk 2 KB of activations and 4 KB of
weights feed 32 DPAS, about 43 int8 operations per byte loaded against ~21. H8R only (one scale per column and per
token, a split's dot product one integer sum), so the epilogue is a single multiply and the results are gemm.cl's
bit for bit (checked at start).

| Tile (`_TM` x 16 `_NC`) | ocloc `bmg`, 256 registers | 128 registers | `ptl-h`, no register option |
|---|---|---|---|
| 32 x 64 (default) | no spills, 128 DPAS a superblock, 473 sends | 5.6 KB spilled | 192 registers, no spills |
| 32 x 32 | no spills | no spills | 128 |
| 16 x 64 | no spills | no spills | 128 |
| 16 x 32, 32 x 16 | no spills | no spills | 96 (10 threads per vector engine) |

**Beyond 32 x 64 (2026-11-13, for a sweep; `bench\xw_sweep.ps1 -Model <gguf> -Wide`).** The 32 x 64 tile's
accumulators already take 128 of the 256 registers, so the next tile with more area (32 x 128 or 64 x 64) needs all
256 for its accumulators alone. ocloc `bmg-g21`, 256 registers, with `-DX2D`:

| Tile | `ARC_GEMM_XW_TM` / `_NC` | Spills | Loads per DPAS against 32 x 64 |
|---|---|---|---|
| 32 x 64 (default) | 32 / 4 | none | 1 |
| 32 x 96 | 32 / 6 | ~2 bytes | 0.83 |
| 16 x 128 | 16 / 8 | none | 1.25 |
| 64 x 32 | 64 / 2 | none | 1.25 |
| 32 x 128 | 32 / 8 | heavy (~660 scratch lines in the assembly) | 0.75 |
| 64 x 64 | 64 / 4 | heavy (~590) | 0.75 |

32 x 96 is the only wider tile that both fits and loads less per DPAS. The 27B's widths (1024, 5120, 17408) are not
multiples of 96, so with `_NC=6` a GEMM runs its first 96-column blocks with the 96-wide kernel and the rest (a
multiple of 64) with the 64-wide one, from a column offset (`g_base`, `xw_piece`). The others in the table are there so
the sweep reaches the widest tiles that build. Every shape passes the start-up check bit for bit, and with DPAS
emulated on the laptop the 4B's logits are identical to gemm.cl's at every shape; the speed needs a B60.

The 140T's `gemm_xw8` (8 lanes, ocloc `arl-h`, 256 registers) takes 64 tokens and 64 columns too: 16 x 64 does not
spill; 64 x 16 spills plainly but not with `XW_PACKA` + `XW_PACKW` (`bin\xw_peak.exe`'s packed variants); 32 x 64 and
64 x 32 spill. `xw_sweep.ps1 -Wide` adds 16 x 64, 64 x 16 and 32 x 64 to its 140T rows. `xw_peak` lists them all with
the packed forms (one minute: `bin\xw_peak.exe`).

**To measure (a B60):** `.\bench\xw_sweep.ps1 -Model <27B gguf>`: the prompt at 512 / 4K / 16K tokens for the
default and four tile shapes, each shape's logits against the default's (identical expected), and a `--prof` of the
default and the fastest shape at 4K. Left after that: the weights double-buffered in registers, the W8 and MoE
expert layouts, and an 8-lane form for the Arc 140T.

## 12. The Arc 140T (Xe-LPG Plus): what its XMX is, and where the prompt GEMM loses (2026-11-11)

The 140T (Arrow Lake H, Core Ultra 9 285H) has an improved first-generation Xe GPU with XMX added, the Xe-HPG
(Alchemist) kind: per Xe-core 16 vector engines and 16 matrix engines of 1024 bits, 4,096 int8 or 2,048 f16
operations a clock (Intel's oneAPI GPU optimization guide). 8 cores at up to 2.35 GHz: ~77 int8 TOPS, the figure
Intel gives. Its DPAS takes 8-lane sub-groups (8 x 8 x 32 int8 per instruction, half of a Xe2 DPAS).

- **Where the time goes:** the 27B's prompt on the 140T preset (XMX + H8, 221-231 tok/s) is ~12 int8 TOPS, ~15 % of
  that peak. ocloc `arl-h` for `gemm_x8.cl` H8R: ~110 instructions and ~13 sends per DPAS (preset: 3,517 / 425 / 32
  DPAS; `X8W`: 100 and 11), against `gemm_xw.cl` on a B60 (`bmg`): ~60 instructions and 3.7 sends per DPAS of twice
  the work. The weights go in with one 4-byte read per lane per word (gemm.cl's 16-lane layout read by 8 lanes: 16
  reads per column half and 64 values), the activations with two 32-byte reads per row.
- **Built 2026-11-11 (QUEUE 33.4b, `ARC_GEMM_XW=1`, `gemm_xw8.cl`; 16 x 16 at 128 registers: 90 instructions and 6.3 sends per DPAS):** an 8-lane form of the wide tile for H8R: the weights of both column halves by
  `block_read8` (4 words of both halves a read: 4 reads instead of 32), each activation row by one `block_read2`, and
  more column halves per sub-group. The same values into the same registers: bit-identical. Checked with ocloc
  `arl-h` (spills, sends per DPAS) and on the laptop with DPAS emulated (`ARC_XMX_TRY=2`), its speed on a 140T.
- **f16 DPAS precision (open):** oneDNN says f16 work on GPUs older than Xe-HPC "may accumulate to f16". If the
  140T's f16 DPAS keeps fewer bits inside than f32, that would explain the chunked DeltaNet on XMX missing its
  1e-4 check there (its f16 operands are split into hi + lo halves) with a right layout; the 8-lane form passes on
  the laptop with DPAS emulated (4.45e-07). `f33_sweep` now prints the size of the difference.
- **Toward ~80 % of the XMX peak (2026-11-12, issue #5): `bin\xw_peak.exe`.** With 32 x 16 tiles the 27B's FFN
  GEMMs run at ~23 int8 TOPS (~30 % of the owner's measured 76.75), the whole 4K prompt ~16 TOPS. A 64-value step
  of a sub-group does 16 DPAS and 36 reads (32 activation rows one at a time, 4 weight reads), ~190 bytes per DPAS;
  a 140T Xe-core runs about one DPAS a clock. `xw_peak` times the GEMM alone at the FFN shapes against the peak it
  measures, for the default and variants of `gemm_xw8.cl` (the same bits required):

  | Variant (ocloc `arl-h`, 256 registers) | Spill | Loop instructions per DPAS | Register moves in the loop | Loads per DPAS |
  |---|---|---|---|---|
  | default 32 x 16 | 0 | 11.8 | 399 | ~2.6 |
  | `XW_PACKA` 32 x 16 (activations in the DPAS order) | 0 | 8.8 | 180 | ~1.6 |
  | `XW_PACKW` 32 x 16 (weights in the DPAS order) | 0 | 8.8 | 194 | ~2.5 |
  | both, 32 x 16 | 0 | 5.7 | 0 | ~1.6 |
  | both, 32 x 32 | 0 | 4.1 | 0 | ~1.0 |
  | both, 16 x 32 | 0 | 4.9 | 0 | ~1.6 |
  | plain 32 x 32 (the sweep's slower shape) | 0 | 9.5 | 649 | ~1.5 |

  (Loads: a `block_read8` of 8 lanes, 256 bytes, is two messages of 128.) Not used by the engine yet: the ones that
  win on a 140T go in with the H8 activation kernel and the weight decode writing the packed layouts. The bench also
  prints what the f16 DPAS keeps (products, the 16-term sum, the accumulator), to choose the DeltaNet fix: f32 adds
  of DPAS results from zero if only the accumulator rounds, int8 DPAS on 3-part values if the products do.
