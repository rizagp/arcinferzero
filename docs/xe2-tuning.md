# Tuning on Arc B-series GPUs (Xe2: B580, Pro B60, Pro B70)

All the switches (environment variables; each is read once per process):

| Switch | Default | What it does | Section |
|---|---|---|---|
| `ARC_GEMM_GRF=256` | 128 on discrete cards | the dense prompt GEMM's register file | 1 |
| `ARC_MOE_GRF=128` | 256 | the MoE expert GEMMs' register file | 1 |
| `ARC_GEMM_TM` / `_RG` / `_WGY` / `_SPLIT` / `_CG` | 16 / 4 / 8 / 17 / 1 | the int8 prompt GEMM's tiles | 1 |
| `ARC_NO_XMX=1` | XMX where the GPU has it | DP4a only | 1 |
| `ARC_X16=1` | off | the f16 XMX GEMM for W8 weights | 6 |
| `ARC_XMX_FORCE=1` | off | on a GPU with 8-lane sub-groups (Arrow Lake H): XMX even if the start-time speed check finds it no faster | 8 |
| `ARC_GEMM_2D=0` / `ARC_GEMM_2D_PF=0` | on, where the driver offers them with XMX | the XMX prompt GEMM's activations by 2D block loads (one load for 8 token rows x 64 bytes), with a prefetch of the next 64 bytes; these switches turn them off | 8 |
| `ARC_H8=1` | off | the H8 GEMM (QUEUE 31.12): weights and activations Hadamard-rotated, int8 with one scale per column and per token; DPAS on XMX, DP4a else; iGPU too. Qwen models rotate the activations from f32 (`ARC_H8_INT=1`: from int8). `ARC_H8_SB=1`: one scale per 256 values, as the B60 ran it on 2026-11-06. `ARC_H8_D=1`: with random signs. Slower than the default on the laptop; this form not yet run on a B60 | 8 |
| `ARC_X16_TM` / `_GRF` / `_CG` | 32 / 256 / 1 | its tiles and register file | 6 |
| `ARC_GEMV_GRF=256` | 128 | the single-token GEMV's register file | 2 |
| `ARC_GEMV_MT_GRF=256` | 128 on discrete cards | the 2-8 token GEMV's register file (MTP, DFlash2, batches) | 7 |
| `ARC_GVROWS=1/2/4` | by matrix size | rows per sub-group of the codebook GEMV | 2 |
| `ARC_GEMV_PF=1` | off | the single-token GEMV issues all its rows' reads of a step first (bit-identical; the laptop iGPU, forced: 8 % slower, 5.23 -> 4.82 tok/s on the 27B; a B60, 2026-11-06, best of 2: 27B 22.39 -> 23.11, 35B 84.08 -> 83.88) | 7 |
| `ARC_LAUNCH_STATS=1` | off | kernel launches and host time per token | 5 |
| `ARC_TP_*` | | the multi-GPU sums and split (`ARC_TP_SUM`, `ARC_TP_RS_MIN`, `ARC_TP_RG_GU`, `ARC_TP_TIMEOUT`, `ARC_TP_ATT_QSPLIT`) | [multi-gpu.md](multi-gpu.md) |

The tools: `bench\gemm_sweep.ps1` (prompt), `bench\decode_try.ps1` (generation), `bin\mgpu_probe.exe` and
`bin\event_probe.exe` (several GPUs), `bin\replay_probe.exe` (kernel launches), `bin\membw_probe.exe` (memory read
bandwidth of the CPU, the GPU and both; docs/speed-progress.md), `bin\clinfo.exe` (what the driver
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
**A B60, 2026-11-06 (4,096 floats):** 700 kernels 1.790 ms appended, 0.459 ms replayed (2.56 -> 0.66 us a kernel). The
35B takes 632 launches and 2.89 ms of host time per 12.0 ms token (4.6 us a launch). So a replay could save 1.2-2.5 ms
of a token, ~10-20 %, if the host's time is on the token's critical path and a step can be replayed at all: the
experts change every token. Not measured in the engine, not built.
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
  says "with 2D block loads" when they pass. **Not yet checked:** logits on a real model bit for bit on a B60, and any
  run of them over several GPUs. Before a release:

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

**What a B60 should run next** (none of this form has run on an XMX GPU; the kernels build for Battlemage with ocloc,
and a failure of the start check now only leaves H8 out of the XMX GEMMs, with the rest of XMX kept for runs without
`ARC_H8`):

```powershell
$env:ARC_H8=1; .\bin\runz.exe <27B gguf> --bench-pp 4096 --n 16 --prof
$env:ARC_GEMM_TM=32; .\bin\runz.exe <27B gguf> --bench-pp 4096 --n 16 --prof; Remove-Item Env:ARC_GEMM_TM
$env:ARC_GEMM_GRF=256; .\bin\runz.exe <27B gguf> --bench-pp 4096 --n 16 --prof; Remove-Item Env:ARC_GEMM_GRF
.\bin\runz.exe <27B gguf> --perplexity USAGE.md --ppl-chunks 4 --ppl-kl base.kl; Remove-Item Env:ARC_H8
```

If the GEMMs reach far more than the 1.33x, the next steps are the weights in 2D loads and urakozz's wider tile.

**iGPUs with XMX (2026-11-07): the same defaults as the discrete cards, none of it measured on one yet.**

| GPU | XMX GEMM in ocloc 2026.1 | 2D loads | Registers for the GEMM |
|---|---|---|---|
| Lunar Lake (140V), `lnl-m` | real `dpas.8x8` | yes | 128 |
| Panther Lake, `ptl-h` | builds; its code not read | yes | 128 |
| Arrow Lake H (140T), `arl-h` | builds, but **no `dpas` in the code** (`mul`, `mad`), at 16 and at 8 lanes: it looks emulated | no | 256 (128 spills: its registers are 32 bytes) |
| Meteor Lake (125H) | the driver does not offer it | no | unchanged, 256: logits bit-identical |

- Whether Arrow Lake's driver gives real DPAS is not known here: ocloc may compile for a device without it. So on a
  GPU with 8-lane sub-groups the start compares a stored W8 GEMM on the check's small data (best of 6, 16 launches
  each), and XMX is taken only at least 5 % faster. The log says "XMX speed check". It measures the kernels' own speed,
  not memory limits, so the prompt's real gain can be smaller.
- The 140T's numbers so far, 180 tok/s (pp2048) and 6.0 tok/s, are without XMX: `ClRt::xmx` refused every GPU with
  8-lane sub-groups until now. Its `ARC_H8=1` runs came from an older package (f9e62e6, before H8): no "H8 on" line, no
  `h8_*` kernels.

## Not done yet (needs an Xe2 card to develop on)

- XMX for the 2-8 token verification passes of MTP / DFlash2 (now the GEMV-style kernels).
- Several models' worth of GPUs as replicas (one engine per GPU, requests spread over them) and pipeline
  parallelism for prompts.
