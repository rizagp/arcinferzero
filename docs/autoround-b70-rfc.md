# RFC: AutoRound INT4 routed-expert hybrid on a B70

The benchmark in this report uses an **AutoRound quantization**: Intel's
calibrated symmetric INT4/group-128 routed experts, with all non-expert tensors,
tokenizer and chat metadata retained from UD-Q4_K_XL. The separate Q4_0 MTP head
is unchanged. This is a hybrid, not the complete Intel checkpoint and not stock
UD-Q4_K_XL running faster solely because of engine changes.

The control's down projections are Q5_1; the candidate changes them to calibrated
INT4. The smaller aligned expert set fits the 64 GiB host tier. Those quant and
residency changes are part of the comparison, not hidden implementation details.

## Proposed code and upstream status

Checked against upstream main `c6552692def54da2cd854c6b4a6f5fb4ebe6ab3b`:
the private type, importer and native decode cases are absent. This draft adds
the isolated importer, package writer, CPU/reference decoding and generic native
expert decode/GEMV plumbing. It leaves all existing model defaults unchanged.
It requires precise Flash Next (`--precise`), rejecting the default mode; it runs on discrete GPUs (one, or a split over several) and on iGPUs (Xe-LPG, Xe-LPG+, Xe3), through the same expert kernels.

**This is a format/runtime prototype, not a ready-to-merge performance PR.**
The downstream measured profile also includes source-preserving mixer storage,
two-pass expert XMX, growing KV, adaptive tiles/MTP and expert residency. Those
optimizations are not all included here. Do not attribute the measured rates
below to this branch or to unmodified upstream. Precision work is proposed in
PR #21; Q5_1 CPU reference decoding is proposed separately in PR #20.

Feedback requested: private type allocation, package versioning/compatibility,
whether this native calibration-preserving import belongs upstream, and the
preferred order for integrating and validating the remaining optimized paths.

## Source and representation

Pinned source: [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound),
revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`.
Import only the 14 shards containing target routed experts, 72,296,476,000 bytes.
The original BF16 PLE table and source MTP tensors are not imported.

The importer verifies checkpoint identity, source SHA-256, shapes/dtypes,
constant symmetric zero points, finite F16 scales, and every source code and
scale bit after repacking. Stored GPTQ zero-point-minus-one 7 means zero point 8.
It refuses activation-order indices and existing output files. It streams through
bounded working arrays rather than loading the entire checkpoint at once.

`AR4_128` (`0x41523447`) is a **private inferzero extension**, not a standard GGML
quant identifier. CPU blocks contain one source F16 scale and 128 original
four-bit codes in Q4_0 nibble order. A GPU superblock groups two scales, 256
codes and four alignment bytes: 136 bytes total. K=640 down rows pad to 768.
No extra requantization or expert pruning is performed.

The aligned expert file is 68,451,045,376 bytes (63.75 GiB plus a 4 KiB header),
with 2,785,280-byte expert records. The retained non-expert package is 31.960 GiB.
The manifest writer copies all 1,080 non-expert tensors, metadata and tokenizer.
No model weights or expert caches are distributed in this PR.

Required local inputs are config.json, model.safetensors.index.json,
model-info.json from Hugging Face's model API with SHA-256 sibling metadata,
and the referenced expert shards at the pinned revision. NumPy is required.

```sh
python tools/import_autoround_experts.py SOURCE_DIRECTORY AutoRound.experts --experimental-format
g++ -std=c++20 -O2 -mavx2 -mfma -mf16c -Ithird_party tools/import_autoround_pack.cpp -o /tmp/import_autoround_pack
/tmp/import_autoround_pack SOURCE_Q4XL.gguf AutoRound.gguf AutoRound.experts 4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
```

Use the matching experimental runtime; stock upstream cannot load this private
format. The explicit importer flag acknowledges that compatibility requirement.

## Downstream benchmark: AutoRound quant hybrid versus Q4XL

Hardware: one Arc Pro B70 32 GB, i7-14700K, 96 GB system RAM, native Linux
Level Zero/OpenCL engine. Both profiles use precise mode, medium reasoning,
262,144 context capacity, 64 GiB host expert tier and adaptive MTP capacity four.
One server sequence per quant, after downloads/import/CPU-reference jobs finish;
these are individual measurements, not confidence intervals. Archive requests
use temperature 0; tool requests use temperature 1, top-p 0.95, top-k 20, seed 42.
Generated histories can differ across quants.

PP is new prompt tokens divided by forward time; cached tokens are excluded.
TG excludes the first token. The archive contains 15,599 tokens.

| Workload | UD-Q4_K_XL control PP / TG | AutoRound INT4 expert hybrid PP / TG |
|---|---:|---:|
| First-use archive | 484.48 / 15.43 | 300.79 / 36.02 |
| Warm archive replay | 695.63 / 25.78 | 783.59 / 42.36 |
| Weighted incremental turns | 102.41 / 24.53 | 194.15 / 38.53 |
| Plain archive decode | — / 23.31 | — / 30.79 |
| Tool call | 89.92 / 22.58 | 174.68 / 35.69 |
| Tool result | 110.71 / 24.44 | 240.65 / 37.80 |

Mean incremental wall time is 3.461 -> 2.137 s (38% lower). Expert records shrink
77.60 -> 63.75 GiB. First-use PP regresses; cold data and JIT costs were not
separately isolated. Warm gains are not universal startup/TTFT gains. Both
quants pass all eight archive/replay/tool cases and within-model plain/MTP checks.

## Accuracy and behavior limits

F32 reference values decode each model's own quantized weights, not original
BF16 weights. Means below CLI resolution are reported as <0.000005.

| Fixture | Q4XL engine mean KL vs own F32 | AutoRound hybrid engine mean KL vs own F32 | AutoRound maximum |
|---|---:|---:|---:|
| Coding, 31 positions | <0.000005 | <0.000005 | <0.000005 |
| Workflow, 209 positions | 0.00700 | 0.01074 | 0.83681 |
| Tools, 48 positions | 0.00001 | <0.000005 | <0.000005 |
| Forced top-8 sparse attention, 48 positions | 0.00021 | 0.00018 | 0.00419 |

All four candidate mean checks pass 0.015, but individual-position outliers
remain (workflow control maximum 0.58798). F32-to-F32 cross-quant workflow KL
is 0.149171, top-1 agreement 206/209. Distribution differences do not establish
which quant is closer to BF16, or lossless capability relative to Q4XL.

Historical downstream import checks cover all 120,795,955,200 expert codes/scale
bits; a real 4,915,200-weight source/CPU/GPU fixture matches bit for bit. The
small coding/tool suite passes 17/17 functional checks. These are sanity checks,
not a broad coding benchmark.

An actual **AutoRound hybrid** 260,988-token request retrieves early/midpoint keys
correctly at 532.52 PP / 25.70 TG, complete wall time 497.49 s. Its cached follow-up
reuses 260,988 tokens, processes 214 new tokens and answers correctly in 3.885 s
(154.39 PP / 28.81 TG). Full 262,144 physical KV backing works, with minimum
25.77 GiB system RAM available. This candidate-only test has no paired Q4XL
timing or full-window KL measurement.

Actual Pi with the AutoRound hybrid, medium reasoning and temperature 1 passes
72 tool calls, 11 automatic compactions and 24/24 exact final audit results in
17.3 minutes. No tool errors, extra inspections or guard stops; all original
instructions remain in the 31 main provider requests. Weighted main rates are
737.51 PP / 44.36 TG. The isolated diagnostic compacts around 24k tokens (largest
prompt 23,218), retaining full advertised capacity. It is a synthetic read-only
audit, not 11 compactions at 262k depth or proof of long coding-task parity.

## Validation gates for this draft

Current-main merged builds and CPU/package checks are recorded in the PR body.
Both native Linux engine binaries, the reference, package writer and independent
test executables build. The CPU layout fixture passes 9,216 values across five
widths, including K=640, with independent code/scale expectations, padding and
invalid-width checks. The actual candidate package check matches all 1,080
non-expert tensors (34,305,722,880 bytes), 144 virtual expert descriptors,
metadata and tokenizer against the original Q4XL. No new model file is generated
or overwritten for that check. The importer also requires explicit acknowledgement
of its experimental format; its packing algorithm matches the validated local
importer, with that CLI guard added.
Downstream GPU checks above are historical, not execution of this upstream
prototype. Rebased model KL, native kernel execution, Windows/iGPU/TP checks,
default-off regression dumps and full-context integration remain gates before
merging. The running downstream Pi service is unaffected by this submission.
