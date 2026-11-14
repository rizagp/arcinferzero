# Flash Next's precise mode: source values, F32 DeltaNet gates, QSA in float attention (2026-11-13)

Pull request #21 by chumbleyjl, submitted as a draft on 2026-11-13 against main at `c655269`, merged 2026-11-14. It
is the correctness part of the repair of precise mode (`--precise`, `ARC_FN_PRECISE=1`) in his own B70 build
("downstream").

## What it changes

- **Mixer values kept as the source has them** (zero/engine_load.h `hc_exact_weight`, kernels/ops_flashnext.cl
  `-DHC_F32`): precise mode stored the hyper-connection mixers' dequantized rows as BF16, but a Q8 source's
  dequantized values need not fit BF16. They are now kept in F32 unless every value is exactly a BF16 value; then
  the compact BF16 path stays.
- **DeltaNet's alpha / beta projections in F32** (precise mode on one GPU; zero/engine_load.h, engine_qwen.h
  `gdn_ab_input`): computed from the f32 normalized input and left out of the Q8 qkv / z stack.
- **Float attention with the QSA masks** (kernels/ops_attn.cl `attn_fa` with `FA_QSA`, `qsa_fa_combine`): precise
  mode's attention runs in float over the QSA blocks chosen and the uncompressed tail. Tiles excluded whole are
  skipped, and an empty partial accumulator stays finite. Short decode and verification calls are split over at most
  64 key ranges, whose numerators and denominators are merged stably.

**Kept as they were:**

- main's H8 and logging changes;
- dense models: the batched float-input change applies only to Flash Next;
- `--attn-f32` in Flash Next's default mode: still refused (`f32 QSA attention requires Flash Next precise mode`), so
  the pull request does not open that perplexity configuration, which was never supported.

**Not part of it**, and not needed for it to build: his build's native mixer compression, expert XMX, the private
AutoRound types (pull request #22 since, [autoround.md](autoround.md)), virtual KV and local API / Pi changes.

## The author's checks at submission

- **Builds:** native Linux runz and serverz from the branch; tools/ref_forward with `-D_fseeki64=fseeko` (for its
  seek spelling).
- **A masked-attention fixture** (`tests/test_qsa_float.cpp`) builds against the branch. It needs an Intel GPU, so it
  cannot run on the model-free GitHub Windows CI runner. It covers attention without a mask, leading tiles excluded,
  sparse blocks, incomplete tails, several query rows, the split and unsplit paths and addresses near 262K.
- **Not run:** GPU runs and model KL on the rebased code (his B70 Pi service was left running); the Windows, iGPU,
  split and batched-slot regressions. He kept the pull request a draft until those were done, with new checks to be
  recorded as new rather than earlier results labelled again.
- **Q5_1:** a reference with Q5_1 tensors needs the CPU reference fix of pull request #20, which this one does not
  copy.

## Measured on his B70 build before the pull request

Not results of the rebased branch. The model: **UD-Q4_K_XL with Q5_1 expert-down tensors**, with its Q4_0 MTP head;
these are not AutoRound measurements (those are in [autoround.md](autoround.md), with another expert quant). The f32
reference decodes the same quantized weights, not the BF16 ones.

| Prompt | Before | After |
|---|---:|---:|
| Eight fixed ids, with the corrected Q5_1 CPU reference: mean KL | 0.11597 | <0.000005 |
| 209-position workflow, DeltaNet projections in F32: prefill mean KL | 0.07673 | 0.00324 |
| The same workflow, sequential decode: mean KL | 0.04018 | 0.00864 |
| Tool conversation, forced top-8 sparse attention: mean KL | not measured as a pair | 0.00021 |

- **The workflow's maxima:** 0.19503 prefill, 0.32222 decode. A mean within 0.015 does not hold for every position.
- **The masked float-attention fixture** passed 74 cases there: a result of his build, not a run of this branch.

## Checked here before the merge (laptop, 2026-11-14)

ISTA-DASLab's Flash Next GSQ-RCO IQ3_S against tools/ref_forward, release 2026.11.14.07 -> merged: precise KL on 160
code tokens 0.00455 / 0.02056 -> 0.00173 / 0.00221 (prompt / generation); the default mode and the other models
bit-identical to .07. The rest (maxima, the split, speed, memory, long prompts): [before-after.md](before-after.md),
"2026-11-14, Flash Next precise mode repaired".
