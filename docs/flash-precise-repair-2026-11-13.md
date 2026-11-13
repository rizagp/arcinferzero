# Flash Next precise source values and QSA attention

Rebased onto main `c6552692def54da2cd854c6b4a6f5fb4ebe6ab3b`.
This draft contains the correctness part of the downstream B70 repair:

- Preserve dequantized mixer values in F32 unless every value is exactly
  representable in BF16. A Q8 source does not imply its dequantized values fit
  BF16. Keep the existing compact path when it is exact.
- Preserve precise single-GPU DeltaNet alpha/beta projections in F32, evaluate
  them from the F32 normalized input, and exclude them from the Q8 qkv/z stack.
- Use float attention for Flash precise mode, applying the selected QSA blocks
  and the uncompressed tail. Skip wholly excluded tiles and keep empty partial
  accumulators finite. Split short decode/verification calls across at most
  64 key ranges and merge numerators/denominators stably.

Current main's H8 and logging changes are preserved. The batched float-input
change is restricted to Flash Next; dense-model behavior is retained. Normal
non-precise Flash --attn-f32 remains refused, so this PR no longer enables
that previously unsupported perplexity configuration. Native mixer compression,
expert XMX, private AutoRound types, virtual KV and local API/Pi changes are
separate work, not prerequisites for this source patch to compile.

## Checks

Both native Linux runz and serverz build from this branch. The CPU reference
builds with `-D_fseeki64=fseeko` for its existing seek spelling. The independent
masked-attention fixture builds against this branch. It requires an Intel GPU
and is not suitable for the model-free GitHub Windows CI runner. Its coverage includes
unmasked attention, excluded leading tiles, sparse blocks, incomplete tails,
multiple query rows, split/unsplit paths and near-262k addresses.

At initial submission the production B70 Pi service was left running, so GPU
execution and model KL on the rebased source were not run. Review follow-up
will record new checks explicitly rather than relabeling historical evidence. Windows, iGPU, tensor-parallel and
batched-slot regressions remain pending. Keep this PR a draft until those gates
are met. The Q5_1 reference fix is separately required for references containing
that quant; this branch does not silently duplicate it.

## Historical downstream evidence, not rebased-main results

The model here is **UD-Q4_K_XL with Q5_1 expert-down tensors**, with its existing
Q4_0 MTP head. These precision measurements are not AutoRound benchmarks.
The F32 reference decodes the same quantized weights, not original BF16 weights.

| Fixture | Before | After |
|---|---:|---:|
| Eight fixed IDs, corrected Q5_1 CPU reference, mean KL | 0.11597 | <0.000005 |
| 209-position workflow, before/after F32 DeltaNet projections, prefill mean KL | 0.07673 | 0.00324 |
| Same workflow, sequential decode mean KL | 0.04018 | 0.00864 |
| Tool conversation, forced top-8 sparse attention mean KL | Not a paired measurement | 0.00021 |

The workflow maxima are 0.19503 prefill and 0.32222 decode. Mean KL passing
0.015 is not a guarantee for every position. The downstream independent masked
float-attention fixture passed 74 cases; that historical result must not be
presented as execution of this rebased branch. The separate AutoRound report
explicitly identifies its changed expert quant and different measurements.
