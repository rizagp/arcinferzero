# Kernel audit with ocloc (2026-10-27)

The GPU assembly of the kernels the engine runs, read for work that is not needed. The binaries are the driver's own,
taken from the kernel cache (`%USERPROFILE%\.arcinferzero\kernels`, file = FNV-1a of source + options + device, as
`zero/clrt.h` names it; `ARC_KLIST=1` prints the file and options of every kernel built) and disassembled with
`ocloc disasm`; variants were compiled with ocloc 2026.1 (a newer compiler than the driver's). Nothing measured on
the GPU yet: every gain below is an estimate until measured, and every fix has to keep the logits bit-identical
unless it says otherwise. Ranked by plausible real gain; ALU-only savings in memory-bound loops come last (on this GPU
they made decode loops slower three times: [lessons](lessons-intel-arc-igpu.md)).

## Outcome (2026-10-27)

Measured on the GPU: #6, #7 (HS <= 3) and #9 kept, all bit-identical: 35B prompt +2 %, 27B prompt attention
-9 %, ~1 ms less per 35B token in the quantization kernels; #1 not a regression (the small MoE tiles gave the 35B's
MTP +3.5 %); #2, #3, #5, #8, #10 no faster or slower, not kept; #4 not worth doing. What the assembly showed as
waits or wasted reads mostly hit the caches: only the changes that removed real memory traffic (#9) or gave a
compute-bound loop more independent chains (#6, #7) paid.

## To fix or measure first

| # | Kernel | What is not needed | Fix (bit-identical) | Estimate |
|---|---|---|---|---|
| 1 | `moe_down_q` TM 8, GROWS 4 (2026-10-27's small tiles) | spills 768-832 B per thread, 16 / 8 scratch messages per loop pass; hit by 6-8 rows: the 35B's MTP verification (5 drafts), DFlash2 (7 drafts), batches of 5-8 | **measured, kept as it is**: 35B 5 / 6 slots moe_down 16.8-17.1 / 18.1-18.4 ms per step with GROWS 4, 18.4 / 20.2 with GROWS 2 (no spill), 23.3 / 25.5 with the old TM 16; 35B MTP (5 drafts, 256 tokens) 26.8-26.9 -> 27.8 tok/s against TM 16, logits bit-identical | done: not a regression |
| 2 | `hc_up1` (Flash Next) | weight loads one at a time: HC and KP / 256 are run-time arguments, so the loops do not unroll; ~12 memory round trips in a row per sub-group; 37 GB/s of ~75 | **measured, not kept**: E / HC / superblocks as defines and every weight loaded at entry before the barrier (no spill at one row, 1,012 instructions): 9.2 -> 9.6 ms per token; at 2 rows 76 scratch messages, 9.7 -> 16.6 ms | not the bottleneck: the serial loads were not what limits it |
| 3 | `hc_down1` (Flash Next) | ~20 waited loads and ~10 barriers before the weights stream; every x-group recomputes the same norm | **measured, not kept**: the same (E as a define, the 5 superblocks prefetched before the norm; 659 instructions, no spill): 7.0 -> 7.8 ms per token | left: the norm computed once instead of in all 21 x-groups (untried) |
| 4 | Flash Next shared expert | F 640 padded to 768: 128 zero rows in gate / up, 128 zero columns in down read every token (~38 MB) | **not done**: the stacked `rgu` GEMV is off because Flash Next's shared gate / up are IQ4_XS (it stacks Q8_0 only), not because of the padding; what is left is ~0.9 ms of ~100 per token (< 1 %, and Flash Next generation waits on the SSD), and rows 640-767 of the working buffer would have to stay zero for the down projection | not worth it |
| 5 | `moe_down_q` TM 2-8 | each slot's activations and scales loaded again for each of the 4 rows (TM 8: 104 loads, 24 distinct) | **measured, not kept**: token-outer (`-DDN_TO`: the GROWS rows decoded first, each slot loaded once): 35B moe_down 2 slots 7.8-8.1 -> 9.2 ms, 5 slots 16.6-17.9 -> 19.3 ms (GROWS 2: 8.8 / 18.3) | the reloads hit L1; more registers live cost more |
| 6 | `moe_rg` (prompt MoE, 35B 320-1279-token chunks) | the scale multiply `c0 * (short)a0` is an emulated 32-bit multiply (mul + mach) | **done**: the multiply in float (exact: abs(c0) < 2^20; ocloc IQ3_S DP4a without a wait 288 -> 764 of 1024). 35B, 11,598-token prompt in 1024-token chunks: moe_gateup (IQ3_S) 7.58-7.62 -> 7.25-7.30 s, moe_down (IQ4_XS) unchanged, 523-525 -> 532-537 tok/s; logits bit-identical; `ARC_RG_IMUL=1` the int form | +2 % prompt (35B) |
| 7 | `attn_q8` QK (prompt attention) | HS dependent DP4a chains, 64 long each (35B: 320 of 512 DP4a wait) | **done for HS <= 3**: two integer sums per head (even / odd words), added before the conversion (exact). 27B (HS 3), 11,598-token prompt: attention 5.09-5.14 -> 4.52-4.72 s (-9 %); 35B (HS 4) 1.88 -> 1.90 s and 9B 1.52 -> 1.52 s: no gain, HS >= 4 keeps one chain; logits bit-identical on all three | 27B long prompts |
| 8 | GEMV scales at T = 1 | `asc[b * M + m]` with M = 16: each lane's scale gather touches 16 cache lines for 64 useful bytes | **measured, not kept**: one token with Mp = 1 (every producer and consumer then uses M = 1; bit-identical on 9B / 27B / 35B): wall time within noise, GPU time 27B 14.7-15.0 -> 15.1-15.7 s per 64 tokens, 9B mixed | the gathers hit L1 |
| 9 | quantization kernels (`swiglu_quant`, `gated_norm_quant`, `attn_gate_quant`, `rmsnorm_quant`) | per-element branches on flags uniform for the work-group and 16 scalar loads per lane | **done**: the flags tested once, `vload16` per lane. Per 64 decode tokens: swiglu_quant 35B 71.4-71.9 -> 22.9-23.0 ms, 9B 90.5 -> 40.9-47.8; rmsnorm_quant -8 %; gated_norm_quant -8 %. 27B 11,598-token prompt: rmsnorm_quant 968 -> 636 ms, gated_norm 654 -> 572, attn_gate 298 -> 260. Logits bit-identical on 9B, 27B, 35B, Flash Next, Gemma 4 E4B | ~1 ms per decode token (35B ~2 %) |
| 10 | `moe_down_q` scales (T = 1 and TM) | scale stride S = slots_ (max_chunk x NU): 16 lines in 16 pages per load | **measured, not kept**: S = n for the per-tile down projection of 1-8 tokens (bit-identical): 35B moe_down 1-6 slots the same or slower (6 slots 17.7-18.4 -> 18.4-18.5 ms per step) | the reads hit the cache |

## Smaller

- `moe_down_q` Flash Next at TM > 1: its second superblock step is ~37 % padding (the T = 1 path packs these blocks
  already, `-DFV` DN_PACK): port it.
- Flash Next's `.experts` store down rows 768 wide: 5.7 % of each record read from the SSD is zeros (a compact
  640-wide form needs a new file version).
- 35B `rgu` carries 112 zero router rows per layer (~9.8 MB per token): gate / up first, router last.
- Raw superblocks padded to 8 bytes: Q6_K +2.8 % (every lm_head), IQ3_XXS +6 %, IQ2 +8-9 % (if the loads need only 4).
- Dead `absum` stores in prompt-chunk quantization (no GEMM layout with mins is used at T > 8).
- `moe_gateup_q` TM > 1 reloads `stok[r0 + m]` every iteration; hoist it.
- `attn_prep`: precise pow / cos / sin per element (1,024 mach, 112 scratch messages); a per-token RoPE table.
- IQ4_NL GEMV through `gemv` instead of `gemv_v` (Flash Next's down_sh only).
- `moe_route1` two reductions per round; one zero-fill per layer before `moe_combine`; `qsa_select`'s serial scan.
- Not taken: the IQ3_S `apply_signs4` multiply (tried 17 Oct, slower); per-16 GEMM chains and MOE_TILES pipelining
  (uncertain, pipelining rejected before).

## Nothing worth changing (0 spill unless noted)

GEMV: Q8_0 (G1 102 instructions / 24 sends per loop), Q6_K, IQ4_XS, Q4_K, Q5_K, IQ3_S, Q4_0. MoE T = 1: `moe_gateup_q`
(Flash Next 626, 35B 759), `moe_down_q` (1,324 / 1,091), moe.cl kernels, `asc_transpose`. Prompt: GEMM QTYPE 8 (35B
dense, 937 / 1024 DP4a without a wait) and 101 (915), `decode_w8_sl` Q6_K. Others: `gdn_recur`, `attn_decode`,
`attn_combine`, `gdn_conv` (21 scratch; 256 GRF rejected before), `conv_state_update`, `gdn_gates`, `hc_combine`,
`ple_conv_add`, `ple_hist_update`, `qsa_tail`, `qsa_score(_rt)`.

Not covered: the QSA `attn_decode` and the 35B's `attn_dsplit` (not in the cache / out of scope).
