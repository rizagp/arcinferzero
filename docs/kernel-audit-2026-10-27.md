# Kernel audit with ocloc (2026-10-27)

The GPU assembly of the kernels the engine runs, read for work that is not needed. The binaries are the driver's own,
taken from the kernel cache (`%USERPROFILE%\arcinferzero\kernels`, file = FNV-1a of source + options + device, as
`zero/clrt.h` names it; `ARC_KLIST=1` prints the file and options of every kernel built) and disassembled with
`ocloc disasm`; variants were compiled with ocloc 2026.1 (a newer compiler than the driver's). Nothing measured on
the GPU yet: every gain below is an estimate until measured, and every fix has to keep the logits bit-identical
unless it says otherwise. Ranked by plausible real gain; ALU-only savings in memory-bound loops come last (on this GPU
they made decode loops slower three times: [lessons](lessons-intel-arc-igpu.md)).

## To fix or measure first

| # | Kernel | What is not needed | Fix (bit-identical) | Estimate |
|---|---|---|---|---|
| 1 | `moe_down_q` TM 8, GROWS 4 (2026-10-27's small tiles) | spills 768-832 B per thread, 16 / 8 scratch messages per loop pass; hit by 6-8 rows: the 35B's MTP verification (5 drafts), DFlash2 (7 drafts), batches of 5-8 | **measured, kept as it is**: 35B 5 / 6 slots moe_down 16.8-17.1 / 18.1-18.4 ms per step with GROWS 4, 18.4 / 20.2 with GROWS 2 (no spill), 23.3 / 25.5 with the old TM 16; 35B MTP (5 drafts, 256 tokens) 26.8-26.9 -> 27.8 tok/s against TM 16, logits bit-identical | done: not a regression |
| 2 | `hc_up1` (Flash Next) | weight loads one at a time: HC and KP / 256 are run-time arguments, so the loops do not unroll; ~12 memory round trips in a row per sub-group; 37 GB/s of ~75 | HC and the superblock count as defines; load all weights at entry; epilogue loads above the loop | up to ~5 ms of ~100 ms per token |
| 3 | `hc_down1` (Flash Next) | ~20 waited loads and ~10 barriers before the weights stream; every x-group recomputes the same norm | E as a define (loops unroll), prefetch the 5 superblocks before the norm | up to ~2.6 ms per token |
| 4 | Flash Next shared expert | F 640 padded to 768: 128 zero rows in gate / up, 128 zero columns in down read every token (~38 MB); the padding also turns off the stacked router + gate / up GEMV (`rgu`, +3-4 % on the 35B) | gate / up over 640 rows, zeros written past 640, down skips K >= 640; `rgu` with 640-row pieces | ~0.9 ms + `rgu` per token |
| 5 | `moe_down_q` TM 2-8 | each slot's activations and scales loaded again for each of the 4 rows (TM 8: 104 loads, 24 distinct) | rows decoded once, each slot loaded once ("token-outer", 27B verification GEMV 23 -> 43 GB/s) | verification / batched steps of MoE models |
| 6 | `moe_rg` (prompt MoE, 35B 320-1279-token chunks, Flash Next down) | the scale multiply `c0 * (short)a0` is an emulated 32-bit multiply (mul + mach) | the multiply in float (exact: abs(c0) < 2^20): ocloc rg21 mach 408 -> 152 and DP4a without a wait 288 -> 764 of 1024; rg23 557 -> 888 | compute-bound: measure (the dense GEMM's similar change once went 3x slower) |
| 7 | `attn_q8` QK (prompt attention) | HS dependent DP4a chains, 64 long each (35B: 320 of 512 DP4a wait) | two accumulators per head (c even / odd), summed before the float conversion; ocloc 27B shape waits 169 -> 42, 35B shape 320 -> 206 but 6 scratch messages: check | long prompts (attention grows with the square) |
| 8 | GEMV scales at T = 1 | `asc[b * M + m]` with M = 16: each lane's scale gather touches 16 cache lines for 64 useful bytes (Q4_K / Q5_K also absum); Q4_K / Q5_K reach ~62 GB/s, Q8_0 77 | write and read the T = 1 scales with M = 1 (every T = 1 producer and consumer) | 27B (Q4_K / Q5_K, Q5_K head), Flash Next gemm_o |
| 9 | quantization kernels (`swiglu_quant`, `gated_norm_quant`, `attn_gate_quant`, `rmsnorm_quant`) | per-element branches on flags uniform for the work-group and 16 scalar loads per lane: loads not issued together | branch once outside the loop; vload16 per lane | decode ~1 ms per Flash Next token; prompts ~1 % (27B swiglu 1.3 s per 11K vs ~0.67 floor) |
| 10 | `moe_down_q` scales (T = 1 and TM), `moe_gateup_q` T = 1 | scale stride S = slots_ (max_chunk x NU): 16 lines in 16 pages per load | S = n for T <= 8 outside two-pass mode (host only); ASCT for the T = 1 gate / up | uncertain (cache hits) |

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
