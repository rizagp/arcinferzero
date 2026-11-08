# Self audit 2026-11-08

What was read: the code changes since commit 25381e5 (2026-11-07) in `zero/` and `bench/`, plus the uncommitted tree
of the same day. This was a static review by a separate agent. Nothing was run for the review itself; the fixes below
were then built and checked on the laptop.

## Fixed

1. **XMX was off on discrete Alchemist cards too (A770 / A750 / A380).** The gate meant for the 140T tested
   "8-lane sub-groups", which DG2 also has. Now only iGPUs with 8-lane DPAS keep DP4a by default (`zero/engine.h`).
   A discrete 8-lane card runs the bit-for-bit check, then the model-sized speed check in `xmx_check`, since no one has
   measured `gemm_x8` on one.
2. **`bench\b60_sweep.ps1` swept `ARC_H8_FUSE` on the base without H8.** The runs measured nothing, and drift could
   put `FUSE=0` into "best without H8". FUSE now runs on the H8 base only.
3. **`bench\igpu_sweep.ps1` and `bench\igpu_grf_sweep.ps1`: the path column could say "XMX differed!" for an f16
   failure.** The patterns for the int8 check's messages also matched the f16 check's. Only the label was wrong, not
   the tok/s.
4. **`ARC_MOE_CG` x `ARC_MOE_WY` had no upper bound.** A large product went past the largest work-group and ended the
   run. CG is now 1-4 and WY at most 32 / CG (`zero/engine_moe.h`).

## Noted, not changed

- `ARC_X16_TRY=1` on the 125H (which has 8-lane sub-groups) now tests the 8-lane f16 kernel. The 16-lane one needs
  `ARC_NO_SG8=1` there.
- The fused H8 decode sets the staging owner without a W8 header in `stage_h_`. No path reads it today: `h8_ok`
  depends on the weight alone, so an H8 weight never reaches the plain W8 GEMM.

## Checked and found right

- **`decode.cl h8r_decode` against `decode_w8_sb` + `h8.cl h8r_load` / `h8r_weights`:** they match line by line.
  This covers the lanes' value ranges at HB 512 and 1024, the sign index, the scale and min bytes (per-16 and per-32
  layouts), the per-superblock `d`, the FWHT and the packing, and the output layout. The logits were also
  bit-identical on the laptop (27B, 4B).
- **MoE tiles with global ids:** the same set of (column group, tile) pairs; no barriers or local memory.
- **`ARC_GEMM_WPF`:** the prefetch stays within the column group's own superblocks, and the XMX check covers it.
- **The other kernels:** `gemm_x8` with X8W (the arithmetic order is unchanged, and the early return comes before any
  barrier) and `gemm_x16` at 8 lanes (operand layouts, half offsets, the emulation).
- **The iGPU speed check:** DP4a is built with the switches restored, its buffers are freed, and its result is shared
  per process.
- **The sweep scripts:** their regexes match runz's output, the environment is cleared between runs, and a failed run
  reports 0 instead of stopping.
