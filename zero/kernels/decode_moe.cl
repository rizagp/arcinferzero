// Part of the decode kernels (zero/kernels/decode*.cl, KernelParts in zero/clrt.h): built after decode.cl, whose
// decoders it uses. The mixture-of-experts kernels on the raw GGUF expert rows (single tokens and prompt chunks).

// ---- Mixture-of-experts on standard GGUF expert tensors (Unsloth / llama.cpp quantizations of Qwen3.5 / 3.6 MoE):
// the expert matrices [NE][N][K] of one type, in the GPU raw form. Same slots and tiles as kernels/moe.cl (tiles of
// up to TM slots of one expert); each sub-group computes GROWS rows of the tile's expert for its TM slots, decoding
// the weights as gemv does (two superblocks per step staged in local memory, lane l decodes 32-block l % 8 of the
// superblock l / 8). Only types without per-block mins (the IQ types, Q6_K, Q8_0, Q4_0).
// grid (ceil(rows / GROWS) * 16 rounded to 256, max_tiles), local (256, 1).
#ifdef MOE_Q
#ifndef TM
#define TM 8
#endif
#if QT == Q2_K || QT == Q4_K || QT == Q5_K
#define MOE_MIN 1  // (types with per-block mins: the activation sums per 16 are computed here, there is no absum)
#endif
// (types with a vector decoder, vblock: the block is read straight into registers instead of a staged window)
#if (QT == IQ4_XS || QT == Q6_K || QT == Q4_0 || QT == Q8_0 || QT == IQ4_NL || QT == Q2_0 || defined(GRID_OFF)) && !defined(MOE_WINDOW)
#define MOE_VB 1
#endif
inline void moe_stage(__local uint* lgrid, __local uint* laux, __global const uint* tabs) {
#ifdef GRID_OFF
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
#endif
#ifdef AUX_OFF
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
}
#ifdef GRID_OFF
#define MOE_LGRID __local uint lgrid[GRID_N];
#define MOE_GRID ((const uint*)lgrid)
#else
#define MOE_LGRID __local uint lgrid[1];
#define MOE_GRID ((const uint*)tabs)
#endif
#ifdef AUX_OFF
#define MOE_LAUX __local uint laux[AUX_N];
#define MOE_AUX ((const uint*)laux)
#else
#define MOE_LAUX __local uint laux[1];
#define MOE_AUX ((const uint*)tabs)
#endif

// dot of the lane's decoded block w (sc0 / sc1 halves) with an int8 activation block a8
inline int moe_dot(uint8 a8, const uint* w, int sc0, int sc1) {
  int lo = dot_4x8packed_ss_int(a8.s0, w[0]);
  lo += dot_4x8packed_ss_int(a8.s1, w[1]);
  lo += dot_4x8packed_ss_int(a8.s2, w[2]);
  lo += dot_4x8packed_ss_int(a8.s3, w[3]);
  int hi = dot_4x8packed_ss_int(a8.s4, w[4]);
  hi += dot_4x8packed_ss_int(a8.s5, w[5]);
  hi += dot_4x8packed_ss_int(a8.s6, w[6]);
  hi += dot_4x8packed_ss_int(a8.s7, w[7]);
  return sc0 * lo + sc1 * hi;
}
// d * (scaled dot) - dmin * (mins x activation sums): the lane's block term before the activation scale
inline float moe_term(uint8 a8, const uint* w, float d, float dmin, int sc0, int sc1, int mn0, int mn1) {
#ifdef MOE_MIN
  const uint one = 0x01010101u;
  int slo = dot_4x8packed_ss_int(a8.s0, one), shi = dot_4x8packed_ss_int(a8.s4, one);
  slo += dot_4x8packed_ss_int(a8.s1, one);
  slo += dot_4x8packed_ss_int(a8.s2, one);
  slo += dot_4x8packed_ss_int(a8.s3, one);
  shi += dot_4x8packed_ss_int(a8.s5, one);
  shi += dot_4x8packed_ss_int(a8.s6, one);
  shi += dot_4x8packed_ss_int(a8.s7, one);
  return d * (float)moe_dot(a8, w, sc0, sc1) - dmin * (float)(mn0 * slo + mn1 * shi);
#else
  return d * (float)moe_dot(a8, w, sc0, sc1);
#endif
}

#ifndef HLD  // row stride of h (Flash Next: F = 640 rows, stored in rows of 768 for the padded down projection)
#define HLD F
#endif
// h[slot][F] = silu(x W_gate_e^T) * (x W_up_e^T); x rows are the slots' tokens (aq [token][K], per-32 scales asc
// [K/32][M]); rg / ru: [NE * F] rows of K.
// One slot per tile (TM 1, token generation) with the vector decoders: the slot's token, its activation row and
// the lanes' row pointers are fixed before the loop, and each step loads the lane's activation block before the
// weights are decoded, so its latency overlaps the weight loads (same terms, same order: bit-identical).
#if TM == 1 && defined(MOE_VB)
#define MOE_T1 1
#endif
#ifndef DN_PACK
#define DN_PACK 1
#endif
#ifndef GU_ORDER
#define GU_ORDER 1  // 1: gate decoded and accumulated before the up block is decoded (fewer live registers)
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_gateup_q(__global const uchar* rg, __global const uchar* ru, int F, int K, __global const uint* tabs,
                           __global const char* aq, __global const float* asc, int M, __global const int* tiles,
                           __global const int* stok, __global float* h XARGS_GU) {
  __local uint win_all[16][WIN_CALLS * 16];
  __local uint* win = win_all[get_sub_group_id() & 15];
  MOE_LGRID
  MOE_LAUX
  const int ti = get_group_id(1);
  if (ti >= tiles[0]) return;  // (uniform over the work-group)
#ifdef MOE_VB
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
#else
  moe_stage(lgrid, laux, tabs);
#endif
  __global const int* tp = tiles + 1 + 3 * ti;
  const int e = tp[0], r0 = tp[1], nr = tp[2];
  const int lane = get_sub_group_local_id(), n0 = (get_global_id(0) / 16) * GROWS, nsb = K / 256;
  __global const uchar* eg = XBASE(rg, xog, e, F);
  __global const uchar* eu = XBASE(ru, xou, e, F);
  float ag[GROWS][TM], au[GROWS][TM];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) ag[r][m] = au[r][m] = 0.0f;
#ifdef Q2F  // (two-pass mode: the residual's terms, from the same decoded blocks; see Q2F above)
  float bg[GROWS][TM], bu[GROWS][TM];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) bg[r][m] = bu[r][m] = 0.0f;
#endif
#ifdef MOE_T1
  {
    const int tok = stok[r0], hs = lane / 8, ib = lane % 8;
    __global const uint* ap = (__global const uint*)(aq + (size_t)tok * K);
    __global const float* sap = asc + tok;
#ifdef Q2F
    __global const uint* ap2 = (__global const uint*)(aq + Q2_AQ + (size_t)tok * K);
    __global const float* sap2 = asc + Q2_SC + tok;
#endif
    __global const uchar* pg[GROWS];
    __global const uchar* pu[GROWS];
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(n0 + r, F - 1);
      pg[r] = XR(eg, n, 0);
      pu[r] = XR(eu, n, 0);
    }
    for (int sp = 0; sp < nsb; sp += 2) {
      const bool has = sp + hs < nsb;
      const int sbi = has ? sp + hs : sp, bc = sbi * 8 + ib;  // (a lane past the row: superblock sp, a8 = sa = 0)
      uint8 a8 = vload8(0, ap + 8 * bc);
      float sa = sap[(size_t)bc * M];
      a8 = has ? a8 : (uint8)0;
      sa = has ? sa : 0.0f;
#ifdef Q2F
      uint8 r8 = vload8(0, ap2 + 8 * bc);
      float sr = sap2[(size_t)bc * M];
      r8 = has ? r8 : (uint8)0;
      sr = has ? sr : 0.0f;
#endif
#pragma unroll
      for (int r = 0; r < GROWS; ++r) {
        uint wg[8], wu[8];
        int gs0, gs1, us0, us1, gm0 = 0, gm1 = 0, um0 = 0, um1 = 0;
        float dg, du, gdmin = 0.0f, udmin = 0.0f;
        vblock(pg[r] + sbi * BBP, ib, kt, wg, &dg, &gdmin, &gs0, &gs1, &gm0, &gm1);
#if GU_ORDER
        ag[r][0] += sa * moe_term(a8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
#ifdef Q2F
        bg[r][0] += sr * moe_term(r8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
#endif
#endif
        vblock(pu[r] + sbi * BBP, ib, kt, wu, &du, &udmin, &us0, &us1, &um0, &um1);
#if !GU_ORDER
        ag[r][0] += sa * moe_term(a8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
#ifdef Q2F
        bg[r][0] += sr * moe_term(r8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
#endif
#endif
        au[r][0] += sa * moe_term(a8, wu, du, udmin, us0, us1, um0, um1);
#ifdef Q2F
        bu[r][0] += sr * moe_term(r8, wu, du, udmin, us0, us1, um0, um1);
#endif
      }
    }
  }
#else
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + lane / 8 < nsb;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(n0 + r, F - 1);
      // the gate and up blocks decoded once, then used for every slot of the tile
      uint wg[8], wu[8];
      int gs0, gs1, us0, us1, gm0 = 0, gm1 = 0, um0 = 0, um1 = 0;
      float dg, du, gdmin = 0.0f, udmin = 0.0f;
#ifdef MOE_VB
      {
        const int sbi = has ? sp + lane / 8 : sp;
        vblock(XR(eg, n, sbi), lane % 8, kt, wg, &dg, &gdmin, &gs0, &gs1, &gm0, &gm1);
        vblock(XR(eu, n, sbi), lane % 8, kt, wu, &du, &udmin, &us0, &us1, &um0, &um1);
      }
#else
#pragma unroll
      for (int gu = 0; gu < 2; ++gu) {
        __global const uint* src = (__global const uint*)(gu ? XR(eu, n, sp) : XR(eg, n, sp));
#pragma unroll
        for (int c = 0; c < WIN_CALLS; ++c) win[c * 16 + lane] = intel_sub_group_block_read(src + c * 16);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        if (gu) decode32((const uchar*)win + (has ? lane / 8 : 0) * BBP, lane % 8, MOE_GRID, MOE_AUX, wu, &us0, &us1, &um0, &um1, &du, &udmin);
        else decode32((const uchar*)win + (has ? lane / 8 : 0) * BBP, lane % 8, MOE_GRID, MOE_AUX, wg, &gs0, &gs1, &gm0, &gm1, &dg, &gdmin);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      }
#endif
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        if (m < nr) {
          const int tok = stok[r0 + m];
          const uint8 a8 = has ? vload8(0, (__global const uint*)(aq + (size_t)tok * K + 32 * b)) : (uint8)0;
#ifdef ASCT  // (the scales token by token, [M][K/32]: the lanes' blocks side by side)
          const float sa = has ? asc[(size_t)tok * (K / 32) + b] : 0.0f;
#else
          const float sa = has ? asc[(size_t)b * M + tok] : 0.0f;
#endif
          ag[r][m] += sa * moe_term(a8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
#ifdef Q2F
          const uint8 r8 = has ? vload8(0, (__global const uint*)(aq + Q2_AQ + (size_t)tok * K + 32 * b)) : (uint8)0;
          const float sr = has ? asc[Q2_SC + (size_t)b * M + tok] : 0.0f;
          bg[r][m] += sr * moe_term(r8, wg, dg, gdmin, gs0, gs1, gm0, gm1);
#endif
          au[r][m] += sa * moe_term(a8, wu, du, udmin, us0, us1, um0, um1);
#ifdef Q2F
          bu[r][m] += sr * moe_term(r8, wu, du, udmin, us0, us1, um0, um1);
#endif
        }
      }
    }
  }
#endif
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#ifdef Q2F  // (the value pass's sum, then the residual's added, as the two passes stored them)
      const float g = sub_group_reduce_add(ag[r][m]) + sub_group_reduce_add(bg[r][m]);
      const float u = sub_group_reduce_add(au[r][m]) + sub_group_reduce_add(bu[r][m]);
#else
      const float g = sub_group_reduce_add(ag[r][m]), u = sub_group_reduce_add(au[r][m]);
#endif
      const int n = n0 + r;
#ifdef MOE_GELU  // (Gemma 4 experts: gelu_tanh(gate) * up)
      if (lane == 0 && n < F && m < nr)
        h[(size_t)(r0 + m) * HLD + n] = 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g))) * u;
#elif defined(SWI_CLAMP)  // (DeepSeek-V4.1: the gate clamped from above, the up both ways)
      if (lane == 0 && n < F && m < nr) {
        const float gc = fmin(g, SWI_CLAMP), uc = clamp(u, -SWI_CLAMP, SWI_CLAMP);
        h[(size_t)(r0 + m) * HLD + n] = gc / (1.0f + exp(-gc)) * uc;
      }
#else
      if (lane == 0 && n < F && m < nr) h[(size_t)(r0 + m) * HLD + n] = g / (1.0f + exp(-g)) * u;
#endif
    }
}

// y[slot][N] = weight(slot) * (h[slot] W_down_e^T); h quantized per slot (hq [slot][K], scales hsc [K/32][S]).
// TM 1 with -DFV=<columns that hold values> (Flash Next: 640 of the 768): the last superblock's REM value blocks
// of the sub-group's GROWS rows are spread over the lanes (lane = row * REM + block) instead of each row taking a
// whole step with 16 - REM lanes on padding or past the row; each lane's term goes back to the lane that held it
// before (sub_group_shuffle) and is added there in the same order: bit-identical, as the skipped blocks' terms are
// exactly zero (their activations quantize to 0 with scale 0).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_down_q(__global const uchar* rd, int N, int K, __global const uint* tabs, __global const char* hq,
                         __global const float* hsc, int S, __global const int* tiles, __global const float* swt,
                         __global float* y XARGS_D) {
  __local uint win_all[16][WIN_CALLS * 16];
  __local uint* win = win_all[get_sub_group_id() & 15];
  MOE_LGRID
  MOE_LAUX
  const int ti = get_group_id(1);
  if (ti >= tiles[0]) return;
#ifdef MOE_VB
  __local uint kt[VB_TABLES];
  vb_tables(kt, tabs);
#else
  moe_stage(lgrid, laux, tabs);
#endif
  __global const int* tp = tiles + 1 + 3 * ti;
  const int e = tp[0], r0 = tp[1], nr = tp[2];
  const int lane = get_sub_group_local_id(), n0 = (get_global_id(0) / 16) * GROWS, nsb = K / 256;
  __global const uchar* ed = XBASE(rd, xod, e, N);
  float acc[GROWS][TM];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) acc[r][m] = 0.0f;
#ifdef Q2F  // (two-pass mode: h's residual at -DQ2M bytes / -DQ2MS floats, as moe_quant writes it)
  float acc2[GROWS][TM];
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) acc2[r][m] = 0.0f;
#endif
#ifdef MOE_T1
  {
    const int hs = lane / 8, ib = lane % 8;
    __global const uint* ap = (__global const uint*)(hq + (size_t)r0 * K);
    __global const float* sap = hsc + r0;
#ifdef Q2F
    __global const uint* ap2 = (__global const uint*)(hq + Q2M + (size_t)r0 * K);
    __global const float* sap2 = hsc + Q2MS + r0;
#endif
    __global const uchar* pr[GROWS];
#pragma unroll
    for (int r = 0; r < GROWS; ++r) pr[r] = XR(ed, min(n0 + r, N - 1), 0);
#if defined(FV) && DN_PACK
#define DN_FULL (2 * ((FV) / 512))            // superblocks in whole value pairs
#define DN_REM ((FV) / 32 - 8 * DN_FULL)       // value blocks after them
    const int spe = DN_FULL;
#else
    const int spe = nsb;
#endif
    for (int sp = 0; sp < spe; sp += 2) {
      const bool has = sp + hs < nsb;
      const int sbi = has ? sp + hs : sp, bc = sbi * 8 + ib;
      uint8 a8 = vload8(0, ap + 8 * bc);
      float sa = sap[(size_t)bc * S];
      a8 = has ? a8 : (uint8)0;
      sa = has ? sa : 0.0f;
#ifdef Q2F
      uint8 r8 = vload8(0, ap2 + 8 * bc);
      float sr = sap2[(size_t)bc * S];
      r8 = has ? r8 : (uint8)0;
      sr = has ? sr : 0.0f;
#endif
#pragma unroll
      for (int r = 0; r < GROWS; ++r) {
        uint w[8];
        int sc0, sc1, mn0 = 0, mn1 = 0;
        float d, dmin = 0.0f;
        vblock(pr[r] + sbi * BBP, ib, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
        acc[r][0] += sa * moe_term(a8, w, d, dmin, sc0, sc1, mn0, mn1);
#ifdef Q2F
        acc2[r][0] += sr * moe_term(r8, w, d, dmin, sc0, sc1, mn0, mn1);
#endif
      }
    }
#if defined(FV) && DN_PACK
#if DN_REM > 0 && DN_REM <= 8 && DN_REM * GROWS <= 16
    {
      const int rr = min(lane / DN_REM, GROWS - 1), j = lane % DN_REM, bc = 8 * DN_FULL + j;
      const uint8 a8 = vload8(0, ap + 8 * bc);
      const float sa = sap[(size_t)bc * S], st = lane < DN_REM ? sa : 0.0f;
      uint w[8];
      int sc0, sc1, mn0 = 0, mn1 = 0;
      float d, dmin = 0.0f;
      vblock(XR(ed, min(n0 + rr, N - 1), DN_FULL), j, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
      const float t = moe_term(a8, w, d, dmin, sc0, sc1, mn0, mn1);
#pragma unroll
      for (int r = 0; r < GROWS; ++r) acc[r][0] += st * intel_sub_group_shuffle(t, r * DN_REM + j);
#ifdef Q2F
      const uint8 r8 = vload8(0, ap2 + 8 * bc);
      const float sr = sap2[(size_t)bc * S], st2 = lane < DN_REM ? sr : 0.0f;
      const float t2 = moe_term(r8, w, d, dmin, sc0, sc1, mn0, mn1);
#pragma unroll
      for (int r = 0; r < GROWS; ++r) acc2[r][0] += st2 * intel_sub_group_shuffle(t2, r * DN_REM + j);
#endif
    }
#elif DN_REM > 0
    {
      const int sp = DN_FULL;
      const bool has = sp + hs < nsb;
      const int sbi = has ? sp + hs : sp, bc = sbi * 8 + ib;
      uint8 a8 = vload8(0, ap + 8 * bc);
      float sa = sap[(size_t)bc * S];
      a8 = has ? a8 : (uint8)0;
      sa = has ? sa : 0.0f;
#ifdef Q2F
      uint8 r8 = vload8(0, ap2 + 8 * bc);
      float sr = sap2[(size_t)bc * S];
      r8 = has ? r8 : (uint8)0;
      sr = has ? sr : 0.0f;
#endif
#pragma unroll
      for (int r = 0; r < GROWS; ++r) {
        uint w[8];
        int sc0, sc1, mn0 = 0, mn1 = 0;
        float d, dmin = 0.0f;
        vblock(pr[r] + sbi * BBP, ib, kt, w, &d, &dmin, &sc0, &sc1, &mn0, &mn1);
        acc[r][0] += sa * moe_term(a8, w, d, dmin, sc0, sc1, mn0, mn1);
#ifdef Q2F
        acc2[r][0] += sr * moe_term(r8, w, d, dmin, sc0, sc1, mn0, mn1);
#endif
      }
    }
#endif
#endif
  }
#else
  for (int sp = 0; sp < nsb; sp += 2) {
    const int b = sp * 8 + lane;
    const bool has = sp + lane / 8 < nsb;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(n0 + r, N - 1);
      uint w[8];
      int sc0, sc1, mn0 = 0, mn1 = 0;
      float d, dmin = 0.0f;
#ifdef MOE_VB
      vblock(XR(ed, n, (has ? sp + lane / 8 : sp)), lane % 8, kt, w, &d, &dmin, &sc0, &sc1,
             &mn0, &mn1);
#else
      __global const uint* src = (__global const uint*)XR(ed, n, sp);
#pragma unroll
      for (int c = 0; c < WIN_CALLS; ++c) win[c * 16 + lane] = intel_sub_group_block_read(src + c * 16);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      decode32((const uchar*)win + (has ? lane / 8 : 0) * BBP, lane % 8, MOE_GRID, MOE_AUX, w, &sc0, &sc1, &mn0, &mn1, &d, &dmin);
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
#endif
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        if (m < nr) {
          const int row = r0 + m;
          const uint8 a8 = has ? vload8(0, (__global const uint*)(hq + (size_t)row * K + 32 * b)) : (uint8)0;
          const float sa = has ? hsc[(size_t)b * S + row] : 0.0f;
          acc[r][m] += sa * moe_term(a8, w, d, dmin, sc0, sc1, mn0, mn1);
#ifdef Q2F
          const uint8 r8 = has ? vload8(0, (__global const uint*)(hq + Q2M + (size_t)row * K + 32 * b)) : (uint8)0;
          const float sr = has ? hsc[Q2MS + (size_t)b * S + row] : 0.0f;
          acc2[r][m] += sr * moe_term(r8, w, d, dmin, sc0, sc1, mn0, mn1);
#endif
        }
      }
    }
  }
#endif
#pragma unroll
  for (int r = 0; r < GROWS; ++r)
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const float v = sub_group_reduce_add(acc[r][m]);
      const int n = n0 + r;
#ifdef Q2F  // (as the two passes: the weighted value sum, then the weighted residual sum added)
      const float v2 = sub_group_reduce_add(acc2[r][m]);
      if (lane == 0 && n < N && m < nr) y[(size_t)(r0 + m) * N + n] = swt[r0 + m] * v + swt[r0 + m] * v2;
#else
      if (lane == 0 && n < N && m < nr) y[(size_t)(r0 + m) * N + n] = swt[r0 + m] * v;
#endif
    }
}

#endif  // MOE_Q

// ---- Mixture-of-experts prompt chunks straight from the raw expert rows (-DMOE_RG): the per-tile int8 GEMM of
// kernels/gemm.cl (-DMOE_TILES, W8 layouts 101 = per-32 and 100 = per-16 scales) with each lane decoding its own
// weight row's superblocks in registers instead of reading decode_w8's copy. The same int8 weights and scales and the
// same order of operations, so the results are those of decode_w8 + gemm, bit for bit, without the decode pass and
// its writes. Lanes = 16 rows of one expert; a tile's tokens are broadcast as in the GEMM. Types without per-block
// mins that decode_w8 does not requantize (not Q2_K / Q4_K / Q5_K). The per-32 float types (Q8_0, Q4_0, IQ4_NL,
// Q2_0) as the GEMM's native Q8_0 form: the block scale times the activation scale per 32-block.
// C[r][gc * 16 + lane] for the tiles tstart[0] .. tstart[ne] (moe_scan), looped with the launch's y size as stride;
// accumulate: 0 store, 1 add, 2 C = silu(C) * result; kv: the columns that hold values (K, or less for rows padded to
// whole superblocks). grid (N, ny), local (16, 1).
#ifdef MOE_RG
#define RG_SC32 (QT == IQ4_XS || QT == IQ3_S || QT == IQ3_XXS || QT == IQ2_XXS || PER32)
#ifndef RG_TP
#define RG_TP 4  // tokens per step (no spills with 256 registers for IQ3_XXS / IQ4_XS / Q3_K; 8 or 16 spill)
#endif
#define RG_TM 16
#define RG_B(x, i) intel_sub_group_shuffle((x), (i))
// (Q4_0 / Q8_0: each 32-block read straight from global memory with vblock, gemv_v's decoder, instead of copying the
// whole superblock into registers first: that copy (36 / 68 words per lane) made moe_rg spill 2.6 KB per thread on
// Gemma 4's Q4_0 experts. The same bytes and the same weights: bit-identical. -DRG_COPY: the copy, for A/B)
#if (QT == Q4_0 || QT == Q8_0) && !defined(RG_COPY)
#define RG_VB 1
#endif
#ifdef XCACHE
#define XARGS_RG , __global const ulong* xaddr, int xoff
#else
#define XARGS_RG
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void moe_rg(__global const uchar* raw, int N, int K, __global const uint* tabs, __global const char* restrict aq,
                     __global const float* restrict asc, __global float* restrict C, int M, int ldc, int accumulate,
                     __global const int* restrict tiles, __global const int* restrict tstart, int ne,
                     __global const int* restrict rowmap, int use_map, __global const float* restrict rowscale,
                     int use_scale, int kv XARGS_RG) {
  const int gy = get_group_id(1), ngy = get_num_groups(1);
  if (tstart[0] + gy >= tstart[ne]) return;  // (no tile for this work-group: before the staging)
#ifdef GRID_OFF
  __local uint lgrid[GRID_N];
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
  const uint* grid = lgrid;
#else
  const uint* grid = tabs;
#endif
#ifdef AUX_OFF
  __local uint laux[AUX_N];
  for (int i = get_local_id(0); i < AUX_N; i += get_local_size(0)) laux[i] = tabs[AUX_OFF + i];
  const uint* aux = laux;
#else
  const uint* aux = tabs;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RG_VB
  __local uint kt[VB_TABLES];  // (unused by these types)
#endif
  const int lane = get_sub_group_local_id(), gc = get_group_id(0), nsb = K / 256, t_end = tstart[ne];
  for (int t = tstart[0] + gy; t < t_end; t += ngy) {
    const int e = tiles[1 + 3 * t], r0 = tiles[2 + 3 * t], nr = tiles[3 + 3 * t];
    const int rl = r0 + min(lane, nr - 1);  // this lane's slot (rows past the tile repeat its last slot)
    const int tokl = use_map ? rowmap[rl] : rl;
    const int n = gc * 16 + lane;  // this lane's weight row
#ifdef XCACHE
    __global const uchar* wrow = (__global const uchar*)xaddr[e] + xoff + (size_t)n * nsb * BBP;
#else
    __global const uchar* wrow = raw + ((size_t)e * N + n) * nsb * BBP;
#endif
    float res[RG_TM];
#pragma unroll
    for (int m = 0; m < RG_TM; ++m) res[m] = 0.0f;
    for (int s = 0; s < nsb; ++s) {
#ifdef RG_VB
      __global const uchar* sbp = wrow + (size_t)s * BBP;
#else
      uint sbu[BBP / 4];
      {
        __global const uint* src = (__global const uint*)(wrow + (size_t)s * BBP);
#pragma unroll
        for (int i = 0; i < BBP / 8; ++i) {
          const uint2 v = vload2(i, src);
          sbu[2 * i] = v.x;
          sbu[2 * i + 1] = v.y;
        }
      }
      const uchar* p = (const uchar*)sbu;
#endif
      float tq[RG_TM];
#pragma unroll
      for (int m = 0; m < RG_TM; ++m) tq[m] = 0.0f;
      float d = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (s * 256 + 64 * j >= kv) break;  // (columns past kv: padding, Flash Next's down projection; kv % 64 == 0)
        uint lo[8], hi[8];
        int a0, a1, b0, b1, mn0, mn1;
        float dl, dmin;
#ifdef RG_VB
        vblock(sbp, 2 * j, kt, lo, &dl, &dmin, &a0, &a1, &mn0, &mn1);
        vblock(sbp, 2 * j + 1, kt, hi, &d, &dmin, &b0, &b1, &mn0, &mn1);
#else
        decode32(p, 2 * j, grid, aux, lo, &a0, &a1, &mn0, &mn1, &dl, &dmin);
        decode32(p, 2 * j + 1, grid, aux, hi, &b0, &b1, &mn0, &mn1, &d, &dmin);
#endif
        // the decoded scales as the W8 header holds them (int8)
        a0 = (int)(char)a0; a1 = (int)(char)a1; b0 = (int)(char)b0; b1 = (int)(char)b1;
        const float avl = asc[((size_t)s * 8 + 2 * j) * M + tokl], avh = asc[((size_t)s * 8 + 2 * j + 1) * M + tokl];
#pragma unroll
        for (int mg = 0; mg < RG_TM; mg += RG_TP) {
          uint aw[RG_TP];
#pragma unroll
          for (int q = 0; q < RG_TP; ++q)
            aw[q] = intel_sub_group_block_read((__global const uint*)(aq + (size_t)RG_B(tokl, mg + q) * K + s * 256 + j * 64));
#if RG_SC32
          int c0[RG_TP], c1[RG_TP];
#pragma unroll
          for (int q = 0; q < RG_TP; ++q) c0[q] = c1[q] = 0;
#pragma unroll
          for (int w = 0; w < 8; ++w)
#pragma unroll
            for (int q = 0; q < RG_TP; ++q) {
              c0[q] += dot_4x8packed_ss_int(RG_B(aw[q], w), lo[w]);
              c1[q] += dot_4x8packed_ss_int(RG_B(aw[q], w + 8), hi[w]);
            }
#pragma unroll
          for (int q = 0; q < RG_TP; ++q) {
            const int m = mg + q;
#if PER32
            res[m] = fma(RG_B(avl, m) * dl, (float)c0[q], fma(RG_B(avh, m) * d, (float)c1[q], res[m]));
#elif defined(RG_IMUL)
            const int plo = c0[q] * (short)a0, phi = c1[q] * (short)b0;
            tq[m] = fma(RG_B(avl, m), (float)plo, fma(RG_B(avh, m), (float)phi, tq[m]));
#else
            // (the scale multiply in float: the int product is exact (abs(c0) < 2^20) and both forms round it once,
            //  so the same values; the int32 multiply is emulated (mul + mach) and broke the DP4a chains: ocloc
            //  IQ3_S 288 -> 764 of 1024 DP4a without a wait; -DRG_IMUL: the int form)
            const float plo = (float)c0[q] * (float)a0, phi = (float)c1[q] * (float)b0;
            tq[m] = fma(RG_B(avl, m), plo, fma(RG_B(avh, m), phi, tq[m]));
#endif
          }
#else
          int c0[RG_TP], c1[RG_TP], c2[RG_TP], c3[RG_TP];
#pragma unroll
          for (int q = 0; q < RG_TP; ++q) c0[q] = c1[q] = c2[q] = c3[q] = 0;
#pragma unroll
          for (int w = 0; w < 4; ++w)
#pragma unroll
            for (int q = 0; q < RG_TP; ++q) {
              c0[q] += dot_4x8packed_ss_int(RG_B(aw[q], w), lo[w]);
              c1[q] += dot_4x8packed_ss_int(RG_B(aw[q], w + 4), lo[w + 4]);
              c2[q] += dot_4x8packed_ss_int(RG_B(aw[q], w + 8), hi[w]);
              c3[q] += dot_4x8packed_ss_int(RG_B(aw[q], w + 12), hi[w + 4]);
            }
#pragma unroll
          for (int q = 0; q < RG_TP; ++q) {
            const int m = mg + q;
            const int plo = c0[q] * (short)a0 + c1[q] * (short)a1, phi = c2[q] * (short)b0 + c3[q] * (short)b1;
            tq[m] = fma(RG_B(avl, m), (float)plo, fma(RG_B(avh, m), (float)phi, tq[m]));
          }
#endif
        }
      }
#if !PER32
#pragma unroll
      for (int m = 0; m < RG_TM; ++m) res[m] = fma(d, tq[m], res[m]);
#endif
    }
    const float rsc = use_scale ? rowscale[rl] : 1.0f;
#pragma unroll
    for (int m = 0; m < RG_TM; ++m) {
      if (m >= nr) break;
      __global uint* cp = (__global uint*)(C + (size_t)(r0 + m) * ldc + gc * 16);
      float v = res[m] * RG_B(rsc, m);
      if (accumulate == 1) {
        v += as_float(intel_sub_group_block_read(cp));
      } else if (accumulate == 2) {
        const float g = as_float(intel_sub_group_block_read(cp));
        v *= g / (1.0f + exp(-g));
      }
      intel_sub_group_block_write(cp, as_uint(v));
    }
  }
}
#endif  // MOE_RG
