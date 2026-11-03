// Part of the non-GEMM kernels (zero/kernels/ops*.cl, KernelParts in zero/clrt.h): built after the parts before
// it, whose definitions it uses. Float attention of Gemma 4 and Qwen3 dense (attn_prep_g, attn_fa_g / _gc / _gh, rope_table), then the
// residual norms (add_norm_quant, add_rmsnorm), Gemma's PLE and the logit soft cap.

// ---------------------------------------------------------------- Gemma 4
// RoPE angles of rows pos0 .. pos0 + M_valid - 1 for one layer type (head size hd, base, optional frequency
// factors): rt[m][i] = (cos, sin) of theta = pos * base^(-2i/hd) / ff[i], as attn_prep_g computes them, once per
// forward pass instead of in every layer. grid (hd / 2, M_valid).
__kernel void rope_table(__global float2* rt, int hd, int pos0, float theta_base, __global const float* ff,
                         int use_ff) {
  const int fi = get_global_id(0), m = get_global_id(1);
  float theta = (float)(pos0 + m) * pow(theta_base, -2.0f * (float)fi / (float)hd);
  if (use_ff) theta /= ff[fi];
  rt[(size_t)m * (hd / 2) + fi] = (float2)(cos(theta), sin(theta));
}

// n (a multiple of 8) consecutive floats into a register array with 32-byte vector loads: a lane's run of a row
// read as one access instead of n scalar ones (same values; the per-lane runs are 64-128 bytes apart, so scalar
// loads touched 16 cache lines per instruction)
#define LOADF(dst, p, n)                                                                  \
  _Pragma("unroll") for (int c_ = 0; c_ < (n) / 8; ++c_) {                              \
    const float8 t_ = vload8(0, (p) + 8 * c_);                                          \
    (dst)[8 * c_] = t_.s0; (dst)[8 * c_ + 1] = t_.s1; (dst)[8 * c_ + 2] = t_.s2;         \
    (dst)[8 * c_ + 3] = t_.s3; (dst)[8 * c_ + 4] = t_.s4; (dst)[8 * c_ + 5] = t_.s5;     \
    (dst)[8 * c_ + 6] = t_.s6; (dst)[8 * c_ + 7] = t_.s7;                               \
  }
// Attention inputs for one (token, head) per sub-group: RMS norm per head (weighted for q and k), NEOX RoPE over all
// HD dims (theta = pos * base^(-2i/HD) / ff[i] with optional frequency factors), V = RMS norm without weight of the
// V projection, or of the raw K projection when the layer has none (has_v = 0); has_v = 2: the V projection as it
// is, no norm. theta_base <= 0: no RoPE. int8 copies with a per-row scale:
// q -> qi/qsc, K/V -> the 16-key-block cache (slot = pos % cap). Lane l holds dims [VPL*l, VPL*l + VPL).
// qkv rows: [q nh*HD | k nkv*HD | v nkv*HD]. With rt (rope_table of these rows) the angles come from the table.
// grid (16*(nh+nkv), M), local (16, 1).
#define VPL (HD / 16)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_prep_g(__global const float* qkv, int ldq, int nh, int nkv, int has_v, __global const float* qn,
                          __global const float* kn, int M_valid, int pos0, float eps, float theta_base,
                          __global const float* ff, int use_ff, float q_scale, __global char* qi,
                          __global float* qsc, __global uint* kq, __global float* ksc, __global uchar* vq,
                          __global float* vsc, int cap, __global const float2* rt, int f16kv, __global float* qf,
                          __global KC_T* kc, __global VC_T* vc) {
  const int m = get_global_id(1), hs = get_global_id(0) / SG, lane = get_sub_group_local_id();
  if (m >= M_valid) return;
  const bool isq = hs < nh;
  const int kvh = hs - nh, pos = pos0 + m;
  __global const float* row = qkv + (size_t)m * ldq;
  __global const float* src = isq ? row + (size_t)hs * HD : row + (size_t)(nh + kvh) * HD;
  float v[VPL];
  float ss = 0.0f;
  LOADF(v, src + VPL * lane, VPL)
#pragma unroll
  for (int i = 0; i < VPL; ++i) ss += v[i] * v[i];
  ss = sub_group_reduce_add(ss);
  float r = rsqrt(ss / HD + eps);
  __global const float* nw = isq ? qn : kn;
  // (norm weights, RoPE pairs and table in pieces of 8: whole-row copies spilled 4.6 KB per thread at HD 512)
#define PRP (VPL < 8 ? VPL : 8)
#pragma unroll
  for (int c0 = 0; c0 < VPL; c0 += PRP) {
    float nwv[PRP];
    LOADF(nwv, nw + VPL * lane + c0, PRP)
#pragma unroll
    for (int i = 0; i < PRP; ++i) v[c0 + i] *= r * nwv[i];
  }
  // NEOX RoPE: element e pairs with e +- HD/2, which lives in lane l ^ 8
  if (theta_base > 0.0f) {  // (uniform)
  __global const float* rtl = (__global const float*)(rt + (size_t)m * (HD / 2) + (VPL * lane) % (HD / 2));
#pragma unroll
  for (int c0 = 0; c0 < VPL; c0 += PRP) {
    float pv[PRP], rtv[2 * PRP];  // this lane's (cos, sin) pairs: angles (VPL * lane) % (HD / 2) + c0 ..
#pragma unroll
    for (int i = 0; i < PRP; ++i) pv[i] = intel_sub_group_shuffle_xor(v[c0 + i], 8);
#ifdef RT_TABLE  // (-DRT_TABLE: the caller always passes the table; the angles' code below (pow / cos / sin, ~8,000 of
                 //  the kernel's 8,500 instructions) is left out. Without it: the Gemma 4 drafter's layers, no table)
    LOADF(rtv, rtl + 2 * c0, 2 * PRP)
#else
    if (rt) { LOADF(rtv, rtl + 2 * c0, 2 * PRP) }
#endif
#pragma unroll
    for (int i = 0; i < PRP; ++i) {
      const int e = VPL * lane + c0 + i, fi = e % (HD / 2);
      float c, sn;
#ifdef RT_TABLE
      c = rtv[2 * i];
      sn = rtv[2 * i + 1];
      (void)fi;
#else
      if (rt) {
        c = rtv[2 * i];
        sn = rtv[2 * i + 1];
      } else {
        float theta = (float)pos * pow(theta_base, -2.0f * (float)fi / (float)HD);
        if (use_ff) theta /= ff[fi];
        c = cos(theta);
        sn = sin(theta);
      }
#endif
      v[c0 + i] = e < HD / 2 ? v[c0 + i] * c - pv[i] * sn : v[c0 + i] * c + pv[i] * sn;
    }
  }
  }
#undef PRP
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < VPL; ++i) {
    if (isq) v[i] *= q_scale;
    amax = fmax(amax, fabs(v[i]));
  }
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, o));  // scale block
  float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  uint pk[VPL / 4];
#pragma unroll
  for (int c = 0; c < VPL / 4; ++c) pk[c] = 0;
#pragma unroll
  for (int i = 0; i < VPL; ++i) pk[i / 4] |= (uint)(convert_int_rte(v[i] * inv) & 0xFF) << (8 * (i % 4));
  // (f16kv bit 0: the f32 query and the float K / V rows for attn_fa_g as well: kc / vc [kv head][cap][HD], slot =
  //  pos % cap)
  if (f16kv & 1) {
    if (isq) {
      __global float* qd = qf + ((size_t)m * nh + hs) * HD + VPL * lane;
#pragma unroll
      for (int i = 0; i < VPL; ++i) qd[i] = v[i];
    } else {
      __global KC_T* kd = kc + ((size_t)kvh * cap + pos % cap) * HD + VPL * lane;
#pragma unroll
      for (int i = 0; i < VPL; ++i) KST1(v[i], kd + i);
    }
  }
  if (isq) {
    __global uint* dst = (__global uint*)(qi + ((size_t)m * nh + hs) * HD) + (VPL / 4) * lane;
#pragma unroll
    for (int c = 0; c < VPL / 4; ++c) dst[c] = pk[c];
    if (lane % (16 / SB) == 0) qsc[((size_t)m * nh + hs) * SB + lane / (16 / SB)] = amax / 127.0f;
    return;
  }
  const int slot = pos % cap, kk = slot % 16;
  const size_t blk = (size_t)kvh * (cap / 16) + slot / 16;
  // (f16kv bit 1: this layer keeps no int8 K / V copy, only the drafter's layers do)
  const bool i8 = !(f16kv & 2);
#pragma unroll
  for (int c = 0; c < VPL / 4; ++c)
    if (i8) kq[(blk * (HD / 4) + (VPL / 4) * lane + c) * 16 + kk] = pk[c];
  // V: the V projection, or the raw K projection, RMS-normalized without weight
  // (the raw K projection is read again from qkv, an input only: a register copy of it spilled at HD 512)
  float raw[VPL];
  if (has_v) {
    __global const float* vs = row + (size_t)(nh + nkv + kvh) * HD;
    ss = 0.0f;
    LOADF(raw, vs + VPL * lane, VPL)
#pragma unroll
    for (int i = 0; i < VPL; ++i) ss += raw[i] * raw[i];
    ss = sub_group_reduce_add(ss);
  } else {
    LOADF(raw, src + VPL * lane, VPL)
  }
  r = has_v == 2 ? 1.0f : rsqrt(ss / HD + eps);
  float vmax = 0.0f;
#pragma unroll
  for (int i = 0; i < VPL; ++i) { raw[i] *= r; vmax = fmax(vmax, fabs(raw[i])); }
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) vmax = fmax(vmax, intel_sub_group_shuffle_xor(vmax, o));
  inv = vmax > 0.0f ? 127.0f / vmax : 0.0f;
#pragma unroll
  for (int i = 0; i < VPL; ++i)
    if (i8) vq[(blk * HD + VPL * lane + i) * 16 + kk] = (uchar)convert_int_rte(raw[i] * inv);
  if (i8 && lane % (16 / SB) == 0) {  // scales [block][scale block][key]
    ksc[(blk * SB + lane / (16 / SB)) * 16 + kk] = amax / 127.0f;
    vsc[(blk * SB + lane / (16 / SB)) * 16 + kk] = vmax / 127.0f;
  }
  if (f16kv & 1) {
    __global VC_T* vd = vc + ((size_t)kvh * cap + pos % cap) * HD + VPL * lane;
#pragma unroll
    for (int i = 0; i < VPL; ++i) VST1(raw[i], vd + i);
  }
}

// Gemma 4 attention in float (two-pass mode, QUEUE F21.3): f32 queries (qf [M][nh][HD]) against f16 K / V rows
// (kc / vc [kv head][cap][HD], slot = position % cap, written by attn_prep_g), f32 scores, softmax and P.V; no
// 1/sqrt(d) (Gemma 4). Query position p0 + m sees the keys lo .. hi: lo = p - win + 1 (win > 0, sliding) or 0,
// hi = p, or nc_hi inside a non-causal span [nc_lo, nc_hi] (an image). Each row is computed alone (its result does
// not depend on the other rows of the launch): verification rows equal single tokens. One sub-group per (GQ q heads
// of one kv head, row); keys in blocks of 16, lane = key for the scores, lane = 16 dims per 256 for P.V
// (head 128: 8 dims per lane).
// grid (16, nh / GQ, M), local (16, 1, 1).
// -DFA_ROWS=R (prompt chunks): a work-group of R consecutive rows, one sub-group each as above, loads every block of
// 16 keys and values once into local memory for all of them (each row read them all from global memory before:
// 4K-token prompt 14.7 s of attention); a row still takes exactly its own blocks with the same arithmetic, so the
// results are bit for bit those of R = 1. grid (16, nh / GQ, ceil(M / R) * R), local (16, 1, R).
#ifdef FA_G
#if HD >= 256
#define FDC (HD / 256)  // 256-dim chunks of a row, 16 dims per lane each
#define FVW 16
#define FVT float16
#define FVLOAD_H(p) VLD16(p)
#define FVSTORE vstore16
#else  // (head 128: one chunk, 8 dims per lane)
#define FDC 1
#define FVW 8
#define FVT float8
#define FVLOAD_H(p) VLD8(p)
#define FVSTORE vstore8
#endif
// -DFA_SPLIT (generation / verification, <= 8 rows): flash-decoding. A row's key blocks are cut into splits of at
// least FA_BPS blocks, at most FA_SMAX of them, a function of that row's own range only (so a row gives the same
// result alone and in a verification launch); each split runs the loop below over its blocks and writes its
// unnormalized output, its max and its sum to part, and attn_fa_gc combines them. One row alone used to be 8
// sub-groups scanning every key: 26B at 14K context, 142 ms of attention per token.
#define FA_SMAX 64
#define FA_BPS 8
inline void fa_split_range(int lo, int hi, int sp, int* b0, int* b1) {  // blocks [b0, b1) of split sp (empty: b0 >= b1)
  const int nb = hi / 16 - lo / 16 + 1, ns = min(FA_SMAX, (nb + FA_BPS - 1) / FA_BPS), per = (nb + ns - 1) / ns;
  *b0 = lo / 16 + sp * per;
  *b1 = min(lo / 16 + (sp + 1) * per, hi / 16 + 1);
}
#ifndef FA_ROWS
#define FA_ROWS 1
#endif
#if FA_ROWS > 1 && defined(K_F16)
#error "FA_ROWS copies f32 K rows as 32-bit words: not with K_F16"
#endif
#if FA_ROWS > 1
#define FA_KS (HD + 1)  // (a key's row stride in local memory: lane = key reads without bank conflicts)
#define FA_KW uint
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_fa_g(__global const float* qf, __global const KC_T* kc, __global const VC_T* vc, int cap, int nh,
                        int nkv, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi
#ifdef FA_SPLIT
                        , int sgrid, __global float* part
#endif
                        ) {
  const int lane = get_sub_group_local_id();
#ifdef FA_SPLIT
  const int hg = get_global_id(1), m = get_global_id(2) / sgrid, sp = get_global_id(2) % sgrid;
#else
  const int hg = get_global_id(1), m = get_global_id(2);
#endif
#if FA_ROWS > 1
  __local FA_KW sk[16 * FA_KS];
  __local VC_RAW sv[16 * HD];
  const int mr = min(m, M_valid - 1);  // (rows past the launch take part in the loads; nothing is stored for them)
#else
  if (m >= M_valid) return;
  const int mr = m;
#endif
  const int pos = pos0 + mr, g = (hg * GQ) / (nh / nkv);
  const int lo = win > 0 ? max(0, pos - win + 1) : 0;
  const int hi = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;
  __global const float* qp = qf + ((size_t)mr * nh + hg * GQ) * HD;
  __global const KC_T* kb = kc + (size_t)g * cap * HD;
  __global const VC_T* vb = vc + (size_t)g * cap * HD;
  FVT o[GQ][FDC];
  float mx[GQ], lp[GQ];
#pragma unroll
  for (int r = 0; r < GQ; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < FDC; ++c) o[r][c] = 0.0f;
  }
#if FA_ROWS > 1
  // the work-group's blocks: its first row's lo to its last row's hi (both grow with the position)
  const int m0 = get_group_id(2) * FA_ROWS, m1 = min(m0 + FA_ROWS, M_valid) - 1, q0 = pos0 + m0, q1 = pos0 + m1;
  const int blo = (win > 0 ? max(0, q0 - win + 1) : 0) / 16, bhi = ((q1 >= nc_lo && q1 <= nc_hi) ? nc_hi : q1) / 16;
  const int tid = get_local_id(2) * 16 + lane;
  for (int b = blo; b <= bhi; ++b) {
    barrier(CLK_LOCAL_MEM_FENCE);
    // (cap is a multiple of 16: a block's 16 keys are 16 consecutive slots, one modulo per block)
    const int bslot = (b * 16) % cap;
    for (int i = tid; i < 16 * HD; i += 16 * FA_ROWS) {
      const int kk = i / HD, d = i % HD;
      const size_t at = (size_t)(bslot + kk) * HD + d;
      sk[kk * FA_KS + d] = ((__global const FA_KW*)kb)[at];
      sv[kk * HD + d] = ((__global const VC_RAW*)vb)[at];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < lo / 16 || b > hi / 16) continue;  // (uniform over the sub-group: its row's own blocks only)
#elif defined(FA_SPLIT)
  int bs0, bs1;
  fa_split_range(lo, hi, sp, &bs0, &bs1);
  if (bs0 >= bs1) return;  // (a split past this row's)
  for (int b = bs0; b < bs1; ++b) {
#else
  for (int b = lo / 16; b <= hi / 16; ++b) {
#endif
    const int t0 = b * 16, p = t0 + lane, t0s = t0 % cap;  // (t0s + j: key t0 + j's slot, cap a multiple of 16)
    const bool valid = p >= lo && p <= hi;
    float s[GQ];
#pragma unroll
    for (int r = 0; r < GQ; ++r) s[r] = 0.0f;
#if FA_ROWS > 1
    __local const KC_T* krow = (__local const KC_T*)sk + lane * FA_KS;
#else
    __global const KC_T* krow = kb + (size_t)(t0s + lane) * HD;
#endif
    for (int c = 0; c < HD / 16; ++c) {
      const float16 kv = valid ? KLD16(krow + 16 * c) : (float16)0.0f;
#pragma unroll
      for (int r = 0; r < GQ; ++r) {
        const float qc = as_float(intel_sub_group_block_read((__global const uint*)(qp + r * HD + 16 * c)));
        float a = s[r];
        a = fma(BC16(qc, 0), kv.s0, a);  a = fma(BC16(qc, 1), kv.s1, a);
        a = fma(BC16(qc, 2), kv.s2, a);  a = fma(BC16(qc, 3), kv.s3, a);
        a = fma(BC16(qc, 4), kv.s4, a);  a = fma(BC16(qc, 5), kv.s5, a);
        a = fma(BC16(qc, 6), kv.s6, a);  a = fma(BC16(qc, 7), kv.s7, a);
        a = fma(BC16(qc, 8), kv.s8, a);  a = fma(BC16(qc, 9), kv.s9, a);
        a = fma(BC16(qc, 10), kv.sa, a); a = fma(BC16(qc, 11), kv.sb, a);
        a = fma(BC16(qc, 12), kv.sc, a); a = fma(BC16(qc, 13), kv.sd, a);
        a = fma(BC16(qc, 14), kv.se, a); a = fma(BC16(qc, 15), kv.sf, a);
        s[r] = a;
      }
    }
#pragma unroll
    for (int r = 0; r < GQ; ++r) {
      const float sv = valid ? s[r] : -INFINITY;
      const float mn = fmax(mx[r], sub_group_reduce_max(sv));
      const float corr = exp(mx[r] - mn);
      const float pr = valid ? exp(sv - mn) : 0.0f;
      lp[r] = lp[r] * corr + pr;
#pragma unroll
      for (int c = 0; c < FDC; ++c) o[r][c] *= corr;
      mx[r] = mn;
      s[r] = pr;
    }
    for (int j = 0; j < 16; ++j) {
      const int pj = t0 + j;
      if (pj < lo || pj > hi) continue;  // (the same for every lane)
#if FA_ROWS > 1
      __local const VC_T* vrow = (__local const VC_T*)sv + j * HD + FVW * lane;
#else
      __global const VC_T* vrow = vb + (size_t)(t0s + j) * HD + FVW * lane;
#endif
#pragma unroll
      for (int c = 0; c < FDC; ++c) {
        const FVT vv = FVLOAD_H(vrow + 16 * FVW * c);
#pragma unroll
        for (int r = 0; r < GQ; ++r) o[r][c] = fma((FVT)sub_group_broadcast(s[r], j), vv, o[r][c]);
      }
    }
  }
  if (m >= M_valid) return;
#pragma unroll
  for (int r = 0; r < GQ; ++r) {
    const float L = sub_group_reduce_add(lp[r]);
#ifdef FA_SPLIT  // (this split's unnormalized output, then its sum and max)
    __global float* pp = part + (((size_t)m * nh + hg * GQ + r) * FA_SMAX + sp) * (HD + 2);
#pragma unroll
    for (int c = 0; c < FDC; ++c) FVSTORE(o[r][c], 0, pp + 16 * FVW * c + FVW * lane);
    if (lane == 0) {
      pp[HD] = L;
      pp[HD + 1] = mx[r];
    }
#else
#pragma unroll
    for (int c = 0; c < FDC; ++c)
      FVSTORE(o[r][c] / L, 0, out + ((size_t)m * nh + hg * GQ + r) * HD + 16 * FVW * c + FVW * lane);
#endif
  }
}
// The FA_SPLIT partials of each (row, head) combined: out = sum_s o_s e^(mx_s - mx) / sum_s L_s e^(mx_s - mx), the
// splits in order. grid (HD, nh, M).
__kernel void attn_fa_gc(__global const float* part, int nh, int pos0, int M_valid, __global float* out, int win,
                         int nc_lo, int nc_hi
                         ) {
  const int d = get_global_id(0), h = get_global_id(1), m = get_global_id(2);
  if (m >= M_valid) return;
  const int pos = pos0 + m;
  const int lo = win > 0 ? max(0, pos - win + 1) : 0;
  const int hi = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;
  __global const float* pp = part + ((size_t)m * nh + h) * FA_SMAX * (HD + 2);
  float mx = -INFINITY;
  for (int sp = 0; sp < FA_SMAX; ++sp) {
    int b0, b1;
    fa_split_range(lo, hi, sp, &b0, &b1);
    if (b0 >= b1) break;
    mx = fmax(mx, pp[(size_t)sp * (HD + 2) + HD + 1]);
  }
  float num = 0.0f, den = 0.0f;
  for (int sp = 0; sp < FA_SMAX; ++sp) {
    int b0, b1;
    fa_split_range(lo, hi, sp, &b0, &b1);
    if (b0 >= b1) break;
    __global const float* q = pp + (size_t)sp * (HD + 2);
    const float w = exp(q[HD + 1] - mx);
    num = fma(q[d], w, num);
    den = fma(q[HD], w, den);
  }
  out[((size_t)m * nh + h) * HD + d] = num / den;
}
#undef FDC
#undef FVW
#undef FVT
#undef FVLOAD_H
#undef FVSTORE
#endif  // FA_G

// attn_fa_g for prompt chunks of the wide-head layers (-DFA_GH, two-pass mode; Gemma 4's global layers, head 512, 8
// q heads per kv head): the same results bit for bit with each key row read once per row instead of once per pair of
// q heads. A work-group = one row x one kv head, FS sub-groups. Per step the FS sub-groups take FS consecutive blocks
// of 16 keys, one each, and score it for all HPG q heads (lane = key, each score the same fma chain over the dims as
// attn_fa_g), into local memory. Then every sub-group runs the blocks' softmax updates in order for all heads (the
// same operations, so the same values in each) and P.V for its HD / FS dims (lane = HD / FS / 16 dims). attn_fa_g
// holds HD / 16 output floats per lane per (row, head), which limited it to 2 heads per sub-group.
// grid (16 * FS, nkv, M), local (16 * FS, 1, 1).
#ifdef FA_GH
#define GD (HD / FS / 16)  // output dims per lane
#ifndef GR
#define GR 1  // query rows per work-group: each key / value row read once for GR rows x HPG heads
#endif
#define GRH (GR * HPG)
#ifndef KB
#define KB 1  // blocks of 16 keys per sub-group and step: each q load serves KB key rows
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_fa_gh(__global const float* qf, __global const KC_T* kc, __global const VC_T* vc, int cap, int nh,
                         int nkv, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi
                         ) {
  __local float sc[FS * KB][GRH][16];
  const int lane = get_sub_group_local_id(), sgi = get_sub_group_id();
  const int g = get_group_id(1), m0 = get_group_id(2) * GR;
  // each row's key range [lo, hi]; the work-group walks the blocks of their union
  int lor[GR], hir[GR];
#pragma unroll
  for (int r = 0; r < GR; ++r) {
    const int pos = pos0 + min(m0 + r, M_valid - 1);
    lor[r] = win > 0 ? max(0, pos - win + 1) : 0;
    hir[r] = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;
    if (m0 + r >= M_valid) lor[r] = 1, hir[r] = 0;  // (padding row: no keys)
  }
  int lo = lor[0], hi = hir[0];
#pragma unroll
  for (int r = 1; r < GR; ++r)
    if (hir[r] >= lor[r]) lo = min(lo, lor[r]), hi = max(hi, hir[r]);
  __global const KC_T* kb = kc + (size_t)g * cap * HD;
  __global const VC_T* vb = vc + (size_t)g * cap * HD;
  float mx[GRH], lp[GRH];
#if GD == 4
  float4 o[GRH];
#elif GD == 8
  float8 o[GRH];
#else
  float16 o[GRH];
#endif
#pragma unroll
  for (int h = 0; h < GRH; ++h) {
    mx[h] = -INFINITY;
    lp[h] = 0.0f;
    o[h] = 0.0f;
  }
  const int b0 = lo / 16, b1 = hi / 16;
  for (int bs = b0; bs <= b1; bs += FS * KB) {
    {  // scores of blocks bs + KB sgi + kk
      __global const KC_T* krow[KB];
      bool any[KB];
      float s[KB][GRH];
#pragma unroll
      for (int kk = 0; kk < KB; ++kk) {
        const int b = bs + KB * sgi + kk, p = b * 16 + lane;
        any[kk] = b <= b1 && p >= lo && p <= hi;
        // (head 512: one modulo per block, cap a multiple of 16; 26B, 13,921 tokens: global attention 16.4 -> 15.3 s.
        //  Head 256 keeps one per key: per block it was 5 % slower there; the same slots either way)
#if HD >= 512
        krow[kk] = kb + (size_t)((b * 16) % cap + lane) * HD;
#else
        krow[kk] = kb + (size_t)(p % cap) * HD;
#endif
#pragma unroll
        for (int h = 0; h < GRH; ++h) s[kk][h] = 0.0f;
      }
      for (int c = 0; c < HD / 16; ++c) {
        float16 kv[KB];
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) kv[kk] = any[kk] ? KLD16(krow[kk] + 16 * c) : (float16)0.0f;
#pragma unroll
        for (int h = 0; h < GRH; ++h) {
          const int r = h / HPG, m = min(m0 + r, M_valid - 1);
          const float qc = as_float(intel_sub_group_block_read((__global const uint*)(qf + ((size_t)m * nh + g * HPG + h % HPG) * HD + 16 * c)));
#pragma unroll
          for (int kk = 0; kk < KB; ++kk) {
            float a = s[kk][h];
            a = fma(BC16(qc, 0), kv[kk].s0, a);  a = fma(BC16(qc, 1), kv[kk].s1, a);
            a = fma(BC16(qc, 2), kv[kk].s2, a);  a = fma(BC16(qc, 3), kv[kk].s3, a);
            a = fma(BC16(qc, 4), kv[kk].s4, a);  a = fma(BC16(qc, 5), kv[kk].s5, a);
            a = fma(BC16(qc, 6), kv[kk].s6, a);  a = fma(BC16(qc, 7), kv[kk].s7, a);
            a = fma(BC16(qc, 8), kv[kk].s8, a);  a = fma(BC16(qc, 9), kv[kk].s9, a);
            a = fma(BC16(qc, 10), kv[kk].sa, a); a = fma(BC16(qc, 11), kv[kk].sb, a);
            a = fma(BC16(qc, 12), kv[kk].sc, a); a = fma(BC16(qc, 13), kv[kk].sd, a);
            a = fma(BC16(qc, 14), kv[kk].se, a); a = fma(BC16(qc, 15), kv[kk].sf, a);
            s[kk][h] = a;
          }
        }
      }
#pragma unroll
      for (int kk = 0; kk < KB; ++kk) {
        const int b = bs + KB * sgi + kk, p = b * 16 + lane;
#pragma unroll
        for (int h = 0; h < GRH; ++h) {
          const int r = h / HPG;
          sc[KB * sgi + kk][h][lane] = b <= b1 && p >= lor[r] && p <= hir[r] ? s[kk][h] : -INFINITY;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < FS * KB && bs + i <= b1; ++i) {  // the blocks in order, as attn_fa_g takes them
      const int t0 = (bs + i) * 16, p = t0 + lane, t0s = t0 % cap;  // (t0s + j: key t0 + j's slot)
      float pr[GRH];
#pragma unroll
      for (int h = 0; h < GRH; ++h) {
        const int r = h / HPG;
        const bool valid = p >= lor[r] && p <= hir[r];
        const float sv = sc[i][h][lane];
        const float mn = fmax(mx[h], sub_group_reduce_max(sv));
        // (a block before a row's first key leaves it untouched: no exp(-inf + inf))
        const float corr = mn == -INFINITY ? 1.0f : exp(mx[h] - mn);
        pr[h] = valid ? exp(sv - mn) : 0.0f;
        lp[h] = lp[h] * corr + pr[h];
        o[h] *= corr;
        mx[h] = mn;
      }
      for (int j = 0; j < 16; ++j) {
        const int pj = t0 + j;
        if (pj < lo || pj > hi) continue;  // (the same for every lane)
#if HD >= 512
        __global const VC_T* vrow = vb + (size_t)(t0s + j) * HD + sgi * (HD / FS) + GD * lane;
#else
        __global const VC_T* vrow = vb + (size_t)(pj % cap) * HD + sgi * (HD / FS) + GD * lane;
#endif
#if GD == 4
        const float4 vv = VLD4(vrow);
#pragma unroll
        for (int h = 0; h < GRH; ++h) o[h] = fma((float4)sub_group_broadcast(pr[h], j), vv, o[h]);
#elif GD == 8
        const float8 vv = VLD8(vrow);
#pragma unroll
        for (int h = 0; h < GRH; ++h) o[h] = fma((float8)sub_group_broadcast(pr[h], j), vv, o[h]);
#else
        const float16 vv = VLD16(vrow);
#pragma unroll
        for (int h = 0; h < GRH; ++h) o[h] = fma((float16)sub_group_broadcast(pr[h], j), vv, o[h]);
#endif
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (int h = 0; h < GRH; ++h) {
    const int m = m0 + h / HPG;
    const float L = sub_group_reduce_add(lp[h]);
    if (m < M_valid) {
      __global float* op = out + ((size_t)m * nh + g * HPG + h % HPG) * HD + sgi * (HD / FS) + GD * lane;
#if GD == 4
      vstore4(o[h] / L, 0, op);
#elif GD == 8
      vstore8(o[h] / L, 0, op);
#else
      vstore16(o[h] / L, 0, op);
#endif
    }
  }
}
#undef GD
#undef GRH
#undef KB
#endif  // FA_GH

// Single-token rows: x[m] = (x[m] + rmsnorm(t[m]) * w) * s (as add_rmsnorm), then the next GEMV input from the new
// x[m]: rmsnorm(x[m]) * w2 (or x[m] itself with w2_norm = 0) quantized like rmsnorm_quant. The new values stay in
// registers for the second step (sub-group j, lane l: elements 256 (j + 16 i) + 16 l .. +15; K <= 16384): reading
// them back from x could return the pre-update values from L1. grid (256, M_valid), local (256, 1).
#ifndef ANQ_MAX  // (superblocks per sub-group: the engine passes ceil(K / 256 / 16))
#define ANQ_MAX 4
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void add_norm_quant(__global float* x, __global const float* t, __global const float* w, int K, float eps,
                             float s, __global const float* w2, int w2_norm, int M, __global char* aq,
                             __global float* asc, __global ushort* absum, int act32) {
  const int m = get_group_id(1), lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();
  __global const float* tr = t + (size_t)m * K;
  __global float* xr = x + (size_t)m * K;
  float ss = 0.0f;
  for (int i = get_local_id(0); i < K; i += get_local_size(0)) ss += tr[i] * tr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  float xv[ANQ_MAX][16], s2 = 0.0f;
#pragma unroll
  for (int j = 0; j < ANQ_MAX; ++j) {
    const int sb = sg + j * nsg;
    if (sb < K / 256) {
      const int base = sb * 256 + 16 * lane;
      float xo[16], to[16], wo[16];
      LOADF(xo, xr + base, 16)
      LOADF(to, tr + base, 16)
      LOADF(wo, w + base, 16)
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float v = (xo[i] + to[i] * r * wo[i]) * s;
        xv[j][i] = v;
        s2 += v * v;
      }
      vstore8((float8)(xv[j][0], xv[j][1], xv[j][2], xv[j][3], xv[j][4], xv[j][5], xv[j][6], xv[j][7]), 0, xr + base);
      vstore8((float8)(xv[j][8], xv[j][9], xv[j][10], xv[j][11], xv[j][12], xv[j][13], xv[j][14], xv[j][15]), 0,
              xr + base + 8);
    }
  }
  s2 = work_group_reduce_add(s2);
  const float r2 = w2_norm ? rsqrt(s2 / K + eps) : 1.0f;
#pragma unroll
  for (int j = 0; j < ANQ_MAX; ++j) {
    const int sb = sg + j * nsg;
    if (sb < K / 256) {
      const int base = sb * 256 + 16 * lane;
      float v[16], w2v[16];
      if (w2_norm) { LOADF(w2v, w2 + base, 16) }
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = w2_norm ? xv[j][i] * r2 * w2v[i] : xv[j][i];
      quant_store16(v, sb, m, M, K, aq, asc, absum, act32);
    }
  }
}

// x[m] = (x[m] + rmsnorm(t[m]) * w) * s: Gemma's post-attention / post-FFN norm, residual add and layer output
// scale. grid (256, M_valid), local (256, 1).
__kernel void add_rmsnorm(__global float* x, __global const float* t, __global const float* w, int K, float eps,
                          float s) {
  const int m = get_group_id(1), l = get_local_id(0);
  __global const float* tr = t + (size_t)m * K;
  __global float* xr = x + (size_t)m * K;
  float ss = 0.0f;
  for (int i = l; i < K; i += get_local_size(0)) ss += tr[i] * tr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  for (int i = l; i < K; i += get_local_size(0)) xr[i] = (xr[i] + tr[i] * r * w[i]) * s;
}

// Gemma 4 per-layer embeddings (E2B / E4B). pl rows hold the per-layer projection of the token embedding
// (n_layer groups of NPL values); tok rows the scaled per-layer token embeddings. In place:
// pl[g] = (rmsnorm(pl[g] * pre_scale) * w + tok[g]) * out_scale. grid (16 * n_layer, M), local (16, 1).
#define NPL 256
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ple_combine(__global float* pl, __global const float* tok, __global const float* w, int n_layer,
                          int M_valid, float pre_scale, float eps, float out_scale) {
  const int m = get_global_id(1), l = get_global_id(0) / SG, lane = get_sub_group_local_id();
  if (m >= M_valid) return;
  __global float* p = pl + ((size_t)m * n_layer + l) * NPL + lane * 16;
  __global const float* t = tok + ((size_t)m * n_layer + l) * NPL + lane * 16;
  float v[16], ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) { v[i] = p[i] * pre_scale; ss += v[i] * v[i]; }
  ss = sub_group_reduce_add(ss);
  const float r = rsqrt(ss / NPL + eps);
#pragma unroll
  for (int i = 0; i < 16; ++i) p[i] = (v[i] * r * w[lane * 16 + i] + t[i]) * out_scale;
}

// Per-layer input gate of layer `il`: v = gelu(g[m]) * pl[m][il], quantized for the following projection
// (K = NPL, one superblock). grid (16, M), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ple_gate_quant(__global const float* g, __global const float* pl, int il, int n_layer, int M,
                             int M_valid, __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_global_id(1), lane = get_sub_group_local_id();
  float v[16];
  const bool valid = m < M_valid;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane * 16 + i;
    v[i] = valid ? gelu_tanh(g[(size_t)m * NPL + e]) * pl[((size_t)m * n_layer + il) * NPL + e] : 0.0f;
  }
  quant_store16(v, 0, m, M, NPL, aq, asc, absum, act32);
}

// Logit softcap of rows r of C (row stride ld, from column off): v = cap * tanh(v / cap), as the head GEMVs' HEAD_OUT
// (two-pass mode: after both passes). grid (ceil(N / 256) * 256, rows).
__kernel void softcap_rows(__global float* C, int ld, int off, int N, float cap) {
  const int n = get_global_id(0), r = get_global_id(1);
  if (n >= N) return;
  __global float* p = C + (size_t)r * ld + off + n;
  *p = cap * tanh(*p * (1.0f / cap));
}

