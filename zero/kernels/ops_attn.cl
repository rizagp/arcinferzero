// Part of the non-GEMM kernels (zero/kernels/ops*.cl, KernelParts in zero/clrt.h): built after the parts before
// it, whose definitions it uses. Qwen3.5 / 3.6 / 3.8 attention in int8: prep (RoPE, int8 q / K / V), the prompt kernels (attn_fa, attn_q8,
// attn_ps) and the decode / verification ones (attn_split, attn_decode, attn_dsplit, attn_combine), MTP helpers.

// Full-attention prep: per-head RMSNorm of q and k, NeoX RoPE on the first n_rot dims, q scaled by kq_scale.
// qkv row: [q|gate interleaved per head (nh*512) | k (nkv*256) | v (nkv*256)], row stride ldq.
// Writes q to qout [M][nh][256] and k, v (f16) to the cache [nkv][ctx][256] at pos0 + m. Also writes int8 copies of
// q (qi [M][nh][256], scale qsc [M][nh]) and k (kq [nkv][ctx][256], scale ksc [nkv][ctx]) for attn_q8.
// One sub-group per (head slot, token): slots [0, nh) are q heads, [nh, nh+nkv) k/v heads.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_prep(__global const float* qkv, int ldq, int nh, int nkv, __global const float* qn,
                        __global const float* kn, int M_valid, int pos0, float eps, float theta_scale, int n_rot,
                        float kq_scale, __global float* qout, __global half* kc, __global half* vc, int ctx,
                        __global char* qi, __global float* qsc, __global uint* kq, __global float* ksc,
                        __global uchar* vq, __global float* vsc, int f32_attn, int rope_shift,
                        __global const int* rpos, int use_rpos, int4 sect) {
  const int m = get_global_id(1), hs = get_global_id(0) / SG, lane = get_sub_group_local_id();
  if (m >= M_valid) return;
  const bool isq = hs < nh;
  __global const float* row = qkv + (size_t)m * ldq;
  __global const float* src = isq ? row + hs * 512 : row + nh * 512 + (hs - nh) * 256;
  __global const float* nw = isq ? qn : kn;
  float v[16];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = src[16 * lane + i];
    ss += v[i] * v[i];
  }
  ss = sub_group_reduce_add(ss);
  const float r = rsqrt(ss / 256.0f + eps);
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] *= r * nw[16 * lane + i];

  // RoPE (n_rot = 64): pairs (e, e + n_rot/2) live in lanes l and l ^ 2 for l < 4. The rotary position is the cache
  // position plus rope_shift (text after images, whose tokens advance the rotary position by max(grid w, h) only);
  // with use_rpos the rows carry their own (t, h, w) positions (image tokens, Qwen3-VL interleaved M-RoPE: pair fi
  // takes h when fi % 3 == 1 and fi < 3 * sect.y, w when fi % 3 == 2 and fi < 3 * sect.z, else t)
  const int pos = pos0 + m, half_rot = n_rot / 2, rp_text = pos + rope_shift;
  float pv[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) pv[i] = intel_sub_group_shuffle_xor(v[i], 2);
  if (16 * lane < n_rot) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int e = 16 * lane + i, fi = e % half_rot;
      int rp = rp_text;
      if (use_rpos) {
        const int sector = fi % (sect.x + sect.y + sect.z + sect.w);
        const int comp = (sector % 3 == 1 && sector < 3 * sect.y) ? 1 : (sector % 3 == 2 && sector < 3 * sect.z) ? 2 : 0;
        rp = rpos[3 * m + comp];
      }
      const float theta = (float)rp * pow(theta_scale, (float)fi);
      const float c = cos(theta), sn = sin(theta);
      v[i] = e < half_rot ? v[i] * c - pv[i] * sn : v[i] * c + pv[i] * sn;
    }
  }
  if (isq)
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] *= kq_scale;
  const int kvh = hs - nh;
  __global const float* vs = row + nh * 512 + nkv * 256 + kvh * 256 + 16 * lane;
  if (f32_attn) {  // f32 q and f16 K/V rows for attn_fa
    if (isq) {
#pragma unroll
      for (int i = 0; i < 16; ++i) qout[((size_t)m * nh + hs) * 256 + 16 * lane + i] = v[i];
    } else {
      __global half* kd = kc + ((size_t)kvh * ctx + pos) * 256 + 16 * lane;
      __global half* vd = vc + ((size_t)kvh * ctx + pos) * 256 + 16 * lane;
#pragma unroll
      for (int i = 0; i < 16; ++i) { vstore_half(v[i], i, kd); vstore_half(vs[i], i, vd); }
    }
    return;
  }
  // int8 copy of the row (absmax scale per 64-dim block = 4 lanes)
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, o));  // scale block
  float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  uint pk[4] = {0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 16; ++i) pk[i / 4] |= (uint)(convert_int_rte(v[i] * inv) & 0xFF) << (8 * (i % 4));
  if (isq) {
    vstore4((uint4)(pk[0], pk[1], pk[2], pk[3]), 0, (__global uint*)(qi + ((size_t)m * nh + hs) * 256 + 16 * lane));
    if (lane % (16 / SB) == 0) qsc[((size_t)m * nh + hs) * SB + lane / (16 / SB)] = amax / 127.0f;
    return;
  }
  // K and V of this position go to 16-key blocks: K as [word 64][key 16] uints, V as [dim 256][key 16] bytes
  const size_t blk = (size_t)kvh * (ctx / 16) + pos / 16;
  const int kk = pos % 16;
#pragma unroll
  for (int c = 0; c < 4; ++c) kq[(blk * 64 + 4 * lane + c) * 16 + kk] = pk[c];
  float vv[16], vmax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) vv[i] = vs[i];
#pragma unroll
  for (int i = 0; i < 16; ++i) vmax = fmax(vmax, fabs(vv[i]));
#pragma unroll
  for (int o = 1; o < 16 / SB; o <<= 1) vmax = fmax(vmax, intel_sub_group_shuffle_xor(vmax, o));
  inv = vmax > 0.0f ? 127.0f / vmax : 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) vq[(blk * 256 + 16 * lane + i) * 16 + kk] = (uchar)convert_int_rte(vv[i] * inv);
  if (lane % (16 / SB) == 0) {  // scales [block][scale block][key]
    ksc[(blk * SB + lane / (16 / SB)) * 16 + kk] = amax / 127.0f;
    vsc[(blk * SB + lane / (16 / SB)) * 16 + kk] = vmax / 127.0f;
  }
}

// Flash-style causal attention. One sub-group per (group of GQ q heads sharing a kv head, token m).
// Keys go in blocks of 16: scores with lane = key (q broadcast from lanes), then P*V with lane = 16 head dims.
// Work-group = 16 consecutive tokens of one head group, so their K/V streams hit in L1.
// K/V rows beyond pos must be finite (the cache is zero-initialized); ctx must be a multiple of 16.
// GQ (q heads per sub-group) must divide nh/nkv. grid (16, nh/GQ, M).
#ifndef GQ
#define GQ 4
#endif
#define BC16(x, i) intel_sub_group_shuffle((x), (i))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_fa(__global const float* q, __global const half* kc, __global const half* vc, int ctx, int nh,
                      int nkv, int pos0, int M_valid, __global float* out) {
  const int lane = get_sub_group_local_id();
  const int hg = get_global_id(1), m = get_global_id(2);
  if (m >= M_valid) return;
  const int pos = pos0 + m, g = (hg * GQ) / (nh / nkv);
  __global const float* qp = q + ((size_t)m * nh + hg * GQ) * 256;
  __global const half* kb = kc + (size_t)g * ctx * 256;
  __global const half* vb = vc + (size_t)g * ctx * 256;
  float16 o[GQ];
  float mx[GQ], lp[GQ];
#pragma unroll
  for (int r = 0; r < GQ; ++r) { o[r] = 0.0f; mx[r] = -INFINITY; lp[r] = 0.0f; }

  const int nblk = pos / 16 + 1;
  for (int b = 0; b < nblk; ++b) {
    const int t0 = b * 16;
    float s[GQ];
#pragma unroll
    for (int r = 0; r < GQ; ++r) s[r] = 0.0f;
    __global const half* krow = kb + (size_t)(t0 + lane) * 256;
#pragma unroll 2
    for (int c = 0; c < 16; ++c) {
      const float16 kv = vload_half16(0, krow + 16 * c);
#pragma unroll
      for (int r = 0; r < GQ; ++r) {
        const float qc = as_float(intel_sub_group_block_read((__global const uint*)(qp + r * 256 + 16 * c)));
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
    const bool valid = t0 + lane <= pos;
#pragma unroll
    for (int r = 0; r < GQ; ++r) {
      const float sv = valid ? s[r] : -INFINITY;
      const float mn = fmax(mx[r], sub_group_reduce_max(sv));
      const float corr = exp(mx[r] - mn);
      const float p = valid ? exp(sv - mn) : 0.0f;
      lp[r] = lp[r] * corr + p;
      o[r] *= corr;
      mx[r] = mn;
      s[r] = p;
    }
#pragma unroll 2
    for (int j = 0; j < 16; ++j) {
      const float16 vv = vload_half16(0, vb + (size_t)(t0 + j) * 256 + 16 * lane);
#pragma unroll
      for (int r = 0; r < GQ; ++r) o[r] = fma((float16)sub_group_broadcast(s[r], j), vv, o[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < GQ; ++r) {
    const float L = sub_group_reduce_add(lp[r]);
    vstore16(o[r] / L, 0, out + ((size_t)m * nh + hg * GQ + r) * 256 + 16 * lane);
  }
}

// y = attn_out * sigmoid(gate), gate from the q|gate interleaved projection; quantized (K = nh*256).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_gate_quant(__global const float* att, __global const float* qkv, int ldq, int K, int M,
                              int M_valid, __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  for (int s = get_sub_group_id(); s < K / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane, h = base / 256, d = base % 256;
    float v[16];
    if (valid) {  // (the row flag tested once, vector loads; the same arithmetic)
      const float16 ga = vload16(0, qkv + (size_t)m * ldq + h * 512 + 256 + d), aa = vload16(0, att + (size_t)m * K + base);
      const float gv[16] = {ga.s0, ga.s1, ga.s2, ga.s3, ga.s4, ga.s5, ga.s6, ga.s7, ga.s8, ga.s9, ga.sa, ga.sb, ga.sc, ga.sd, ga.se, ga.sf};
      const float av[16] = {aa.s0, aa.s1, aa.s2, aa.s3, aa.s4, aa.s5, aa.s6, aa.s7, aa.s8, aa.s9, aa.sa, aa.sb, aa.sc, aa.sd, aa.se, aa.sf};
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = av[i] / (1.0f + exp(-gv[i]));
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    }
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
  }
}

// Causal attention, int8 throughout. Scores S = (q8 . k8) * qscale * kscale via DP4a, f32 online softmax, and
// P*V via DP4a as well: per 16-key tile, P * vscale (per key) is quantized to 8 bits with one scale per q row and
// dotted with the int8 V bytes (SageAttention-style block quantization of P). K and V live in 16-key blocks
// (K: [HD/4 words][16 keys] uints, V: [HD dims][16 keys] bytes; written by attn_prep*), so a tile is two
// contiguous copies of 4*HD uints into SLM shared by the whole work-group.
// HD = head dim (256 or 512). Optional sliding window `win` (a query at p sees keys p-win+1..p; 0 = unlimited)
// and cache capacity `cap` (positions; key position k lives in slot k % cap, so cap < context = ring buffer).
// Work-group = TPW consecutive tokens x NSP sub-groups; each sub-group handles HS of the HPG q heads that share
// the kv head, over all HD dims (lane l owns dims 16*i + l). grid (ceil(M/TPW) * 16*NSP*TPW, nkv),
// local (16*NSP*TPW, 1). Cache entries never written must be finite (zero-initialized).
#ifndef HPG
#define HPG 4
#endif
#ifndef HS
#define HS (HPG / 2)
#endif
#define NSP (HPG / HS)
#ifndef TPW
#define TPW 8
#endif
#ifndef HD
#define HD 256
#endif
// K/V tiles in SLM: a single buffer (two barriers per tile) leaves room for twice the work-groups per Xe-core of a
// double buffer (one barrier per tile, -DDOUBLE_BUF): HD=512 decode 1.5x faster, prefill the same or faster.
#ifdef DOUBLE_BUF
#define TILE_BUFS 2
#else
#define TILE_BUFS 1
#endif
#define KW (HD / 4)      // int8x4 words per key row
#define KC (HD / 64)     // q words per lane
#define DPL (HD / 16)    // output dims per lane
#define BLK_U (4 * HD)   // uints per 16-key block of K (and of V)

// Packs P (lane = key, values 0..255) of a 16-key tile into 4 words held by every lane: word w = keys 4w..4w+3.
inline uint4 pack_p(int q) {
  uint x = (uint)(q & 0xFF) << (8 * (get_sub_group_local_id() & 3));
  x |= intel_sub_group_shuffle_xor(x, 1);
  x |= intel_sub_group_shuffle_xor(x, 2);
  return (uint4)(intel_sub_group_shuffle(x, 0), intel_sub_group_shuffle(x, 4), intel_sub_group_shuffle(x, 8),
                 intel_sub_group_shuffle(x, 12));
}

#ifdef PRECISE_EXP
#define EXP2 exp2
#else
#define EXP2 native_exp2
#endif
// Scales: q [row][head][SB], K and V [cache block][SB][16 keys], SB scale blocks per row (-DSB): 1 = one per row
// (Qwen), KC = one per 64-dim block (Gemma 4, whose V rows have a few large dims that a per-row scale would let
// swamp the rest). The GEMM-like loops always go over the KC 64-dim chunks of a row.
// Scores of a 16-key tile for HS q heads (lane = key): int8 DP4a sums scaled by q and K scales -> sacc.
// KW_EXPR reads K word w of chunk c for this lane's key; sob = scale index of (tile, block 0, this lane).
#if SB == 1 && HS >= 4
#define QK_TILE(KW_EXPR, sob)                                                           \
  {                                                                                     \
    int acc[HS];                                                                        \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) acc[r] = 0;                          \
    _Pragma("unroll") for (int c = 0; c < KC; ++c)                                      \
      _Pragma("unroll") for (int w = 0; w < 16; ++w) {                                  \
        const uint kw = KW_EXPR;                                                        \
        _Pragma("unroll") for (int r = 0; r < HS; ++r)                                  \
          acc[r] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[r][c], w), kw);     \
      }                                                                                 \
    const float ks = ksc[sob];                                                          \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = (float)acc[r] * qs[r][0] * ks; \
  }
#elif SB == 1
// (HS <= 3: two integer sums per head, even and odd words, added before the conversion: exact, so the same scores,
//  with twice the independent DP4a chains; 27B (HS 3) 11.6K prompt attention 5.09-5.14 -> 4.61-4.72 s. HS 4 (35B):
//  no faster, one chain per head kept)
#define QK_TILE(KW_EXPR, sob)                                                           \
  {                                                                                     \
    int acc[HS], acc1[HS];                                                              \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) acc[r] = acc1[r] = 0;                \
    _Pragma("unroll") for (int c = 0; c < KC; ++c)                                      \
      _Pragma("unroll") for (int w = 0; w < 16; w += 2) {                               \
        {                                                                               \
          const uint kw = KW_EXPR;                                                      \
          _Pragma("unroll") for (int r = 0; r < HS; ++r)                                \
            acc[r] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[r][c], w), kw);   \
        }                                                                               \
        {                                                                               \
          const int w0_ = w;                                                            \
          const int w = w0_ + 1;                                                        \
          const uint kw = KW_EXPR;                                                      \
          _Pragma("unroll") for (int r = 0; r < HS; ++r)                                \
            acc1[r] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[r][c], w), kw);  \
        }                                                                               \
      }                                                                                 \
    const float ks = ksc[sob];                                                          \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = (float)(acc[r] + acc1[r]) * qs[r][0] * ks; \
  }
#else
#define QK_TILE(KW_EXPR, sob)                                                           \
  _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = 0.0f;                        \
  _Pragma("unroll") for (int c = 0; c < KC; ++c) {                                      \
    int acc[HS];                                                                        \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) acc[r] = 0;                          \
    _Pragma("unroll") for (int w = 0; w < 16; ++w) {                                    \
      const uint kw = KW_EXPR;                                                          \
      _Pragma("unroll") for (int r = 0; r < HS; ++r)                                    \
        acc[r] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[r][c], w), kw);       \
    }                                                                                   \
    const float ks = ksc[(sob) + c * 16];                                               \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = fma((float)acc[r], qs[r][c] * ks, sacc[r]); \
  }
#endif

// Online-softmax step for one q head over a 16-key tile (lane = key). Scores are in the log2 domain (log2(e) is
// folded into the q scales). The accumulators are rescaled only when the running max grows (a uniform branch).
// Leaves pk = this key's weight relative to the running max and pb = exp2(tile max - running max) >= pk.
#define SOFTMAX_TILE(r, sv, valid)                                                      \
  {                                                                                     \
    const float tm = sub_group_reduce_max(sv);                                          \
    if (tm > mx[r]) {                                                                   \
      const float corr = EXP2(mx[r] - tm);                                              \
      lp[r] *= corr;                                                                    \
      _Pragma("unroll") for (int i = 0; i < DPL; ++i) o[r][i] *= corr;                  \
      mx[r] = tm;                                                                       \
    }                                                                                   \
    pk[r] = (valid) ? EXP2((sv) - mx[r]) : 0.0f;                                        \
    lp[r] += pk[r];                                                                     \
    pb[r] = EXP2(tm - mx[r]);                                                           \
  }

// P is quantized to 0..255 (unsigned x signed DP4a) or, with -DP7, to 0..127 (signed DP4a).
#ifdef P7
#define PQMAX 127.0f
#define PDOT(p, v) dot_4x8packed_ss_int(p, v)
#define PDOT_ACC(p, v, a) dot_acc_sat_4x8packed_ss_int(p, v, a)
#else
#define PQMAX 255.0f
#define PDOT(p, v) dot_4x8packed_us_int(p, v)
#define PDOT_ACC(p, v, a) dot_acc_sat_4x8packed_us_int(p, v, a)
#endif
// o[r][i] += P . V[dim 16*i + lane] for the HS heads; vt points at the tile's V block ([HD dims][4 words]).
// P * vscale(key, block) is quantized to 8 bits against the bound pb * max vscale of the tile and dotted with the
// int8 V bytes (SageAttention-style block quantization of P); with SB = KC once per 64-dim block (dims 16*i..,
// i in 4c..4c+3).
// The HS divisions run once (lane r for head r; the bounds are equal in every lane), and the dot products
// accumulate inside the DP4a with the heads interleaved (|sum| <= 16*255*128: never saturates). Same values;
// 4B prompt attention at 64K 936 -> 777 ms per layer, spill 576 -> 0 bytes (16 Oct).
#define PV_PACK(c, valid, sob)                                                          \
  const float vs = (valid) ? vsc[(sob) + (c) * 16] : 0.0f, vmt = sub_group_reduce_max(vs); \
  uint4 pw[HS];                                                                         \
  float ps[HS];                                                                         \
  {                                                                                     \
    float pmv = 0.0f;                                                                   \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) pmv = lane == r ? pb[r] * vmt : pmv; \
    const float invv = pmv > 0.0f ? PQMAX / pmv : 0.0f;                                 \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) {                                    \
      pw[r] = pack_p(convert_int_rte(pk[r] * vs * sub_group_broadcast(invv, r)));       \
      ps[r] = sub_group_broadcast(pmv, r) * (1.0f / PQMAX);                             \
    }                                                                                   \
  }
#define PV_DIM(vt, i)                                                                   \
  {                                                                                     \
    const uint4 v = vt[16 * (i) + lane];                                                \
    int a[HS];                                                                          \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT(pw[r].x, v.x);           \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT_ACC(pw[r].y, v.y, a[r]); \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT_ACC(pw[r].z, v.z, a[r]); \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) a[r] = PDOT_ACC(pw[r].w, v.w, a[r]); \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) o[r][i] = fma((float)a[r], ps[r], o[r][i]);\
  }
#if SB == 1
#define PV_TILE(vt, sob, valid)                                                         \
  {                                                                                     \
    PV_PACK(0, valid, sob)                                                              \
    _Pragma("unroll") for (int i = 0; i < DPL; ++i) PV_DIM(vt, i)                       \
  }
#else
#define PV_TILE(vt, sob, valid)                                                         \
  _Pragma("unroll") for (int c = 0; c < KC; ++c) {                                      \
    PV_PACK(c, valid, sob)                                                              \
    _Pragma("unroll") for (int ii = 0; ii < 4; ++ii) PV_DIM(vt, 4 * c + ii)             \
  }
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
#ifdef QSA  // Flash Next sparse attention: a query sees the keys of its selected blocks and of its tail block
#define QSA_ARGS , __global const uint* qmask, int qmw
// (block size QR; mask row `row`: bit b of word b / 32 = block b)
#define QSA_KEEP(row, qpos, key) ((key) >= ((qpos) + 1) / QR * QR || ((qmask[(size_t)(row) * qmw + ((key) / QR) / 32] >> (((key) / QR) & 31)) & 1u))
#else
#define QSA_ARGS
#endif
__kernel void attn_q8(__global const char* qi, __global const float* qsc, __global const uint* kq,
                      __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                      int nh, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi QSA_ARGS) {
  // double-buffered tile: [K BLK_U uints | V BLK_U uints]; the next tile is fetched into registers while the current
  // one is processed. The per-key scales are read by every sub-group straight from the cache (L1): keeping them out
  // of SLM leaves an HD=512 tile pair at exactly 32 KB, so two work-groups fit an Xe-core's 64 KB.
  __local uint tile[TILE_BUFS][2 * BLK_U];
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16, lsz = 16 * NSP * TPW;
  const int tl = sg / NSP, hh = sg % NSP, g = get_group_id(1);
  const int m = get_group_id(0) * TPW + tl;
  const bool active = m < M_valid;
  const int pos = pos0 + m, h0 = g * HPG + hh * HS;
  const int first = pos0 + (int)get_group_id(0) * TPW;
  const int last = pos0 + min(M_valid - 1, (int)get_group_id(0) * TPW + TPW - 1);
  const int lo = win > 0 ? pos - win + 1 : 0;  // first key this query sees
  // queries inside a non-causal span [nc_lo, nc_hi] (an image) also see the later keys of that span
  const int hi = (pos >= nc_lo && pos <= nc_hi) ? nc_hi : pos;

  uint qw[HS][KC];
  float qs[HS][SB];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global const uint* qp = (__global const uint*)(qi + ((size_t)(active ? m : 0) * nh + h0 + r) * HD);
#pragma unroll
    for (int c = 0; c < KC; ++c) qw[r][c] = active ? qp[c * 16 + lane] : 0u;
#pragma unroll
    for (int c = 0; c < SB; ++c) qs[r][c] = active ? qsc[((size_t)m * nh + h0 + r) * SB + c] * M_LOG2E_F : 0.0f;
  }
  float o[HS][DPL], mx[HS], lp[HS];
  float pk[HS], pb[HS], sacc[HS];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[r][i] = 0.0f;
  }

  const int nbc = cap / 16;                       // blocks in the cache
  const size_t blk0 = (size_t)g * nbc;
  const int last_hi = (nc_hi >= 0 && first <= nc_hi && last >= nc_lo) ? max(last, nc_hi) : last;
  const int b_first = win > 0 ? max(0, first - win + 1) / 16 : 0, b_last = last_hi / 16;
#define NL (2 * BLK_U / (16 * NSP * TPW))
  __global const uint* vqu = (__global const uint*)vq;
  uint rk[NL];
  {
    const size_t nb = (blk0 + b_first % nbc) * BLK_U;
#pragma unroll
    for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
  }
  for (int b = b_first; b <= b_last; ++b) {
    const int t0 = b * 16;
    __local uint* tb = tile[b % TILE_BUFS];
    if (TILE_BUFS == 1 && b > b_first) barrier(CLK_LOCAL_MEM_FENCE);  // everyone is done with the previous tile
#pragma unroll
    for (int j = 0; j < NL; ++j) tb[lid + j * lsz] = rk[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < b_last) {
      const int bs = (b + 1) % nbc;
      const size_t nb = (blk0 + bs) * BLK_U;
#pragma unroll
      for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
    }
    if (!active || t0 > hi || t0 + 15 < lo) continue;
    __local const uint* Kt = tb;
    __local const uint4* Vt = (__local const uint4*)(tb + BLK_U);
    const size_t sob = ((size_t)g * nbc + (size_t)(b % nbc)) * SB * 16 + lane;  // this tile's scales
    const int key = t0 + lane;
#ifdef QSA
    const bool valid = key <= hi && key >= lo && QSA_KEEP(m, pos, key);
    if (sub_group_all(!valid)) continue;  // (no key of the tile selected: nothing to add)
#else
    const bool valid = key <= hi && key >= lo;
#endif
#ifndef ABL_NOQK  // ablations (timing experiments only: wrong results)
#if HS == 4  // one address op per K load instead of two: 4 heads per sub-group spill 608 -> 0 bytes (3 heads: slower)
    __local const uint* KL = Kt + lane;
    QK_TILE(KL[(c * 16 + w) * 16], sob)
#else
    QK_TILE(Kt[(c * 16 + w) * 16 + lane], sob)
#endif
#else
#pragma unroll
    for (int r = 0; r < HS; ++r) sacc[r] = (float)(lane & 3) - (float)r + as_float(Kt[lane]) * 1e-30f;
#endif
#ifndef ABL_NOSM
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float sv = valid ? sacc[r] : -INFINITY;
      SOFTMAX_TILE(r, sv, valid)
    }
#else
#pragma unroll
    for (int r = 0; r < HS; ++r) { pk[r] = valid ? sacc[r] * 0.01f + 0.5f : 0.0f; pb[r] = 1.0f; lp[r] += pk[r]; }
#endif
#ifndef ABL_NOPV
    PV_TILE(Vt, sob, valid)
#else
#pragma unroll
    for (int r = 0; r < HS; ++r) o[r][0] += pk[r] * as_float(Vt[lane].x) * 1e-30f;
#endif
  }
  if (active) {
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float inv = 1.0f / sub_group_reduce_add(lp[r]);
      __global float* dst = out + ((size_t)m * nh + h0 + r) * HD + lane;
#pragma unroll
      for (int i = 0; i < DPL; ++i) dst[16 * i] = o[r][i] * inv;
    }
  }
}

// Prefill attention for long prompts with the head dims split over sub-groups. As attn_q8, a work-group covers TPW
// consecutive tokens of one kv head and shares every 16-key K/V tile through SLM; but a sub-group owns PTT tokens x
// PHG q heads (PRH rows) and only PDS = HD/PNS of the dims. It reads only its slice of each tile and holds q and the
// output of its dims only, so more rows fit in its registers and every byte read from SLM serves PRH rows (attn_q8:
// HS rows, SLM-bandwidth bound at long context; at HD 512 even HS = 2 spills). The partial scores of a row's PNS dim
// groups are summed through SLM (exact int sums with one scale per row: then the scores, the softmax and the output
// are bit-identical to attn_q8); every dim group runs the same online softmax and accumulates P*V for its dims.
// Full causal attention or a window (no non-causal spans). grid (ceil(M/TPW) * PLSZ, nkv), local (PLSZ, 1).
#ifdef PNS
#define PDS (HD / PNS)
#define PCS (PDS / 64)               // 64-dim chunks of a dim group
#define PNHG (HPG / PHG)             // head groups
#define PRH (PTT * PHG)              // (token, head) rows of a sub-group
#define PNRG (PNHG * (TPW / PTT))    // row groups of a work-group
#define PLSZ (16 * PNRG * PNS)       // work-items
#define PNL (2 * BLK_U / PLSZ)       // tile words fetched per work-item
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_ps(__global const char* qi, __global const float* qsc, __global const uint* kq,
                      __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                      int nh, int pos0, int M_valid, __global float* out, int win) {
  __local uint tile[2 * BLK_U];
  __local uint sp[PNRG][PNS][PRH][16];  // partial scores (int sums with SB = 1, else float bits)
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16;
  const int dj = sg % PNS, rg = sg / PNS, hgi = rg % PNHG, tg = rg / PNHG, g = get_group_id(1);
  const int mb = (int)get_group_id(0) * TPW, h0 = g * HPG + hgi * PHG;
  // tokens of this sub-group: q rows mb + tg*PTT + t, positions pos0 + that, keys [tlo, thi]
  int tlo[PTT], thi[PTT];
  bool tact[PTT];
  int any_lo = INT_MAX, any_hi = -1;
#pragma unroll
  for (int t = 0; t < PTT; ++t) {
    const int m = mb + tg * PTT + t, pos = pos0 + m;
    tact[t] = m < M_valid;
    tlo[t] = win > 0 ? pos - win + 1 : 0;
    thi[t] = pos;
    if (tact[t]) {
      any_lo = min(any_lo, tlo[t]);
      any_hi = max(any_hi, thi[t]);
    }
  }
  uint qw[PRH][PCS];
  float qs[PRH][PCS];
#pragma unroll
  for (int x = 0; x < PRH; ++x) {
    const int t = x / PHG, r = x % PHG, m = tact[t] ? mb + tg * PTT + t : 0;
    __global const uint* qp = (__global const uint*)(qi + ((size_t)m * nh + h0 + r) * HD);
#pragma unroll
    for (int cc = 0; cc < PCS; ++cc) {
      const int c = dj * PCS + cc;
      qw[x][cc] = tact[t] ? qp[c * 16 + lane] : 0u;
      qs[x][cc] = tact[t] ? qsc[((size_t)m * nh + h0 + r) * SB + (SB == 1 ? 0 : c)] * M_LOG2E_F : 0.0f;
    }
  }
  float o[PRH][4 * PCS], mx[PRH], lp[PRH], pk[PRH], pb[PRH];
#pragma unroll
  for (int x = 0; x < PRH; ++x) {
    mx[x] = -INFINITY;
    lp[x] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4 * PCS; ++i) o[x][i] = 0.0f;
  }

  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  const int first = pos0 + mb, last = pos0 + min(M_valid - 1, mb + TPW - 1);
  const int b_first = win > 0 ? max(0, first - win + 1) / 16 : 0, b_last = last / 16;
  __global const uint* vqu = (__global const uint*)vq;
  uint rk[PNL];
  {
    const size_t nb = (blk0 + b_first % nbc) * BLK_U;
#pragma unroll
    for (int j = 0; j < PNL; ++j) rk[j] = j < PNL / 2 ? kq[nb + lid + j * PLSZ] : vqu[nb + lid + (j - PNL / 2) * PLSZ];
  }
  __local const uint4* Vt = (__local const uint4*)(tile + BLK_U);
  for (int b = b_first; b <= b_last; ++b) {
    const int t0 = b * 16;
    if (b > b_first) barrier(CLK_LOCAL_MEM_FENCE);  // everyone is done with the previous tile and partial scores
#pragma unroll
    for (int j = 0; j < PNL; ++j) tile[lid + j * PLSZ] = rk[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < b_last) {
      const size_t nb = (blk0 + (b + 1) % nbc) * BLK_U;
#pragma unroll
      for (int j = 0; j < PNL; ++j) rk[j] = j < PNL / 2 ? kq[nb + lid + j * PLSZ] : vqu[nb + lid + (j - PNL / 2) * PLSZ];
    }
    // the same for every dim group of a row group: they write and read the partial scores together
    const bool work = any_hi >= t0 && any_lo <= t0 + 15;
    const size_t sob = (blk0 + (size_t)(b % nbc)) * SB * 16 + lane;  // this tile's scales
    const int key = t0 + lane;
    if (work) {
#if SB == 1
      int ia[PRH];
#pragma unroll
      for (int x = 0; x < PRH; ++x) ia[x] = 0;
#pragma unroll
      for (int cc = 0; cc < PCS; ++cc)
#pragma unroll
        for (int w = 0; w < 16; ++w) {
          const uint kw = tile[((dj * PCS + cc) * 16 + w) * 16 + lane];
#pragma unroll
          for (int x = 0; x < PRH; ++x) ia[x] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[x][cc], w), kw);
        }
#pragma unroll
      for (int x = 0; x < PRH; ++x) sp[rg][dj][x][lane] = (uint)ia[x];
#else
      float fa[PRH];
#pragma unroll
      for (int x = 0; x < PRH; ++x) fa[x] = 0.0f;
#pragma unroll
      for (int cc = 0; cc < PCS; ++cc) {
        const int c = dj * PCS + cc;
        int acc[PRH];
#pragma unroll
        for (int x = 0; x < PRH; ++x) acc[x] = 0;
#pragma unroll
        for (int w = 0; w < 16; ++w) {
          const uint kw = tile[(c * 16 + w) * 16 + lane];
#pragma unroll
          for (int x = 0; x < PRH; ++x) acc[x] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[x][cc], w), kw);
        }
        const float ks = ksc[sob + c * 16];
#pragma unroll
        for (int x = 0; x < PRH; ++x) fa[x] = fma((float)acc[x], qs[x][cc] * ks, fa[x]);
      }
#pragma unroll
      for (int x = 0; x < PRH; ++x) sp[rg][dj][x][lane] = as_uint(fa[x]);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!work) continue;
    // full scores and the online softmax of every row that sees a key of this tile (the others get P = 0)
#pragma unroll
    for (int x = 0; x < PRH; ++x) {
      const int t = x / PHG;
      pk[x] = 0.0f;
      pb[x] = 0.0f;
      if (tact[t] && t0 <= thi[t] && t0 + 15 >= tlo[t]) {
        const bool valid = key <= thi[t] && key >= tlo[t];
#if SB == 1
        int s = 0;
#pragma unroll
        for (int j = 0; j < PNS; ++j) s += (int)sp[rg][j][x][lane];
        const float sc = (float)s * qs[x][0] * ksc[sob];
#else
        float sc = 0.0f;
#pragma unroll
        for (int j = 0; j < PNS; ++j) sc += as_float(sp[rg][j][x][lane]);
#endif
        const float sv = valid ? sc : -INFINITY;
        const float tm = sub_group_reduce_max(sv);
        if (tm > mx[x]) {
          const float corr = EXP2(mx[x] - tm);
          lp[x] *= corr;
#pragma unroll
          for (int i = 0; i < 4 * PCS; ++i) o[x][i] *= corr;
          mx[x] = tm;
        }
        pk[x] = valid ? EXP2(sv - mx[x]) : 0.0f;
        lp[x] += pk[x];
        pb[x] = EXP2(tm - mx[x]);
      }
    }
    // P * V for this sub-group's dims; P quantized against each token's bound as in PV_PACK
    uint4 pw[PRH];
    float pq[PRH];
#pragma unroll
    for (int cc = 0; cc < PCS; ++cc) {
      const int c = dj * PCS + cc;
      if (SB > 1 || cc == 0) {
        const float vsv = vsc[sob + (SB == 1 ? 0 : c * 16)];
        float vst[PTT], vmt[PTT];
#pragma unroll
        for (int t = 0; t < PTT; ++t) {
          vst[t] = key <= thi[t] && key >= tlo[t] ? vsv : 0.0f;
          vmt[t] = sub_group_reduce_max(vst[t]);
        }
#pragma unroll
        for (int x = 0; x < PRH; ++x) {
          const int t = x / PHG;
          const float pm = pb[x] * vmt[t];
          pw[x] = pack_p(convert_int_rte(pk[x] * vst[t] * (pm > 0.0f ? PQMAX / pm : 0.0f)));
          pq[x] = pm * (1.0f / PQMAX);
        }
      }
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        const uint4 v = Vt[16 * (4 * c + ii) + lane];
#pragma unroll
        for (int x = 0; x < PRH; ++x) {
          int a = PDOT(pw[x].x, v.x);
          a += PDOT(pw[x].y, v.y);
          a += PDOT(pw[x].z, v.z);
          a += PDOT(pw[x].w, v.w);
          o[x][4 * cc + ii] = fma((float)a, pq[x], o[x][4 * cc + ii]);
        }
      }
    }
  }
#pragma unroll
  for (int x = 0; x < PRH; ++x) {
    const int t = x / PHG, r = x % PHG;
    if (tact[t]) {
      const float inv = 1.0f / sub_group_reduce_add(lp[x]);
      __global float* dst = out + ((size_t)(mb + tg * PTT + t) * nh + h0 + r) * HD + lane;
#pragma unroll
      for (int cc = 0; cc < PCS; ++cc)
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) dst[16 * (4 * (dj * PCS + cc) + ii)] = o[x][4 * cc + ii] * inv;
    }
  }
}
#endif

// Flash-decoding for a few query rows (single tokens, speculative verification) with the tile scheme of attn_q8
// (q rows qrow0 + m of qi):
// a work-group covers one key split of one kv head for all M_valid (<= TPW) rows x NSP head slices, so each K/V
// tile is read from memory once into SLM and used by every q head of every row. Row m (position pos0 + m) sees
// keys [max(lo(m), split start), min(pos0 + m, split end)]. Partial results (max, sum, unnormalized o) go to
// part[((m * nsplit + split) * nh + h) * (HD + 2)]; attn_combine merges each row's splits.
// grid (nsplit * 16*NSP*TPW, nkv), local (16*NSP*TPW, 1). kstart and slen: multiples of 16.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_split(__global const char* qi, __global const float* qsc, __global const uint* kq,
                         __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                         int nh, int pos0, int M_valid, __global float* part, int win, int kstart, int slen, int nsplit,
                         int qrow0) {
  __local uint tile[TILE_BUFS][2 * BLK_U];
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16, lsz = 16 * NSP * TPW;
  const int m = sg / NSP, hh = sg % NSP, g = get_group_id(1), split = get_group_id(0);
  const bool active = m < M_valid;
  const int pos = pos0 + m, h0 = g * HPG + hh * HS;
  const int s0 = kstart + split * slen, s1 = s0 + slen - 1;  // this split's keys
  const int lo = max(s0, win > 0 ? pos - win + 1 : 0), hi = min(s1, pos);

  uint qw[HS][KC];
  float qs[HS][SB];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global const uint* qp = (__global const uint*)(qi + ((size_t)(qrow0 + (active ? m : 0)) * nh + h0 + r) * HD);
#pragma unroll
    for (int c = 0; c < KC; ++c) qw[r][c] = active ? qp[c * 16 + lane] : 0u;
#pragma unroll
    for (int c = 0; c < SB; ++c) qs[r][c] = active ? qsc[((size_t)(qrow0 + m) * nh + h0 + r) * SB + c] * M_LOG2E_F : 0.0f;
  }
  float o[HS][DPL], mx[HS], lp[HS];
  float pk[HS], pb[HS], sacc[HS];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[r][i] = 0.0f;
  }
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  const int lo_all = max(s0, win > 0 ? pos0 - win + 1 : 0), hi_all = min(s1, pos0 + M_valid - 1);
  const int b_first = max(lo_all, 0) / 16, b_last = hi_all / 16;
  __global const uint* vqu = (__global const uint*)vq;
  uint rk[NL];
  if (b_first <= b_last) {
    const size_t nb = (blk0 + b_first % nbc) * BLK_U;
#pragma unroll
    for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
  }
  for (int b = b_first; b <= b_last; ++b) {
    const int t0 = b * 16;
    __local uint* tb = tile[b % TILE_BUFS];
    if (TILE_BUFS == 1 && b > b_first) barrier(CLK_LOCAL_MEM_FENCE);  // everyone is done with the previous tile
#pragma unroll
    for (int j = 0; j < NL; ++j) tb[lid + j * lsz] = rk[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (b < b_last) {
      const int bs = (b + 1) % nbc;
      const size_t nb = (blk0 + bs) * BLK_U;
#pragma unroll
      for (int j = 0; j < NL; ++j) rk[j] = j < NL / 2 ? kq[nb + lid + j * lsz] : vqu[nb + lid + (j - NL / 2) * lsz];
    }
    if (!active || t0 > hi || t0 + 15 < lo) continue;
    __local const uint* Kt = tb;
    __local const uint4* Vt = (__local const uint4*)(tb + BLK_U);
    const size_t sob = ((size_t)g * nbc + (size_t)(b % nbc)) * SB * 16 + lane;  // this tile's scales
    const int key = t0 + lane;
    const bool valid = key <= hi && key >= lo;
    QK_TILE(Kt[(c * 16 + w) * 16 + lane], sob)
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float sv = valid ? sacc[r] : -INFINITY;
      SOFTMAX_TILE(r, sv, valid)
    }
    PV_TILE(Vt, sob, valid)  // (int8 V: attn_split is not used with --v4, see attn_decode_rows)
  }
  if (active) {
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      __global float* dst = part + (((size_t)m * nsplit + split) * nh + h0 + r) * (HD + 2);
      const float L = sub_group_reduce_add(lp[r]);
      if (lane == 0) { dst[0] = mx[r]; dst[1] = L; }
#pragma unroll
      for (int i = 0; i < DPL; ++i) dst[2 + 16 * i + lane] = o[r][i];
    }
  }
}

// Single-token attention split over the key range (flash-decoding): one sub-group per (key split, kv head,
// head slice) computes, for HS q heads, the max score, softmax denominator and unnormalized P*V over its keys
// [kstart + split*slen, min(pos+1, kstart + (split+1)*slen)), restricted to the window, with the same int8 math
// as attn_q8. attn_combine merges the splits. grid (16 * nsplit, nkv * NSP), local (16, NSP): the head slices of a
// split share its K/V reads through L1.
// part: [nsplit][nh][HD + 2] = (max, sum, o[HD]). slen: multiple of 16; kstart: multiple of slen at or below the
// window start.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode(__global const char* qi, __global const float* qsc, __global const uint* kq,
                          __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                          int nh, int pos, __global float* part, int qrow, int win, int kstart, int slen QSA_ARGS
                         ) {
  const int lane = get_sub_group_local_id(), split = get_global_id(0) / SG;
  const int g = get_global_id(1) / NSP, h0 = g * HPG + (get_global_id(1) % NSP) * HS;
  const int lo = win > 0 ? max(0, pos - win + 1) : 0;
  const int k0 = max(kstart + split * slen, lo & ~15), k1 = min(pos + 1, kstart + (split + 1) * slen);
  uint qw[HS][KC];
  float qs[HS][SB];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global const uint* qp = (__global const uint*)(qi + ((size_t)qrow * nh + h0 + r) * HD);
#pragma unroll
    for (int c = 0; c < KC; ++c) qw[r][c] = qp[c * 16 + lane];
#pragma unroll
    for (int c = 0; c < SB; ++c) qs[r][c] = qsc[((size_t)qrow * nh + h0 + r) * SB + c] * M_LOG2E_F;
  }
  float o[HS][DPL], mx[HS], lp[HS];
  float pk[HS], pb[HS], sacc[HS];
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    mx[r] = -INFINITY;
    lp[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[r][i] = 0.0f;
  }
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  for (int t0 = k0; t0 < k1; t0 += 16) {
    const int bs = (t0 / 16) % nbc;
    const size_t b = blk0 + bs, sob = b * SB * 16 + lane;
    const int key = t0 + lane;
#ifdef QSA
    const bool valid = key < k1 && key >= lo && QSA_KEEP(qrow, pos, key);
    if (sub_group_all(!valid)) continue;  // (no key of the tile selected: its K/V are not read)
#else
    const bool valid = key < k1 && key >= lo;
#endif
    QK_TILE(intel_sub_group_block_read(kq + (b * KW + c * 16 + w) * 16), sob)
#pragma unroll
    for (int r = 0; r < HS; ++r) {
      const float sv = valid ? sacc[r] : -INFINITY;
      SOFTMAX_TILE(r, sv, valid)
    }
    __global const uint4* vt = vq + b * HD;
    PV_TILE(vt, sob, valid)
  }
#pragma unroll
  for (int r = 0; r < HS; ++r) {
    __global float* dst = part + ((size_t)split * nh + h0 + r) * (HD + 2);
    const float L = sub_group_reduce_add(lp[r]);
    if (lane == 0) { dst[0] = mx[r]; dst[1] = L; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) dst[2 + 16 * i + lane] = o[r][i];
  }
}

// Flash-decoding of up to TR query rows (single tokens, speculative verification) with the head dims split over the
// sub-groups of a work-group: a work-group covers one key split of one kv head; sub-group (head group hg, dim group
// dj) owns HG q heads of every row and the DS = HD/NSUB dims [dj*DS, (dj+1)*DS). Per 16-key tile each sub-group
// reads only its dims of K and V (every byte of the cache is read once for all rows and heads), computes partial
// scores, and the NSUB partials of a (row, head) are summed through SLM; the online softmax then runs in every dim
// group of the head group (same values, same order: identical results), and each sub-group accumulates P*V for its
// dims. Row m (position pos0 + m, q row qrow0 + m) sees keys [max(lo(m), split start), min(pos0 + m, split end)].
// Same int8 math as attn_decode. Partials go to part[((m * nsplit + split) * nh + h) * (HD + 2)] for attn_combine.
// Built with -DHG, -DNSUB (DS a multiple of 64) and -DTR. grid (nsplit * 16*NHG*NSUB, nkv), local (16*NHG*NSUB, 1).
// kstart and slen: multiples of 16.
#ifdef NSUB
#define DS (HD / NSUB)
#define CPS (DS / 64)
#define NHG (HPG / HG)
#ifndef TR
#define TR 1
#endif
#define RH (TR * HG)  // (row, head) pairs of a sub-group
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_dsplit(__global const char* qi, __global const float* qsc, __global const uint* kq,
                          __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                          int nh, int pos0, int M_valid, __global float* part, int win, int kstart, int slen,
                          int nsplit, int qrow0) {
  __local float sp[2][NHG][NSUB][RH][SG];
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), hgi = sg / NSUB, dj = sg % NSUB;
  const int g = get_group_id(1), split = get_group_id(0), h0 = g * HPG + hgi * HG;
  const int s0 = kstart + split * slen, s1 = s0 + slen - 1;  // this split's keys
  const int lo_all = max(s0, win > 0 ? pos0 - win + 1 : 0), hi_all = min(s1, pos0 + M_valid - 1);
  uint qw[RH][CPS];
  float qs[RH][CPS];
#pragma unroll
  for (int x = 0; x < RH; ++x) {
    const int m = min(x / HG, M_valid - 1), r = x % HG;
    __global const uint* qp = (__global const uint*)(qi + ((size_t)(qrow0 + m) * nh + h0 + r) * HD);
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const int c = dj * CPS + cc;
      qw[x][cc] = qp[c * 16 + lane];
      qs[x][cc] = qsc[((size_t)(qrow0 + m) * nh + h0 + r) * SB + (SB == 1 ? 0 : c)] * M_LOG2E_F;
    }
  }
  float o[RH][4 * CPS], mx[RH], lp[RH], pk[RH], pb[RH];
#pragma unroll
  for (int x = 0; x < RH; ++x) {
    mx[x] = -INFINITY;
    lp[x] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4 * CPS; ++i) o[x][i] = 0.0f;
  }
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;
  int it = 0;
  for (int t0 = lo_all & ~15; t0 <= hi_all; t0 += 16, ++it) {
    const size_t b = blk0 + (t0 / 16) % nbc, sob = b * SB * 16 + lane;
    const int key = t0 + lane;
    // partial scores of this sub-group's dims
    float ps_[RH];
#pragma unroll
    for (int x = 0; x < RH; ++x) ps_[x] = 0.0f;
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const int c = dj * CPS + cc;
      int acc[RH];
#pragma unroll
      for (int x = 0; x < RH; ++x) acc[x] = 0;
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const uint kw = intel_sub_group_block_read(kq + (b * KW + c * 16 + w) * 16);
#pragma unroll
        for (int x = 0; x < RH; ++x) acc[x] += dot_4x8packed_ss_int(intel_sub_group_shuffle(qw[x][cc], w), kw);
      }
      const float ks = ksc[sob + (SB == 1 ? 0 : c * 16)];
#pragma unroll
      for (int x = 0; x < RH; ++x) ps_[x] = fma((float)acc[x], qs[x][cc] * ks, ps_[x]);
    }
    __local float* spb = &sp[it & 1][hgi][0][0][0];
#pragma unroll
    for (int x = 0; x < RH; ++x) spb[(dj * RH + x) * SG + lane] = ps_[x];
    barrier(CLK_LOCAL_MEM_FENCE);
    // full scores and the online softmax of every (row, head) of the head group; a row with no key of this tile
    // (beyond its position, or before its window) is left as it is
#pragma unroll
    for (int x = 0; x < RH; ++x) {
      const int m = x / HG, pos = pos0 + m, lo = max(s0, win > 0 ? pos - win + 1 : 0), hi = min(s1, pos);
      pk[x] = 0.0f;
      pb[x] = 0.0f;
      if (m < M_valid && t0 <= hi && t0 + 15 >= lo) {
        const bool valid = key <= hi && key >= lo;
        float sv = 0.0f;
#pragma unroll
        for (int j = 0; j < NSUB; ++j) sv += spb[(j * RH + x) * SG + lane];
        sv = valid ? sv : -INFINITY;
        const float tm = sub_group_reduce_max(sv);
        if (tm > mx[x]) {
          const float corr = EXP2(mx[x] - tm);
          lp[x] *= corr;
#pragma unroll
          for (int i = 0; i < 4 * CPS; ++i) o[x][i] *= corr;
          mx[x] = tm;
        }
        pk[x] = valid ? EXP2(sv - mx[x]) : 0.0f;
        lp[x] += pk[x];
        pb[x] = EXP2(tm - mx[x]);
      }
    }
    // P * V for this sub-group's dims (P re-quantized per 64-dim block with SB > 1, as PV_PACK); keys outside every
    // row's range have pk = 0
    __global const uint4* vt = vq + b * HD;
#define DS_VSC(c) vsc[sob + (SB == 1 ? 0 : (c) * 16)]
#define DS_REPACK (SB > 1)
    uint4 pw[RH];
    float pq[RH];
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const int c = dj * CPS + cc;
      if (DS_REPACK || cc == 0) {
#ifdef ROWVMT
        // (-DROWVMT, Qwen: the bound of each row covers only the keys that row sees, so a row of a verification pass
        //  gets exactly the values of decoding its token alone: speculative decoding stays bit-identical)
        const float vraw = DS_VSC(c);
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          const int m = x / HG, pos = pos0 + m, lo = max(s0, win > 0 ? pos - win + 1 : 0), hi = min(s1, pos);
          const float vs = m < M_valid && key >= lo && key <= hi ? vraw : 0.0f, vmt = sub_group_reduce_max(vs);
          const float pm = pb[x] * vmt;
          pw[x] = pack_p(convert_int_rte(pk[x] * vs * (pm > 0.0f ? PQMAX / pm : 0.0f)));
          pq[x] = pm * (1.0f / PQMAX);
        }
#else
        const bool used = key >= lo_all && key <= hi_all;  // the bound covers only keys some row sees
        const float vs = used ? DS_VSC(c) : 0.0f, vmt = sub_group_reduce_max(vs);
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          const float pm = pb[x] * vmt;
          pw[x] = pack_p(convert_int_rte(pk[x] * vs * (pm > 0.0f ? PQMAX / pm : 0.0f)));
          pq[x] = pm * (1.0f / PQMAX);
        }
#endif
      }
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        const uint4 v = vt[64 * c + 16 * ii + lane];
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          int a = PDOT(pw[x].x, v.x);
          a += PDOT(pw[x].y, v.y);
          a += PDOT(pw[x].z, v.z);
          a += PDOT(pw[x].w, v.w);
          o[x][4 * cc + ii] = fma((float)a, pq[x], o[x][4 * cc + ii]);
        }
      }
    }
  }
#undef DS_VSC
#undef DS_REPACK
#pragma unroll
  for (int x = 0; x < RH; ++x) {
    const int m = x / HG, r = x % HG;
    const float L = sub_group_reduce_add(lp[x]);
    if (m < M_valid) {
      __global float* dst = part + (((size_t)m * nsplit + split) * nh + h0 + r) * (HD + 2);
      if (dj == 0 && lane == 0) { dst[0] = mx[x]; dst[1] = L; }
#pragma unroll
      for (int cc = 0; cc < CPS; ++cc)
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) dst[2 + 64 * (dj * CPS + cc) + 16 * ii + lane] = o[x][4 * cc + ii];
    }
  }
}
#endif

// Merges the per-split partial results of attn_decode / attn_split (partials row prow) into output row orow.
// One work-item per (q head, output dim); grid (nh * hd), local 64 (hd: multiple of 64). The lanes of a sub-group
// take the split maxima and weights 16 at a time, then every lane sums its dim over the splits with the weights
// broadcast (independent loads, no data-dependent branches). Splits with no valid key (max = -inf) weigh 0.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_combine(__global const float* part, int nsplit, int nh, __global float* out, int orow, int hd,
                           int prow) {
  const int h = get_global_id(0) / hd, d = get_global_id(0) % hd, lane = get_sub_group_local_id();
  __global const float* p = part + ((size_t)prow * nsplit * nh + h) * (hd + 2);
  const size_t stride = (size_t)nh * (hd + 2);
  float M = -INFINITY;
  for (int s = lane; s < nsplit; s += SG) M = fmax(M, p[s * stride]);
  M = sub_group_reduce_max(M);
  float L = 0.0f, acc = 0.0f;
  for (int s0 = 0; s0 < nsplit; s0 += SG) {
    const int s = s0 + lane;
    const float m = s < nsplit ? p[s * stride] : -INFINITY;
    const float w = m == -INFINITY ? 0.0f : exp2(m - M);
    L = fma(s < nsplit ? p[s * stride + 1] : 0.0f, w, L);
    const int n = min(SG, nsplit - s0);
    __global const float* q = p + s0 * stride + 2 + d;
    if (n == SG) {
#pragma unroll
      for (int j = 0; j < SG; ++j) acc = fma(q[j * stride], sub_group_broadcast(w, j), acc);
    } else {
      for (int j = 0; j < n; ++j) acc = fma(q[j * stride], sub_group_broadcast(w, j), acc);
    }
  }
  out[((size_t)orow * nh + h) * hd + d] = acc / sub_group_reduce_add(L);
}

// MTP (nextn) block input: out[m] = [rmsnorm(e[m]) * ew | rmsnorm(h[m]) * hw] (f32, 2E wide). grid (256, n).
__kernel void mtp_concat(__global const float* e, __global const float* h, __global const float* ew,
                         __global const float* hw, int E, float eps, __global float* out) {
  const int m = get_group_id(1), l = get_local_id(0);
  __global const float* er = e + (size_t)m * E;
  __global const float* hr = h + (size_t)m * E;
  float se = 0.0f, sh = 0.0f;
  for (int i = l; i < E; i += get_local_size(0)) { se += er[i] * er[i]; sh += hr[i] * hr[i]; }
  se = work_group_reduce_add(se);
  sh = work_group_reduce_add(sh);
  const float re = rsqrt(se / E + eps), rh = rsqrt(sh / E + eps);
  __global float* o = out + (size_t)m * 2 * E;
  for (int i = l; i < E; i += get_local_size(0)) {
    o[i] = er[i] * re * ew[i];
    o[E + i] = hr[i] * rh * hw[i];
  }
}

