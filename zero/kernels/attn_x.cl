// attn_q8's integer sums on XMX (QUEUE F33.2, ARC_ATTN_XMX=1; zero/xe2/engine_xe2.h): placed in front of the ops
// kernel parts when attn_q8 is built with -DATTN_XMX (a kernel file of its own: discrete-card code). Discrete Xe2 /
// Xe3 cards, 16-lane DPAS; -DATTN_XMX_EMU emulates it with shuffles (tests on GPUs and CPUs without XMX).
//
// attn_q8's two integer dot products per 16-key tile become DPAS (k = 32 int8 values, M = 4 rows = up to 4 q heads of
// the sub-group, N = 16 lanes):
//   scores: acc[head][key] = q . k over the head dims        (A: q, 2 dims a lane; B: a key's 32 dims, lane = key)
//   P.V:    acc[head][dim] = P . V over the tile's 16 keys    (A: P as u8, keys 16..31 zero; B: Vt as attn_q8 holds it,
//                                                              lane = dim)
// DPAS sums integers exactly, as DP4a does, and every float operation around them is attn_q8's in the same order:
// the results are bit-identical to attn_q8 (one scale per row, SB = 1, Qwen; HS <= 4). P quantized to 0..255 (u8),
// attn_q8's default (not -DP7).

#if SB != 1 || HS > 4
#error "attn_x: one scale per row (SB = 1) and at most 4 q heads per sub-group"
#endif

#ifdef ATTN_XMX_EMU
// one DPAS 4 x 16 x 32 (int8 / u8 x int8, int sums): a: lane l holds row r's values 2l (low byte), 2l + 1 (high);
// b: lane n = column n, its 32 values as 8 words (value 4i + j in byte j of word i); result: component r = row r, lane = column
// (each lane's two bytes are taken apart before they are shuffled, as whole ints: a shuffled word shifted by 8 * (k & 1)
//  in the unrolled loop gave wrong sums on the 125H's GPU compiler (2026-11-09: every output of attn_q8's check
//  apart), while the CPU's OpenCL got it right; split first, both agree)
inline int4 ax_i8(short4 a, int8 b, int4 acc) {
  const int bw[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};
  int c[4] = {acc.x, acc.y, acc.z, acc.w};
  const int a0[4] = {(int)(char)(a.x & 0xFF), (int)(char)(a.y & 0xFF), (int)(char)(a.z & 0xFF), (int)(char)(a.w & 0xFF)};
  const int a1[4] = {(int)(char)((a.x >> 8) & 0xFF), (int)(char)((a.y >> 8) & 0xFF), (int)(char)((a.z >> 8) & 0xFF),
                     (int)(char)((a.w >> 8) & 0xFF)};
#pragma unroll
  for (int p = 0; p < 16; ++p) {  // (values 2p, 2p + 1: lane p's low and high byte)
    const int b0 = (int)(char)(bw[(2 * p) >> 2] >> (8 * ((2 * p) & 3)));
    const int b1 = (int)(char)(bw[(2 * p + 1) >> 2] >> (8 * ((2 * p + 1) & 3)));
#pragma unroll
    for (int r = 0; r < 4; ++r) c[r] += intel_sub_group_shuffle(a0[r], p) * b0 + intel_sub_group_shuffle(a1[r], p) * b1;
  }
  return (int4)(c[0], c[1], c[2], c[3]);
}
inline int4 ax_u8(ushort4 a, int8 b, int4 acc) {
  const int bw[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};
  int c[4] = {acc.x, acc.y, acc.z, acc.w};
  const int a0[4] = {a.x & 0xFF, a.y & 0xFF, a.z & 0xFF, a.w & 0xFF};
  const int a1[4] = {(a.x >> 8) & 0xFF, (a.y >> 8) & 0xFF, (a.z >> 8) & 0xFF, (a.w >> 8) & 0xFF};
#pragma unroll
  for (int p = 0; p < 16; ++p) {
    const int b0 = (int)(char)(bw[(2 * p) >> 2] >> (8 * ((2 * p) & 3)));
    const int b1 = (int)(char)(bw[(2 * p + 1) >> 2] >> (8 * ((2 * p + 1) & 3)));
#pragma unroll
    for (int r = 0; r < 4; ++r) c[r] += intel_sub_group_shuffle(a0[r], p) * b0 + intel_sub_group_shuffle(a1[r], p) * b1;
  }
  return (int4)(c[0], c[1], c[2], c[3]);
}
#else
#define ax_i8(a, b, acc) intel_sub_group_i8_i8_matrix_mad_k32((a), (b), (acc))
#define ax_u8(a, b, acc) intel_sub_group_u8_i8_matrix_mad_k32((a), (b), (acc))
#endif

// q as the scores' A operand, once per work-item: qa[kb] row r = head r's dims 32 kb + 2 lane, + 1 (from qw[r][kb / 2],
// whose lane w holds dims 64 c + 4w .. 4w + 3); rows past HS zero
#define QK_PREP_X                                                                       \
  short4 qa[HD / 32];                                                                   \
  _Pragma("unroll") for (int kb = 0; kb < HD / 32; ++kb) {                              \
    short v_[4];                                                                        \
    _Pragma("unroll") for (int r = 0; r < 4; ++r) {                                     \
      const uint w_ = intel_sub_group_shuffle(qw[r < HS ? r : 0][kb / 2], (kb & 1) * 8 + lane / 2); \
      v_[r] = r < HS ? (short)(w_ >> (16 * (lane & 1))) : (short)0;                     \
    }                                                                                   \
    qa[kb] = (short4)(v_[0], v_[1], v_[2], v_[3]);                                      \
  }

// QK_TILE's scores (lane = key): the K words of chunk c, word w through KW_EXPR as attn_q8 reads them
#define QK_TILE_X(KW_EXPR, sob)                                                         \
  {                                                                                     \
    int4 acc_ = (int4)0;                                                                \
    _Pragma("unroll") for (int kb = 0; kb < HD / 32; ++kb) {                            \
      const int c = kb / 2;                                                             \
      uint k8_[8];                                                                      \
      _Pragma("unroll") for (int ww = 0; ww < 8; ++ww) {                                \
        const int w = (kb & 1) * 8 + ww;                                                \
        k8_[ww] = KW_EXPR;                                                              \
      }                                                                                 \
      acc_ = ax_i8(qa[kb], as_int8((uint8)(k8_[0], k8_[1], k8_[2], k8_[3], k8_[4], k8_[5], k8_[6], k8_[7])), acc_); \
    }                                                                                   \
    const float ks = ksc[sob];                                                          \
    const int ai_[4] = {acc_.x, acc_.y, acc_.z, acc_.w};                                \
    _Pragma("unroll") for (int r = 0; r < HS; ++r) sacc[r] = (float)ai_[r] * qs[r][0] * ks; \
  }

// PV_TILE's update (lane = dim): P packed by PV_PACK (pw[r] word w = keys 4w..4w+3, the same in every lane), as the A
// operand (lane l < 8: keys 2l, 2l + 1; lanes 8..15: keys 16..31, zero), dotted with the tile's V words
#define PV_TILE_X(vt, sob, valid)                                                       \
  {                                                                                     \
    PV_PACK(0, valid, sob)                                                              \
    ushort4 pa_;                                                                        \
    {                                                                                   \
      ushort v_[4];                                                                     \
      _Pragma("unroll") for (int r = 0; r < 4; ++r) {                                   \
        const uint4 pr_ = pw[r < HS ? r : 0];                                           \
        const int q_ = (lane >> 1) & 3;                                                 \
        const uint ws_ = q_ == 0 ? pr_.x : q_ == 1 ? pr_.y : q_ == 2 ? pr_.z : pr_.w;   \
        v_[r] = r < HS && lane < 8 ? (ushort)(ws_ >> (16 * (lane & 1))) : (ushort)0;    \
      }                                                                                 \
      pa_ = (ushort4)(v_[0], v_[1], v_[2], v_[3]);                                      \
    }                                                                                   \
    _Pragma("unroll") for (int i = 0; i < DPL; ++i) {                                   \
      const uint4 v = vt[16 * (i) + lane];                                              \
      const int4 a_ = ax_u8(pa_, as_int8((uint8)(v.x, v.y, v.z, v.w, 0u, 0u, 0u, 0u)), (int4)0); \
      const int ai_[4] = {a_.x, a_.y, a_.z, a_.w};                                      \
      _Pragma("unroll") for (int r = 0; r < HS; ++r) o[r][i] = fma((float)ai_[r], ps[r], o[r][i]); \
    }                                                                                   \
  }
