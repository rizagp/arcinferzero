// H8 (QUEUE 31.12): the prompt GEMM's weights and activations rotated by a normalized Walsh-Hadamard transform in
// blocks of HB values along K (the same rotation on both sides: (Hx).(Hw) = x.w), then rounded to int8 with one scale
// per 256-superblock. The GEMM (gemm.cl QTYPE 110) then runs pure int8 dot products over each superblock, one float
// multiply-add per superblock, instead of rescaling every 16 or 32 values (the scale epilogue: GEMM 9.7 -> 11.0 TOPS
// on the 125H without it; on a B60 the DPAS accumulates 256 values at a time). The rotation spreads the activations'
// outliers: plain per-row int8 had a 27B KL of 0.038 against f32, rotated in blocks of 1024 0.00057 (tools/ref_forward
// REF_HAD, 2026-11-03). Built with -DHB=<256 | 512 | 1024> (a power of two dividing K) and, for the weights,
// -DSRC=<the layout they were decoded into: 8 = Q8_0 repacked, 100..103 = W8>.
//
// Layouts (gemm.cl): per 16-column group g and superblock s, value e (0..255) of column l is byte e % 4 of uint
// ((g * nsb + s) * 64 + e / 4) * 16 + l of wq; the W8 header is HDR uints per lane (word k of lane l at
// ((g * nsb + s) * HDR + k) * 16 + l): Q8_0 8 per-32 float scales; W8 d, dmin, 4 words of int8 scales (per 16, or per
// 32 in bytes 0 and 2 of a word), 4 words of uint8 mins (per 16). H8: the same values layout, one float per
// superblock and lane (header HDR 1).
#define SG 16
#ifndef HB
#error "HB: the rotation block (256, 512 or 1024)"
#endif
#define NW (HB / 4)  // work-items: one int8x4 word each

// in-place FWHT of the HB values in x[], unnormalized (the 1 / sqrt(HB) is folded into the scales)
inline void fwht_local(__local float* x, int t) {
  for (int len = 1; len < HB; len <<= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int k = 0; k < 2; ++k) {  // (HB / 2 butterflies over HB / 4 items)
      const int p = 2 * t + k, i = (p / len) * 2 * len + (p % len), j = i + len;
      const float a = x[i], b = x[j];
      x[i] = a + b;
      x[j] = a - b;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

// the largest |value| of each 256-superblock of the block: per sub-group, then 4 sub-groups a superblock
inline float superblock_amax(__local float* mx, float m, int t) {
  const float sgm = sub_group_reduce_max(m);
  if (get_sub_group_local_id() == 0) mx[t / SG] = sgm;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int s0 = (t / 64) * 4;
  return fmax(fmax(mx[s0], mx[s0 + 1]), fmax(mx[s0 + 2], mx[s0 + 3]));
}

inline uint pack_q(const float v[4], float inv) {
  uint w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = clamp((int)rint(v[i] * inv), -127, 127);
    w |= ((uint)(q & 0xFF)) << (8 * i);
  }
  return w;
}

#ifdef SRC
// The weights of one column and one rotation block, decoded from the W8 / Q8_0 layout, rotated and written back in
// place as H8 (each work-group reads only the words it then writes), their superblock scales into hh.
// grid (N * NW, K / HB), local (NW, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void h8_weights(__global uint* restrict wq, __global const uint* restrict wh, __global float* restrict hh,
                         int N, int K) {
  __local float x[HB];
  __local float mx[NW / SG];
  const int t = get_local_id(0), n = get_group_id(0), blk = get_group_id(1);
  const int g = n / SG, l = n % SG, nsb = K / 256;
  const int e0 = blk * HB + 4 * t, s = e0 / 256, e = e0 % 256;
  __global uint* wp = wq + (((size_t)g * nsb + s) * 64 + e / 4) * SG + l;
  const uint word = *wp;
  float v[4];
#if SRC == 8
  __global const uint* hp = wh + ((size_t)g * nsb + s) * (8 * SG);
  const float d = as_float(hp[(e / 32) * SG + l]);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = d * (float)(char)(word >> (8 * i));
#else
  __global const uint* hp = wh + ((size_t)g * nsb + s) * (10 * SG);
  const float d = as_float(hp[l]), dmin = as_float(hp[SG + l]);
  const int b16 = e / 16;  // (the 4 values share their 16-block)
  const uint hs = hp[(2 + b16 / 4) * SG + l];
#if SRC & 1
  const int sc = (int)(char)(hs >> ((b16 & 2) ? 16 : 0));  // (per-32 scales: bytes 0 and 2 of the word)
#else
  const int sc = (int)(char)(hs >> (8 * (b16 & 3)));
#endif
#if SRC >= 102
  const float mn = (float)((hp[(6 + b16 / 4) * SG + l] >> (8 * (b16 & 3))) & 0xFF);
#else
  const float mn = 0.0f;
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = d * (float)sc * (float)(char)(word >> (8 * i)) - dmin * mn;
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i) x[4 * t + i] = v[i];
  fwht_local(x, t);
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) m = fmax(m, fabs(v[i] = x[4 * t + i]));
  const float amax = superblock_amax(mx, m, t);
  *wp = pack_q(v, amax > 0.0f ? 127.0f / amax : 0.0f);
  if (t % 64 == 0) hh[((size_t)g * nsb + s) * SG + l] = amax / (127.0f * sqrt((float)HB));
}
#endif

// The activations of one token row and one rotation block: the engine's int8 rows (aq [M][K], scales asc per 32
// values with ACT32, else per 256: [K / 32][M] or [K / 256][M]) as floats, rotated, rounded again to int8 into aqh
// [M][K] with one scale per 256 values in asch [K / 256][M].
// grid (NW, M, K / HB), local (NW, 1, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void h8_act(__global const uint* restrict aq, __global const float* restrict asc, int act32,
                     __global uint* restrict aqh, __global float* restrict asch, int M, int K) {
  __local float x[HB];
  __local float mx[NW / SG];
  const int t = get_local_id(0), m = get_global_id(1), blk = get_global_id(2);
  const int e0 = blk * HB + 4 * t;
  const uint word = aq[((size_t)m * K + e0) / 4];
  const float sc = act32 ? asc[(size_t)(e0 / 32) * M + m] : asc[(size_t)(e0 / 256) * M + m];
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[4 * t + i] = sc * (float)(char)(word >> (8 * i));
  fwht_local(x, t);
  float mm = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) mm = fmax(mm, fabs(v[i] = x[4 * t + i]));
  const float amax = superblock_amax(mx, mm, t);
  aqh[((size_t)m * K + e0) / 4] = pack_q(v, amax > 0.0f ? 127.0f / amax : 0.0f);
  if (t % 64 == 0) asch[(size_t)(e0 / 256) * M + m] = amax / (127.0f * sqrt((float)HB));
}
