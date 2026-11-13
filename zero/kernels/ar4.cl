// Private inferzero package layout for exact AutoRound symmetric INT4/group-128.
// GPU superblock: two F16 scales, 8 Q4_0-order 32-value nibble blocks, 4 pad bytes.
#define AR4_128 0x41523447
#if QT == AR4_128
inline void ar4_unpack(uint4 q, uint* w) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint v = q[k];
    const uint lo = v & 0x0f0f0f0fu, hi = (v >> 4) & 0x0f0f0f0fu;
    w[k] = ((lo | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
    w[k + 4] = ((hi | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
  }
}
#endif
