// With no arguments (or --self-test), runs the model-free Q5_1 regression used by CI.
// Prints the CPU dequantization of one row of a GGUF tensor (first values, sum, sum of squares), to compare the
// decoders in src/quants.h with an independent implementation. usage: test_quant_rows model.gguf TENSOR [row] [n]
// With PAD=1 and a small-block type (IQ4_NL / Q2_0) it also checks pad_small_block_rows: the padded row must
// dequantize to the same K values followed by zeros.
#include "../src/gguf.h"
#include "../src/quants.h"
#include <cstdio>
#include <vector>

static int q51_reference_test() {
  int checked = 0;
  for (int K : {32, 64, 128, 256, 512, 640, 768, 1024}) {
    std::vector<uint8_t> source((size_t)K / 32 * 24);
    std::vector<float> expected(K), actual(K + 16, -123456.0f);
    for (int b = 0; b < K / 32; ++b) {
      // Exact binary fractions: expected values do not use the production decoder.
      const float d = b % 4 == 0 ? 0.0f : (b % 4 == 1 ? 0.25f : (b % 4 == 2 ? 0.5f : 2.0f));
      const float m = (b % 3 - 1) * 0.5f;
      uint8_t* p = source.data() + b * 24;
      const uint16_t dh = f32_to_fp16(d), mh = f32_to_fp16(m);
      memcpy(p, &dh, 2); memcpy(p + 2, &mh, 2);
      uint32_t high = 0;
      for (int j = 0; j < 32; ++j) {
        const int q = (j * 13 + b * 7) % 32;
        high |= (uint32_t)(q >> 4) << j;
        p[8 + j % 16] |= (q & 15) << (j < 16 ? 0 : 4);
        expected[b * 32 + j] = d * q + m;
      }
      memcpy(p + 4, &high, 4);
    }
    dequant_row(GGML_Q5_1, source.data(), K, actual.data() + 8);
    for (int i = 0; i < K; ++i) {
      if (actual[8 + i] != expected[i]) {
        std::printf("FAIL K=%d i=%d got=%g expected=%g\n", K, i, actual[8 + i], expected[i]);
        return 1;
      }
      ++checked;
    }
    for (int i = 0; i < 8; ++i)
      if (actual[i] != -123456.0f || actual[K + 8 + i] != -123456.0f) return 2;
  }
  std::printf("test_quant_rows: all passed (%d Q5_1 values, eight row widths, output guards unchanged)\n", checked);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--self-test")) return q51_reference_test();
  if (argc < 3) { fprintf(stderr, "usage: test_quant_rows model.gguf TENSOR [row] [n]\n"); return 1; }
  GgufFile f(argv[1]);
  const GgufTensor& t = f.get(argv[2]);
  const uint64_t row = argc > 3 ? strtoull(argv[3], nullptr, 10) : 0;
  const int n = argc > 4 ? atoi(argv[4]) : 8;
  int blk = 0, bytes = 0;
  ggml_block_info(t.type, blk, bytes);
  const int K = (int)t.dims[0];
  const size_t row_bytes = (size_t)K / blk * bytes;
  std::vector<float> v(K);
  dequant_row(t.type, t.data + row * row_bytes, K, v.data());
  double s = 0, s2 = 0;
  for (float x : v) { s += x; s2 += (double)x * x; }
  printf("%s %s K=%d row %llu: sum %.6f sumsq %.6f first:", argv[2], ggml_type_name(t.type), K, (unsigned long long)row, s, s2);
  for (int i = 0; i < n && i < K; ++i) printf(" %.6f", v[i]);
  printf("\n");
  if (getenv("PAD") && is_small_block_type(t.type)) {
    const int Kp = padded_k(K);
    std::vector<uint8_t> padded((size_t)Kp / blk * bytes);
    pad_small_block_rows(t.type, t.data + row * row_bytes, 1, K, padded.data());
    std::vector<float> p(Kp);
    dequant_row(t.type, padded.data(), Kp, p.data());
    int bad = 0;
    for (int i = 0; i < Kp; ++i) {
      const float want = i < K ? v[i] : 0.0f;
      if (p[i] != want) ++bad;
    }
    printf("padded to K=%d (%zu bytes): %s\n", Kp, padded.size(), bad ? "MISMATCH" : "identical + zeros");
    return bad ? 1 : 0;
  }
  return 0;
}
