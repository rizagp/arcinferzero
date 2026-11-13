// Independent Q5_1 row fixture, including Flash Next's K=640 expert-down tail.
#include "../src/quants.h"
#include <cstdio>

int main() {
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
  std::printf("PASS: %d Q5_1 values, eight row widths, output guards unchanged\n", checked);
}
