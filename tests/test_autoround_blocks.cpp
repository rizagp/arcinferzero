// Independent source-code oracle for the private AR4_128 CPU/GPU layouts.
#include "../src/quants.h"
#include <cstdio>

int main() {
  int count = 0;
  if (gguf_sb_bytes(GGML_AR4_128) != 132 || raw_padded_sb_bytes(GGML_AR4_128) != 136) return 1;
  for (int K : {128, 256, 640, 768, 1024}) {
    const int Kp = padded_k(K), rows = 3;
    std::vector<uint8_t> source((size_t)rows * K / 128 * 66, 0);
    std::vector<float> expected((size_t)rows * K);
    for (int row = 0; row < rows; ++row) for (int b = 0; b < K / 128; ++b) {
      const float d = (b + row) % 2 ? 0.25f : 2.0f;
      const uint16_t h = f32_to_fp16(d);
      uint8_t* p = source.data() + ((size_t)row * K / 128 + b) * 66;
      memcpy(p, &h, 2);
      for (int j = 0; j < 128; ++j) {
        const int code = (j * 7 + b * 3 + row) % 16;
        p[2 + j / 32 * 16 + j % 16] |= code << (j % 32 < 16 ? 0 : 4);
        expected[(size_t)row * K + b * 128 + j] = d * (code - 8);
      }
    }
    std::vector<uint8_t> padded((size_t)rows * Kp / 128 * 66);
    pad_small_block_rows(GGML_AR4_128, source.data(), rows, K, padded.data());
    for (int row = 0; row < rows; ++row) {
      std::vector<float> decoded(Kp);
      dequant_row(GGML_AR4_128, padded.data() + (size_t)row * Kp / 128 * 66, Kp, decoded.data());
      for (int j = 0; j < Kp; ++j) {
        const float want = j < K ? expected[(size_t)row * K + j] : 0;
        if (decoded[j] != want) return 2;
      }
      for (int sb = 0; sb < Kp / 256; ++sb) {
        uint8_t gpu[136];
        gpu_raw_sb(GGML_AR4_128, padded.data() + ((size_t)row * Kp / 256 + sb) * 132, gpu);
        for (int j = 0; j < 256; ++j) {
          const int code = (gpu[4 + j / 32 * 16 + j % 16] >> (j % 32 < 16 ? 0 : 4)) & 15;
          const float got = fp16_to_f32(rd16(gpu + (j / 128) * 2)) * (code - 8);
          if (got != decoded[sb * 256 + j]) return 3;
          ++count;
        }
        for (int j = 132; j < 136; ++j) if (gpu[j]) return 4;
      }
    }
  }
  float output[32] = {};
  bool rejected = false;
  try { dequant_row(GGML_AR4_128, nullptr, 32, output); }
  catch (const std::runtime_error&) { rejected = true; }
  if (!rejected) return 5;
  std::printf("PASS: %d AR4 values, independent oracle, padding and invalid width\n", count);
}
