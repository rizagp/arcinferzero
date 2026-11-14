// AR4_128 (AutoRound INT4 experts, pull request #22): one imported expert's gate / up / down, three ways.
//   1. the source checkpoint's own f32 values (dumped by the importer's checks, the "golden" files);
//   2. the CPU decoder (dequant_row) on the expert file's records turned back into GGUF blocks;
//   3. the GPU's vblock (zero/kernels/decode.cl, as gemv_v and the expert kernels read it).
// All three must be equal bit for bit, the down projection's padding (K 640 -> 768) included.
// usage: test_autoround_decode EXPERTS_FILE GATE.f32 UP.f32 DOWN.f32   (run from the repository: it reads zero/kernels)
// Exit: 0 passed, 1 a mismatch or a file too short, 2 wrong arguments.
#include "../zero/clrt.h"
#include "../src/quants.h"
#include <cstdio>
#include <fstream>

namespace {

// The check kernel: work-item (row, 32-block) decodes its block with vblock and writes the 32 floats.
const char* kCheck = R"(
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ar4_check(__global const uchar* raw, int N, int K, __global const uint* tabs, __global float* out) {
  __local uint table[VB_TABLES];
  vb_tables(table, tabs);
  const int row = get_global_id(0), block = get_global_id(1), s = block / 8, ib = block % 8;
  if (row >= N) return;
  uint w[8];
  float d, dm;
  int sc0, sc1, m0, m1;
  vblock(raw + ((size_t)row * (K / 256) + s) * BBP, ib, table, w, &d, &dm, &sc0, &sc1, &m0, &m1);
  for (int j = 0; j < 32; ++j) out[(size_t)row * K + block * 32 + j] = d * (float)(char)(w[j / 4] >> (8 * (j % 4)));
}
)";

// A GPU superblock (136 bytes: 2 scales, 2 x 64 nibble bytes, 4 pad) back to its two GGUF blocks (2 x 66 bytes).
void gpu_to_gguf(const uint8_t* p, uint8_t* q) {
  memcpy(q, p, 2);            // first 128-group: its scale
  memcpy(q + 2, p + 4, 64);   //                  its nibbles
  memcpy(q + 66, p + 2, 2);   // second group: its scale
  memcpy(q + 68, p + 68, 64); //               its nibbles
}

int fail(const char* what, int part) {
  fprintf(stderr, "test_autoround_decode: %s (part %d)\n", what, part);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: test_autoround_decode EXPERTS_FILE GATE.f32 UP.f32 DOWN.f32\n");
    return 2;
  }
  ClRt cl;
  if (!cl.init()) return 1;
  cl_program program = cl.build(read_file("zero/kernels/decode.cl") + kCheck,
                                "-cl-std=CL3.0 -DQT=" + std::to_string(GGML_AR4_128));
  cl_kernel kernel = cl.kernel(program, "ar4_check");
  const std::vector<uint32_t> table = build_iq_tables();

  std::ifstream input(argv[1], std::ios::binary);
  input.seekg(4096);  // (the expert file's header and layer table; the first record follows)
  size_t checked = 0;
  for (int part = 0; part < 3; ++part) {  // gate, up (640 x 2560), down (2560 x 640)
    const int N = part == 2 ? 2560 : 640, K = part == 2 ? 640 : 2560, Kp = padded_k(K);
    const size_t row_bytes = (size_t)Kp / 256 * 136;
    std::vector<uint8_t> raw((size_t)N * row_bytes);
    input.read((char*)raw.data(), raw.size());
    if (!input) return fail("expert file too short", part);
    std::vector<float> golden((size_t)N * K), want((size_t)N * Kp), got(want.size());
    std::ifstream ref(argv[part + 2], std::ios::binary);
    ref.read((char*)golden.data(), golden.size() * 4);
    if (!ref) return fail("source dump too short", part);

    // 2. the CPU decoder against the source values
    for (int row = 0; row < N; ++row) {
      std::vector<uint8_t> gguf((size_t)Kp / 128 * 66);
      for (int s = 0; s < Kp / 256; ++s) gpu_to_gguf(raw.data() + row * row_bytes + s * 136, gguf.data() + s * 132);
      dequant_row(GGML_AR4_128, gguf.data(), Kp, want.data() + (size_t)row * Kp);
      if (memcmp(want.data() + (size_t)row * Kp, golden.data() + (size_t)row * K, (size_t)K * 4))
        return fail("CPU decoder differs from the source", part);
    }

    // 3. the GPU's vblock against the CPU decoder (the padding too)
    cl_mem weights = cl.buffer(CL_MEM_READ_ONLY, raw.size(), raw.data());
    cl_mem tabs = cl.buffer(CL_MEM_READ_ONLY, table.size() * 4, table.data());
    cl_mem output = cl.buffer(CL_MEM_READ_WRITE, got.size() * 4);
    clSetKernelArg(kernel, 0, sizeof(weights), &weights);
    clSetKernelArg(kernel, 1, 4, &N);
    clSetKernelArg(kernel, 2, 4, &Kp);
    clSetKernelArg(kernel, 3, sizeof(tabs), &tabs);
    clSetKernelArg(kernel, 4, sizeof(output), &output);
    const size_t global[2] = {(size_t)(N + 63) / 64 * 64, (size_t)Kp / 32}, local[2] = {64, 1};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue, output, CL_TRUE, 0, got.size() * 4, got.data(), 0, nullptr, nullptr));
    for (cl_mem b : {weights, tabs, output}) clReleaseMemObject(b);
    if (memcmp(want.data(), got.data(), got.size() * 4)) return fail("GPU vblock differs from the CPU decoder", part);

    checked += golden.size();
    printf("part=%d source/CPU/GPU bit-identical (%zu weights)\n", part, golden.size());
    fflush(stdout);
  }
  printf("PASS: %zu real source weights, including down padding\n", checked);
  return 0;
}
