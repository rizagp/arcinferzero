// Model-free package policy checks: a same-shape expert file from another checkpoint must not load silently.
#include "../zero/expert_store.h"
#include <cstdio>
#include <filesystem>

int main() {
  const auto path = std::filesystem::temp_directory_path() /
                    ("inferzero-revision-" + std::to_string(plat::process_id()) + ".bin");
  struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); } } cleanup{path};
  {  // empty GGUF v3: only the policy's metadata/tensors below are needed (no model weights or GPU)
    FILE* out = fopen(path.string().c_str(), "wb");
    if (!out) return 1;
    const uint32_t header[8] = {0x46554747, 3, 0, 0, 0, 0, 0, 0};
    const bool ok = fwrite(header, sizeof header, 1, out) == 1;
    fclose(out);
    if (!ok) return 2;
  }
  GgufFile f(path.string());
  const std::string revision = "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897";
  constexpr uint64_t fp = 0x4c67bf686b7f7fd3ull;
  int checks = 0;
  auto expect = [&](uint64_t actual, bool fail, const char* message = "") {
    bool threw = false;
    try { check_autoround_expert_revision(f, actual); }
    catch (const std::runtime_error& e) {
      threw = true;
      if (std::string(e.what()).find(message) == std::string::npos) return false;
    }
    ++checks;
    return threw == fail;
  };
  // Non-packages and ordinary packages preserve the old fingerprint policy, even with unrelated metadata.
  if (!expect(fp ^ 1, false)) return 3;
  f.kv["inferzero.package"] = (int64_t)1;
  if (!expect(fp ^ 1, false)) return 4;
  f.kv["inferzero.autoround.revision"] = std::string("unrelated metadata");
  if (!expect(fp ^ 1, false)) return 5;
  GgufTensor t; t.type = GGML_AR4_128; f.tensors.push_back(t);
  f.kv["inferzero.autoround.revision"] = revision;
  if (!expect(fp, false) || !expect(fp ^ 1, true, "does not match")) return 6;
  f.kv["inferzero.autoround.revision"] = std::string("4C67BF686B7F7FD386BAE6B07AB59E8FF1D5B897");
  if (!expect(fp, false)) return 7;
  for (const std::string& invalid : {std::string(""), revision.substr(0,16), revision + "0",
                                   std::string("g") + revision.substr(1), revision.substr(0,39) + "z"}) {
    f.kv["inferzero.autoround.revision"] = invalid;
    if (!expect(fp, true, "40-digit hexadecimal")) return 8;
  }
  f.kv.erase("inferzero.autoround.revision");
  if (!expect(fp, true, "40-digit hexadecimal")) return 9;
  f.kv["inferzero.autoround.revision"] = (int64_t)42;
  if (!expect(fp, true, "40-digit hexadecimal")) return 10;
  f.kv["inferzero.autoround.revision"] = std::string(40, 'f');
  if (!expect(UINT64_MAX, false)) return 11;
  f.kv["inferzero.autoround.revision"] = std::string(40, '0');
  if (!expect(0, false)) return 12;
  // v1 only stores the prefix: do not claim whole-SHA or payload authentication.
  f.kv["inferzero.autoround.revision"] = revision.substr(0,16) + std::string(24,'0');
  if (!expect(fp, false)) return 13;
  // Exercise the actual expert-store load gate too, with an empty synthetic v1 file (no model weights).
  const auto experts = std::filesystem::path(path.string() + ".experts");
  Cleanup expert_cleanup{experts};
  unsigned char header[4096] = {};
  const uint32_t magic[2] = {ExpertStore::kMagic, ExpertStore::kVersion};
  const uint64_t fields[4] = {fp, 0, 1, sizeof header};
  memcpy(header, magic, sizeof magic);
  memcpy(header + sizeof magic, fields, sizeof fields);
  FILE* out = fopen(experts.string().c_str(), "wb");
  if (!out) return 14;
  const bool written = fwrite(header, sizeof header, 1, out) == 1;
  fclose(out);
  if (!written) return 15;
  f.kv["inferzero.experts_file"] = experts.filename().string();
  auto load = [&](bool fail, const char* message = "") {
    bool threw = false;
    try { ExpertStore store; store.open_or_build(f, path.string(), {}, 1); }
    catch (const std::runtime_error& e) {
      threw = true;
      if (std::string(e.what()).find(message) == std::string::npos) return false;
    }
    ++checks;
    return threw == fail;
  };
  f.kv["inferzero.autoround.revision"] = revision;
  if (!load(false)) return 16;
  f.kv["inferzero.autoround.revision"] = std::string(40, '0');
  if (!load(true, "does not match")) return 17;
  f.kv.erase("inferzero.autoround.revision");
  if (!load(true, "40-digit hexadecimal")) return 18;
  f.tensors[0].type = GGML_F32;
  if (!load(false)) return 19;
  f.tensors[0].type = GGML_AR4_128;
  f.kv["inferzero.autoround.revision"] = revision;
  const uint64_t other_fingerprint = fp ^ 1;
  memcpy(header + 8, &other_fingerprint, sizeof other_fingerprint);
  out = fopen(experts.string().c_str(), "wb");
  if (!out) return 20;
  const bool replaced = fwrite(header, sizeof header, 1, out) == 1;
  fclose(out);
  if (!replaced || !load(true, "does not match")) return 21;
  std::filesystem::resize_file(experts, 39);  // incomplete 40-byte v1 header
  if (!load(true, "expert file")) return 22;
  f.kv.erase("inferzero.package");
  if (!expect(other_fingerprint, false)) return 23;
  printf("test_autoround_revision: %d checks passed\n", checks);
}
