// AR4_128 (AutoRound INT4 experts, pull request #22): the hybrid package (tools/import_autoround_pack) against the
// UD-Q4_K_XL model it was made from. Every tensor but the experts must be the same bytes, every expert tensor a
// virtual AR4_128 one (its data in the separate expert file), and every metadata value the same (split.* and the
// package's own inferzero.* keys aside).
// usage: test_autoround_package SOURCE.gguf PACKAGE.gguf
// Exit: 0 passed (1,080 tensors kept, 144 virtual), 1 a difference, 2 wrong arguments.
#include "../src/gguf.h"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: test_autoround_package SOURCE.gguf PACKAGE.gguf\n");
    return 2;
  }
  try {
    GgufFile source(argv[1]), candidate(argv[2]);
    size_t kept = 0, experts = 0;
    uint64_t bytes = 0;
    for (const auto& tensor : source.tensors) {
      const auto& target = candidate.get(tensor.name);
      if (target.dims != tensor.dims) throw std::runtime_error("shape differs: " + tensor.name);
      if (tensor.name.find("_exps.weight") != std::string::npos) {
        if (target.type != GGML_AR4_128 || target.data) throw std::runtime_error("invalid virtual expert: " + tensor.name);
        ++experts;
      } else {
        if (target.type != tensor.type || target.nbytes != tensor.nbytes || !target.data ||
            memcmp(target.data, tensor.data, tensor.nbytes))
          throw std::runtime_error("retained tensor differs: " + tensor.name);
        ++kept;
        bytes += tensor.nbytes;
      }
    }
    for (const auto& key : source.kv_order) {
      if (key.rfind("split.", 0) == 0 || key.rfind("inferzero.", 0) == 0) continue;
      if (source.kv.at(key) != candidate.kv.at(key) || source.kv_types.at(key) != candidate.kv_types.at(key))
        throw std::runtime_error("retained metadata differs: " + key);
    }
    printf("PASS: %zu unchanged tensors (%llu bytes), %zu virtual expert tensors, metadata/tokenizer unchanged\n", kept,
           (unsigned long long)bytes, experts);
    return kept == 1080 && experts == 144 ? 0 : 1;
  } catch (const std::exception& error) {
    fprintf(stderr, "test_autoround_package: %s\n", error.what());
    return 1;
  }
}
