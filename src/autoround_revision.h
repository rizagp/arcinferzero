// An AutoRound package and its separate expert file must name the same import revision. Expert-file v1 stores
// the first 16 hexadecimal digits (64 bits) of the full checkpoint SHA, not a hash of the weights themselves.
#pragma once
#include "gguf.h"

inline void check_autoround_expert_revision(const GgufFile& f, uint64_t fingerprint) {
  if (!f.has("inferzero.package")) return;
  bool ar4 = false;
  for (const auto& t : f.tensors) if (t.type == GGML_AR4_128) { ar4 = true; break; }
  if (!ar4) return;
  const std::string revision = f.get_str("inferzero.autoround.revision");
  if (revision.size() != 40)
    throw std::runtime_error("AutoRound package: inferzero.autoround.revision must be a full 40-digit hexadecimal checkpoint SHA");
  uint64_t expected = 0;
  for (size_t i = 0; i < revision.size(); ++i) {
    const char c = revision[i];
    const int digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                      c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
    if (digit < 0)
      throw std::runtime_error("AutoRound package: inferzero.autoround.revision must be a full 40-digit hexadecimal checkpoint SHA");
    if (i < 16) expected = (expected << 4) | (uint64_t)digit;
  }
  if (fingerprint != expected)
    throw std::runtime_error("AutoRound package: expert-file revision fingerprint does not match inferzero.autoround.revision; "
                             "use the expert file imported from checkpoint " + revision);
}
