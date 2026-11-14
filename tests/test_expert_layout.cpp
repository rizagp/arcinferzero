// Expert-file v1 has four non-field bytes after type[]. They must not decide whether a package matches.
#include "../zero/expert_store.h"
#include <cstddef>
#include <cstdio>

int main() {
  using L = ExpertStore::LayerInfo;
  static_assert(sizeof(L) == 56 && offsetof(L, part) == 16, "expert-file v1 layout changed");
  L a{}, b{};
  for (int i = 0; i < 3; ++i) { a.type[i] = b.type[i] = 10 + i; a.part[i] = b.part[i] = 4096 * (i + 1); }
  a.rec = b.rec = 24576;
  a.off = b.off = 4096;
  for (size_t i = 12; i < offsetof(L, part); ++i) {
    reinterpret_cast<unsigned char*>(&a)[i] = 0x35;
    reinterpret_cast<unsigned char*>(&b)[i] = 0xac;
  }
  if (!(a == b)) return 1;
  for (int i = 0; i < 3; ++i) {
    ++b.type[i]; if (a == b) return 2; --b.type[i];
    ++b.part[i]; if (a == b) return 3; --b.part[i];
  }
  ++b.rec; if (a == b) return 4; --b.rec;
  ++b.off; if (a == b) return 5;
  std::puts("expert-file fields checked; padding ignored; v1 layout unchanged");
}
