// CPU-only checks of src/pbkdf2.h (the remote access password's hash on Linux, tools/server_remote.h) against
// PBKDF2-HMAC-SHA256 values: RFC 7914 section 11's first vector, its 80,000-round one, a password longer than one
// SHA-256 block (the key is hashed first) and a key length that is not a multiple of 32. The values were made with
// Python's hashlib.pbkdf2_hmac (2026-11-14, audit 2026-11-14 I4).
#include "../src/pbkdf2.h"
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
static void check(const char* name, bool ok) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", name);
  fails += !ok;
}
static std::string hex(const std::vector<uint8_t>& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : v) s += d[b >> 4], s += d[b & 15];
  return s;
}
static std::vector<uint8_t> bytes(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

int main() {
  check("RFC 7914: passwd / salt, 1 round, 64 bytes",
        hex(pbkdf2::derive("passwd", bytes("salt"), 1, 64)) ==
            "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783");
  check("RFC 7914: Password / NaCl, 80000 rounds, 64 bytes",
        hex(pbkdf2::derive("Password", bytes("NaCl"), 80000, 64)) ==
            "4ddcd8f60b98be21830cee5ef22701f9641a4418d04c0414aeff08876b34ab56a1d425a1225833549adb841b51c9b3176a272bdebba1d078478f62b397f33c8d");
  check("a 100-byte password (hashed first), 1000 rounds, 32 bytes",
        hex(pbkdf2::derive(std::string(100, 'x'), bytes("a salt of some length"), 1000, 32)) ==
            "9979a6f428ef07c06edf942e5cbad525fbe53b76349e88ea98174bde678d8834");
  check("40 bytes (two blocks, the second cut to 8)",
        hex(pbkdf2::derive("pw", bytes("salt"), 3, 40)) == "3c671234e386cb307220d70ee15c4985160b8a8dba62f247001b07974ad52ed1785bf34b5aa5dd58");
  if (fails) {
    printf("test_pbkdf2: %d failed\n", fails);
    return 1;
  }
  printf("test_pbkdf2: all passed\n");
  return 0;
}
