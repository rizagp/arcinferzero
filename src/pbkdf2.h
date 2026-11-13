// SHA-256 (FIPS 180-4), HMAC-SHA256 (RFC 2104) and PBKDF2-HMAC-SHA256 (RFC 8018) in plain C++: the remote access
// password's hash on Linux (tools/server_remote.h), where Windows uses CNG (BCryptDeriveKeyPBKDF2) for the same
// function, so a password set on one system is read the same on the other. Checked against RFC 7914's PBKDF2 vectors
// and against CNG (docs/linux-port.md).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace pbkdf2 {

struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf[64] = {};
  size_t n = 0;
  uint64_t len = 0;
  static uint32_t rotr(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
  void add(const uint8_t* p, size_t k) {
    len += k;
    while (k) {
      const size_t take = std::min<size_t>(64 - n, k);
      memcpy(buf + n, p, take);
      n += take, p += take, k -= take;
      if (n == 64) block(buf), n = 0;
    }
  }
  void finish(uint8_t out[32]) {
    const uint64_t bits = len * 8;
    const uint8_t one = 0x80, zero = 0;
    add(&one, 1);
    while (n != 56) add(&zero, 1);
    uint8_t l[8];
    for (int i = 0; i < 8; ++i) l[i] = (uint8_t)(bits >> (56 - 8 * i));
    add(l, 8);
    for (int i = 0; i < 8; ++i)
      for (int k = 0; k < 4; ++k) out[4 * i + k] = (uint8_t)(h[i] >> (24 - 8 * k));
  }
};

// HMAC-SHA256 with the key's inner and outer pads hashed once (PBKDF2 uses the same key for every round)
struct Hmac {
  Sha256 in, out;
  explicit Hmac(const std::string& key) {
    uint8_t k[64] = {};
    if (key.size() > 64) {
      Sha256 s;
      s.add((const uint8_t*)key.data(), key.size());
      s.finish(k);
    } else {
      memcpy(k, key.data(), key.size());
    }
    uint8_t ip[64], op[64];
    for (int i = 0; i < 64; ++i) ip[i] = k[i] ^ 0x36, op[i] = k[i] ^ 0x5c;
    in.add(ip, 64);
    out.add(op, 64);
  }
  void mac(const uint8_t* p, size_t n, uint8_t res[32]) const {
    Sha256 a = in, b = out;
    uint8_t t[32];
    a.add(p, n);
    a.finish(t);
    b.add(t, 32);
    b.finish(res);
  }
};

// PBKDF2-HMAC-SHA256: dk_len bytes from the password, the salt and the round count
inline std::vector<uint8_t> derive(const std::string& password, const std::vector<uint8_t>& salt, unsigned iters, size_t dk_len) {
  const Hmac h(password);
  std::vector<uint8_t> dk;
  for (uint32_t blk = 1; dk.size() < dk_len; ++blk) {
    std::vector<uint8_t> s = salt;
    for (int i = 3; i >= 0; --i) s.push_back((uint8_t)(blk >> (8 * i)));
    uint8_t u[32], t[32];
    h.mac(s.data(), s.size(), u);
    memcpy(t, u, 32);
    for (unsigned i = 1; i < iters; ++i) {
      h.mac(u, 32, u);
      for (int k = 0; k < 32; ++k) t[k] ^= u[k];
    }
    dk.insert(dk.end(), t, t + std::min<size_t>(32, dk_len - dk.size()));
  }
  return dk;
}

}  // namespace pbkdf2
