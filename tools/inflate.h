// DEFLATE (RFC 1951) decoding, zlib streams (RFC 1950) and ZIP archives, for reading the documents users upload: a
// DOCX is a ZIP of deflated XML parts, a PDF keeps its pages in FlateDecode streams. Small and dependency-free; a
// damaged stream gives what was decoded up to the damage (ok = false), never reads outside its input, and stops at
// max_out bytes (a "zip bomb" cannot fill the memory).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace infl {

struct Bits {
  const unsigned char* p;
  size_t n, pos = 0;  // (pos in bits)
  bool bad = false;
  unsigned get(int k) {
    unsigned v = 0;
    for (int i = 0; i < k; ++i, ++pos) {
      if (pos >> 3 >= n) {
        bad = true;
        return 0;
      }
      v |= ((p[pos >> 3] >> (pos & 7)) & 1u) << i;
    }
    return v;
  }
};

// a canonical Huffman code: counts per length and the symbols in code order
struct Huff {
  uint16_t count[16] = {};
  uint16_t sym[320] = {};
  bool build(const uint8_t* len, int n) {
    memset(count, 0, sizeof count);
    for (int i = 0; i < n; ++i) count[len[i]]++;
    count[0] = 0;
    uint16_t offs[16] = {};
    int left = 1;
    for (int l = 1; l < 16; ++l) {
      left <<= 1;
      left -= count[l];
      if (left < 0) return false;  // (over-subscribed)
      offs[l] = (uint16_t)(l == 1 ? 0 : offs[l - 1] + count[l - 1]);
    }
    for (int i = 0; i < n; ++i)
      if (len[i]) sym[offs[len[i]]++] = (uint16_t)i;
    return true;
  }
  int decode(Bits& b) const {
    int code = 0, first = 0, index = 0;
    for (int l = 1; l < 16; ++l) {
      code |= (int)b.get(1);
      const int c = count[l];
      if (code - c < first) return sym[index + (code - first)];
      index += c;
      first = (first + c) << 1;
      code <<= 1;
      if (b.bad) return -1;
    }
    return -1;
  }
};

// raw DEFLATE data into out; false: damaged (out holds what came before) or longer than max_out
inline bool inflate(const unsigned char* data, size_t n, std::string& out, size_t max_out = 256u << 20) {
  static const uint16_t lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const uint8_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
  static const uint16_t dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
  static const uint8_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  Bits b{data, n};
  int last = 0;
  do {
    last = (int)b.get(1);
    const unsigned type = b.get(2);
    if (b.bad) return false;
    if (type == 0) {  // (stored)
      b.pos = (b.pos + 7) & ~size_t(7);
      if ((b.pos >> 3) + 4 > n) return false;
      const unsigned char* q = data + (b.pos >> 3);
      const unsigned len = q[0] | q[1] << 8, nlen = q[2] | q[3] << 8;
      if ((len ^ 0xffff) != nlen || (b.pos >> 3) + 4 + len > n) return false;
      if (out.size() + len > max_out) return false;
      out.append((const char*)q + 4, len);
      b.pos += (size_t)(4 + len) * 8;
      continue;
    }
    if (type == 3) return false;
    Huff lit, dist;
    uint8_t len[320] = {};
    if (type == 1) {  // (fixed codes)
      for (int i = 0; i < 144; ++i) len[i] = 8;
      for (int i = 144; i < 256; ++i) len[i] = 9;
      for (int i = 256; i < 280; ++i) len[i] = 7;
      for (int i = 280; i < 288; ++i) len[i] = 8;
      lit.build(len, 288);
      for (int i = 0; i < 30; ++i) len[i] = 5;
      dist.build(len, 30);
    } else {  // (dynamic codes)
      const int hlit = (int)b.get(5) + 257, hdist = (int)b.get(5) + 1, hclen = (int)b.get(4) + 4;
      static const uint8_t ord[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
      uint8_t cl[19] = {};
      for (int i = 0; i < hclen; ++i) cl[ord[i]] = (uint8_t)b.get(3);
      Huff ch;
      if (b.bad || !ch.build(cl, 19)) return false;
      for (int i = 0; i < hlit + hdist;) {
        const int s = ch.decode(b);
        if (s < 0) return false;
        if (s < 16) len[i++] = (uint8_t)s;
        else {
          int rep = 0;
          uint8_t v = 0;
          if (s == 16) {
            if (i == 0) return false;
            v = len[i - 1];
            rep = 3 + (int)b.get(2);
          } else if (s == 17) rep = 3 + (int)b.get(3);
          else rep = 11 + (int)b.get(7);
          if (i + rep > hlit + hdist) return false;
          while (rep--) len[i++] = v;
        }
      }
      if (!lit.build(len, hlit) || !dist.build(len + hlit, hdist)) return false;
    }
    while (true) {
      const int s = lit.decode(b);
      if (s < 0) return false;
      if (s < 256) {
        if (out.size() >= max_out) return false;
        out += (char)s;
      } else if (s == 256) break;
      else {
        const int li = s - 257;
        if (li >= 29) return false;
        const size_t l = lbase[li] + b.get(lext[li]);
        const int di = dist.decode(b);
        if (di < 0 || di >= 30) return false;
        const size_t d = dbase[di] + b.get(dext[di]);
        if (b.bad || d > out.size() || out.size() + l > max_out) return false;
        const size_t from = out.size() - d;
        for (size_t k = 0; k < l; ++k) out += out[from + k];  // (may overlap: byte by byte)
      }
    }
  } while (!last);
  return true;
}

// a zlib stream (2-byte header, DEFLATE, Adler-32): PDF's FlateDecode
inline bool zlib(const std::string& in, std::string& out, size_t max_out = 256u << 20) {
  if (in.size() < 2) return false;
  const unsigned char* p = (const unsigned char*)in.data();
  const bool hdr = (p[0] & 0x0f) == 8 && ((p[0] << 8) | p[1]) % 31 == 0;
  return hdr ? inflate(p + 2, in.size() - 2, out, max_out) : inflate(p, in.size(), out, max_out);  // (some writers leave it out)
}

// one entry of a ZIP archive by name (stored or deflated); false: no such entry or damaged
inline bool zip_entry(const std::string& zip, const std::string& name, std::string& out, size_t max_out = 256u << 20) {
  const unsigned char* z = (const unsigned char*)zip.data();
  auto u16 = [&](size_t o) { return o + 2 <= zip.size() ? (unsigned)(z[o] | z[o + 1] << 8) : 0u; };
  auto u32 = [&](size_t o) { return o + 4 <= zip.size() ? (uint32_t)(z[o] | z[o + 1] << 8 | z[o + 2] << 16 | (uint32_t)z[o + 3] << 24) : 0u; };
  // the end of central directory record: in the last 64 KB + 22 bytes
  if (zip.size() < 22) return false;
  size_t e = zip.size() - 22;
  const size_t stop = zip.size() > 65557 ? zip.size() - 65557 : 0;
  while (u32(e) != 0x06054b50) {
    if (e == stop) return false;
    --e;
  }
  size_t cd = u32(e + 16);
  const unsigned entries = u16(e + 10);
  for (unsigned k = 0; k < entries && cd + 46 <= zip.size() && u32(cd) == 0x02014b50; ++k) {
    const unsigned method = u16(cd + 10), nl = u16(cd + 28), xl = u16(cd + 30), cl = u16(cd + 32);
    const uint32_t csize = u32(cd + 20), local = u32(cd + 42);
    if (cd + 46 + nl > zip.size()) return false;
    if (zip.compare(cd + 46, nl, name) == 0) {
      if ((size_t)local + 30 > zip.size()) return false;
      const size_t data = local + 30 + u16(local + 26) + u16(local + 28);
      if (data + csize > zip.size()) return false;
      if (method == 0) {
        if (csize > max_out) return false;
        out.assign(zip, data, csize);
        return true;
      }
      return method == 8 && inflate(z + data, csize, out, max_out);
    }
    cd += 46 + nl + xl + cl;
  }
  return false;
}

}  // namespace infl
