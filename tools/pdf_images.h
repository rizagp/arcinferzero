// Scanned PDF pages (an image without a text layer) for a vision model: the image a page draws, taken out of the PDF
// on the CPU without a library, as bytes the Windows Imaging Component decodes (src/image.h img::decode):
//   DCTDecode            the JPEG bytes as they are
//   FlateDecode or none  the raw samples (DeviceGray / RGB / CMYK, ICCBased, CalGray / CalRGB, Indexed, image masks;
//                        1-16 bits; PNG predictors) as a BMP
//   CCITTFaxDecode       the fax data (Group 4, Group 3 1-D and 2-D) behind a minimal TIFF header
//   JBIG2, JPEG 2000, LZW ...: a note only
// The server's read_file has the loaded vision model transcribe such a page (tools/server.cpp); the text is kept per
// file and page (tools/server_files.h) and joins the document's paragraphs here, so search_file finds it.
//   scan_pages(data, doc)   doc.scans: the pages with (almost) no text that draw an image of 0.05 megapixels or more
//   page_image(data, page)  the largest image the page (1-based) draws, ready for img::decode; or a note why not
//   add_pages(doc, texts)   transcriptions (page -> text) into the document's paragraphs; doc.unread and doc.note
// Hostile input: image sizes as for uploads (16384 pixels a side, 64 megapixels), raw samples only as far as they
// decoded (each stream's inflate is capped), content streams walked with a step cap, form XObjects 4 deep.
#pragma once
#include "doc_reader.h"

namespace rdoc {
namespace pdfimg {

using pdf::Obj;
using pdf::P;

struct Image {
  std::string bytes, mime;  // (empty: no image, the note says why)
  int w = 0, h = 0;
  std::string filter;  // (how the PDF stores it: DCTDecode, CCITTFaxDecode, FlateDecode, ...)
  std::string note;
};
inline constexpr int kMaxSide = 16384;               // (as img::decode takes uploads)
inline constexpr uint64_t kMaxPixels = 64ull << 20;
inline constexpr size_t kTextChars = 40;              // (a page with less text than this and a large image: a scan)
inline constexpr uint64_t kScanPixels = 50000;      // (224 x 224: no icon or logo; a scan has 1-9 megapixels)

// "1-5, 8, 10-12"
inline std::string ranges(const std::vector<int>& v) {
  std::string o;
  int groups = 0;
  for (size_t k = 0; k < v.size();) {
    size_t e = k;
    while (e + 1 < v.size() && v[e + 1] == v[e] + 1) ++e;
    if (++groups > 12) { o += ", ..."; break; }
    o += (o.empty() ? "" : ", ") + std::to_string(v[k]) + (e > k ? "-" + std::to_string(v[e]) : "");
    k = e + 1;
  }
  return o;
}

// the image XObjects a page draws ("Do" in its content stream and in its form XObjects), each once, with the
// resources they were drawn from
inline std::vector<std::pair<P, P>> drawn(const pdf::File& f, const P& pg, const P& res) {
  std::vector<std::pair<P, P>> out;
  std::set<const Obj*> seen;
  int steps = 0;
  std::function<void(const std::string&, const P&, int)> run = [&](const std::string& cs, const P& r, int depth) {
    if (depth > 4) return;
    const P xobj = f.get(r, "XObject");
    pdf::Lexer lx{cs, 0};
    P last;  // (the operand before the operator)
    while (lx.i < cs.size() && out.size() < 256) {
      if (++steps > 4000000) return;
      std::string op;
      if (P v = lx.value(&op)) { last = v; continue; }
      if (op.empty()) {
        if (lx.i >= cs.size()) break;
        continue;
      }
      if (op == "BI") {  // (an inline image: its data up to EI)
        const size_t e = cs.find("EI", lx.i);
        lx.i = e == std::string::npos ? cs.size() : e + 2;
      } else if (op == "Do" && last && last->t == Obj::Name) {
        const P xo = f.get(xobj, last->s);
        if (xo && !seen.count(xo.get())) {
          seen.insert(xo.get());
          const P sub = f.get(xo, "Subtype");
          if (sub && sub->s == "Image") out.push_back({xo, r});
          else if (sub && sub->s == "Form") {
            const P r2 = f.get(xo, "Resources");
            run(f.decode(xo), r2 ? r2 : r, depth + 1);
          }
        }
      }
      last = nullptr;
    }
  };
  std::string content;
  const P c = f.get(pg, "Contents");
  if (c && c->t == Obj::Arr) {
    for (auto& x : c->a) content += f.decode(f.resolve(x)) + "\n";
  } else content = f.decode(c);
  run(content, res, 0);
  return out;
}

inline int num(const pdf::File& f, const P& d, const char* k, int def) {
  const P v = f.get(d, k);
  return v && v->t == Obj::Num && std::isfinite(v->n) ? (int)std::clamp(v->n, -1e9, 1e9) : def;
}
inline bool flag(const pdf::File& f, const P& d, const char* k) {
  const P v = f.get(d, k);
  return v && v->t == Obj::Bool && v->n != 0;
}

// a color space as the samples are turned into RGB: components (1 gray or ink, 3 RGB, 4 CMYK) and, for Indexed,
// its palette as RGB triples
struct Space {
  int n = 0;
  bool ink = false;     // (Separation: 1 is full ink, dark)
  std::string palette;  // (Indexed: (hival + 1) * 3 bytes of RGB)
  std::string name;
};
inline bool space(const pdf::File& f, P cs, const P& res, Space& s, int depth = 0) {
  cs = f.resolve(cs);
  if (!cs || depth > 4) return false;
  if (cs->t == Obj::Name) {
    const std::string& n = cs->s;
    s.name = n;
    if (n == "DeviceGray" || n == "G" || n == "CalGray") { s.n = 1; return true; }
    if (n == "DeviceRGB" || n == "RGB" || n == "CalRGB") { s.n = 3; return true; }
    if (n == "DeviceCMYK" || n == "CMYK") { s.n = 4; return true; }
    return space(f, f.get(f.get(res, "ColorSpace"), n), res, s, depth + 1);  // (a name in the page's resources)
  }
  if (cs->t != Obj::Arr || cs->a.empty()) return false;
  const P head = f.resolve(cs->a[0]);
  if (!head || head->t != Obj::Name) return false;
  const std::string& n = head->s;
  s.name = n;
  if (n == "CalGray") { s.n = 1; return true; }
  if (n == "CalRGB") { s.n = 3; return true; }
  if (n == "ICCBased" && cs->a.size() >= 2) {
    const int c = num(f, f.resolve(cs->a[1]), "N", 0);
    if (c == 1 || c == 3 || c == 4) { s.n = c; return true; }
    return false;
  }
  if ((n == "Separation" || n == "DeviceN") && cs->a.size() >= 3) {
    const P names = f.resolve(cs->a[1]);
    if (n == "DeviceN" && (!names || names->t != Obj::Arr || names->a.size() != 1)) return false;
    s.n = 1;
    s.ink = true;
    return true;
  }
  if ((n == "Indexed" || n == "I") && cs->a.size() >= 4) {
    Space base;
    if (!space(f, cs->a[1], res, base, depth + 1) || !base.palette.empty()) return false;
    const P hv = f.resolve(cs->a[2]), lut = f.resolve(cs->a[3]);
    if (!hv || hv->t != Obj::Num) return false;
    const int hival = (int)std::clamp(hv->n, 0.0, 255.0);
    std::string t = lut && lut->t == Obj::Str ? lut->s : lut && lut->has_stream ? f.decode(lut) : "";
    t.resize((size_t)(hival + 1) * base.n, '\0');
    s.n = 1;
    for (int k = 0; k <= hival; ++k) {
      const unsigned char* p = (const unsigned char*)t.data() + (size_t)k * base.n;
      int r, g, b;
      if (base.n == 1) r = g = b = base.ink ? 255 - p[0] : p[0];
      else if (base.n == 3) { r = p[0]; g = p[1]; b = p[2]; }
      else { r = (255 - p[0]) * (255 - p[3]) / 255; g = (255 - p[1]) * (255 - p[3]) / 255; b = (255 - p[2]) * (255 - p[3]) / 255; }
      s.palette += (char)r;
      s.palette += (char)g;
      s.palette += (char)b;
    }
    return true;
  }
  return false;
}

// a BMP of w x h pixels: one channel with a palette (pal: 256 RGB triples), or RGB (rgb: w * h * 3, top row first)
inline std::string bmp(int w, int h, const std::string& idx, const std::string& pal, const std::string& rgb) {
  const bool paletted = !pal.empty();
  const size_t row = (((size_t)w * (paletted ? 1 : 3)) + 3) & ~(size_t)3, head = 14 + 40 + (paletted ? 1024 : 0);
  std::string o(head + row * h, '\0');
  auto put = [&](size_t at, uint32_t v, int bytes) {
    for (int k = 0; k < bytes; ++k) o[at + k] = (char)(v >> (8 * k));
  };
  o[0] = 'B';
  o[1] = 'M';
  put(2, (uint32_t)o.size(), 4);
  put(10, (uint32_t)head, 4);
  put(14, 40, 4);
  put(18, (uint32_t)w, 4);
  put(22, (uint32_t)h, 4);  // (positive: the bottom row first)
  put(26, 1, 2);
  put(28, paletted ? 8 : 24, 2);
  put(34, (uint32_t)(row * h), 4);
  put(38, 2835, 4);  // (72 dpi)
  put(42, 2835, 4);
  if (paletted) {
    put(46, 256, 4);
    for (int k = 0; k < 256; ++k) {
      o[54 + k * 4] = pal[k * 3 + 2];
      o[54 + k * 4 + 1] = pal[k * 3 + 1];
      o[54 + k * 4 + 2] = pal[k * 3];
    }
  }
  for (int y = 0; y < h; ++y) {
    char* d = &o[head + row * (size_t)(h - 1 - y)];
    if (paletted) memcpy(d, idx.data() + (size_t)y * w, (size_t)w);
    else
      for (int x = 0; x < w; ++x) {
        const char* s = rgb.data() + ((size_t)y * w + x) * 3;
        d[x * 3] = s[2];
        d[x * 3 + 1] = s[1];
        d[x * 3 + 2] = s[0];
      }
  }
  return o;
}

// raw samples (decoded: FlateDecode or no filter) as a BMP
inline bool raw_image(const pdf::File& f, const P& xo, const P& res, const std::string& data, Image& im) {
  const bool mask = flag(f, xo, "ImageMask");
  Space s;
  if (mask) s.n = 1;
  else if (!space(f, f.get(xo, "ColorSpace"), res, s)) {
    const P cs = f.get(xo, "ColorSpace");
    im.note = "an image in a color space that is not read here (" + (s.name.empty() ? std::string(cs ? "unknown" : "none given") : s.name) + ")";
    return false;
  }
  const int bpc = mask ? 1 : num(f, xo, "BitsPerComponent", 8);
  if (bpc != 1 && bpc != 2 && bpc != 4 && bpc != 8 && bpc != 16) {
    im.note = "an image of " + std::to_string(bpc) + " bits per sample";
    return false;
  }
  const int w = im.w, h = im.h, n = s.n;
  const size_t rowbytes = ((size_t)w * n * bpc + 7) / 8, need = rowbytes * h;
  if (data.size() < need / 2) {  // (a damaged stream: half a page or less is no page to read)
    im.note = "a damaged image stream (" + std::to_string(data.size()) + " of " + std::to_string(need) + " bytes)";
    return false;
  }
  std::string d = data;
  if (d.size() < need) d.resize(need, '\xFF');  // (the missing rows: white)
  const P dec = f.get(xo, "Decode");
  bool inv = false;
  if (dec && dec->t == Obj::Arr && dec->a.size() >= 2) {
    const P a = f.resolve(dec->a[0]), b = f.resolve(dec->a[1]);
    inv = a && b && a->t == Obj::Num && b->t == Obj::Num && a->n > b->n;
  }
  const int maxv = (1 << std::min(bpc, 8)) - 1;
  auto sample = [&](int y, size_t k) -> int {  // (sample k of row y, at most 8 bits: 16-bit samples by their high byte)
    const unsigned char* r = (const unsigned char*)d.data() + rowbytes * (size_t)y;
    if (bpc == 8) return r[k];
    if (bpc == 16) return r[k * 2];
    const size_t bit = k * bpc;
    return (r[bit / 8] >> (8 - bpc - (int)(bit % 8))) & maxv;
  };
  if (n == 1) {
    std::string idx((size_t)w * h, '\0'), pal(768, '\0');
    if (!s.palette.empty()) {
      memcpy(&pal[0], s.palette.data(), std::min<size_t>(768, s.palette.size()));
      const int hival = (int)s.palette.size() / 3 - 1;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) idx[(size_t)y * w + x] = (char)std::min(sample(y, (size_t)x), hival);
    } else {
      for (int k = 0; k < 256; ++k) {  // (sample value -> gray; masks and gray: 0 black; ink: 0 white)
        int g = k > maxv ? 255 : k * 255 / maxv;
        if (inv != s.ink) g = 255 - g;
        pal[k * 3] = pal[k * 3 + 1] = pal[k * 3 + 2] = (char)g;
      }
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) idx[(size_t)y * w + x] = (char)sample(y, (size_t)x);
    }
    im.bytes = bmp(w, h, idx, pal, "");
  } else {
    std::string rgb((size_t)w * h * 3, '\0');
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int v[4] = {};  // (a colour space of n components fills v[0, n): the rest 0, not undefined)
        for (int c = 0; c < n; ++c) {
          v[c] = sample(y, (size_t)x * n + c) * 255 / maxv;
          if (inv) v[c] = 255 - v[c];
        }
        char* o = &rgb[((size_t)y * w + x) * 3];
        if (n == 3) { o[0] = (char)v[0]; o[1] = (char)v[1]; o[2] = (char)v[2]; }
        else
          for (int c = 0; c < 3; ++c) o[c] = (char)((255 - v[c]) * (255 - v[3]) / 255);
      }
    im.bytes = bmp(w, h, "", "", rgb);
  }
  im.mime = "image/bmp";
  return true;
}

// CCITT fax data behind a TIFF header (one strip): Group 4 (K < 0) or Group 3 (K 0: 1-D, K > 0: 2-D)
inline std::string ccitt_tiff(const std::string& data, int w, int h, int k, bool align, bool white_is_one) {
  std::string o = std::string("II*\0", 4);
  auto u16 = [&](uint32_t v) { o += (char)(v & 255); o += (char)(v >> 8 & 255); };
  auto u32 = [&](uint32_t v) { for (int b = 0; b < 4; ++b) o += (char)(v >> (8 * b) & 255); };
  u32(8);
  std::vector<std::array<uint32_t, 3>> tags = {  // (tag, type: 3 SHORT / 4 LONG, value), in tag order
      {256, 4, (uint32_t)w}, {257, 4, (uint32_t)h}, {258, 3, 1}, {259, 3, k < 0 ? 4u : 3u},
      {262, 3, white_is_one ? 1u : 0u},  // (0 WhiteIsZero: as a fax decodes; 1 when the PDF draws it the other way)
      {273, 4, 0}, {277, 3, 1}, {278, 4, (uint32_t)h}, {279, 4, (uint32_t)data.size()}};
  if (k >= 0) tags.push_back({292, 4, (k > 0 ? 1u : 0u) | (align ? 4u : 0u)});  // (T4Options: 2-D, byte-aligned EOLs)
  else tags.push_back({293, 4, 0});                                          // (T6Options)
  const uint32_t at = 8 + 2 + (uint32_t)tags.size() * 12 + 4;  // (the strip right after the directory)
  u16((uint32_t)tags.size());
  for (auto& t : tags) {
    u16(t[0]);
    u16(t[1]);
    u32(1);
    const uint32_t v = t[0] == 273 ? at : t[2];
    if (t[1] == 3) { u16(v); u16(0); }
    else u32(v);
  }
  u32(0);
  return o + data;
}

// image xo as bytes img::decode reads
inline Image image(const pdf::File& f, const P& xo, const P& res) {
  Image im;
  im.w = num(f, xo, "Width", 0);
  im.h = num(f, xo, "Height", 0);
  const std::vector<std::string> fs = f.filters(xo);
  im.filter = fs.empty() ? "none" : fs.back();
  if (im.w <= 0 || im.h <= 0 || im.w > kMaxSide || im.h > kMaxSide || (uint64_t)im.w * im.h > kMaxPixels) {
    im.note = "an image of " + std::to_string(im.w) + " x " + std::to_string(im.h) + " pixels (at most " +
              std::to_string(kMaxSide) + " a side and 64 megapixels)";
    return im;
  }
  std::string last;
  const std::string data = f.decode(xo, &last);
  if (last.empty() && !fs.empty() && fs.back() != "FlateDecode") {
    im.note = "an image stored with " + fs.back() + ", which is not read here";
    return im;
  }
  if (data.empty()) {
    im.note = "an empty or damaged image stream";
    return im;
  }
  if (last == "DCTDecode") {
    im.bytes = data;
    im.mime = "image/jpeg";
  } else if (last == "CCITTFaxDecode") {
    const P pa = f.parms(xo, fs.size() - 1);
    const int k = num(f, pa, "K", 0), cols = num(f, pa, "Columns", 1728), rows = num(f, pa, "Rows", 0);
    im.w = cols;  // (the fax data's own width counts)
    if (rows > 0 && rows <= kMaxSide) im.h = rows;
    if (im.w <= 0 || im.w > kMaxSide || (uint64_t)im.w * im.h > kMaxPixels) {
      im.note = "a fax image of " + std::to_string(im.w) + " columns";
      return im;
    }
    // (decoded, a fax's white runs are 1 bits unless BlackIs1; a gray image or a mask shows 0 as black. A
    //  /Decode [1 0] turns that round once more.)
    const P dec = f.get(xo, "Decode");
    bool inv = false;
    if (dec && dec->t == Obj::Arr && dec->a.size() >= 2) {
      const P a = f.resolve(dec->a[0]), b = f.resolve(dec->a[1]);
      inv = a && b && a->t == Obj::Num && b->t == Obj::Num && a->n > b->n;
    }
    im.bytes = ccitt_tiff(data, im.w, im.h, k, flag(f, pa, "EncodedByteAlign"), flag(f, pa, "BlackIs1") != inv);
    im.mime = "image/tiff";
  } else if (last == "JBIG2Decode" || last == "JPXDecode") {
    im.note = std::string("a ") + (last == "JPXDecode" ? "JPEG 2000" : "JBIG2") + " image, which is not decoded here";
  } else raw_image(f, xo, res, data, im);
  return im;
}

// the largest image page (1-based) draws
inline Image page_image(const std::string& data, int page) {
  Image best;
  pdf::File f(data);
  if (f.encrypted) {
    best.note = "an encrypted PDF";
    return best;
  }
  const auto pages = f.pages();
  if (page < 1 || page > (int)pages.size()) {
    best.note = "no page " + std::to_string(page) + " (the PDF has " + std::to_string(pages.size()) + ")";
    return best;
  }
  const auto imgs = drawn(f, pages[(size_t)page - 1].first, pages[(size_t)page - 1].second);
  const std::pair<P, P>* pick = nullptr;
  uint64_t most = 0;
  for (const auto& x : imgs) {
    const uint64_t px = (uint64_t)std::max(0, num(f, x.first, "Width", 0)) * (uint64_t)std::max(0, num(f, x.first, "Height", 0));
    if (px > most) { most = px; pick = &x; }
  }
  if (!pick) {
    best.note = "page " + std::to_string(page) + " draws no image";
    return best;
  }
  best = image(f, pick->first, pick->second);
  if (imgs.size() > 1 && best.note.empty()) best.note = "the page draws " + std::to_string(imgs.size()) + " images: the largest is read";
  return best;
}

// doc.scans: its pages with less than kTextChars of text that draw an image of kScanPixels or more
inline void scan_pages(const std::string& data, Doc& doc) {
  doc.scans.clear();
  std::vector<size_t> chars((size_t)std::max(0, doc.pages) + 1, 0);
  for (const Para& p : doc.paras)
    if (p.page >= 1 && p.page <= doc.pages) chars[(size_t)p.page] += p.text.size();
  bool any = false;
  for (int p = 1; p <= doc.pages; ++p) any = any || chars[(size_t)p] < kTextChars;
  if (!any) return;
  pdf::File f(data);
  if (f.encrypted) return;
  const auto pages = f.pages();
  for (int p = 1; p <= doc.pages && p <= (int)pages.size(); ++p) {
    if (chars[(size_t)p] >= kTextChars) continue;
    for (const auto& x : drawn(f, pages[(size_t)p - 1].first, pages[(size_t)p - 1].second))
      if ((uint64_t)std::max(0, num(f, x.first, "Width", 0)) * (uint64_t)std::max(0, num(f, x.first, "Height", 0)) >= kScanPixels) {
        doc.scans.push_back(p);
        break;
      }
  }
  doc.unread = doc.scans;
}

// a transcription (Markdown) as paragraphs at its blank lines; a code fence around all of it taken off
inline std::vector<std::string> blocks(std::string t) {
  std::vector<std::string> v;
  t.erase(std::remove(t.begin(), t.end(), '\r'), t.end());
  if (t.rfind("```", 0) == 0) {
    const size_t nl = t.find('\n'), end = t.rfind("```");
    if (nl != std::string::npos && end > nl) t = t.substr(nl + 1, end - nl - 1);
  }
  std::string cur;
  auto end = [&] {
    while (!cur.empty() && (cur.back() == '\n' || cur.back() == ' ')) cur.pop_back();
    if (!cur.empty()) v.push_back(cur);
    cur.clear();
  };
  for (size_t i = 0; i <= t.size();) {
    const size_t e = std::min(t.find('\n', i), t.size());
    std::string line = t.substr(i, e - i);
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.pop_back();
    if (tidy(line).empty()) end();
    else cur += line + "\n";
    i = e + 1;
  }
  end();
  return v;
}

// the scanned pages' transcriptions (page -> text) into doc, in page order; doc.unread: the scans still without
// one; doc.note says which pages are scans and which are read
inline void add_pages(Doc& doc, const std::map<int, std::string>& texts) {
  std::vector<Para> paras;
  size_t k = 0;
  doc.unread.clear();
  for (int p : doc.scans) {
    while (k < doc.paras.size() && doc.paras[k].page <= p) paras.push_back(doc.paras[k++]);
    auto it = texts.find(p);
    if (it == texts.end()) { doc.unread.push_back(p); continue; }
    for (const std::string& b : blocks(it->second))
      if (b != "(no text)") paras.push_back({b, p, 0});
  }
  while (k < doc.paras.size()) paras.push_back(doc.paras[k++]);
  doc.paras.swap(paras);
  if (doc.scans.empty()) return;
  doc.note = std::to_string(doc.scans.size()) + " of " + std::to_string(doc.pages) + " pages are scanned images without a text layer (" +
             (doc.scans.size() == (size_t)doc.pages ? std::string("all") : "pages " + ranges(doc.scans)) + ")";
  if (doc.unread.empty()) doc.note += "; their text was transcribed from the images by a vision model (it can misread)";
  else if (doc.unread.size() < doc.scans.size())
    doc.note += "; transcribed by a vision model (it can misread) except pages " + ranges(doc.unread);
}

}  // namespace pdfimg
}  // namespace rdoc
