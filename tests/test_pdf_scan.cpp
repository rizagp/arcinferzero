// Checks of tools/pdf_images.h (scanned PDF pages for a vision model), model-free: PDFs built here from known
// images (no user files) give back their page images, which the Windows Imaging Component decodes (src/image.h) to
// the same pixels:
//   DCTDecode (a JPEG made by WIC; also behind FlateDecode)    the JPEG bytes as they are
//   FlateDecode raw samples: RGB with PNG predictors, gray 8 bits, gray 1 bit with /Decode [1 0], an image mask,
//   Indexed, CMYK, through a form XObject                      a BMP of the same pixels
//   CCITTFaxDecode Group 4 (WIC's own TIFF G4 strip)           a TIFF of the same pixels
// The scanned pages found (a page with text is not one), transcriptions merged into the paragraphs and searched,
// the notes; hostile input: huge sizes, short and broken streams, a form that draws itself, random damage.
//   test_pdf_scan.exe              the checks
//   test_pdf_scan.exe FILE.pdf     each page's scan status and its image (filter, size, note): to look at any PDF
#include "../tools/pdf_images.h"
#include "../src/image.h"
#include <cstdio>
#include <fstream>
#include <sstream>

static int fails = 0;
static void check(const char* name, bool ok, const std::string& got = "") {
  printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, ok || got.empty() ? "" : "  got: ", ok ? "" : got.c_str());
  fails += !ok;
}

// ---- the known images (w x h: 0.1 megapixels, a scan's size; not a multiple of 8 wide: rows end inside a byte)
static const int W = 403, H = 257;
static bool black(int x, int y) { return x < 2 || y < 2 || x >= W - 2 || y >= H - 2 || ((x / 7 + y / 5) % 3 == 0 && x > 20); }
static uint8_t gray(int x, int y) { return (uint8_t)((x * 3 + y * 2) & 255); }
static void rgb(int x, int y, uint8_t* p) {
  p[0] = (uint8_t)(x * 5);
  p[1] = (uint8_t)(y * 3);
  p[2] = (uint8_t)((x + y) * 2);
}

// ---- zlib without compression (stored blocks), as a FlateDecode stream
static std::string zlib_stored(const std::string& d) {
  std::string o = "\x78\x01";
  size_t i = 0;
  do {
    const size_t n = std::min<size_t>(65535, d.size() - i);
    o += (char)(i + n == d.size() ? 1 : 0);
    o += (char)(n & 255);
    o += (char)(n >> 8);
    o += (char)(~n & 255);
    o += (char)(~n >> 8 & 255);
    o += d.substr(i, n);
    i += n;
  } while (i < d.size());
  uint32_t a = 1, b = 0;
  for (unsigned char c : d) {
    a = (a + c) % 65521;
    b = (b + a) % 65521;
  }
  const uint32_t ad = b << 16 | a;
  for (int k = 3; k >= 0; --k) o += (char)(ad >> (8 * k) & 255);
  return o;
}
// rows of bpp-byte pixels with PNG filters (row y: filter y % 5) in front
static std::string png_filter(const std::string& raw, size_t rowbytes, int bpp) {
  std::string o, prev(rowbytes, '\0');
  for (size_t r = 0; r * rowbytes < raw.size(); ++r) {
    const std::string row = raw.substr(r * rowbytes, rowbytes);
    const int ft = (int)(r % 5);
    o += (char)ft;
    for (size_t k = 0; k < rowbytes; ++k) {
      const int x = (unsigned char)row[k], a = k >= (size_t)bpp ? (unsigned char)row[k - bpp] : 0, b = (unsigned char)prev[k],
                c = k >= (size_t)bpp ? (unsigned char)prev[k - bpp] : 0;
      int pr = 0;
      if (ft == 1) pr = a;
      else if (ft == 2) pr = b;
      else if (ft == 3) pr = (a + b) / 2;
      else if (ft == 4) {
        const int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
        pr = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      o += (char)(x - pr);
    }
    prev = row;
  }
  return o;
}

// ---- WIC encoders: a JPEG, and a TIFF with one CCITT Group 4 strip
static std::string wic_encode(const GUID& container, const GUID& fmt, int w, int h, UINT stride, const std::string& px, bool g4) {
  (void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  IWICImagingFactory* f = nullptr;
  if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)))) return "";
  IStream* st = nullptr;
  IWICBitmapEncoder* enc = nullptr;
  IWICBitmapFrameEncode* fr = nullptr;
  IPropertyBag2* bag = nullptr;
  std::string out;
  HRESULT hr = CreateStreamOnHGlobal(nullptr, TRUE, &st);
  if (SUCCEEDED(hr)) hr = f->CreateEncoder(container, nullptr, &enc);
  if (SUCCEEDED(hr)) hr = enc->Initialize(st, WICBitmapEncoderNoCache);
  if (SUCCEEDED(hr)) hr = enc->CreateNewFrame(&fr, &bag);
  if (SUCCEEDED(hr)) {
    PROPBAG2 opt{};
    VARIANT v{};
    if (g4) {
      opt.pstrName = (LPOLESTR)L"TiffCompressionMethod";
      v.vt = VT_UI1;
      v.bVal = WICTiffCompressionCCITT4;
    } else {
      opt.pstrName = (LPOLESTR)L"ImageQuality";
      v.vt = VT_R4;
      v.fltVal = 0.95f;
    }
    hr = bag->Write(1, &opt, &v);
  }
  if (SUCCEEDED(hr)) hr = fr->Initialize(bag);
  if (SUCCEEDED(hr)) hr = fr->SetSize((UINT)w, (UINT)h);
  WICPixelFormatGUID pf = fmt;
  if (SUCCEEDED(hr)) hr = fr->SetPixelFormat(&pf);
  if (SUCCEEDED(hr) && pf != fmt) hr = E_FAIL;
  if (SUCCEEDED(hr)) hr = fr->WritePixels((UINT)h, stride, (UINT)px.size(), (BYTE*)px.data());
  if (SUCCEEDED(hr)) hr = fr->Commit();
  if (SUCCEEDED(hr)) hr = enc->Commit();
  if (SUCCEEDED(hr)) {
    HGLOBAL g = nullptr;
    if (SUCCEEDED(GetHGlobalFromStream(st, &g))) {
      STATSTG s{};
      st->Stat(&s, STATFLAG_NONAME);
      const char* p = (const char*)GlobalLock(g);
      out.assign(p, (size_t)s.cbSize.QuadPart);
      GlobalUnlock(g);
    }
  }
  if (bag) bag->Release();
  if (fr) fr->Release();
  if (enc) enc->Release();
  if (st) st->Release();
  f->Release();
  return out;
}
// the one strip of a little-endian TIFF ("" when it has several)
static std::string tiff_strip(const std::string& t) {
  auto u16 = [&](size_t at) { return at + 2 <= t.size() ? (uint32_t)(unsigned char)t[at] | (uint32_t)(unsigned char)t[at + 1] << 8 : 0u; };
  auto u32 = [&](size_t at) { return u16(at) | u16(at + 2) << 16; };
  if (t.size() < 8 || t.compare(0, 2, "II") != 0) return "";
  const size_t ifd = u32(4);
  const uint32_t n = u16(ifd);
  uint32_t off = 0, len = 0, count = 0;
  for (uint32_t k = 0; k < n; ++k) {
    const size_t e = ifd + 2 + k * 12;
    const uint32_t tag = u16(e), type = u16(e + 2), cnt = u32(e + 4), val = type == 3 ? u16(e + 8) : u32(e + 8);
    if (tag == 273) { off = val; count = cnt; }
    if (tag == 279) len = val;
  }
  return count == 1 && (size_t)off + len <= t.size() ? t.substr(off, len) : "";
}

// ---- a PDF of pages: each its content stream and image (or form) XObjects; objects without a cross-reference table
struct Obj {
  std::string dict, stream;
  bool has_stream = false;
};
struct Pdf {
  std::vector<Obj> objs;
  int add(const std::string& dict) { objs.push_back({dict, "", false}); return (int)objs.size(); }
  int add(const std::string& dict, const std::string& stream) {
    const size_t e = dict.rfind(">>");  // (the length inside the dictionary)
    objs.push_back({(e == std::string::npos ? dict : dict.substr(0, e)) + " /Length " + std::to_string(stream.size()) + " >>", stream, true});
    return (int)objs.size();
  }
  // a page drawing content with XObjects "/Name n 0 R ..." (filled in later)
  std::vector<std::pair<std::string, std::string>> pages;  // (content, XObject dictionary body)
  std::string build() {
    std::vector<int> ids;
    const int font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    const int pages_id = (int)objs.size() + 1 + (int)pages.size() * 2;
    for (const auto& [content, xo] : pages) {
      const int c = add("<<", content);
      objs.back().dict = "<< /Length " + std::to_string(content.size()) + " >>";
      ids.push_back(add("<< /Type /Page /Parent " + std::to_string(pages_id) + " 0 R /MediaBox [0 0 612 792] /Contents " +
                        std::to_string(c) + " 0 R /Resources << /Font << /F1 " + std::to_string(font) +
                        " 0 R >> /XObject << " + xo + " >> >> >>"));
    }
    std::string kids;
    for (int id : ids) kids += std::to_string(id) + " 0 R ";
    add("<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(ids.size()) + " >>");
    add("<< /Type /Catalog /Pages " + std::to_string(pages_id) + " 0 R >>");
    std::string o = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";
    for (size_t k = 0; k < objs.size(); ++k) {
      o += std::to_string(k + 1) + " 0 obj\n" + objs[k].dict + "\n";
      if (objs[k].has_stream) o += "stream\n" + objs[k].stream + "\nendstream\n";
      o += "endobj\n";
    }
    return o + "trailer\n<< /Root " + std::to_string(objs.size()) + " 0 R >>\n%%EOF\n";
  }
};
static std::string image_dict(const std::string& rest) {
  return "<< /Type /XObject /Subtype /Image /Width " + std::to_string(W) + " /Height " + std::to_string(H) + " " + rest;
}
static const char* kDraw = "q 612 0 0 792 0 0 cm /Im0 Do Q";

// the page's image decoded, against the expected pixels: the largest difference of any sample
static int diff(const rdoc::pdfimg::Image& im, const std::function<void(int, int, uint8_t*)>& want, std::string& why, int h = H) {
  if (im.bytes.empty()) { why = "no image: " + im.note; return 999; }
  ImageRGB d;
  try {
    d = img::decode(std::vector<uint8_t>(im.bytes.begin(), im.bytes.end()));
  } catch (const std::exception& e) {
    why = std::string("decode: ") + e.what();
    return 999;
  }
  if (d.w != W || d.h != h) { why = std::to_string(d.w) + "x" + std::to_string(d.h); return 999; }
  int most = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < W; ++x) {
      uint8_t e[3];
      want(x, y, e);
      for (int c = 0; c < 3; ++c) most = std::max(most, abs((int)d.px[((size_t)y * W + x) * 3 + c] - (int)e[c]));
    }
  why = "largest difference " + std::to_string(most) + " (" + im.mime + ", " + im.filter + ")";
  return most;
}
static double mean_diff(const rdoc::pdfimg::Image& im, const std::function<void(int, int, uint8_t*)>& want) {
  ImageRGB d;
  try {
    d = img::decode(std::vector<uint8_t>(im.bytes.begin(), im.bytes.end()));
  } catch (const std::exception&) {
    return 999;
  }
  if (d.w != W || d.h != H) return 999;
  double s = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      uint8_t e[3];
      want(x, y, e);
      for (int c = 0; c < 3; ++c) s += abs((int)d.px[((size_t)y * W + x) * 3 + c] - (int)e[c]);
    }
  return s / (W * H * 3.0);
}

static std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc > 1) {
    const std::string data = slurp(argv[1]);
    rdoc::Doc d = rdoc::read(argv[1], data);
    rdoc::pdfimg::scan_pages(data, d);
    printf("%d pages, scanned: %s\n", d.pages, d.scans.empty() ? "none" : rdoc::pdfimg::ranges(d.scans).c_str());
    for (int p : d.scans) {
      const auto im = rdoc::pdfimg::page_image(data, p);
      printf("page %d: %s %dx%d, %zu bytes%s%s\n", p, im.filter.c_str(), im.w, im.h, im.bytes.size(), im.note.empty() ? "" : ", ", im.note.c_str());
    }
    return 0;
  }
  using rdoc::pdfimg::page_image;
  auto want_gray = [](int x, int y, uint8_t* p) { p[0] = p[1] = p[2] = gray(x, y); };
  auto want_bw = [](int x, int y, uint8_t* p) { p[0] = p[1] = p[2] = black(x, y) ? 0 : 255; };
  auto want_rgb = [](int x, int y, uint8_t* p) { rgb(x, y, p); };
  std::string raw_rgb, raw_gray, raw_bw, raw_bw_inv, raw_cmyk;
  const size_t bw_row = (W + 7) / 8;
  raw_bw.assign(bw_row * H, '\0');
  raw_bw_inv.assign(bw_row * H, '\0');
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      uint8_t p[3];
      rgb(x, y, p);
      raw_rgb.append((const char*)p, 3);
      raw_gray += (char)gray(x, y);
      if (!black(x, y)) raw_bw[(size_t)y * bw_row + x / 8] |= (char)(0x80 >> (x % 8));  // (1: white)
      else raw_bw_inv[(size_t)y * bw_row + x / 8] |= (char)(0x80 >> (x % 8));       // (1: black)
      raw_cmyk += (char)(255 - p[0]);
      raw_cmyk += (char)(255 - p[1]);
      raw_cmyk += (char)(255 - p[2]);
      raw_cmyk += '\0';
    }

  // ---- 1. raw samples (FlateDecode)
  {
    Pdf pdf;
    const int rgb_png = pdf.add(image_dict("/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                                           "/DecodeParms << /Predictor 15 /Colors 3 /Columns " + std::to_string(W) + " >> >>"),
                                zlib_stored(png_filter(raw_rgb, (size_t)W * 3, 3)));
    const int g8 = pdf.add(image_dict("/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode >>"), zlib_stored(raw_gray));
    const int bw_inv = pdf.add(image_dict("/ColorSpace /DeviceGray /BitsPerComponent 1 /Decode [1 0] /Filter /Fl >>"), zlib_stored(raw_bw_inv));
    const int mask = pdf.add(image_dict("/ImageMask true >>"), raw_bw);  // (no filter; 0 paints black)
    std::string lut, idx;
    for (int k = 0; k < 256; ++k) { lut += (char)k; lut += (char)(255 - k); lut += (char)(k / 2); }
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) idx += (char)gray(x, y);
    const int ind = pdf.add(image_dict("/ColorSpace [/Indexed /DeviceRGB 255 " + std::to_string(pdf.objs.size() + 2) + " 0 R] /BitsPerComponent 8 /Filter /FlateDecode >>"),
                            zlib_stored(idx));
    pdf.add("<< >>", lut);
    const int cmyk = pdf.add(image_dict("/ColorSpace [/ICCBased " + std::to_string(pdf.objs.size() + 2) + " 0 R] /BitsPerComponent 8 /Filter /FlateDecode >>"),
                             zlib_stored(raw_cmyk));
    pdf.add("<< /N 4 >>", "icc");
    const int form = pdf.add("<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /XObject << /Im0 " +
                                 std::to_string(g8) + " 0 R >> >> >>",
                             kDraw);
    for (int id : {rgb_png, g8, bw_inv, mask, ind, cmyk})
      pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(id) + " 0 R"});
    pdf.pages.push_back({"/Fm0 Do", "/Fm0 " + std::to_string(form) + " 0 R"});  // (7: the image inside a form)
    const std::string data = pdf.build();
    std::string why;
    check("Flate RGB with PNG predictors (all five filters): same pixels", diff(page_image(data, 1), want_rgb, why) == 0, why);
    check("Flate gray 8 bits: same pixels", diff(page_image(data, 2), want_gray, why) == 0, why);
    check("gray 1 bit with /Decode [1 0], rows ending inside a byte", diff(page_image(data, 3), want_bw, why) == 0, why);
    check("image mask without a filter", diff(page_image(data, 4), want_bw, why) == 0, why);
    check("Indexed (palette in a stream)", diff(page_image(data, 5), [](int x, int y, uint8_t* p) {
                                             const int k = gray(x, y);
                                             p[0] = (uint8_t)k; p[1] = (uint8_t)(255 - k); p[2] = (uint8_t)(k / 2);
                                           }, why) == 0, why);
    check("ICCBased CMYK (N 4)", diff(page_image(data, 6), want_rgb, why) == 0, why);
    check("an image drawn by a form XObject", diff(page_image(data, 7), want_gray, why) == 0, why);
    check("raw image as a BMP", page_image(data, 2).mime == "image/bmp");

    rdoc::Doc d = rdoc::read("scan.pdf", data);
    rdoc::pdfimg::scan_pages(data, d);
    check("every image page is a scan", d.scans.size() == 7 && d.unread.size() == 7, rdoc::pdfimg::ranges(d.scans));
  }

  // ---- 2. JPEG (DCTDecode), alone and behind FlateDecode; CCITT Group 4; a page with text; JBIG2
  {
    const std::string jpg = wic_encode(GUID_ContainerFormatJpeg, GUID_WICPixelFormat24bppBGR, W, H, W * 3, [&] {
      std::string bgr;
      for (size_t k = 0; k < raw_rgb.size(); k += 3) { bgr += raw_rgb[k + 2]; bgr += raw_rgb[k + 1]; bgr += raw_rgb[k]; }
      return bgr;
    }(), false);
    check("WIC made a JPEG", jpg.size() > 100, std::to_string(jpg.size()));
    // (WIC's BlackWhite: 1 white, 0 black, rows of whole bytes; FH rows: WIC writes strips of 8 KB unpacked, one here)
    const int FH = 150;
    std::string bw_wic = raw_bw.substr(0, bw_row * FH);
    const std::string tif = wic_encode(GUID_ContainerFormatTiff, GUID_WICPixelFormatBlackWhite, W, FH, (UINT)bw_row, bw_wic, true);
    const std::string g4 = tiff_strip(tif);
    check("WIC made a one-strip CCITT G4 TIFF", !g4.empty(), std::to_string(tif.size()) + " bytes");
    const std::string fax_dict = "<< /Type /XObject /Subtype /Image /Width " + std::to_string(W) + " /Height " + std::to_string(FH) + " ";
    Pdf pdf;
    const int j1 = pdf.add(image_dict("/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode >>"), jpg);
    const int j2 = pdf.add(image_dict("/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter [/FlateDecode /DCTDecode] >>"), zlib_stored(jpg));
    const int c4 = pdf.add(fax_dict + "/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode /DecodeParms << /K -1 /Columns " +
                               std::to_string(W) + " /Rows " + std::to_string(FH) + " >> >>",
                           g4);
    const int c4i = pdf.add(fax_dict + "/ImageMask true /Decode [1 0] /Filter [/CCF] /DecodeParms [<< /K -1 /Columns " +
                                std::to_string(W) + " /BlackIs1 true >>] >>",
                            g4);
    const int jb = pdf.add(image_dict("/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode >>"), "jbig2 data");
    pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(j1) + " 0 R"});
    pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(j2) + " 0 R"});
    pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(c4) + " 0 R"});
    pdf.pages.push_back({"BT /F1 12 Tf 72 700 Td (A page with its own text layer, long enough to count as text.) Tj ET " + std::string(kDraw),
                         "/Im0 " + std::to_string(j1) + " 0 R"});  // (4: text over a picture: not a scan)
    pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(c4i) + " 0 R"});
    pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(jb) + " 0 R"});
    const std::string data = pdf.build();
    const auto i1 = page_image(data, 1), i2 = page_image(data, 2);
    check("DCTDecode: the JPEG bytes as they are", i1.bytes == jpg && i1.mime == "image/jpeg");
    check("FlateDecode + DCTDecode: the JPEG inside", i2.bytes == jpg);
    const double md = mean_diff(i1, want_rgb);
    check("the JPEG decodes to the image (mean difference < 4)", md < 4, std::to_string(md));
    std::string why;
    check("CCITT G4 behind a TIFF header: same pixels", diff(page_image(data, 3), want_bw, why, FH) == 0, why);
    check("CCITT as a TIFF", page_image(data, 3).mime == "image/tiff");
    check("CCITT BlackIs1 with /Decode [1 0], an image mask (twice turned: the same)", diff(page_image(data, 5), want_bw, why, FH) == 0, why);
    const auto j = page_image(data, 6);
    check("JBIG2: a note only", j.bytes.empty() && j.note.find("JBIG2") != std::string::npos, j.note);

    // scanned pages, transcriptions, search
    rdoc::Doc d = rdoc::read("scan.pdf", data);
    rdoc::pdfimg::scan_pages(data, d);
    check("a page with a text layer is not a scan", d.scans == std::vector<int>({1, 2, 3, 5, 6}), rdoc::pdfimg::ranges(d.scans));
    rdoc::pdfimg::add_pages(d, {});
    check("note: the scanned pages", d.note.find("5 of 6 pages are scanned images") != std::string::npos && d.note.find("1-3, 5-6") != std::string::npos, d.note);
    rdoc::pdfimg::add_pages(d, {{2, "# Laporan Triwulan\n\nRealisasi anggaran kantor wilayah mencapai 87 persen.\n\n| a | b |\n| 1 | 2 |"},
                                {3, "```markdown\nKeputusan rapat pengawasan\n```"}, {5, "(no text)"}});
    check("transcriptions merged in page order",
          d.paras.size() >= 5 && d.paras[0].page == 2 && d.paras.back().page == 4, std::to_string(d.paras.size()));
    check("a table's rows stay lines", std::any_of(d.paras.begin(), d.paras.end(), [](const rdoc::Para& p) { return p.text == "| a | b |\n| 1 | 2 |"; }));
    check("a code fence around a transcription taken off", std::any_of(d.paras.begin(), d.paras.end(), [](const rdoc::Para& p) { return p.text == "Keputusan rapat pengawasan" && p.page == 3; }));
    check("unread: the scans without a transcription", d.unread == std::vector<int>({1, 6}), rdoc::pdfimg::ranges(d.unread));
    check("note: which pages were transcribed", d.note.find("except pages 1, 6") != std::string::npos, d.note);
    const rdoc::Index ix(d);
    const auto hits = ix.search("realisasi anggaran", 3);
    check("search finds a transcribed page", !hits.empty() && ix.chunks[hits[0].chunk].page_from == 2,
          hits.empty() ? "none" : std::to_string(ix.chunks[hits[0].chunk].page_from));
  }

  // ---- 3. hostile input
  {
    Pdf pdf;
    const int huge = pdf.add("<< /Type /XObject /Subtype /Image /Width 100000 /Height 100000 /ColorSpace /DeviceGray /BitsPerComponent 8 >>", "x");
    const int big = pdf.add("<< /Type /XObject /Subtype /Image /Width 8000 /Height 8000 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode >>",
                            zlib_stored(std::string(1000, '\x7f')));
    const int broken = pdf.add(image_dict("/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode >>"), "\x78\x9c not deflate at all");
    const int fax = pdf.add(image_dict("/ImageMask true /Filter /CCITTFaxDecode /DecodeParms << /K -1 /Columns 999999 >> >>"), "fax");
    const int self = (int)pdf.objs.size() + 1;
    pdf.add("<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources << /XObject << /Fm0 " + std::to_string(self) + " 0 R >> >> >>", "/Fm0 Do /Fm0 Do");
    const int odd = pdf.add(image_dict("/ColorSpace /Lab /BitsPerComponent 8 >>"), std::string((size_t)W * H * 3, '\0'));
    const int bpc = pdf.add(image_dict("/ColorSpace /DeviceGray /BitsPerComponent 7 >>"), std::string((size_t)W * H, '\0'));
    const int cutjpg = pdf.add(image_dict("/ColorSpace /DeviceRGB /Filter /DCTDecode >>"), "\xFF\xD8\xFF\xE0 cut");
    for (int id : {huge, big, broken, fax})
      pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(id) + " 0 R"});
    pdf.pages.push_back({"/Fm0 Do", "/Fm0 " + std::to_string(self) + " 0 R"});
    for (int id : {odd, bpc, cutjpg})
      pdf.pages.push_back({kDraw, "/Im0 " + std::to_string(id) + " 0 R"});
    const std::string data = pdf.build();
    const auto h1 = page_image(data, 1), h2 = page_image(data, 2), h3 = page_image(data, 3), h4 = page_image(data, 4);
    check("100000 x 100000: refused before any memory", h1.bytes.empty() && h1.note.find("16384") != std::string::npos, h1.note);
    check("an 8000 x 8000 RGB image of 1000 bytes: damaged, no 192 MB", h2.bytes.empty() && h2.note.find("damaged") != std::string::npos, h2.note);
    check("a broken Flate stream: a note", h3.bytes.empty() && !h3.note.empty(), h3.note);
    check("a fax of 999999 columns: refused", h4.bytes.empty() && h4.note.find("999999") != std::string::npos, h4.note);
    check("a form that draws itself: no image, no hang", page_image(data, 5).note.find("no image") != std::string::npos, page_image(data, 5).note);
    check("a Lab image: a note", page_image(data, 6).note.find("Lab") != std::string::npos, page_image(data, 6).note);
    check("7 bits a sample: a note", page_image(data, 7).note.find("7 bits") != std::string::npos, page_image(data, 7).note);
    bool threw = false;
    try { img::decode(std::vector<uint8_t>(page_image(data, 8).bytes.begin(), page_image(data, 8).bytes.end())); } catch (const std::exception&) { threw = true; }
    check("a cut JPEG: passed on, the decoder refuses it", threw);
    check("no page 99", page_image(data, 99).note.find("no page") != std::string::npos);
    rdoc::Doc d = rdoc::read("bad.pdf", data);
    rdoc::pdfimg::scan_pages(data, d);
    check("hostile pages listed as scans (their notes say why)", !d.scans.empty(), rdoc::pdfimg::ranges(d.scans));

    // random damage: bytes changed, cut short; nothing may crash
    std::string good;
    {
      Pdf p2;
      const int a = p2.add(image_dict("/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode >>"), zlib_stored(raw_gray));
      const int b = p2.add(image_dict("/ImageMask true >>"), raw_bw);
      p2.pages.push_back({kDraw, "/Im0 " + std::to_string(a) + " 0 R"});
      p2.pages.push_back({kDraw, "/Im0 " + std::to_string(b) + " 0 R"});
      good = p2.build();
    }
    uint32_t r = 7;
    auto rnd = [&] { r = r * 1103515245u + 12345u; return r >> 8; };
    int decoded = 0;
    for (int it = 0; it < 300; ++it) {
      std::string bad = good;
      for (int k = 0; k < 1 + (int)(rnd() % 12); ++k) bad[rnd() % bad.size()] = (char)rnd();
      if (it % 10 == 0) bad.resize(rnd() % bad.size());
      rdoc::Doc dd = rdoc::read("x.pdf", bad);
      rdoc::pdfimg::scan_pages(bad, dd);
      rdoc::pdfimg::add_pages(dd, {{1, "x"}});
      for (int p = 1; p <= 2; ++p) {
        const auto im = page_image(bad, p);
        if (im.bytes.empty()) continue;
        try {
          img::decode(std::vector<uint8_t>(im.bytes.begin(), im.bytes.end()));
          ++decoded;
        } catch (const std::exception&) {
        }
      }
    }
    check("300 damaged PDFs: no crash", true, std::to_string(decoded) + " images decoded");
  }

  if (fails) {
    printf("test_pdf_scan: %d failed\n", fails);
    return 1;
  }
  printf("test_pdf_scan: all passed\n");
  return 0;
}
