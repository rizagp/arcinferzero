// PDF pages as PNG images, rendered by Windows itself (Windows.Data.Pdf through C++/WinRT, Windows 10 / 11): for
// OCR models, which read images only (tools/server_ocr.h). Every kind of page (text, vector drawings, scans), as a
// PDF viewer draws it.
//   Pdf p; p.open(bytes, err)     the file's bytes read, or false and why
//   p.pages()                     pages in it
//   p.render(page, dpi, err)      page (1-based) as PNG bytes at about `dpi`, or "" and why
// Linux: poppler's pdfinfo (pages, page size) and pdftoppm (the page as PNG), when installed (poppler-utils), on a
// temporary copy of the bytes; the same interface and the same 4000-pixel cap.
#pragma once
#include <string>
#ifndef _WIN32
#include "../src/platform_proc.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <unistd.h>
namespace pdfr {
struct Pdf {
  struct Tmp {
    std::string path;
    ~Tmp() {
      if (!path.empty()) unlink(path.c_str());
    }
  };
  std::shared_ptr<Tmp> tmp;
  int n = 0;
  bool open(const std::string& bytes, std::string& err) {
    char path[] = "/tmp/inferzero-pdf-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { err = "no temporary file"; return false; }
    auto t = std::make_shared<Tmp>();
    t->path = path;
    size_t off = 0;
    while (off < bytes.size()) {
      const ssize_t w = ::write(fd, bytes.data() + off, bytes.size() - off);
      if (w <= 0) break;
      off += (size_t)w;
    }
    ::close(fd);
    if (off < bytes.size()) { err = "the temporary file could not be written"; return false; }
    const plat::ProcResult r = plat::run_program({"pdfinfo", t->path}, "", 60000, 1 << 20);
    if (!r.started) { err = r.error + " (Linux renders PDF pages with poppler: apt install poppler-utils)"; return false; }
    const size_t k = r.out.find("Pages:");
    if (r.exit_code != 0 || k == std::string::npos) {
      err = "not a PDF poppler can read" + (r.err.empty() ? std::string() : ": " + r.err.substr(0, r.err.find('\n')));
      return false;
    }
    n = atoi(r.out.c_str() + k + 6);
    tmp = t;
    return true;
  }
  int pages() const { return tmp ? n : 0; }
  std::string render(int page, int dpi, std::string& err) const {
    if (!tmp || page < 1 || page > n) { err = "no page " + std::to_string(page); return ""; }
    const std::string pg = std::to_string(page);
    // (the page's size in points, 72 per inch: its long side at most 4000 px, as on Windows)
    const plat::ProcResult info = plat::run_program({"pdfinfo", "-f", pg, "-l", pg, tmp->path}, "", 60000, 1 << 20);
    double wpt = 612, hpt = 792;
    // ("Page    3 size: 595.28 x 841.89 pts (A4)": with -f / -l only that page's line has a size)
    const size_t k = info.out.find(" size:");
    if (k != std::string::npos) sscanf(info.out.c_str() + k + 6, "%lf x %lf", &wpt, &hpt);
    const double long_px = std::max(wpt, hpt) * dpi / 72.0;
    std::vector<std::string> a = {"pdftoppm", "-f", pg, "-l", pg, "-png", "-singlefile"};
    if (long_px > 4000) a.insert(a.end(), {"-scale-to", "4000"});
    else a.insert(a.end(), {"-r", std::to_string(dpi)});
    a.push_back(tmp->path);
    const plat::ProcResult r = plat::run_program(a, "", 120000, (size_t)256 << 20);
    if (!r.started) { err = r.error + " (apt install poppler-utils)"; return ""; }
    if (r.exit_code != 0 || r.out.compare(1, 3, "PNG") != 0) {
      err = "the page could not be rendered" + (r.err.empty() ? std::string() : ": " + r.err.substr(0, r.err.find('\n')));
      return "";
    }
    return r.out;
  }
};
}  // namespace pdfr
#else
#include <winrt/Windows.Data.Pdf.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Windows.Storage.h>
#pragma comment(lib, "windowsapp.lib")

namespace pdfr {

inline std::wstring wide(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}
// (WinRT on this thread: the server's request threads have no apartment of their own; .get() needs one that is not
//  single-threaded)
inline void apartment() {
  static thread_local bool done = false;
  if (done) return;
  done = true;
  try {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
  } catch (const winrt::hresult_error&) {}  // (already initialized: COM for Media Foundation)
}
// The file's bytes, not its path: a chat's folder is renamed when the chat is first saved, which can come while a
// PDF of it is read (StorageFile then found no file at the path it was given), and a "/" in a path is refused too.
inline winrt::Windows::Data::Pdf::PdfDocument open(const std::string& bytes) {
  using namespace winrt::Windows;
  apartment();
  Storage::Streams::InMemoryRandomAccessStream st;
  Storage::Streams::DataWriter wr(st);
  wr.WriteBytes(winrt::array_view<const uint8_t>((const uint8_t*)bytes.data(), (const uint8_t*)bytes.data() + bytes.size()));
  wr.StoreAsync().get();
  wr.DetachStream();
  st.Seek(0);
  return Data::Pdf::PdfDocument::LoadFromStreamAsync(st).get();
}
// a PDF read once (open: false and why), its pages rendered from memory
struct Pdf {
  winrt::Windows::Data::Pdf::PdfDocument doc{nullptr};
  bool open(const std::string& bytes, std::string& err) {
    try {
      doc = pdfr::open(bytes);
      return true;
    } catch (const winrt::hresult_error& e) {
      err = winrt::to_string(e.message());
    } catch (const std::exception& e) {
      err = e.what();
    }
    return false;
  }
  int pages() const { return doc ? (int)doc.PageCount() : 0; }
  std::string render(int page, int dpi, std::string& err) const;
};
inline std::string Pdf::render(int page, int dpi, std::string& err) const {
  using namespace winrt::Windows;
  try {
    if (!doc || page < 1 || page > (int)doc.PageCount()) {
      err = "no page " + std::to_string(page);
      return "";
    }
    Data::Pdf::PdfPage pg = doc.GetPage((uint32_t)(page - 1));
    Data::Pdf::PdfPageRenderOptions opt;
    const auto sz = pg.Size();  // (device-independent pixels, 96 per inch)
    double w = (std::max)(1.0, sz.Width * dpi / 96.0), h = (std::max)(1.0, sz.Height * dpi / 96.0);
    // (a page of a poster's or a drawing's size: its long side at most 4000 px, not a bitmap of gigabytes; an A4 page at
    //  150 dpi is 1240 x 1754, unchanged)
    const double scale = (std::min)(1.0, 4000.0 / (std::max)(w, h));
    opt.DestinationWidth((uint32_t)(std::max)(1.0, w * scale));
    opt.DestinationHeight((uint32_t)(std::max)(1.0, h * scale));
    // (the image: PNG, the renderer's default)
    Storage::Streams::InMemoryRandomAccessStream st;
    pg.RenderToStreamAsync(st, opt).get();
    const uint32_t n = (uint32_t)st.Size();
    st.Seek(0);
    Storage::Streams::DataReader rd(st);
    rd.LoadAsync(n).get();
    std::string out(n, '\0');
    rd.ReadBytes(winrt::array_view<uint8_t>((uint8_t*)out.data(), (uint8_t*)out.data() + n));
    return out;
  } catch (const winrt::hresult_error& e) {
    err = winrt::to_string(e.message());
  } catch (const std::exception& e) {
    err = e.what();
  }
  return "";
}

}  // namespace pdfr
#endif
