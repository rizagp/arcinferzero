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
#include <utility>
#include <vector>
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
  std::vector<std::pair<double, double>> sizes;  // (each page's width and height in points, read once at open)
  bool open(const std::string& bytes, std::string& err) {
    try {
      auto t = std::make_shared<Tmp>();
      t->path = plat::write_temp("pdf", bytes.data(), bytes.size(), err);
      if (t->path.empty()) return false;
      const plat::ProcResult r = plat::run_program({"pdfinfo", plat::path_arg(t->path)}, "", 60000, 1 << 20);
      if (!r.started) { err = r.error + " (Linux renders PDF pages with poppler: apt install poppler-utils)"; return false; }
      // (the last line that starts with "Pages:": the title, subject and author come before it and may hold a line
      //  break and "Pages:" of their own)
      size_t k = std::string::npos;
      for (size_t at = 0; (at = r.out.find("Pages:", at)) != std::string::npos; ++at)
        if (at == 0 || r.out[at - 1] == '\n') k = at;
      if (r.exit_code != 0 || k == std::string::npos) {
        err = "not a PDF poppler can read" + (r.err.empty() ? std::string() : ": " + plat::last_line(r.err));
        return false;
      }
      // (at most kMaxPages: the OCR reads 50 pages a request and goes on with "lanjut", so every page counts, but a
      //  forged count must not size the list below or the page-size run; 100,000 pages' sizes fit its 16 MB of output)
      constexpr long kMaxPages = 100000;
      n = (int)std::clamp(strtol(r.out.c_str() + k + 6, nullptr, 10), 0L, kMaxPages);
      // (every page's size in one run: "Page    3 size: 595.28 x 841.89 pts (A4)"; a page not listed keeps Letter)
      sizes.assign((size_t)n, {612.0, 792.0});
      if (n > 0) {
        const plat::ProcResult s = plat::run_program({"pdfinfo", "-f", "1", "-l", std::to_string(n), plat::path_arg(t->path)}, "",
                                                     120000, 16 << 20);
        for (size_t at = 0; (at = s.out.find("\nPage ", at)) != std::string::npos; ++at) {
          int pg = 0;
          double w = 0, h = 0;
          if (sscanf(s.out.c_str() + at + 6, "%d size: %lf x %lf", &pg, &w, &h) == 3 && pg >= 1 && pg <= n && w > 0 && h > 0)
            sizes[(size_t)pg - 1] = {w, h};
        }
      }
      tmp = t;
      return true;
    } catch (const std::exception& e) {
      err = e.what();
      return false;
    }
  }
  int pages() const { return tmp ? n : 0; }
  std::string render(int page, int dpi, std::string& err) const {
    try {  // (never throws, as the Windows half: a page that fails is a note, the rest of a batch goes on)
      if (!tmp || page < 1 || page > n) { err = "no page " + std::to_string(page); return ""; }
      const std::string pg = std::to_string(page);
      const auto [wpt, hpt] = sizes[(size_t)page - 1];
      const double long_px = std::max(wpt, hpt) * dpi / 72.0;  // (points: 72 per inch; the long side at most 4000 px)
      std::vector<std::string> a = {"pdftoppm", "-f", pg, "-l", pg, "-png", "-singlefile"};
      if (long_px > 4000) a.insert(a.end(), {"-scale-to", "4000"});
      else a.insert(a.end(), {"-r", std::to_string(dpi)});
      a.push_back(plat::path_arg(tmp->path));
      const plat::ProcResult r = plat::run_program(a, "", 120000, (size_t)256 << 20);
      if (!r.started) { err = r.error + " (apt install poppler-utils)"; return ""; }
      if (r.exit_code != 0 || r.timed_out || r.cut || r.out.size() < 8 || r.out.compare(1, 3, "PNG") != 0) {
        err = r.timed_out ? std::string("the page took too long to render")
                          : "the page could not be rendered" + (r.err.empty() ? std::string() : ": " + plat::last_line(r.err));
        return "";
      }
      return r.out;
    } catch (const std::exception& e) {
      err = e.what();
      return "";
    }
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
