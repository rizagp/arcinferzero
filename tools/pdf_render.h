// PDF pages as PNG images, rendered by Windows itself (Windows.Data.Pdf through C++/WinRT, Windows 10 / 11): for
// OCR models, which read images only (tools/server_ocr.h). Every kind of page (text, vector drawings, scans), as a
// PDF viewer draws it.
//   pdf_page_count(path, err)                 pages in the file, or -1 and why
//   pdf_render_page(path, page, dpi, err)    page (1-based) as PNG bytes at about `dpi`, or "" and why
#pragma once
#include <string>
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
inline winrt::Windows::Data::Pdf::PdfDocument open(const std::string& path) {
  using namespace winrt::Windows;
  apartment();
  const std::wstring full = std::filesystem::absolute(std::filesystem::path(wide(path))).wstring();
  Storage::StorageFile f = Storage::StorageFile::GetFileFromPathAsync(full).get();
  return Data::Pdf::PdfDocument::LoadFromFileAsync(f).get();
}
inline int pdf_page_count(const std::string& path, std::string& err) {
  try {
    return (int)open(path).PageCount();
  } catch (const winrt::hresult_error& e) {
    err = winrt::to_string(e.message());
  } catch (const std::exception& e) {
    err = e.what();
  }
  return -1;
}
inline std::string pdf_render_page(const std::string& path, int page, int dpi, std::string& err) {
  using namespace winrt::Windows;
  try {
    Data::Pdf::PdfDocument doc = open(path);
    if (page < 1 || page > (int)doc.PageCount()) {
      err = "no page " + std::to_string(page);
      return "";
    }
    Data::Pdf::PdfPage pg = doc.GetPage((uint32_t)(page - 1));
    Data::Pdf::PdfPageRenderOptions opt;
    const auto sz = pg.Size();  // (device-independent pixels, 96 per inch)
    opt.DestinationWidth((uint32_t)(std::max)(1.0, sz.Width * dpi / 96.0));
    opt.DestinationHeight((uint32_t)(std::max)(1.0, sz.Height * dpi / 96.0));
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
