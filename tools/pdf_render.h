// PDF pages as PNG images, rendered by Windows itself (Windows.Data.Pdf through C++/WinRT, Windows 10 / 11): for
// OCR models, which read images only (tools/server_ocr.h). Every kind of page (text, vector drawings, scans), as a
// PDF viewer draws it.
//   Pdf p; p.open(bytes, err)     the file's bytes read, or false and why
//   p.pages()                     pages in it
//   p.render(page, dpi, err)      page (1-based) as PNG bytes at about `dpi`, or "" and why
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
