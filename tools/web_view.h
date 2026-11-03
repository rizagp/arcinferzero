// web_view (tools/server.cpp, POST /v1/tools/web_view): a page as a browser shows it. Microsoft Edge (on every
// Windows 11) runs headless: one run writes a screenshot, another the page's HTML after its scripts ran (--dump-dom),
// so pages built by scripts read as text too. Edge loads what the page asks for (frames, scripts, images), and the
// checks of web_tools.h hold only for the first URL, so all of Edge's traffic goes through a small proxy in this
// process that connects to public addresses only (as web_fetch: no loopback, private, link-local... address, checked
// on the address it connects to), on ports 80 and 443. Edge gets a new empty profile each run (no cookies, logins or
// history of the user's), no extensions, no sync, WebRTC UDP off (it would bypass the proxy).
#pragma once
#include <wincodec.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "web_tools.h"
#pragma comment(lib, "windowscodecs.lib")

namespace webv {

// ---- the proxy
class Proxy {
 public:
  static Proxy& get() {
    static Proxy p;
    return p;
  }
  // its port, started at the first call (0: it could not start)
  int port() {
    std::call_once(once_, [this] { start(); });
    return port_;
  }

 private:
  static constexpr int kMaxConns = 64;
  int port_ = 0;
  std::once_flag once_;
  std::atomic<int> active_{0};

  void start() {
    WSADATA wd;
    WSAStartup(MAKEWORD(2, 2), &wd);
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // (this computer only)
    a.sin_port = 0;
    int len = sizeof a;
    if (bind(s, (sockaddr*)&a, sizeof a) != 0 || listen(s, 64) != 0 || getsockname(s, (sockaddr*)&a, &len) != 0) {
      closesocket(s);
      return;
    }
    port_ = ntohs(a.sin_port);
    std::thread([this, s] {
      for (;;) {
        SOCKET c = accept(s, nullptr, nullptr);
        if (c == INVALID_SOCKET) {  // (a failing listener: no busy loop on one core)
          Sleep(100);
          continue;
        }
        if (active_ >= kMaxConns) { closesocket(c); continue; }
        ++active_;
        std::thread([this, c] {
          serve(c);
          closesocket(c);
          --active_;
        }).detach();
      }
    }).detach();
  }
  static void timeouts(SOCKET s, DWORD ms) {
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof ms);
  }
  static bool send_all(SOCKET s, const char* p, size_t n) {
    while (n > 0) {
      const int k = send(s, p, (int)std::min<size_t>(n, 1 << 20), 0);
      if (k <= 0) return false;
      p += k, n -= (size_t)k;
    }
    return true;
  }
  static void refuse(SOCKET c, const std::string& why) {
    const std::string b = "HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: " +
                          std::to_string(why.size()) + "\r\n\r\n" + why;
    send_all(c, b.data(), b.size());
  }
  // a connection to host:port when every address it resolves to is public, else INVALID_SOCKET and why
  static SOCKET open(const std::string& host, int port, std::string& why) {
    if (port != 80 && port != 443) { why = "port " + std::to_string(port) + " refused (80 and 443 only)"; return INVALID_SOCKET; }
    const std::string h = webt::lower(host);
    if (h.empty() || h == "localhost" || (h.size() > 10 && h.compare(h.size() - 10, 10, ".localhost") == 0)) { why = "local address refused"; return INVALID_SOCKET; }
    addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) { why = "cannot resolve " + host; return INVALID_SOCKET; }
    for (addrinfo* a = res; a; a = a->ai_next)
      if (!webt::public_addr(a->ai_addr)) {
        freeaddrinfo(res);
        why = "local or private address refused (" + host + ")";
        return INVALID_SOCKET;
      }
    SOCKET s = INVALID_SOCKET;
    for (addrinfo* a = res; a && s == INVALID_SOCKET; a = a->ai_next) {  // (to the addresses just checked: no second lookup)
      s = socket(a->ai_family, SOCK_STREAM, IPPROTO_TCP);
      if (s == INVALID_SOCKET) continue;
      timeouts(s, 15000);
      if (connect(s, a->ai_addr, (int)a->ai_addrlen) != 0) { closesocket(s); s = INVALID_SOCKET; }
    }
    freeaddrinfo(res);
    if (s == INVALID_SOCKET) why = "cannot connect to " + host;
    return s;
  }
  // bytes both ways until either side closes, 60 s without any, or 10 minutes
  static void pump(SOCKET a, SOCKET b) {
    std::vector<char> buf(64 << 10);
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      if (std::chrono::steady_clock::now() - t0 > std::chrono::minutes(10)) return;
      fd_set r;
      FD_ZERO(&r);
      FD_SET(a, &r);
      FD_SET(b, &r);
      timeval tv{60, 0};
      const int n = select(0, &r, nullptr, nullptr, &tv);
      if (n <= 0) return;
      for (auto [from, to] : {std::pair{a, b}, std::pair{b, a}})
        if (FD_ISSET(from, &r)) {
          const int k = recv(from, buf.data(), (int)buf.size(), 0);
          if (k <= 0 || !send_all(to, buf.data(), (size_t)k)) return;
        }
    }
  }
  static void serve(SOCKET c) {
    timeouts(c, 30000);
    std::string head;
    char tmp[4096];
    while (head.find("\r\n\r\n") == std::string::npos) {
      if (head.size() > (32 << 10)) return;
      const int k = recv(c, tmp, sizeof tmp, 0);
      if (k <= 0) return;
      head.append(tmp, (size_t)k);
    }
    const size_t he = head.find("\r\n\r\n") + 4;
    const std::string rest = head.substr(he);
    head.resize(he);
    const size_t le = head.find("\r\n");
    const std::string line = head.substr(0, le);
    const size_t s1 = line.find(' '), s2 = line.find(' ', s1 + 1);
    if (s1 == std::string::npos || s2 == std::string::npos) return refuse(c, "bad request");
    const std::string method = line.substr(0, s1), target = line.substr(s1 + 1, s2 - s1 - 1);
    // host and port from "host:port", "[v6]:port"
    auto split = [](const std::string& hp, int def, std::string& host, int& port) {
      port = def;
      // (a port's digits, out of range or none: -1, refused; atoi's overflow is undefined)
      auto num = [](const char* p) {
        char* e = nullptr;
        const long v = strtol(p, &e, 10);
        return e == p || *e != '\0' || v < 1 || v > 65535 ? -1 : (int)v;
      };
      if (!hp.empty() && hp[0] == '[') {
        const size_t e = hp.find(']');
        host = hp.substr(1, e == std::string::npos ? std::string::npos : e - 1);
        if (e != std::string::npos && e + 1 < hp.size() && hp[e + 1] == ':') port = num(hp.c_str() + e + 2);
        return;
      }
      const size_t k = hp.rfind(':');
      host = k == std::string::npos ? hp : hp.substr(0, k);
      if (k != std::string::npos) port = num(hp.c_str() + k + 1);
    };
    std::string host, why;
    int port = 0;
    if (method == "CONNECT") {
      split(target, 443, host, port);
      SOCKET up = open(host, port, why);
      if (up == INVALID_SOCKET) return refuse(c, why);
      const char ok[] = "HTTP/1.1 200 Connection established\r\n\r\n";
      if (send_all(c, ok, sizeof ok - 1) && (rest.empty() || send_all(up, rest.data(), rest.size()))) pump(c, up);
      closesocket(up);
      return;
    }
    if (webt::lower(target.substr(0, 7)) != "http://") return refuse(c, "only http:// targets and CONNECT");
    const size_t ps = target.find('/', 7);
    split(target.substr(7, ps == std::string::npos ? std::string::npos : ps - 7), 80, host, port);
    const std::string path = ps == std::string::npos ? "/" : target.substr(ps);
    SOCKET up = open(host, port, why);
    if (up == INVALID_SOCKET) return refuse(c, why);
    // the request again in origin form, one request per connection (a second one on it could name another host)
    std::string out = method + " " + path + " " + line.substr(s2 + 1) + "\r\n";
    size_t p = le + 2;
    while (p < he - 2) {
      const size_t e = head.find("\r\n", p);
      const std::string h = head.substr(p, e - p), hl = webt::lower(h);
      p = e + 2;
      if (hl.rfind("proxy-", 0) == 0 || hl.rfind("connection:", 0) == 0 || hl.rfind("keep-alive:", 0) == 0) continue;
      out += h + "\r\n";
    }
    out += "Connection: close\r\n\r\n" + rest;
    if (send_all(up, out.data(), out.size())) pump(c, up);
    closesocket(up);
  }
};

// ---- Edge
inline std::wstring edge_path() {
  for (const char* v : {"ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"}) {
    const char* b = getenv(v);
    if (!b) continue;
    const std::filesystem::path p = std::filesystem::path(b) / "Microsoft" / "Edge" / "Application" / "msedge.exe";
    std::error_code ec;
    if (std::filesystem::exists(p, ec)) return p.wstring();
  }
  return {};
}

// one headless Edge run with args; its stdout (out) when capture; within ms, the whole process tree ended after
inline bool edge_run(const std::wstring& args, int ms, std::string* out, std::string& err) {
  const std::wstring exe = edge_path();
  if (exe.empty()) { err = "Microsoft Edge not found (it renders the page)"; return false; }
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
  li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
  SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (out) CreatePipe(&rd, &wr, &sa, 0), SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
  HANDLE nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
  // (Edge inherits these two handles only: with every inheritable handle it got the server's sockets too, other
  //  clients' connections held open by Edge's processes until they end)
  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof si;
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdOutput = out && wr ? wr : nul;
  si.StartupInfo.hStdError = nul;
  si.StartupInfo.hStdInput = nullptr;
  HANDLE inherit[2];
  DWORD ninherit = 0;
  if (nul != INVALID_HANDLE_VALUE) inherit[ninherit++] = nul;
  if (out && wr) inherit[ninherit++] = wr;
  SIZE_T asz = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &asz);
  std::vector<char> abuf(asz);
  si.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)abuf.data();
  const bool attrs = ninherit > 0 && InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &asz) &&
                     UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, ninherit * sizeof(HANDLE), nullptr, nullptr);
  PROCESS_INFORMATION pi{};
  std::wstring cmd = L"\"" + exe + L"\" " + args;
  // (no attribute list: nothing inherited at all, Edge's output then lost: --dump-dom gives no text, the screenshot works)
  const BOOL ok = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, attrs ? TRUE : FALSE,
                                 CREATE_NO_WINDOW | CREATE_SUSPENDED | (attrs ? EXTENDED_STARTUPINFO_PRESENT : 0), nullptr, nullptr,
                                 &si.StartupInfo, &pi);
  if (attrs) DeleteProcThreadAttributeList(si.lpAttributeList);
  if (wr) CloseHandle(wr);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (!ok) {
    err = "Edge did not start (error " + std::to_string(GetLastError()) + ")";
    if (rd) CloseHandle(rd);
    CloseHandle(job);
    return false;
  }
  AssignProcessToJobObject(job, pi.hProcess);
  ResumeThread(pi.hThread);
  std::thread reader;
  if (out)
    reader = std::thread([&] {
      char b[65536];
      DWORD n = 0;
      while (ReadFile(rd, b, sizeof b, &n, nullptr) && n > 0)
        if (out->size() < (16u << 20)) out->append(b, n);
    });
  const bool done = WaitForSingleObject(pi.hProcess, (DWORD)ms) == WAIT_OBJECT_0;
  TerminateJobObject(job, 1);  // (its renderer and GPU processes too)
  if (reader.joinable()) reader.join();
  if (rd) CloseHandle(rd);
  CloseHandle(pi.hThread), CloseHandle(pi.hProcess), CloseHandle(job);
  if (!done) err = "the page did not finish within " + std::to_string(ms / 1000) + " s";
  return done;
}

// rows [y0, y0 + h) of a PNG, as a PNG (WIC); false: not decodable or the rows are past its end
inline bool png_rows(const std::string& png, int y0, int h, std::string& out, int& full_h) {
  // (COM for this call, undone at its end when this call started it: a connection's thread asks many times)
  const HRESULT ci = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  struct Un { bool on; ~Un() { if (on) CoUninitialize(); } } un{SUCCEEDED(ci)};
  if (png.size() > 0xFFFFFFFFull) return false;
  IWICImagingFactory* f = nullptr;
  if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)))) return false;
  IWICStream* in = nullptr;
  IWICBitmapDecoder* dec = nullptr;
  IWICBitmapFrameDecode* fr = nullptr;
  IWICBitmapClipper* clip = nullptr;
  IStream* mem = nullptr;
  IWICBitmapEncoder* enc = nullptr;
  IWICBitmapFrameEncode* fe = nullptr;
  UINT w = 0, hh = 0;
  HRESULT hr = f->CreateStream(&in);
  if (SUCCEEDED(hr)) hr = in->InitializeFromMemory((BYTE*)png.data(), (DWORD)png.size());
  if (SUCCEEDED(hr)) hr = f->CreateDecoderFromStream(in, nullptr, WICDecodeMetadataCacheOnDemand, &dec);
  if (SUCCEEDED(hr)) hr = dec->GetFrame(0, &fr);
  if (SUCCEEDED(hr)) hr = fr->GetSize(&w, &hh);
  full_h = (int)hh;
  if (SUCCEEDED(hr) && (y0 < 0 || (UINT)y0 >= hh)) hr = E_INVALIDARG;
  const WICRect rc{0, y0, (INT)w, (INT)std::min<UINT>((UINT)h, hh - (UINT)std::max(0, y0))};
  if (SUCCEEDED(hr)) hr = f->CreateBitmapClipper(&clip);
  if (SUCCEEDED(hr)) hr = clip->Initialize(fr, &rc);
  if (SUCCEEDED(hr)) hr = CreateStreamOnHGlobal(nullptr, TRUE, &mem);
  if (SUCCEEDED(hr)) hr = f->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc);
  if (SUCCEEDED(hr)) hr = enc->Initialize(mem, WICBitmapEncoderNoCache);
  if (SUCCEEDED(hr)) hr = enc->CreateNewFrame(&fe, nullptr);
  if (SUCCEEDED(hr)) hr = fe->Initialize(nullptr);
  if (SUCCEEDED(hr)) hr = fe->SetSize((UINT)rc.Width, (UINT)rc.Height);
  WICPixelFormatGUID pf = GUID_WICPixelFormat32bppBGRA;
  if (SUCCEEDED(hr)) hr = fe->SetPixelFormat(&pf);
  if (SUCCEEDED(hr)) hr = fe->WriteSource(clip, nullptr);
  if (SUCCEEDED(hr)) hr = fe->Commit();
  if (SUCCEEDED(hr)) hr = enc->Commit();
  if (SUCCEEDED(hr)) {
    HGLOBAL g = nullptr;
    STATSTG stt{};
    mem->Stat(&stt, STATFLAG_NONAME);
    GetHGlobalFromStream(mem, &g);
    const void* p = GlobalLock(g);
    out.assign((const char*)p, (size_t)stt.cbSize.QuadPart);
    GlobalUnlock(g);
  }
  for (IUnknown* u : std::initializer_list<IUnknown*>{fe, enc, mem, clip, fr, dec, in, f})
    if (u) u->Release();
  return SUCCEEDED(hr);
}

inline bool read_bytes(const std::filesystem::path& p, std::string& out) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

struct View {
  std::string png;     // the screen asked for (screen 0: the top)
  std::string html;    // the page after its scripts ran ("" when that run failed)
  int page_height = 0; // the rendered height the screenshot covered (screens * height when the page is longer)
  std::string error;   // the screenshot's (empty: png holds it)
};

// url (http(s), its host already checked public) at width x height, screen n (0-5) down: two Edge runs at once
inline View render(const std::string& url, int screen, int width = 1280, int height = 900) {
  View v;
  static std::mutex mu;  // (one render at a time: each is ~2 Edge processes and their memory)
  std::lock_guard<std::mutex> g(mu);
  const int port = Proxy::get().port();
  if (!port) { v.error = "the proxy for Edge could not start"; return v; }
  static std::atomic<int> seq{0};
  std::error_code ec;
  const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec);
  // (the first render: the folders an earlier server left, more than an hour old, removed; each is a whole Edge
  //  profile, ~10-40 MB)
  static std::once_flag swept;
  std::call_once(swept, [&] {
    std::error_code e2;
    for (const auto& d : std::filesystem::directory_iterator(tmp, e2)) {
      if (!d.is_directory(e2) || d.path().filename().string().rfind("arcinferzero-view-", 0) != 0) continue;
      const auto age = std::filesystem::file_time_type::clock::now() - d.last_write_time(e2);
      if (!e2 && age > std::chrono::hours(1)) std::filesystem::remove_all(d.path(), e2);
    }
  });
  const std::filesystem::path dir = tmp / ("arcinferzero-view-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(++seq));
  std::filesystem::create_directories(dir / "p1", ec);
  std::filesystem::create_directories(dir / "p2", ec);
  const std::wstring wurl = webt::widen(url);
  auto common = [&](const char* profile) {
    return L"--headless=new --disable-gpu --hide-scrollbars --mute-audio --no-first-run --no-default-browser-check "
           L"--disable-extensions --disable-sync --disable-background-networking --disable-component-update "
           L"--disable-default-apps --no-pings --disable-features=Translate,MediaRouter,AutofillServerCommunication "
           L"--webrtc-ip-handling-policy=disable_non_proxied_udp --force-webrtc-ip-handling-policy "
           L"--proxy-server=http://127.0.0.1:" + std::to_wstring(port) + L" --proxy-bypass-list=\"<-loopback>\" "
           L"--user-data-dir=\"" + (dir / profile).wstring() + L"\" --virtual-time-budget=8000 ";
  };
  const int h = height * (std::clamp(screen, 0, 5) + 1);
  const std::filesystem::path shot = dir / "shot.png";
  const std::wstring a1 = common("p1") + L"--window-size=" + std::to_wstring(width) + L"," + std::to_wstring(h) +
                          L" --screenshot=\"" + shot.wstring() + L"\" \"" + wurl + L"\"";
  const std::wstring a2 = common("p2") + L"--window-size=" + std::to_wstring(width) + L"," + std::to_wstring(height) +
                          L" --dump-dom \"" + wurl + L"\"";
  std::string e2;
  std::thread t2([&] { edge_run(a2, 45000, &v.html, e2); });
  std::string e1;
  edge_run(a1, 45000, nullptr, e1);
  t2.join();
  std::string full;
  if (!read_bytes(shot, full) || full.empty()) {
    v.error = e1.empty() ? "Edge wrote no screenshot" : e1;
  } else if (!png_rows(full, height * std::clamp(screen, 0, 5), height, v.png, v.page_height)) {
    v.error = "the screenshot could not be cut";
  }
  // (Edge's processes are ended, but Windows may hold their files a moment longer: tried again for a second)
  for (int k = 0; k < 10; ++k) {
    std::filesystem::remove_all(dir, ec);
    if (!std::filesystem::exists(dir, ec)) break;
    Sleep(100);
  }
  return v;
}

}  // namespace webv
