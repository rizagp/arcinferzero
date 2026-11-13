// web_view (tools/server.cpp, POST /v1/tools/web_view): a page as a browser shows it. Microsoft Edge (on every
// Windows 11) runs headless: one run writes a screenshot, another the page's HTML after its scripts ran (--dump-dom),
// so pages built by scripts read as text too. Edge loads what the page asks for (frames, scripts, images), and the
// checks of web_tools.h hold only for the first URL, so all of Edge's traffic goes through a small proxy in this
// process that connects to public addresses only (as web_fetch: no loopback, private, link-local... address, checked
// on the address it connects to), on web ports (webt::web_port_ok). Edge gets a new empty profile each run (no cookies, logins or
// history of the user's), no extensions, no sync, WebRTC UDP off (it would bypass the proxy).
// Only Edge's processes of a render running now may use the proxy: a connection is served when the process at its
// other end is in one of those runs' jobs (its PID from the TCP table, then IsProcessInJob), so no other program of
// the computer, nor a page in the user's own browser, gets a way to the web through it (self audit G7).
#pragma once
#include "web_tools.h"  // (first: it brings winsock2.h before windows.h; iphlpapi.h after both, self audit R11)
#include "../src/platform_net.h"  // (the proxy's sockets)
#include <wincodec.h>
#include <iphlpapi.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "iphlpapi.lib")

namespace webv {

// the jobs of the Edge runs going on now
struct Jobs {
  static std::mutex& mu() { static std::mutex m; return m; }
  static std::vector<HANDLE>& list() { static std::vector<HANDLE> v; return v; }
  static void add(HANDLE j) { std::lock_guard<std::mutex> g(mu()); list().push_back(j); }
  static void remove(HANDLE j) {
    std::lock_guard<std::mutex> g(mu());
    auto& v = list();
    v.erase(std::remove(v.begin(), v.end(), j), v.end());
  }
  // is the process pid in one of them
  static bool has(DWORD pid) {
    HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!p) return false;
    bool in = false;
    {
      std::lock_guard<std::mutex> g(mu());
      for (HANDLE j : list()) {
        BOOL b = FALSE;
        if (IsProcessInJob(p, j, &b) && b) { in = true; break; }
      }
    }
    CloseHandle(p);
    return in;
  }
};

// the process at the other end of a loopback connection to port `to` from port `from` (0: not found)
inline DWORD peer_pid(USHORT from, USHORT to) {
  ULONG sz = 0;
  GetExtendedTcpTable(nullptr, &sz, FALSE, AF_INET, TCP_TABLE_OWNER_PID_CONNECTIONS, 0);
  for (int tries = 0; tries < 3; ++tries) {
    std::vector<char> b(sz + 4096);
    sz = (ULONG)b.size();
    if (GetExtendedTcpTable(b.data(), &sz, FALSE, AF_INET, TCP_TABLE_OWNER_PID_CONNECTIONS, 0) != NO_ERROR) continue;
    const auto* t = (const MIB_TCPTABLE_OWNER_PID*)b.data();
    for (DWORD k = 0; k < t->dwNumEntries; ++k) {
      const MIB_TCPROW_OWNER_PID& r = t->table[k];
      if (ntohs((USHORT)r.dwLocalPort) == from && ntohs((USHORT)r.dwRemotePort) == to && r.dwLocalAddr == htonl(INADDR_LOOPBACK))
        return r.dwOwningPid;
    }
    return 0;
  }
  return 0;
}

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
  static constexpr uint64_t kMaxTunnelBytes = 256ull << 20;  // (a page's one connection: a stream stops there)
  int port_ = 0;
  std::once_flag once_;
  std::atomic<int> active_{0};

  void start() {
    static const bool wsa = plat::net_init();  // (once: self audit L2)
    if (!wsa) return;
    plat::socket_t s = plat::sock_open(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == plat::kNoSocket) return;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // (this computer only)
    a.sin_port = 0;
    sockaddr_storage got{};
    if (!plat::sock_bind_listen(s, (sockaddr*)&a, sizeof a, 64) || !plat::sock_name(s, got)) {
      plat::sock_close(s);
      return;
    }
    port_ = ntohs(((sockaddr_in*)&got)->sin_port);
    std::thread([this, s] {
      for (;;) {
        plat::socket_t c = plat::sock_accept(s);
        if (c == plat::kNoSocket) {  // (a failing listener: no busy loop on one core)
          Sleep(100);
          continue;
        }
        if (active_.fetch_add(1) >= kMaxConns) {  // (taken, then given back when over: no burst past 64)
          --active_;
          plat::sock_close(c);
          continue;
        }
        const int me = port_;
        std::thread([this, c, me] {
          sockaddr_storage pa{};
          // (Edge's processes of a running render only, self audit G7. The TCP table can lag a new connection by a
          //  moment: looked up again for up to ~100 ms before a refusal, and a refusal is logged, R5)
          bool ok = false;
          DWORD pid = 0;
          if (plat::sock_peer(c, pa))
            for (int k = 0; k < 5 && !ok; ++k) {
              pid = peer_pid(ntohs(((sockaddr_in*)&pa)->sin_port), (USHORT)me);
              ok = pid && Jobs::has(pid);
              if (!ok) Sleep(20);
            }
          if (ok) serve(c);
          else {
            static std::atomic<int> logged{0};
            const int n = ++logged;  // (the first 20, then every 100th with the count: never silent, never flooding, R19)
            if (n <= 20 || n % 100 == 0)
              zlog::eprintf("web_view: proxy refused a connection from process %lu (not the web_view browser; %d so far)\n",
                      (unsigned long)pid, n);
            refuse(c, "this proxy serves the web_view browser only");
          }
          plat::sock_close(c);
          --active_;
        }).detach();
      }
    }).detach();
  }
  static void timeouts(plat::socket_t s, unsigned ms) {
    plat::sock_recv_timeout(s, ms);
    plat::sock_send_timeout(s, ms);
  }
  static bool send_all(plat::socket_t s, const char* p, size_t n) {
    while (n > 0) {
      const int k = plat::sock_send(s, p, (int)std::min<size_t>(n, 1 << 20));
      if (k <= 0) return false;
      p += k, n -= (size_t)k;
    }
    return true;
  }
  static void refuse(plat::socket_t c, const std::string& why) {
    const std::string b = "HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: " +
                          std::to_string(why.size()) + "\r\n\r\n" + why;
    send_all(c, b.data(), b.size());
  }
  // a connection to host:port when every address it resolves to is public, else plat::kNoSocket and why
  static plat::socket_t open(const std::string& host, int port, std::string& why) {
    // (any web port: the proxy serves only Edge's processes of a render, G7, so other ports than 80 / 443 are as safe
    //  as in web_fetch; the services' ports refused as browsers do)
    if (!webt::web_port_ok(port)) { why = "port " + std::to_string(port) + " refused (a port of another service, not the web)"; return plat::kNoSocket; }
    const std::string h = webt::lower(host);
    if (h.empty() || h == "localhost" || (h.size() > 10 && h.compare(h.size() - 10, 10, ".localhost") == 0)) { why = "local address refused"; return plat::kNoSocket; }
    addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) { why = "cannot resolve " + host; return plat::kNoSocket; }
    for (addrinfo* a = res; a; a = a->ai_next)
      if (!webt::public_addr(a->ai_addr)) {
        freeaddrinfo(res);
        why = "local or private address refused (" + host + ")";
        return plat::kNoSocket;
      }
    plat::socket_t s = plat::kNoSocket;
    for (addrinfo* a = res; a && s == plat::kNoSocket; a = a->ai_next) {  // (to the addresses just checked: no second lookup)
      s = plat::sock_open(a->ai_family, SOCK_STREAM, IPPROTO_TCP);
      if (s == plat::kNoSocket) continue;
      timeouts(s, 15000);
      if (!plat::sock_connect(s, a->ai_addr, (size_t)a->ai_addrlen)) { plat::sock_close(s); s = plat::kNoSocket; }
    }
    freeaddrinfo(res);
    if (s == plat::kNoSocket) why = "cannot connect to " + host;
    return s;
  }
  // bytes both ways until either side closes, 60 s without any, 10 minutes, or kMaxTunnelBytes (self audit G8)
  static void pump(plat::socket_t a, plat::socket_t b) {
    std::vector<char> buf(64 << 10);
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t moved = 0;
    for (;;) {
      if (std::chrono::steady_clock::now() - t0 > std::chrono::minutes(10) || moved > kMaxTunnelBytes) return;
      bool ra = false, rb = false;
      const int n = plat::sock_wait_read2(a, b, 60000, ra, rb);
      if (n <= 0) return;
      for (auto [from, to, ready] : {std::tuple{a, b, ra}, std::tuple{b, a, rb}})
        if (ready) {
          const int k = plat::sock_recv(from, buf.data(), (int)buf.size());
          if (k <= 0 || !send_all(to, buf.data(), (size_t)k)) return;
          moved += (uint64_t)k;
        }
    }
  }
  static void serve(plat::socket_t c) {
    timeouts(c, 30000);
    std::string head;
    char tmp[4096];
    while (head.find("\r\n\r\n") == std::string::npos) {
      if (head.size() > (32 << 10)) return;
      const int k = plat::sock_recv(c, tmp, (int)sizeof tmp);
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
      plat::socket_t up = open(host, port, why);
      if (up == plat::kNoSocket) return refuse(c, why);
      const char ok[] = "HTTP/1.1 200 Connection established\r\n\r\n";
      if (send_all(c, ok, sizeof ok - 1) && (rest.empty() || send_all(up, rest.data(), rest.size()))) pump(c, up);
      plat::sock_close(up);
      return;
    }
    if (webt::lower(target.substr(0, 7)) != "http://") return refuse(c, "only http:// targets and CONNECT");
    const size_t ps = target.find('/', 7);
    split(target.substr(7, ps == std::string::npos ? std::string::npos : ps - 7), 80, host, port);
    const std::string path = ps == std::string::npos ? "/" : target.substr(ps);
    plat::socket_t up = open(host, port, why);
    if (up == plat::kNoSocket) return refuse(c, why);
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
    plat::sock_close(up);
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

static constexpr size_t kMaxDom = 16u << 20;  // (--dump-dom output kept; more: cut, and *cut says so)

// one headless Edge run with args; its stdout (out) when capture, *cut when it was longer than kMaxDom; within ms, the
// whole process tree ended after. Its job: killed with its handle, at most 3 GB of memory together and 32 processes
// (a heavy page fails in Edge, not the server: self audit O4); the proxy serves the processes in it only.
inline bool edge_run(const std::wstring& args, int ms, std::string* out, std::string& err, bool* cut = nullptr) {
  const std::wstring exe = edge_path();
  if (exe.empty()) { err = "Microsoft Edge not found (it renders the page)"; return false; }
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (!job) { err = "no job object for Edge (error " + std::to_string(GetLastError()) + ")"; return false; }
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
  li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
  li.JobMemoryLimit = (SIZE_T)3 << 30;
  li.BasicLimitInformation.ActiveProcessLimit = 32;
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
  // (not in the job, e.g. the server itself in a job that allows no nesting: Edge is not run at all, as nothing
  //  would end its children, and the proxy would not serve them: self audit G1)
  if (!AssignProcessToJobObject(job, pi.hProcess)) {
    err = "Edge could not be put in a job (error " + std::to_string(GetLastError()) + ")";
    TerminateProcess(pi.hProcess, 1);
    if (rd) CloseHandle(rd);
    CloseHandle(pi.hThread), CloseHandle(pi.hProcess), CloseHandle(job);
    return false;
  }
  Jobs::add(job);
  ResumeThread(pi.hThread);
  std::thread reader;
  if (out)
    reader = std::thread([&] {
      char b[65536];
      DWORD n = 0;
      while (ReadFile(rd, b, sizeof b, &n, nullptr) && n > 0) {
        if (out->size() + n <= kMaxDom) out->append(b, n);
        else if (cut) *cut = true;
      }
    });
  const bool done = WaitForSingleObject(pi.hProcess, (DWORD)ms) == WAIT_OBJECT_0;
  {  // (Edge stopped by the job's memory limit fails inside itself, silently: said here, R7)
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION q{};
    if (QueryInformationJobObject(job, JobObjectExtendedLimitInformation, &q, sizeof q, nullptr) &&
        q.PeakJobMemoryUsed + (16u << 20) >= li.JobMemoryLimit) {  // (within 16 MB of it: an allocation was refused, R18)
      err = "the page reached the browser's 3 GB memory limit: it was stopped (the screenshot may be blank)";
      zlog::eprintf("web_view: %s\n", err.c_str());
    }
  }
  TerminateJobObject(job, 1);  // (its renderer and GPU processes too)
  if (!done) TerminateProcess(pi.hProcess, 1);
  Jobs::remove(job);
  if (reader.joinable()) {
    // (a process that kept the pipe open past the kill: the read is cancelled, so the join cannot outlive the wait)
    for (int k = 0; k < 50 && WaitForSingleObject(reader.native_handle(), 100) == WAIT_TIMEOUT; ++k)
      CancelSynchronousIo(reader.native_handle());
    reader.join();
  }
  if (rd) CloseHandle(rd);
  CloseHandle(pi.hThread), CloseHandle(pi.hProcess), CloseHandle(job);
  if (!done)  // (after the memory note, not over it, R18)
    err = (err.empty() ? std::string() : err + "; ") + "the page did not finish within " + std::to_string(ms / 1000) + " s";
  return done;
}

// rows [y0, y0 + h) of a PNG, as a PNG (WIC); false: not decodable or the rows are past its end
inline bool png_rows(const std::string& png, int y0, int h, std::string& out) {
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
  if (SUCCEEDED(hr)) {  // (each step checked: a null lock with a size would make assign undefined, self audit C1)
    HGLOBAL g = nullptr;
    STATSTG stt{};
    hr = mem->Stat(&stt, STATFLAG_NONAME);
    if (SUCCEEDED(hr)) hr = GetHGlobalFromStream(mem, &g);
    const void* p = SUCCEEDED(hr) ? GlobalLock(g) : nullptr;
    if (p && stt.cbSize.QuadPart <= GlobalSize(g)) out.assign((const char*)p, (size_t)stt.cbSize.QuadPart);
    else hr = E_FAIL;
    if (p) GlobalUnlock(g);
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
  std::string png;        // the screen asked for (screen 0: the top)
  std::string html;       // the page after its scripts ran ("" when that run failed)
  bool html_cut = false;  // (it was longer than kMaxDom)
  std::string error;      // the screenshot's (empty: png holds it)
  std::string note;       // what else went wrong in a run that still gave something (the memory limit)
};

// the URL as one command-line argument: its host in ASCII form (an internationalized name as punycode, which is what
// DNS and the checks see, R4; a host with any other odd byte: refused), and in the rest every byte that could end its
// quotes or the line, or that the shell may read (a space, ", \, <, >, |, ^, `, {, }, a control byte, NUL, anything
// not ASCII) percent-encoded; "" for a URL that is not http(s) (self audit G2)
inline std::string arg_url(const std::string& url) {
  const std::string l = webt::lower(url.substr(0, 8));
  if (l.rfind("http://", 0) != 0 && l.rfind("https://", 0) != 0) return {};
  const size_t hs = url.find("://") + 3, he = url.find_first_of("/?#", hs);
  const std::string auth = url.substr(hs, he == std::string::npos ? std::string::npos : he - hs);
  // (user@host:port: the host part converted, the rest of the authority kept as it is, but only plain characters)
  const size_t at = auth.rfind('@');
  std::string host = auth.substr(at == std::string::npos ? 0 : at + 1), tail;
  if (!host.empty() && host[0] != '[' && host.rfind(':') != std::string::npos) tail = host.substr(host.rfind(':')), host.resize(host.rfind(':'));
  const std::string ah = webt::ascii_host(host);
  const std::string auth2 = (at == std::string::npos ? std::string() : auth.substr(0, at + 1)) + ah + tail;
  if (ah.empty()) return {};
  for (unsigned char c : auth2)
    if (c <= 0x20 || c >= 0x7F || strchr("\"\\<>|^`{}", c)) return {};
  static const char* hex = "0123456789ABCDEF";
  std::string o = url.substr(0, hs) + auth2;
  for (size_t k = he == std::string::npos ? url.size() : he; k < url.size(); ++k) {
    const unsigned char c = (unsigned char)url[k];
    if (c <= 0x20 || c >= 0x7F || strchr("\"\\<>|^`{}", c)) o += '%', o += hex[c >> 4], o += hex[c & 15];
    else o += (char)c;
  }
  return o;
}

// url (http(s), its host already checked public) at width x height, screen n (0-5) down: two Edge runs at once
inline View render(const std::string& url, int screen, int width = 1280, int height = 900) {
  View v;
  static std::mutex mu;  // (one render at a time: each is ~2 Edge processes and their memory)
  std::lock_guard<std::mutex> g(mu);
  const int port = Proxy::get().port();
  if (!port) { v.error = "the proxy for Edge could not start"; return v; }
  const std::string safe = arg_url(url);
  const std::wstring wurl = webt::widen(safe);
  if (safe.empty() || wurl.empty()) { v.error = "not an http(s) URL"; return v; }
  static std::atomic<int> seq{0};
  std::error_code ec;
  const std::filesystem::path tmp = std::filesystem::temp_directory_path(ec);
  // (every render: the profile folders left behind, more than 30 minutes old, removed (a render lasts 45 s at most,
  //  one at a time); each is a whole Edge profile, ~10-40 MB: self audit L1)
  {
    std::error_code e2;
    for (const auto& d : std::filesystem::directory_iterator(tmp, e2)) {
      const std::string fn = d.path().filename().string();
      if (!d.is_directory(e2) || (fn.rfind("inferzero-view-", 0) != 0 && fn.rfind("arcinferzero-view-", 0) != 0)) continue;
      const auto age = std::filesystem::file_time_type::clock::now() - d.last_write_time(e2);
      if (!e2 && age > std::chrono::minutes(30)) std::filesystem::remove_all(d.path(), e2);
    }
  }
  const std::filesystem::path dir = tmp / ("inferzero-view-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(++seq));
  std::filesystem::create_directories(dir / "p1", ec);
  std::filesystem::create_directories(dir / "p2", ec);
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
  std::thread t2([&] { edge_run(a2, 45000, &v.html, e2, &v.html_cut); });
  std::string e1;
  edge_run(a1, 45000, nullptr, e1);
  t2.join();
  for (const std::string* e : {&e1, &e2})
    if (e->find("3 GB") != std::string::npos) v.note = *e;
  std::string full;
  if (!read_bytes(shot, full) || full.empty()) {
    v.error = e1.empty() ? "Edge wrote no screenshot" : e1;
  } else if (!png_rows(full, height * std::clamp(screen, 0, 5), height, v.png)) {  // (the window's height: Edge shoots the window, R17)
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
