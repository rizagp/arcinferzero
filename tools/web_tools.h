// Web search and page fetch for the web UI's tools (server endpoints /v1/tools/web_search and /v1/tools/web_fetch):
// WinHTTP requests, result parsing for Brave Search (Bing when Brave gives nothing) and HTML to plain text. Only
// public http(s) addresses are fetched: every host, redirects included, is resolved first and loopback, private,
// link-local, multicast and reserved addresses are refused, and the address the connection went to is checked again
// before anything is read, so a model cannot reach the machine or the local network.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winhttp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "normaliz.lib")  // (IdnToAscii)

namespace webt {

struct Page {
  int status = 0;
  std::string url, content_type, body, error;  // url: after redirects
  std::string disposition;                     // (Content-Disposition: a download's file name)
  bool cut = false;                            // (the body stopped at max_bytes: there was more)
  bool too_big = false;                        // (whole or nothing, and its Content-Length was over the limit: not read)
  bool timed_out = false;                      // (the body stopped early: the deadline, a stall or a dropped connection)
};

struct Result {
  std::string title, url, snippet;
};

inline std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}

inline std::string narrow(const wchar_t* p, size_t len, UINT cp = CP_UTF8) {
  if (!len) return {};
  const int n = WideCharToMultiByte(cp, 0, p, (int)len, nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(cp, 0, p, (int)len, s.data(), n, nullptr, nullptr);
  return s;
}

inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

inline std::string url_encode(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char)c;
    else if (c == ' ') o += '+';
    else { o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
  }
  return o;
}

// A public address: not loopback, private (RFC 1918, carrier-grade NAT, IPv6 unique local), link-local, multicast,
// unspecified, reserved or for documentation / benchmarks; an IPv6 address that carries an IPv4 one (mapped,
// compatible, NAT64, 6to4) is judged by that IPv4 address, Teredo is refused
inline bool public_v4(const uint8_t* b) {
  if (b[0] == 0 || b[0] == 10 || b[0] == 127 || b[0] >= 224) return false;  // (224+: multicast and 240/4 reserved)
  if (b[0] == 169 && b[1] == 254) return false;
  if (b[0] == 172 && (b[1] & 0xf0) == 16) return false;
  if (b[0] == 192 && b[1] == 168) return false;
  if (b[0] == 100 && (b[1] & 0xc0) == 64) return false;
  if (b[0] == 192 && b[1] == 0 && (b[2] == 0 || b[2] == 2)) return false;  // 192.0.0.0/24, 192.0.2.0/24
  if (b[0] == 198 && (b[1] & 0xfe) == 18) return false;                      // 198.18.0.0/15
  if (b[0] == 198 && b[1] == 51 && b[2] == 100) return false;                // 198.51.100.0/24
  if (b[0] == 203 && b[1] == 0 && b[2] == 113) return false;                 // 203.0.113.0/24
  return true;
}
inline bool public_addr(const sockaddr* sa) {
  if (sa->sa_family == AF_INET) return public_v4((const uint8_t*)&((const sockaddr_in*)sa)->sin_addr);
  if (sa->sa_family == AF_INET6) {
    const uint8_t* b = ((const sockaddr_in6*)sa)->sin6_addr.s6_addr;
    static const uint8_t v4map[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    static const uint8_t v4compat[12] = {};
    static const uint8_t nat64[12] = {0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0};
    if (!memcmp(b, v4map, 12) || !memcmp(b, v4compat, 12) || !memcmp(b, nat64, 12)) return public_v4(b + 12);  // (:: and ::1 too)
    if (b[0] == 0x20 && b[1] == 0x02) return public_v4(b + 2);                 // 6to4 2002::/16
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0 && b[3] == 0) return false;  // Teredo 2001::/32
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8) return false;  // documentation 2001:db8::/32
    if (b[0] == 0x01 && b[1] == 0 && !b[2] && !b[3] && !b[4] && !b[5] && !b[6] && !b[7]) return false;  // discard 100::/64
    if ((b[0] & 0xfe) == 0xfc) return false;                   // fc00::/7
    if (b[0] == 0xfe && (b[1] & 0x80) == 0x80) return false;  // fe80::/10 link-local, fec0::/10 site-local
    if (b[0] == 0xff) return false;                            // multicast
    return true;
  }
  return false;
}

// A host name as DNS knows it: an internationalized one (münchen.de) in its ASCII form (xn--mnchen-3ya.de), so the
// name checked, the name connected to and the name Edge opens are one (self audit R4); "" when it has no such form
inline std::string ascii_host(const std::string& h) {
  if (std::all_of(h.begin(), h.end(), [](char c) { return (unsigned char)c < 0x80; })) return h;
  const std::wstring w = widen(h);
  wchar_t out[256];
  const int n = w.empty() ? 0 : IdnToAscii(0, w.c_str(), (int)w.size(), out, 256);
  return n > 0 ? narrow(out, (size_t)n) : std::string();
}

// Every address the host resolves to is public (empty error), else why not
inline std::string check_host(const std::string& host_in) {
  const std::string host = ascii_host(host_in);
  if (host.empty()) return "no host";
  const std::string h = lower(host);
  if (h == "localhost" || (h.size() > 10 && h.compare(h.size() - 10, 10, ".localhost") == 0)) return "local address refused";
  static const bool wsa = [] { WSADATA wd; return WSAStartup(MAKEWORD(2, 2), &wd) == 0; }();  // (once: self audit L2)
  if (!wsa) return "no network (WSAStartup failed)";
  addrinfo hints{}, *res = nullptr;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return "cannot resolve " + host;
  std::string e;
  for (addrinfo* a = res; a && e.empty(); a = a->ai_next)
    if (!public_addr(a->ai_addr)) e = "local or private address refused (" + host + ")";
  freeaddrinfo(res);
  return e;
}

// A redirect target against the URL it came from (absolute, scheme-relative, absolute path or relative path)
inline std::string resolve_url(const std::string& base, const std::string& loc) {
  if (loc.rfind("http://", 0) == 0 || loc.rfind("https://", 0) == 0) return loc;
  const size_t se = base.find("://");
  if (se == std::string::npos) return loc;
  if (loc.rfind("//", 0) == 0) return base.substr(0, se + 1) + loc;
  const size_t he = base.find('/', se + 3);
  const std::string origin = he == std::string::npos ? base : base.substr(0, he);
  if (!loc.empty() && loc[0] == '/') return origin + loc;
  std::string dir = he == std::string::npos ? "/" : base.substr(he, base.find_first_of("?#", he) - he);
  dir = dir.substr(0, dir.rfind('/') + 1);
  return origin + dir + loc;
}

// A port the web tools may connect to: any, except the ones browsers refuse because other services listen there
// (Chrome's restricted ports: FTP, SSH, telnet, mail, DNS, NetBIOS / SMB, LDAP, RPC, NFS, SIP, IRC, X11...), so a
// page or a model cannot make the server or its browser talk to such a service on a public host
inline bool web_port_ok(int port) {
  static const int bad[] = {1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79, 87, 95, 101,
                            102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427,
                            445, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989,
                            990, 993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 5060, 5061, 6000, 6566, 6665, 6666, 6667,
                            6668, 6669, 6697, 10080};
  if (port < 1 || port > 65535) return false;
  for (int b : bad) if (b == port) return false;
  return true;
}

// An http(s) URL's host and port, as get() parses it (WinHttpCrackUrl): the one parser of a URL's host for the web
// tools (self audit C3); false: not an http(s) URL
inline bool url_host_port(const std::string& url, std::string& host, int& port) {
  const std::wstring wu = widen(url);
  if (wu.empty()) return false;
  URL_COMPONENTS uc{};
  uc.dwStructSize = sizeof uc;
  wchar_t h[256] = {};
  uc.lpszHostName = h, uc.dwHostNameLength = 256;
  uc.dwUrlPathLength = (DWORD)-1, uc.dwExtraInfoLength = (DWORD)-1;
  if (!WinHttpCrackUrl(wu.c_str(), (DWORD)wu.size(), 0, &uc) || (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS))
    return false;
  host = narrow(h, wcslen(h));
  port = uc.nPort;
  return !host.empty();
}

// GET a public http(s) URL: redirects followed by hand (each host checked), at most max_bytes of body, the whole
// request within timeout_ms. limit, when given: the bytes this body may have, decided from its headers (status,
// content type, file name) before any of it is read (0: none: a type that is refused); `whole` set by it: a file that
// is whole or useless, so a Content-Length over the limit reads nothing (too_big) instead of its first part (a page
// is still read up to the limit). cut: stopped at the limit; timed_out: the deadline came first (self audit O2, G3,
// G5; R1-R3: the Content-Length test only for whole-or-nothing files, never for a page, a search or the GitHub API,
// whose limits it does not concern; Content-Length counts compressed bytes, the limit decompressed ones)
inline Page get(std::string url, size_t max_bytes = 4u << 20, int timeout_ms = 15000, const std::string& accept_lang = "en-US,en;q=0.9",
                const std::string& accept = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                const std::function<size_t(const Page&, bool& whole)>& limit = {}) {
  Page p;
  const auto t0 = std::chrono::steady_clock::now();
  HINTERNET ses = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
                              // (direct connections only: through a proxy the address checked below would be the
                              //  proxy's, and the proxy could fetch a local address)
                              WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!ses) { p.error = "WinHttpOpen failed"; return p; }
  DWORD dec = WINHTTP_DECOMPRESSION_FLAG_ALL;
  WinHttpSetOption(ses, WINHTTP_OPTION_DECOMPRESSION, &dec, sizeof dec);
  DWORD h2 = WINHTTP_PROTOCOL_FLAG_HTTP2;
  WinHttpSetOption(ses, WINHTTP_OPTION_ENABLE_HTTP_PROTOCOL, &h2, sizeof h2);
  WinHttpSetTimeouts(ses, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
  // (WinHTTP's timeouts are per operation: each step gets only what is left of timeout_ms, over every redirect. 0 is no
  //  timeout to WinHTTP, so at least 1 ms)
  const auto deadline = t0 + std::chrono::milliseconds(timeout_ms);
  auto left = [&] {
    return (int)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
  };
  auto receive_within = [&](HINTERNET rq) {
    DWORD ms = (DWORD)std::max(1, left());
    WinHttpSetOption(rq, WINHTTP_OPTION_RECEIVE_TIMEOUT, &ms, sizeof ms);
    WinHttpSetOption(rq, WINHTTP_OPTION_RECEIVE_RESPONSE_TIMEOUT, &ms, sizeof ms);
  };
  for (int hop = 0; hop < 6; ++hop) {
    const int ms = left();
    p = Page{};
    p.url = url;
    if (ms <= 0) {
      p.error = "timed out";
      break;
    }
    const std::wstring wu = widen(url);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof uc;
    wchar_t host[256] = {}, path[8192] = {}, extra[8192] = {};
    uc.lpszHostName = host, uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path, uc.dwUrlPathLength = 8192;
    uc.lpszExtraInfo = extra, uc.dwExtraInfoLength = 8192;
    if (!WinHttpCrackUrl(wu.c_str(), (DWORD)wu.size(), 0, &uc) || (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS)) {
      p.error = "not an http(s) URL";
      break;
    }
    const std::string h8 = narrow(host, wcslen(host));
    if (!web_port_ok(uc.nPort)) {  // (each hop: a redirect to such a port too)
      p.error = "port " + std::to_string(uc.nPort) + " refused (a port of another service, not the web)";
      break;
    }
    if (!(p.error = check_host(h8)).empty()) break;
    // (the name connected to is the one checked: an internationalized host in its punycode form, R15)
    const std::wstring whost = widen(ascii_host(h8));
    HINTERNET con = whost.empty() ? nullptr : WinHttpConnect(ses, whost.c_str(), uc.nPort, 0);
    const std::wstring target = std::wstring(path) + extra;
    HINTERNET rq = con ? WinHttpOpenRequest(con, L"GET", target.empty() ? L"/" : target.c_str(), nullptr, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0)
                       : nullptr;
    if (!rq) {
      p.error = "cannot connect to " + h8;
      if (con) WinHttpCloseHandle(con);
      break;
    }
    DWORD noredir = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(rq, WINHTTP_OPTION_DISABLE_FEATURE, &noredir, sizeof noredir);
    const std::wstring hdr = widen("Accept: " + accept + "\r\nAccept-Language: " + accept_lang + "\r\n");
    const int third = std::max(1, ms / 3);  // (resolve, connect and send: a third of what is left each)
    WinHttpSetTimeouts(rq, third, third, third, std::max(1, ms));
    bool ok = WinHttpSendRequest(rq, hdr.c_str(), (DWORD)hdr.size(), WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (ok) {
      receive_within(rq);
      ok = left() > 0 && WinHttpReceiveResponse(rq, nullptr);
    }
    if (ok) {
      // the address WinHTTP actually connected to (it resolves the host again: a name that answered with a public
      // address for check_host and a local one now, DNS rebinding, gets nothing back)
      WINHTTP_CONNECTION_INFO ci{};
      DWORD cn = sizeof ci;
      if (!WinHttpQueryOption(rq, WINHTTP_OPTION_CONNECTION_INFO, &ci, &cn) || !public_addr((const sockaddr*)&ci.RemoteAddress)) {
        p.error = "local or private address refused (" + h8 + ", at connection)";
        WinHttpCloseHandle(rq), WinHttpCloseHandle(con);
        break;
      }
      DWORD st = 0, sz = sizeof st;
      WinHttpQueryHeaders(rq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &st, &sz, WINHTTP_NO_HEADER_INDEX);
      p.status = (int)st;
      wchar_t buf[4096];
      sz = sizeof buf;
      if (WinHttpQueryHeaders(rq, WINHTTP_QUERY_CONTENT_TYPE, WINHTTP_HEADER_NAME_BY_INDEX, buf, &sz, WINHTTP_NO_HEADER_INDEX))
        p.content_type = narrow(buf, sz / sizeof(wchar_t));
      sz = sizeof buf;
      if (WinHttpQueryHeaders(rq, WINHTTP_QUERY_CONTENT_DISPOSITION, WINHTTP_HEADER_NAME_BY_INDEX, buf, &sz, WINHTTP_NO_HEADER_INDEX))
        p.disposition = narrow(buf, sz / sizeof(wchar_t));
      if (st >= 300 && st < 400) {
        sz = sizeof buf;
        if (WinHttpQueryHeaders(rq, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, buf, &sz, WINHTTP_NO_HEADER_INDEX)) {
          url = resolve_url(url, narrow(buf, sz / sizeof(wchar_t)));
          WinHttpCloseHandle(rq), WinHttpCloseHandle(con);
          continue;
        }
        if (st != 304) p.error = "a redirect (HTTP " + std::to_string(st) + ") without a Location";  // (G9; a 304 is no redirect, R6)
      }
      bool whole = false;
      if (limit) max_bytes = std::min(max_bytes, limit(p, whole));
      if (whole) {
        DWORD cl = 0;  // (a whole-or-nothing file announcing more than its limit: nothing read)
        sz = sizeof cl;
        if (WinHttpQueryHeaders(rq, WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &cl, &sz, WINHTTP_NO_HEADER_INDEX) &&
            cl > max_bytes)
          p.too_big = true;
      }
      for (; !p.too_big && max_bytes > 0;) {
        if (left() <= 0) {  // (the deadline in the middle of the body: not whole)
          p.timed_out = true;
          break;
        }
        receive_within(rq);
        DWORD avail = 0;
        // (a failed query or read in the middle of the body, a stall that reached the deadline or a dropped
        //  connection, leaves it not whole: timed_out, so a cut file is never saved as a whole one, R14. A successful
        //  query of 0 bytes is the end)
        if (!WinHttpQueryDataAvailable(rq, &avail)) { p.timed_out = true; break; }
        if (!avail) break;
        const size_t old = p.body.size();
        avail = (DWORD)std::min<size_t>(avail, max_bytes - old);  // (old < max_bytes: the loop stops at max_bytes)
        p.body.resize(old + avail);
        DWORD got = 0;
        if (!WinHttpReadData(rq, p.body.data() + old, avail, &got)) { p.body.resize(old); p.timed_out = true; break; }
        p.body.resize(old + got);
        if (got == 0) break;  // (a read of nothing: the end, not a reason to ask again until the deadline, G6)
        if (p.body.size() >= max_bytes) {
          p.body.resize(max_bytes);
          DWORD more = 0;  // (exactly max_bytes long is not cut)
          p.cut = WinHttpQueryDataAvailable(rq, &more) && more > 0;
          break;
        }
      }
    } else {
      const DWORD e = GetLastError();
      p.error = e == ERROR_WINHTTP_TIMEOUT || left() <= 0 ? "request to " + h8 + " timed out"
                                                          : "request to " + h8 + " failed (error " + std::to_string(e) + ")";
    }
    WinHttpCloseHandle(rq), WinHttpCloseHandle(con);
    break;
  }
  // (a 3xx with a Location still at the 6th hop; a 304 is left to the caller, which takes only 200)
  if (p.error.empty() && p.status >= 300 && p.status < 400 && p.status != 304) p.error = "too many redirects";
  WinHttpCloseHandle(ses);
  // a body in Latin-1 / Windows-1252 (the content type or a meta tag says so) becomes UTF-8; a text only (a page, a
  // text file): a PDF or an image fetched to be saved keeps its bytes, whatever its first 2 KB hold
  const std::string ct = lower(p.content_type), head = lower(p.body.substr(0, 2048));
  const bool textual = (ct.empty() || ct.rfind("text/", 0) == 0 || ct.find("html") != std::string::npos ||
                        ct.find("xml") != std::string::npos || ct.find("json") != std::string::npos) &&
                       p.body.compare(0, 5, "%PDF-") != 0 && p.body.compare(0, 2, "PK") != 0 &&
                       p.body.find('\0') == std::string::npos;  // (no type given: a binary file is no text)
  if (textual && (ct.find("iso-8859-1") != std::string::npos || ct.find("windows-1252") != std::string::npos ||
      (ct.find("charset") == std::string::npos && (head.find("charset=iso-8859-1") != std::string::npos || head.find("charset=windows-1252") != std::string::npos)))) {
    const int n = MultiByteToWideChar(1252, 0, p.body.data(), (int)p.body.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(1252, 0, p.body.data(), (int)p.body.size(), w.data(), n);
    p.body = narrow(w.data(), w.size());
  }
  return p;
}

inline void put_utf8(std::string& o, uint32_t c) {
  if (c < 0x80) o += (char)c;
  else if (c < 0x800) o += (char)(0xc0 | c >> 6), o += (char)(0x80 | (c & 63));
  else if (c < 0x10000) o += (char)(0xe0 | c >> 12), o += (char)(0x80 | (c >> 6 & 63)), o += (char)(0x80 | (c & 63));
  else if (c < 0x110000) o += (char)(0xf0 | c >> 18), o += (char)(0x80 | (c >> 12 & 63)), o += (char)(0x80 | (c >> 6 & 63)), o += (char)(0x80 | (c & 63));
}

// HTML character references (numeric and the common named ones)
inline std::string decode_entities(const std::string& s) {
  static const std::pair<const char*, uint32_t> named[] = {
      {"amp", '&'}, {"lt", '<'}, {"gt", '>'}, {"quot", '"'}, {"apos", '\''}, {"nbsp", ' '}, {"ndash", 0x2013},
      {"mdash", 0x2014}, {"hellip", 0x2026}, {"lsquo", 0x2018}, {"rsquo", 0x2019}, {"ldquo", 0x201c}, {"rdquo", 0x201d},
      {"laquo", 0xab}, {"raquo", 0xbb}, {"copy", 0xa9}, {"reg", 0xae}, {"trade", 0x2122}, {"middot", 0xb7},
      {"bull", 0x2022}, {"times", 0xd7}, {"deg", 0xb0}, {"euro", 0x20ac}, {"pound", 0xa3}, {"rarr", 0x2192}};
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '&') { o += s[i]; continue; }
    const size_t semi = std::string_view(s).substr(i + 1, 10).find(';');
    if (semi == std::string::npos) { o += '&'; continue; }  // (the next 10 bytes only: no scan to the end per '&')
    const size_t e = i + 1 + semi;
    const std::string n = s.substr(i + 1, e - i - 1);
    uint32_t c = 0;
    if (n.size() > 1 && n[0] == '#') {
      c = (n[1] == 'x' || n[1] == 'X') ? (uint32_t)strtoul(n.c_str() + 2, nullptr, 16) : (uint32_t)strtoul(n.c_str() + 1, nullptr, 10);
    } else {
      for (auto& [k, v] : named) if (n == k) c = v;
    }
    if (!c) { o += '&'; continue; }
    put_utf8(o, c == 0xa0 ? ' ' : c);
    i = e;
  }
  return o;
}

// Collapse spaces within lines, trim lines, at most one blank line in a row
inline std::string tidy(const std::string& s) {
  std::string o;
  int nl = 2;
  bool sp = false;
  for (char c : s) {
    if (c == '\r') continue;
    if (c == '\n') {
      while (!o.empty() && o.back() == ' ') o.pop_back();
      if (nl < 2) o += '\n';
      ++nl, sp = false;
    } else if (c == ' ' || c == '\t' || c == '\f' || c == '\v') {
      sp = true;
    } else {
      if (sp && nl == 0) o += ' ';
      o += c, nl = 0, sp = false;
    }
  }
  while (!o.empty() && (o.back() == '\n' || o.back() == ' ')) o.pop_back();
  return o;
}

// The '>' that ends the tag starting at i (quoted attribute values may hold '>'), npos when none
inline size_t tag_end(const std::string& s, size_t i) {
  char q = 0;
  for (size_t j = i + 1; j < s.size(); ++j) {
    const char c = s[j];
    if (q) {
      if (c == q) q = 0;
    } else if ((c == '"' || c == '\'') && s[j - 1] == '=') {
      q = c;
    } else if (c == '>') {
      return j;
    }
  }
  return std::string::npos;
}

inline std::string strip_tags(const std::string& s) {
  std::string o;
  bool in = false;
  for (char c : s) {
    if (c == '<') in = true;
    else if (c == '>') in = false;
    else if (!in) o += c;
  }
  return tidy(decode_entities(o));
}

// The value of attribute `name` in the tag lo[ts..te] (lo: the page lowercased; the value from html, its case kept,
// entities still in it); "" when the tag has none
inline std::string attr_of(const std::string& html, const std::string& lo, size_t ts, size_t te, const char* name) {
  const std::string key = std::string(name) + "=";
  for (size_t k = lo.find(key, ts); k != std::string::npos && k < te; k = lo.find(key, k + 1)) {
    if (!isspace((unsigned char)lo[k - 1])) continue;  // ("data-href=" is not "href=")
    size_t v = k + key.size();
    if (v >= te) return {};
    if (lo[v] == '"' || lo[v] == '\'') {
      const size_t q = lo.find(lo[v], v + 1);
      return q == std::string::npos || q > te ? std::string() : html.substr(v + 1, q - v - 1);
    }
    size_t e = v;
    while (e < te && !isspace((unsigned char)lo[e]) && lo[e] != '>') ++e;
    return html.substr(v, e - v);
  }
  return {};
}

// A link's target as an absolute http(s) URL against the page's (base), "" for one that leads nowhere to read: a
// fragment on the same page, javascript:, mailto:, tel:, data:
inline std::string link_target(const std::string& base, std::string h) {
  while (!h.empty() && isspace((unsigned char)h.front())) h.erase(0, 1);
  while (!h.empty() && isspace((unsigned char)h.back())) h.pop_back();
  if (h.empty() || h[0] == '#') return {};
  const std::string l = lower(h.substr(0, 11));
  if (l.rfind("javascript:", 0) == 0 || l.rfind("mailto:", 0) == 0 || l.rfind("tel:", 0) == 0 || l.rfind("data:", 0) == 0) return {};
  if (!base.empty()) h = resolve_url(base, h);
  return h.rfind("http://", 0) == 0 || h.rfind("https://", 0) == 0 ? h : std::string();
}

// A link to a file rather than a page: a document, text, image or archive by its extension, or a download / attachment
// path (GitHub's user-attachments, /releases/download/, raw files)
inline bool file_link(const std::string& url) {
  const std::string u = lower(url.substr(0, url.find_first_of("?#")));
  for (const char* p : {"/user-attachments/", "/attachments/", "/files/", "/download/", "/releases/download/", "/raw/",
                        "raw.githubusercontent.com/", "/uploads/"})
    if (u.find(p) != std::string::npos) return true;
  const size_t sl = u.rfind('/'), d = u.rfind('.');
  if (d == std::string::npos || (sl != std::string::npos && d < sl)) return false;
  static const char* exts[] = {"pdf", "doc", "docx", "xls", "xlsx", "xlsm", "ppt", "pptx", "odt", "ods", "rtf", "csv",
                               "tsv", "txt", "md", "log", "json", "jsonl", "xml", "yaml", "yml", "png", "jpg", "jpeg",
                               "gif", "webp", "bmp", "svg", "zip", "7z", "gz", "tgz", "rar", "mp3", "wav", "m4a", "aac",
                               "flac", "ogg", "opus", "wma", "webm", "mp4"};  // (the audio as web_fetch's, G4)
  const std::string e = u.substr(d + 1);
  for (const char* x : exts) if (e == x) return true;
  return false;
}

// A link worth its URL on any page: to a GitHub issue, pull request, discussion, commit or release (threads link to
// one another: the way to follow a problem across them), or one whose text is itself a URL or a reference ("#123",
// "owner/repo#45")
inline bool ref_link(const std::string& url, const std::string& text) {
  const std::string u = lower(url);
  for (const char* h : {"https://github.com/", "https://www.github.com/"})
    if (u.rfind(h, 0) == 0) {
      for (const char* p : {"/issues/", "/pull/", "/discussions/", "/commit/", "/releases/", "/compare/"})
        if (u.find(p) != std::string::npos) return true;
    }
  std::string t = text;
  while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
  while (!t.empty() && isspace((unsigned char)t.front())) t.erase(0, 1);
  if (t.rfind("http://", 0) == 0 || t.rfind("https://", 0) == 0) return true;
  const size_t h = t.find('#');
  if (h == std::string::npos || h + 1 >= t.size() || t.size() > 60) return false;
  for (size_t k = h + 1; k < t.size(); ++k) if (!isdigit((unsigned char)t[k])) return false;
  return true;
}

// HTML to readable text: scripts, styles, navigation, forms and the like dropped; headings as Markdown '#', list
// items as '- ', block elements on their own lines. Links resolved against base (the page's URL): links to files (a
// document a page offers, a file attached to an issue) and references (ref_link: another issue, a URL written out)
// as Markdown [text](url) and images with a description as [image: alt](url), so the reader can open them; the other
// links as their text only, counted in *hidden (their URLs made pages 30-75 % longer: Wikipedia, BBC,
// docs.python.org, GitHub, 2026-11-03), or all of them with all_links (to explore a site).
// The page's <main> (or first <article>) when it has enough text.
inline std::string html_to_text(const std::string& html, std::string* title = nullptr, const std::string& base = "",
                                bool all_links = false, int* hidden = nullptr) {
  if (hidden) *hidden = 0;
  const std::string lo = lower(html);
  if (title) {
    const size_t a = lo.find("<title");
    const size_t b = a == std::string::npos ? a : lo.find('>', a);
    const size_t c = b == std::string::npos ? b : lo.find("</title", b);
    *title = c == std::string::npos ? std::string() : strip_tags(html.substr(b + 1, c - b - 1));
  }
  static const char* skip[] = {"script", "style", "noscript", "svg", "template", "head", "nav", "footer", "aside",
                               "form", "select", "button", "iframe", "object", "canvas"};
  static const char* block[] = {"p", "div", "br", "tr", "table", "section", "article", "main", "header", "ul", "ol",
                                "dl", "dt", "dd", "pre", "blockquote", "figure", "figcaption", "hr", "li", "h1", "h2",
                                "h3", "h4", "h5", "h6"};
  auto convert = [&](size_t from, size_t to) {
    if (hidden) *hidden = 0;  // (the <main> part tried first, then maybe the whole page: counted once)
    std::string o;
    std::string link;     // (the open <a>'s target; "" when none is open)
    size_t link_at = 0;   // (where its text starts in o, after the '[')
    auto close_link = [&] {
      if (link.empty()) return;
      std::string t = o.substr(link_at);
      while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
      size_t a = 0;
      while (a < t.size() && isspace((unsigned char)t[a])) ++a;
      t = t.substr(a);
      o.erase(link_at - 1);  // (the '[' and the text, written again below)
      if (!all_links && !file_link(link) && !ref_link(link, t)) {  // (a link to another page: its text only)
        o += t;
        if (hidden && !t.empty()) ++*hidden;
      } else if (t.empty()) o += link;                                  // (an image-only or empty link: its URL alone)
      else if (t == link || t.find("](") != std::string::npos) o += t;  // (its text is the URL, or holds an image link)
      else o += "[" + t + "](" + link + ")";
      link.clear();
    };
    size_t i = from;
    while (i < to) {
      if (lo[i] != '<') {
        const size_t n = std::min(lo.find('<', i), to);
        o.append(html, i, n - i);
        i = n;
        continue;
      }
      if (lo.compare(i, 4, "<!--") == 0) {
        const size_t e = lo.find("-->", i + 4);
        i = e == std::string::npos ? to : e + 3;
        continue;
      }
      const size_t e = tag_end(lo, i);
      if (e == std::string::npos) break;
      const bool close = lo[i + 1] == '/';
      size_t ns = i + (close ? 2 : 1), ne = ns;
      while (ne < e && (isalnum((unsigned char)lo[ne]))) ++ne;
      const std::string name = lo.substr(ns, ne - ns);
      const size_t ts = i;
      i = e + 1;
      if (name == "a") {
        close_link();  // (an <a> left open, or this one's end)
        if (!close) {
          link = link_target(base, attr_of(html, lo, ts, e, "href"));  // (kept or not at its end, by its text too)
          if (!link.empty()) {
            o += '[';
            link_at = o.size();
          }
        }
        continue;
      }
      if (name == "img" && !close) {  // (an image with a description: a figure, a screenshot in an issue)
        std::string alt = attr_of(html, lo, ts, e, "alt");
        for (char& c : alt) if (c == '\n' || c == '\r' || c == '[' || c == ']') c = ' ';
        const std::string src = link_target(base, attr_of(html, lo, ts, e, "src"));
        if (!src.empty() && alt.find_first_not_of(' ') != std::string::npos && (all_links || file_link(src)))
          o += "[image: " + alt + "](" + src + ")";
        continue;
      }
      if (!close && lo[e - 1] != '/') {
        bool sk = false;
        for (const char* t : skip) sk = sk || name == t;
        if (sk) {  // to the matching close tag (nesting of the same tag counted)
          int depth = 1;
          size_t j = i;
          while (depth > 0) {
            const size_t a = lo.find("<" + name, j), b = lo.find("</" + name, j);
            if (b == std::string::npos) { j = to; break; }
            if (a != std::string::npos && a < b) ++depth, j = a + 1;
            else --depth, j = b + 1;
          }
          const size_t c = lo.find('>', j);
          i = c == std::string::npos ? to : std::min(c + 1, to);
          o += '\n';
          continue;
        }
      }
      bool bl = false;
      for (const char* t : block) bl = bl || name == t;
      if (name.size() == 2 && name[0] == 'h' && name[1] >= '1' && name[1] <= '6') {
        o += close ? "\n" : "\n\n" + std::string(name[1] - '0', '#') + " ";
      } else if (name == "li") {
        if (!close) o += "\n- ";
      } else if (name == "p" || name == "pre" || name == "blockquote" || name == "table") {
        o += "\n\n";
      } else if (bl) {
        o += '\n';
      } else if (name == "td" || name == "th") {
        o += close ? "" : " ";
      }
    }
    close_link();
    return tidy(decode_entities(o));
  };
  for (const char* tag : {"<main", "<article"}) {
    const size_t a = lo.find(tag);
    if (a == std::string::npos) continue;
    const std::string closing = std::string("</") + (tag + 1);
    const size_t b = lo.rfind(closing);
    if (b == std::string::npos || b < a) continue;
    std::string t = convert(a, b);
    if (t.size() >= 400) return t;
  }
  return convert(0, lo.size());
}

inline std::string b64url_decode(const std::string& s) {
  std::string o;
  uint32_t acc = 0;
  int bits = 0;
  for (char c : s) {
    int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52
          : (c == '-' || c == '+') ? 62 : (c == '_' || c == '/') ? 63 : -1;
    if (v < 0) break;
    acc = acc << 6 | (uint32_t)v, bits += 6;
    if (bits >= 8) bits -= 8, o += (char)(acc >> bits & 255);
  }
  return o;
}

// The text between the first '>' after `from` and `end_tag`, tags stripped
inline std::string inner_text(const std::string& s, size_t from, const char* end_tag) {
  const size_t a = s.find('>', from);
  if (a == std::string::npos) return {};
  const size_t b = s.find(end_tag, a);
  return b == std::string::npos ? std::string() : strip_tags(s.substr(a + 1, b - a - 1));
}

inline std::vector<Result> parse_brave(const std::string& html, size_t n) {
  std::vector<Result> rs;
  const std::string mark = "data-type=\"web\"";
  size_t p = html.find(mark);
  while (p != std::string::npos && rs.size() < n) {
    const size_t q = html.find(mark, p + mark.size());
    const std::string seg = html.substr(p, (q == std::string::npos ? html.size() : q) - p);
    p = q;
    Result r;
    const size_t a = seg.find("<a href=\"");
    if (a == std::string::npos) continue;
    r.url = decode_entities(seg.substr(a + 9, seg.find('"', a + 9) - a - 9));
    const size_t t = seg.find("class=\"title");
    if (t != std::string::npos) r.title = inner_text(seg, t, "</div>");
    size_t s = seg.find("class=\"content");
    if (s == std::string::npos) s = seg.find("snippet-description");
    if (s != std::string::npos) r.snippet = inner_text(seg, s, "</div>");
    if (r.url.rfind("http", 0) == 0 && !r.title.empty()) rs.push_back(r);
  }
  return rs;
}

inline std::vector<Result> parse_bing(const std::string& html, size_t n) {
  std::vector<Result> rs;
  const std::string mark = "class=\"b_algo\"";
  size_t p = html.find(mark);
  while (p != std::string::npos && rs.size() < n) {
    const size_t q = html.find(mark, p + mark.size());
    const std::string seg = html.substr(p, (q == std::string::npos ? html.size() : q) - p);
    p = q;
    Result r;
    const size_t h = seg.find("<h2");
    const size_t a = h == std::string::npos ? h : seg.find("href=\"", h);
    if (a == std::string::npos) continue;
    std::string href = decode_entities(seg.substr(a + 6, seg.find('"', a + 6) - a - 6));
    const size_t u = href.find("&u=a1");  // bing.com/ck/a?...&u=a1<base64url of the target>&...
    if (href.find("bing.com/ck/a") != std::string::npos && u != std::string::npos)
      href = b64url_decode(href.substr(u + 5, href.find('&', u + 5) - u - 5));
    r.url = href;
    r.title = inner_text(seg, a, "</a>");
    const size_t s = seg.find("<p", a);
    if (s != std::string::npos) r.snippet = inner_text(seg, s, "</p>");
    if (r.url.rfind("http", 0) == 0 && !r.title.empty()) rs.push_back(r);
  }
  return rs;
}

// Search the web: Brave Search, then Bing when Brave answers with nothing usable (rate limit, captcha). engine:
// "auto", "brave" or "bing". *error: why an engine was skipped (with no results: why every one failed).
inline std::vector<Result> search(const std::string& query, size_t n, const std::string& engine, std::string* used, std::string* error) {
  std::vector<Result> rs;
  std::string err;
  if (engine != "bing") {
    const std::string u = "https://search.brave.com/search?q=" + url_encode(query) + "&source=web";
    Page pg = get(u, 4u << 20, 12000);
    if (pg.status == 429) Sleep(1500), pg = get(u, 4u << 20, 12000);  // (a short burst limit)
    if (pg.error.empty() && pg.status == 200) rs = parse_brave(pg.body, n);
    else err = "brave: " + (pg.error.empty() ? "HTTP " + std::to_string(pg.status) : pg.error);
    if (!rs.empty()) { *used = "brave"; return rs; }
    if (err.empty()) err = "brave: no results" + std::string(pg.body.find("captcha") != std::string::npos ? " (captcha)" : "");
  }
  if (engine != "brave") {
    const Page pg = get("https://www.bing.com/search?q=" + url_encode(query), 4u << 20, 12000);
    if (pg.error.empty() && pg.status == 200) rs = parse_bing(pg.body, n);
    if (!rs.empty()) { *used = "bing"; *error = err; return rs; }
    err += std::string(err.empty() ? "" : "; ") + "bing: " + (!pg.error.empty() ? pg.error : pg.status != 200 ? "HTTP " + std::to_string(pg.status) : "no results");
  }
  *error = err;
  return rs;
}

}  // namespace webt
