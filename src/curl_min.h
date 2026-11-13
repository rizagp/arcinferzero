// Minimal libcurl declarations + a dynamic loader for libcurl.so.4 (Linux only): the HTTPS client of the web tools
// (tools/web_tools.h) where Windows has WinHTTP. Only what they need, so the build needs no curl headers; the
// numbers are libcurl's ABI (stable since 7.x; checked against curl 8.5's headers, docs/linux-port.md). libidn2's
// one call for an internationalized host's ASCII form is here too (libidn2.so.0, which libcurl uses itself).
// Both are part of a standard Ubuntu install; without them the web tools answer with an error, the rest runs.
#pragma once
#ifndef _WIN32
#include <cstddef>
#include <sys/socket.h>
#include "platform.h"  // (plat::dl_open / dl_sym)

namespace curlm {

typedef void CURL;
typedef void CURLU;
typedef int CURLcode;
typedef int CURLUcode;
struct curl_slist;
struct curl_sockaddr {  // (addr: addrlen bytes, an IPv6 address included)
  int family, socktype, protocol;
  unsigned int addrlen;
  struct sockaddr addr;
};
typedef int (*opensocket_fn)(void* clientp, int purpose, curl_sockaddr* address);  // (returns a socket or -1)
typedef size_t (*data_fn)(char* p, size_t size, size_t n, void* user);

enum : int {
  OPT_WRITEDATA = 10001, OPT_URL = 10002, OPT_PROXY = 10004, OPT_USERAGENT = 10018, OPT_HTTPHEADER = 10023,
  OPT_HEADERDATA = 10029, OPT_ACCEPT_ENCODING = 10102, OPT_OPENSOCKETDATA = 10164, OPT_NOPROXY = 10177,
  OPT_CURLU = 10282, OPT_PROTOCOLS_STR = 10318, OPT_REDIR_PROTOCOLS_STR = 10319,
  OPT_WRITEFUNCTION = 20011, OPT_HEADERFUNCTION = 20079, OPT_OPENSOCKETFUNCTION = 20163,
  OPT_FOLLOWLOCATION = 52, OPT_HTTP_VERSION = 84, OPT_NOSIGNAL = 99, OPT_TIMEOUT_MS = 155, OPT_CONNECTTIMEOUT_MS = 156,
  OPT_PROTOCOLS = 181,
  INFO_RESPONSE_CODE = 0x200000 + 2,
  HTTP_VERSION_2TLS = 4, PROTO_HTTP = 1, PROTO_HTTPS = 2, GLOBAL_DEFAULT = 3,
  E_OK = 0, E_COULDNT_CONNECT = 7, E_WRITE_ERROR = 23, E_OPERATION_TIMEDOUT = 28,
  UPART_URL = 0, UPART_SCHEME = 1, UPART_HOST = 5, UPART_PORT = 6, UPART_PATH = 7, UPART_QUERY = 8,
  U_DEFAULT_PORT = 1, U_NON_SUPPORT_SCHEME = 8,
  SOCKET_BAD = -1,
};

struct Api {
  bool ok = false;
  CURLcode (*global_init)(long) = nullptr;
  CURL* (*easy_init)() = nullptr;
  CURLcode (*easy_setopt)(CURL*, int, ...) = nullptr;
  CURLcode (*easy_perform)(CURL*) = nullptr;
  CURLcode (*easy_getinfo)(CURL*, int, ...) = nullptr;
  void (*easy_cleanup)(CURL*) = nullptr;
  const char* (*easy_strerror)(CURLcode) = nullptr;
  curl_slist* (*slist_append)(curl_slist*, const char*) = nullptr;
  void (*slist_free_all)(curl_slist*) = nullptr;
  CURLU* (*url)() = nullptr;
  CURLUcode (*url_set)(CURLU*, int, const char*, unsigned) = nullptr;
  CURLUcode (*url_get)(CURLU*, int, char**, unsigned) = nullptr;
  void (*url_cleanup)(CURLU*) = nullptr;
  void (*free)(void*) = nullptr;
  // libidn2: idn2_to_ascii_8z(input, &output, flags) (output freed with idn2_free); nullptr when not there
  int (*idn2_to_ascii_8z)(const char*, char**, int) = nullptr;
  void (*idn2_free)(void*) = nullptr;
};

// loaded once (thread-safe: a function-local static); ok false when libcurl is not installed
inline const Api& api() {
  static const Api a = [] {
    Api x;
    void* h = plat::dl_open(nullptr, "libcurl.so.4");
    if (!h) return x;
#define CURLM_SYM(field, name) x.field = (decltype(x.field))plat::dl_sym(h, name)
    CURLM_SYM(global_init, "curl_global_init");
    CURLM_SYM(easy_init, "curl_easy_init");
    CURLM_SYM(easy_setopt, "curl_easy_setopt");
    CURLM_SYM(easy_perform, "curl_easy_perform");
    CURLM_SYM(easy_getinfo, "curl_easy_getinfo");
    CURLM_SYM(easy_cleanup, "curl_easy_cleanup");
    CURLM_SYM(easy_strerror, "curl_easy_strerror");
    CURLM_SYM(slist_append, "curl_slist_append");
    CURLM_SYM(slist_free_all, "curl_slist_free_all");
    CURLM_SYM(url, "curl_url");
    CURLM_SYM(url_set, "curl_url_set");
    CURLM_SYM(url_get, "curl_url_get");
    CURLM_SYM(url_cleanup, "curl_url_cleanup");
    CURLM_SYM(free, "curl_free");
#undef CURLM_SYM
    // (curl_url is 7.62, 2018: every distribution with a current compute runtime has it)
    x.ok = x.global_init && x.easy_init && x.easy_setopt && x.easy_perform && x.easy_getinfo && x.easy_cleanup &&
           x.easy_strerror && x.slist_append && x.slist_free_all && x.url && x.url_set && x.url_get && x.url_cleanup &&
           x.free && x.global_init(GLOBAL_DEFAULT) == E_OK;
    if (void* i = plat::dl_open(nullptr, "libidn2.so.0")) {
      x.idn2_to_ascii_8z = (decltype(x.idn2_to_ascii_8z))plat::dl_sym(i, "idn2_to_ascii_8z");
      x.idn2_free = (decltype(x.idn2_free))plat::dl_sym(i, "idn2_free");
      if (!x.idn2_free) x.idn2_to_ascii_8z = nullptr;
    }
    return x;
  }();
  return a;
}

}  // namespace curlm
#endif
