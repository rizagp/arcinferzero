// The platform layer's sockets (docs/linux-port.md, step 1): what the HTTP server (src/http.h), the web_view proxy
// (tools/web_view.h) and tests/test_http.cpp need from the operating system's network calls. On Windows every call is
// the Winsock call those files made before, with the same
// arguments (the Windows build behaves the same); on Linux the BSD socket calls, with the few differences they need
// (MSG_NOSIGNAL so a client gone does not kill the server with SIGPIPE, EINTR retried, sockets not inherited by
// child processes). Included before platform.h on Windows: winsock2.h must come before windows.h.
#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace plat {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kNoSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kNoSocket = -1;
#endif

// The socket library started (WSAStartup 2.2; nothing on Linux): false when it cannot be.
inline bool net_init() {
#ifdef _WIN32
  WSADATA w;
  return WSAStartup(MAKEWORD(2, 2), &w) == 0;
#else
  return true;
#endif
}

// A socket of this family / type / protocol (getaddrinfo's), kNoSocket on failure.
inline socket_t sock_open(int family, int type, int protocol) {
#ifdef _WIN32
  return socket(family, type, protocol);
#else
  return socket(family, type | SOCK_CLOEXEC, protocol);
#endif
}
inline void sock_close(socket_t s) {
#ifdef _WIN32
  closesocket(s);
#else
  ::close(s);
#endif
}
// both directions shut (the client sees the end of the stream), before sock_close
inline void sock_shutdown(socket_t s) {
#ifdef _WIN32
  shutdown(s, SD_BOTH);
#else
  shutdown(s, SHUT_RDWR);
#endif
}

// the sending direction shut (the peer sees the end of what was sent; replies can still be read)
inline void sock_shutdown_send(socket_t s) {
#ifdef _WIN32
  shutdown(s, SD_SEND);
#else
  shutdown(s, SHUT_WR);
#endif
}
// a boolean socket option on (SO_REUSEADDR, IPV6_V6ONLY, TCP_NODELAY): a BOOL on Windows, an int on Linux
inline void sock_flag(socket_t s, int level, int opt) {
#ifdef _WIN32
  BOOL yes = TRUE;
  setsockopt(s, level, opt, (const char*)&yes, sizeof yes);
#else
  int yes = 1;
  setsockopt(s, level, opt, &yes, sizeof yes);
#endif
}
// a receive / send time limit in ms (SO_RCVTIMEO / SO_SNDTIMEO): a DWORD of ms on Windows, a timeval on Linux. A call
// past it fails (recv / send <= 0), as on Windows.
inline void sock_timeout(socket_t s, int opt, unsigned ms) {
#ifdef _WIN32
  DWORD t = ms;
  setsockopt(s, SOL_SOCKET, opt, (const char*)&t, sizeof t);
#else
  timeval t{};
  t.tv_sec = (time_t)(ms / 1000);
  t.tv_usec = (suseconds_t)(ms % 1000) * 1000;
  setsockopt(s, SOL_SOCKET, opt, &t, sizeof t);
#endif
}
inline void sock_recv_timeout(socket_t s, unsigned ms) { sock_timeout(s, SO_RCVTIMEO, ms); }
inline void sock_send_timeout(socket_t s, unsigned ms) { sock_timeout(s, SO_SNDTIMEO, ms); }

inline bool sock_bind_listen(socket_t s, const sockaddr* addr, size_t len, int backlog) {
#ifdef _WIN32
  return bind(s, addr, (int)len) == 0 && ::listen(s, backlog) == 0;
#else
  return bind(s, addr, (socklen_t)len) == 0 && ::listen(s, backlog) == 0;
#endif
}
// the next connection, kNoSocket on failure (the caller waits and tries again)
inline socket_t sock_accept(socket_t ls) {
#ifdef _WIN32
  return accept(ls, nullptr, nullptr);
#else
  return accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
#endif
}
// a connection to addr (blocking; the send time limit, set before, bounds it on Linux, as SO_SNDTIMEO does there)
inline bool sock_connect(socket_t s, const sockaddr* addr, size_t len) {
#ifdef _WIN32
  return connect(s, addr, (int)len) == 0;
#else
  return connect(s, addr, (socklen_t)len) == 0;
#endif
}
// the socket's own address (getsockname: the port a bind to port 0 got): false when it is not known
inline bool sock_name(socket_t s, sockaddr_storage& sa) {
#ifdef _WIN32
  int sl = sizeof sa;
#else
  socklen_t sl = sizeof sa;
#endif
  return getsockname(s, (sockaddr*)&sa, &sl) == 0;
}
// Waits up to ms for a or b to be readable (the web_view proxy's two ends): > 0 with ra / rb set, 0 when the time
// passed, < 0 on an error. Windows: select, as the proxy called it; Linux: poll (select's FD_SETSIZE limit).
inline int sock_wait_read2(socket_t a, socket_t b, int ms, bool& ra, bool& rb) {
  ra = rb = false;
#ifdef _WIN32
  fd_set r;
  FD_ZERO(&r);
  FD_SET(a, &r);
  FD_SET(b, &r);
  timeval tv{ms / 1000, (ms % 1000) * 1000};
  const int n = select(0, &r, nullptr, nullptr, &tv);
  if (n > 0) ra = FD_ISSET(a, &r) != 0, rb = FD_ISSET(b, &r) != 0;
  return n;
#else
  pollfd p[2] = {{a, POLLIN, 0}, {b, POLLIN, 0}};
  int n;
  do n = poll(p, 2, ms); while (n < 0 && errno == EINTR);
  // (a hang-up or an error counts as readable: the recv that follows sees it, as select reports it on Windows)
  if (n > 0) ra = (p[0].revents & (POLLIN | POLLHUP | POLLERR)) != 0, rb = (p[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0;
  return n;
#endif
}
// the peer's address: false when it is not known
inline bool sock_peer(socket_t s, sockaddr_storage& sa) {
#ifdef _WIN32
  int sl = sizeof sa;
#else
  socklen_t sl = sizeof sa;
#endif
  return getpeername(s, (sockaddr*)&sa, &sl) == 0;
}

// up to n bytes: the count sent, <= 0 when the connection broke or the send time limit passed
inline int sock_send(socket_t s, const char* d, int n) {
#ifdef _WIN32
  return ::send(s, d, n, 0);
#else
  while (true) {
    const ssize_t r = ::send(s, d, (size_t)n, MSG_NOSIGNAL);
    if (r < 0 && errno == EINTR) continue;
    return (int)r;
  }
#endif
}
// up to n bytes: the count read, 0 at the end of the stream, < 0 on an error or the receive time limit
inline int sock_recv(socket_t s, char* d, int n) {
#ifdef _WIN32
  return recv(s, d, n, 0);
#else
  while (true) {
    const ssize_t r = ::recv(s, d, (size_t)n, 0);
    if (r < 0 && errno == EINTR) continue;
    return (int)r;
  }
#endif
}
// The peer closed the connection (or it broke), asked without blocking or reading anything: the socket polled for
// reading with no wait, and when readable, one byte peeked (0 or an error: the end of the stream). False when nothing
// is known yet (no data, or data waiting that is not the end).
inline bool sock_peer_gone(socket_t s) {
#ifdef _WIN32
  WSAPOLLFD p{s, POLLRDNORM, 0};
  if (WSAPoll(&p, 1, 0) <= 0) return false;
  char c;
  return (p.revents & (POLLERR | POLLHUP | POLLNVAL)) || ((p.revents & POLLRDNORM) && recv(s, &c, 1, MSG_PEEK) <= 0);
#else
  pollfd p{s, POLLIN, 0};
  if (poll(&p, 1, 0) <= 0) return false;
  char c;
  if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) return true;
  if (!(p.revents & POLLIN)) return false;
  const ssize_t r = ::recv(s, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  return r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
#endif
}

}  // namespace plat
