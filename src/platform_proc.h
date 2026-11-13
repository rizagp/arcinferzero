// The platform layer's external programs (docs/linux-port.md, step 3), Linux only: what Windows does through its own
// libraries (Media Foundation for audio, WIC for images, Windows.Data.Pdf for PDF pages) goes on Linux through tools a
// distribution has (ffmpeg, pdftoppm), run here: the bytes in on stdin, the output read from stdout, within a time limit
// and a size limit, the program in a process group of its own so a time-out ends everything it started.
#pragma once
#ifndef _WIN32
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <poll.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace plat {

// A write to a pipe whose reader is gone raises SIGPIPE, which ends the process by default: ignored once for the
// process (the write then fails with EPIPE, which the callers handle; the sockets send with MSG_NOSIGNAL as well).
inline void ignore_sigpipe() {
  static std::once_flag once;
  std::call_once(once, [] { signal(SIGPIPE, SIG_IGN); });
}

// The folder for temporary files: $TMPDIR when set (a system whose /tmp is a small tmpfs points it elsewhere), else /tmp.
inline std::string temp_dir() {
  const char* t = getenv("TMPDIR");
  return t && *t ? std::string(t) : std::string("/tmp");
}
// A new temporary file "<temp_dir>/inferzero-<what>-XXXXXX" holding data (all of it: EINTR retried): its path, or ""
// and err set
inline std::string write_temp(const char* what, const void* data, size_t n, std::string& err) {
  std::string p = temp_dir() + "/inferzero-" + what + "-XXXXXX";
  const int fd = mkstemp(p.data());
  if (fd < 0) {
    err = "no temporary file in " + temp_dir() + " (" + strerror(errno) + ")";
    return "";
  }
  size_t off = 0;
  while (off < n) {
    const ssize_t w = ::write(fd, (const char*)data + off, n - off);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) break;
    off += (size_t)w;
  }
  ::close(fd);
  if (off < n) {
    err = "the temporary file could not be written (" + temp_dir() + " full?)";
    unlink(p.c_str());
    return "";
  }
  return p;
}
// A file path as a program's argument: one that starts with '-' would be read as an option ("./-i", not "-i")
inline std::string path_arg(const std::string& p) { return !p.empty() && p[0] == '-' ? "./" + p : p; }
// a tool's error output as one line: its last non-empty line (poppler and ffmpeg print warnings before the error),
// without ffmpeg's "[mp3 @ 0x55d0...] " prefix
inline std::string last_line(const std::string& err) {
  size_t e = err.size();
  while (e > 0 && (err[e - 1] == '\n' || err[e - 1] == '\r' || err[e - 1] == ' ')) --e;
  const size_t b = err.rfind('\n', e ? e - 1 : 0);
  std::string l = err.substr(b == std::string::npos ? 0 : b + 1, e - (b == std::string::npos ? 0 : b + 1));
  if (l.size() > 2 && l[0] == '[') {
    const size_t c = l.find("] ");
    if (c != std::string::npos && l.find(" @ 0x") < c) l.erase(0, c + 2);
  }
  return l.substr(0, 300);
}

struct ProcResult {
  bool started = false;    // (false: the program is not installed, or could not be run: error says why)
  bool timed_out = false;  // (killed at the time limit, with everything it started)
  bool cut = false;        // (its output reached max_out: the rest not read, the program killed)
  int exit_code = -1;      // (its exit status; 128 + the signal when a signal ended it)
  std::string out, err, error;
};

// A program on PATH (argv[0]) with its arguments: `in` on its stdin, then stdin closed; its stdout and stderr read
// until it ends, at most max_out bytes of stdout (and 64 KB of stderr) within timeout_ms. With on_out, stdout goes
// there as it comes instead of into out (hours of decoded audio never sit in memory); on_out returning false stops
// the program (cut set). on_out must not throw.
inline ProcResult run_program(const std::vector<std::string>& argv, const std::string& in, int timeout_ms,
                              size_t max_out = (size_t)256 << 20,
                              const std::function<bool(const char*, size_t)>& on_out = {}) {
  ProcResult r;
  if (argv.empty()) {
    r.error = "no program";
    return r;
  }
  ignore_sigpipe();
  int pin[2] = {-1, -1}, pout[2] = {-1, -1}, perr[2] = {-1, -1};
  auto close_all = [&] {
    for (int* p : {pin, pout, perr})
      for (int k = 0; k < 2; ++k)
        if (p[k] >= 0) ::close(p[k]), p[k] = -1;
  };
  if (pipe2(pin, O_CLOEXEC) != 0 || pipe2(pout, O_CLOEXEC) != 0 || pipe2(perr, O_CLOEXEC) != 0) {
    r.error = std::string("pipe: ") + strerror(errno);
    close_all();
    return r;
  }
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, pin[0], 0);
  posix_spawn_file_actions_adddup2(&fa, pout[1], 1);
  posix_spawn_file_actions_adddup2(&fa, perr[1], 2);
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);
  posix_spawnattr_setpgroup(&at, 0);  // (a group of its own: killed as one)
  sigset_t def, none;
  sigemptyset(&def);
  sigaddset(&def, SIGPIPE);  // (the child gets SIGPIPE's default back: ignoring it is this process's choice)
  sigemptyset(&none);
  posix_spawnattr_setsigdefault(&at, &def);
  posix_spawnattr_setsigmask(&at, &none);
  std::vector<char*> av;
  for (const std::string& a : argv) av.push_back(const_cast<char*>(a.c_str()));
  av.push_back(nullptr);
  pid_t pid = -1;
  const int sp = posix_spawnp(&pid, av[0], &fa, &at, av.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  ::close(pin[0]), pin[0] = -1;
  ::close(pout[1]), pout[1] = -1;
  ::close(perr[1]), perr[1] = -1;
  if (sp != 0) {
    r.error = sp == ENOENT ? argv[0] + " is not installed (not found on PATH)" : argv[0] + ": " + strerror(sp);
    close_all();
    return r;
  }
  r.started = true;
  for (int fd : {pin[1], pout[0], perr[0]}) fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
  size_t sent = 0;
  if (in.empty()) ::close(pin[1]), pin[1] = -1;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  char buf[65536];
  bool kill_it = false;
  while (pout[0] >= 0 || perr[0] >= 0) {
    const int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
    if (left <= 0) {
      r.timed_out = kill_it = true;
      break;
    }
    pollfd p[3];
    int n = 0, i_in = -1, i_out = -1, i_err = -1;
    if (pin[1] >= 0) i_in = n, p[n++] = {pin[1], POLLOUT, 0};
    if (pout[0] >= 0) i_out = n, p[n++] = {pout[0], POLLIN, 0};
    if (perr[0] >= 0) i_err = n, p[n++] = {perr[0], POLLIN, 0};
    const int k = poll(p, (nfds_t)n, std::min(left, 1000));
    if (k < 0 && errno != EINTR) {  // (poll itself failed: the program is ended, not left running)
      kill_it = true;
      break;
    }
    if (k <= 0) continue;
    if (i_in >= 0 && p[i_in].revents) {
      const ssize_t w = ::write(pin[1], in.data() + sent, std::min<size_t>(in.size() - sent, 1 << 20));
      if (w > 0) sent += (size_t)w;
      if (w < 0 && errno != EAGAIN && errno != EINTR) sent = in.size();  // (EPIPE: it stopped reading; its output decides)
      if (sent >= in.size()) ::close(pin[1]), pin[1] = -1;
    }
    for (auto [idx, fd, dst, cap] : {std::tuple{i_out, &pout[0], &r.out, max_out}, std::tuple{i_err, &perr[0], &r.err, (size_t)65536}}) {
      if (idx < 0 || !p[idx].revents) continue;
      const ssize_t g = ::read(*fd, buf, sizeof buf);
      if (g > 0 && on_out && dst == &r.out) {
        if (!on_out(buf, (size_t)g)) {
          r.cut = kill_it = true;
          break;
        }
      } else if (g > 0) {
        const size_t take = std::min<size_t>((size_t)g, cap - std::min(cap, dst->size()));
        dst->append(buf, take);
        if (take < (size_t)g && dst == &r.out) {
          r.cut = kill_it = true;
          break;
        }
      } else if (g == 0 || (errno != EAGAIN && errno != EINTR)) {
        ::close(*fd), *fd = -1;
      }
    }
    if (kill_it) break;
  }
  close_all();
  // its end, within the same deadline: a program that closed its output and went on running is killed at it. Waited
  // for without reaping (WNOWAIT): its group is killed while it is a zombie, whose id cannot be taken by another
  // process meanwhile; then reaped
  if (!kill_it) {
    for (;;) {
      siginfo_t si{};
      if (waitid(P_PID, pid, &si, WEXITED | WNOHANG | WNOWAIT) == 0 && si.si_pid == pid) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        r.timed_out = kill_it = true;
        break;
      }
      usleep(20000);
    }
  }
  kill(-pid, SIGKILL);  // (the group: all of it at a limit, else what it left running)
  int st = 0;
  while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
  r.exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1;
  return r;
}

}  // namespace plat
#endif
