// A crash written to stderr before the process ends: the exception (for an access violation, the address read or
// written) and the faulting thread's stack, one frame a line as module!function+offset, with file:line where a .pdb
// sits next to the exe. Without it a crash inside a driver left nothing in the server's log (2026-11-10: serverz ended
// in ze_intel_gpu64.dll, 0xC0000005, at the start of a meeting's transcription, three times, without a line).
// Driver frames name the nearest export. Windows Error Reporting still records the crash afterwards.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstring>
#pragma comment(lib, "dbghelp.lib")

namespace crashlog {
inline void (*extra)() = nullptr;  // (the program's own notes, printed after the exception line)
inline LONG WINAPI filter(EXCEPTION_POINTERS* ep) {
  static volatile LONG once = 0;
  if (InterlockedExchange(&once, 1)) return EXCEPTION_CONTINUE_SEARCH;  // (a crash in here, or a second thread)
  const HANDLE proc = GetCurrentProcess(), th = GetCurrentThread();
  const EXCEPTION_RECORD* er = ep->ExceptionRecord;
  fprintf(stderr, "inferzero: crashed: exception 0x%08lx at %p", (unsigned long)er->ExceptionCode, er->ExceptionAddress);
  if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
    const ULONG_PTR how = er->ExceptionInformation[0];
    fprintf(stderr, " (%s %p)", how == 0 ? "reading" : how == 1 ? "writing" : "executing", (void*)er->ExceptionInformation[1]);
  }
  fprintf(stderr, ", thread %lu\n", (unsigned long)GetCurrentThreadId());
  fflush(stderr);
  if (extra) extra();
  fprintf(stderr, "  its stack:\n");
  fflush(stderr);
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
  SymInitialize(proc, nullptr, TRUE);
  CONTEXT ctx = *ep->ContextRecord;
  STACKFRAME64 f{};
  f.AddrPC.Offset = ctx.Rip;
  f.AddrPC.Mode = AddrModeFlat;
  f.AddrFrame.Offset = ctx.Rbp;
  f.AddrFrame.Mode = AddrModeFlat;
  f.AddrStack.Offset = ctx.Rsp;
  f.AddrStack.Mode = AddrModeFlat;
  for (int i = 0; i < 40; ++i) {
    if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, th, &f, &ctx, nullptr, SymFunctionTableAccess64, SymGetModuleBase64,
                     nullptr))
      break;
    const DWORD64 pc = f.AddrPC.Offset;
    if (!pc) break;
    char path[MAX_PATH] = "?";
    const DWORD64 base = SymGetModuleBase64(proc, pc);
    if (base) GetModuleFileNameA((HMODULE)base, path, MAX_PATH);
    const char* mod = strrchr(path, '\\');
    mod = mod ? mod + 1 : path;
    alignas(SYMBOL_INFO) char sbuf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* s = (SYMBOL_INFO*)sbuf;
    s->SizeOfStruct = sizeof(SYMBOL_INFO);
    s->MaxNameLen = 255;
    DWORD64 disp = 0;
    IMAGEHLP_LINE64 ln{};
    ln.SizeOfStruct = sizeof ln;
    DWORD ldisp = 0;
    if (SymFromAddr(proc, pc, &disp, s)) {
      if (SymGetLineFromAddr64(proc, pc, &ldisp, &ln))
        fprintf(stderr, "  #%d %s!%s+0x%llx (%s:%lu)\n", i, mod, s->Name, (unsigned long long)disp, ln.FileName,
                (unsigned long)ln.LineNumber);
      else
        fprintf(stderr, "  #%d %s!%s+0x%llx\n", i, mod, s->Name, (unsigned long long)disp);
    } else {
      fprintf(stderr, "  #%d %s+0x%llx\n", i, mod, (unsigned long long)(base ? pc - base : pc));
    }
  }
  fflush(stderr);
  return EXCEPTION_CONTINUE_SEARCH;
}
inline void install() { SetUnhandledExceptionFilter(filter); }
}  // namespace crashlog
#else
// Linux: the crash signals (SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT) print the signal, the address for a memory
// fault and the stack (backtrace, function names where the binary exports them: link with -rdynamic), then the
// program's own notes, then the signal ends the process as it would have (a core dump where the system keeps them).
// The signal line and the stack are written with write(2) only (no stdio lock, no allocation: the fault may have come
// while another thread held them), on a stack of its own (a stack overflow is reported too). The notes (extra) are not
// async-signal-safe: they come last, so a handler that hangs or faults there has printed the stack already.
#include <csignal>
#include <cstdint>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>
namespace crashlog {
inline void (*extra)() = nullptr;  // (the program's own notes, printed after the stack)
inline void put(const char* s) {
  for (size_t n = strlen(s); n;) {
    const ssize_t w = write(STDERR_FILENO, s, n);
    if (w <= 0) return;
    s += w, n -= (size_t)w;
  }
}
inline void put_num(unsigned long long v, int base) {
  char b[32];
  int i = 31;
  b[i] = 0;
  do b[--i] = "0123456789abcdef"[v % base], v /= base; while (v && i > 0);
  put(b + i);
}
inline void on_signal(int sig, siginfo_t* si, void*) {
  static volatile sig_atomic_t once = 0;
  if (!once) {
    once = 1;
    const char* name = sig == SIGSEGV ? "SIGSEGV" : sig == SIGBUS ? "SIGBUS" : sig == SIGILL ? "SIGILL" : sig == SIGFPE ? "SIGFPE"
                       : sig == SIGABRT ? "SIGABRT" : "a signal";
    put("\ncrashed: signal ");
    put_num((unsigned)sig, 10);
    put(" (");
    put(name);
    put(")");
    if ((sig == SIGSEGV || sig == SIGBUS) && si) {
      put(", address 0x");
      put_num((unsigned long long)(uintptr_t)si->si_addr, 16);
    }
    put("\nstack:\n");
    void* frames[64];
    const int n = backtrace(frames, 64);  // (libgcc's unwinder loaded at install: no allocation here)
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    if (extra) extra();  // (best effort, see above)
  }
  signal(sig, SIG_DFL);  // (then raised again: the default action, which ends the process)
  raise(sig);
}
inline void install() {
  void* warm[1];
  backtrace(warm, 1);  // (loads libgcc's unwinder now, not inside the handler)
  static char alt[64 * 1024];  // (the handler's own stack: a stack overflow's SIGSEGV still reaches it)
  stack_t ss{};
  ss.ss_sp = alt;
  ss.ss_size = sizeof alt;
  sigaltstack(&ss, nullptr);  // (the main thread's; other threads overflowing still die, without the report)
  struct sigaction sa{};
  sa.sa_sigaction = on_signal;
  sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_ONSTACK;
  sigemptyset(&sa.sa_mask);
  for (int s : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(s, &sa, nullptr);
}
}  // namespace crashlog
#endif
