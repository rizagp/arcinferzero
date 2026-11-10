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
namespace crashlog {
inline void install() {}
}  // namespace crashlog
#endif
