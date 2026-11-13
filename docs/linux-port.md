# A Linux port: the plan (2026-11-12)

Asked for in issue #11 (a headless server on a Core Ultra 9 285H) and by the user. Nothing is built yet; this is the
scope, what is Windows-only today, and the order. The goal of the first step is `serverz` running headless on Ubuntu
with the same kernels and results; the launcher (a WinForms window) stays Windows-only.

## What carries over as it is

- **The GPU layer.** The engine talks to the GPU through Level Zero and compiles its kernels through OpenCL; Intel ships
  both for Linux (`libze_loader.so.1`, `libOpenCL.so.1`, the compute runtime). The 22 kernel files (OpenCL C with
  Intel's sub-group and DPAS extensions) are the same on both systems.
- **Most of the C++** (~58,000 lines in `src/`, `zero/`, `tools/`): the engine, the tokenizer, the chat templates,
  the JSON / HTTP parsing, the samplers, the document readers (DOCX / XLSX / PPTX / PDF text), Whisper and Qwen3-ASR,
  the metrics and the log, none of them depend on Windows.

## What is Windows-only today (measured with a search of the sources)

| Part | Where | On Linux |
|---|---|---|
| Loading the GPU runtimes | `zero/ze_min.h`, `src/cl_min.h`, `zero/clrt.h` (`LoadLibraryA`) | `dlopen` of `libze_loader.so.1` / `libOpenCL.so.1` |
| Model files and the expert file | `src/gguf.h`, `zero/expert_store.h`, `zero/engine_drafters.h`, `src/whisper.h` (`CreateFileA` / `W`, `CreateFileMapping` / `MapViewOfFile`, unbuffered overlapped reads) | `open` + `mmap`; the expert stream with `O_DIRECT` + `pread` (or io_uring later) |
| The GPU memory budget | `zero/clrt.h` `video_budget` (DXGI, the WDDM budget) | no WDDM: a discrete card's memory through Level Zero Sysman (`zesMemoryGetState`, which needs `zesInit`); an iGPU (no Sysman memory module) and WSL2 (no sysfs) from the free system RAM only; `--ctx auto` uses the same accounting |
| Paths and the data folder | 21 files use `USERPROFILE` / `GetModuleFileName` | `$HOME/.inferzero`, `/proc/self/exe` |
| Wide-character file names | 10 files (`std::wstring`, `_wfopen`) | UTF-8 paths directly |
| Sockets | `src/http.h` and 7 more (Winsock: the server, web tools, image / audio fetches, the recorder) | BSD sockets: small differences (`close`, `poll`, errors) |
| The terminal tool and MCP servers | `tools/server_terminal.h`, `tools/server_mcp.h` (`CreateProcess`, pipes, job objects, `CancelSynchronousIo`) | `posix_spawn` / `fork` + `exec`, pipes, a process group per child (`kill(-pgid)` for the whole tree), `poll` with time limits |
| The crash report | `src/crash_log.h` (structured exceptions, DbgHelp) | a signal handler with `backtrace` |
| Audio decoding (M4A, MP3, AAC) | `src/audio.h` (Media Foundation) | WAV / FLAC in C++ as now; the others through `ffmpeg` if installed, else refused with a message |
| PDF pages as images (OCR models) | `tools/pdf_render.h` (Windows.Data.Pdf) | `pdftoppm` (poppler) if installed, else OCR of scanned PDFs off; text PDFs read as now |
| `web_view` (a page as a browser shows it) | `tools/web_view.h` (headless Edge, a job-checked proxy) | headless Chromium if installed, with the same proxy; off by default at first |
| The NPU | `zero/npu/` | off (the NPU's Linux driver is another API) |
| Process details | `SetProcessWorkingSetSize`, power requests, the AVX2 check's message box | `madvise`, nothing, a message on stderr |
| The build | `build.ps1` (MSVC `cl`, `/FI` force-includes, `/openmp`) | CMake with GCC or Clang (`-include`, `-fopenmp`, `-mavx2 -mfma`) |
| The scripts and the launcher | the 14 scripts in `scripts/` (4,300 lines; 30 `.ps1` files in all with `bench/` and `tests/`), the WinForms launcher | a small shell script to start `serverz`, a systemd unit; PowerShell 7 runs on Linux for some of the rest (not `vcvars.ps1`, `make-package.ps1` or the launcher's); the launcher stays Windows-only |

## Progress

- 2026-11-13: step 1 started: `src/platform.h` with the libraries' loading (`plat::dl_open` / `dl_loaded` / `dl_sym`
  / `dl_close`, `dlopen` on Linux) and the home folder (`plat::home_env`); the four GPU-runtime loaders and the data
  folder use it. The Windows build bit-identical (runz's logits against release 2026.11.12.19).
- 2026-11-13: `plat::FileMap` (open, size, map; `mmap` on Linux) for GGUF files (split ones too) and the DFlash2
  drafter. Windows bit-identical (4B, a 2-part Flash Next, 27B with DFlash2, against 2026.11.13.01). On Linux a file
  rewritten while mapped faults instead of being locked against writers (no FILE_SHARE_READ), and a folder is refused
  at open as on Windows.
- 2026-11-13: `plat::AsyncFile` for the expert file's unbuffered reads (`zero/expert_store.h`): on Linux `O_DIRECT`
  + `pread` at the start of each read (no reads in flight yet: `io_uring` later, once it runs there). Windows
  bit-identical (the 35B-A3B, Flash Next streaming its experts, the 4B, against 2026.11.13.02). Next in step 1: the
  expert file's writing (`ExpertStore::build`), Whisper's wide paths (`src/whisper.h`), then sockets.
- 2026-11-13: the expert file's writing through `plat::DirectWriter` and small helpers (page-aligned memory, the
  working-set trim, replace / remove a file): `zero/expert_store.h` has no Windows calls left. Windows bit-identical (two
  test models' expert files byte-identical between the builds). Next: Whisper's wide paths, then sockets.
- 2026-11-13: Whisper's model.bin mapped through `plat::FileMap::open_fs` (a `std::filesystem::path`: CreateFileW on
  Windows, its bytes on Linux); transcripts identical. Left: `src/whisper.h` still passes the path through a narrow
  string (`.string()`), so a folder name outside the code page fails on Windows as before; Whisper's decompression
  (compressapi) and Media Foundation audio stay Windows-only for now. Next: sockets (`src/http.h`).
- 2026-11-13: the HTTP server's sockets through `src/platform_net.h` (`plat::socket_t`, `net_init`, `sock_open` /
  `close` / `shutdown`, `sock_flag`, `sock_recv_timeout` / `sock_send_timeout`, `sock_bind_listen`, `sock_accept`,
  `sock_peer`, `sock_send` / `sock_recv`, `sock_peer_gone`), and its media files (`send_file`) through
  `plat::SeqFile` in `src/platform.h`: `src/http.h` has no Windows calls left. On Windows each is the Winsock / Win32
  call made before, with the same arguments. On Linux: `send` with `MSG_NOSIGNAL` (a client gone must not end the
  server by SIGPIPE), `EINTR` retried, sockets made with `SOCK_CLOEXEC` / `accept4` (not inherited by the terminal
  tool's children), time limits as a `timeval`, `poll` + a non-blocking peek for a client gone. Checked on Windows
  with a real server (the 4B) against release 2026.11.13.09's serverz: the same replies, streaming, a stream cancelled
  by the client, two clients at once, a chunked body, file ranges; `test_http` passes. Not compiled on Linux yet (no
  WSL distribution or cross compiler on this laptop).
- 2026-11-13: the web_view proxy (`tools/web_view.h`) and `tests/test_http.cpp`'s client on `src/platform_net.h` too
  (new: `sock_connect`, `sock_name`, `sock_shutdown_send`, `sock_wait_read2`: `select` on Windows as before, `poll` on
  Linux). web_view gives the same replies as release 2026.11.13.09 (https and http pages through the proxy's CONNECT
  and plain paths, a refused port). web_view as a whole stays Windows-only for now: Edge, the job objects and the TCP
  table that tell the proxy which process connects (on Linux: headless Chromium, a process group, `/proc/net/tcp`).
  What is left of the network code is not sockets: `src/audio.h` / `src/image.h` include `winsock2.h` only for the
  header order (their work is Media Foundation / WIC); `tools/zero_recorder.cpp` is a WASAPI tool and stays
  Windows-only; `tools/web_tools.h` (web_search, web_fetch) is a WinHTTP client, which needs an HTTPS client on
  Linux (libcurl through `dlopen` is the likely choice).
- 2026-11-13: the first Linux compiles, in WSL (Ubuntu 24.04, g++ 13.3). `tests/test_http.cpp` builds there with
  `-Wall -Wextra` and no warnings and passes all its cases: the HTTP server (`src/http.h`) runs on Linux. The step-1
  files (`src/platform.h`, `src/platform_net.h`, `src/app_dir.h`, `src/cl_min.h`, `zero/ze_min.h`, `src/gguf.h`,
  `zero/expert_store.h`) pass a syntax check. Fixed for it: the log (`src/log.h`, force-included everywhere) had
  `<io.h>`, `localtime_s` and `_isatty` (now `localtime_r` / `isatty` on Linux; colors on a terminal whose `TERM` is
  not `dumb`), and an unused parameter in `plat::AsyncFile`'s Linux half. The check, from the repository's folder:
  `g++ -std=c++20 -Wall -Wextra -include src/log.h tests/test_http.cpp -o test_http -pthread`.
- 2026-11-13: the web tools (`tools/web_tools.h`: web_fetch, web_search, the files and audio they download) on Linux
  through libcurl, loaded at run time (`src/curl_min.h`: its few declarations, so the build needs no curl headers;
  `libcurl.so.4`, part of a standard Ubuntu install, and `libidn2.so.0` for internationalized names). The same steps
  as the WinHTTP path: redirects by hand with each hop's port and host checked, no proxy (the environment's
  `http_proxy` ignored), http and https only, the deadline over all hops, the body's limit, cut / too_big / timed_out.
  Where WinHTTP checks the address it connected to, libcurl's socket callback checks each address before it connects.
  URLs through libcurl's parser. Without libcurl the web tools answer with an error saying so; the rest runs. Checked
  with one test program built on both systems (18 cases: pages, redirects, a 404, limits, loopback / private / a name
  resolving to 127.0.0.1 / localhost refused, refused ports and schemes, a timeout, an internationalized host, URL
  parsing, a search): the same results on Linux and Windows; the Windows serverz's web_fetch / web_search identical to
  release 2026.11.13.09. Differences left: libidn2 follows IDNA2008 where Windows' IdnToAscii follows IDNA2003 (a few
  names with ß or ς map differently); a URL with a bare IPv6 address fails on both, with another message.
- 2026-11-13: **`runz` builds and runs on Linux** (step 2's first part), in WSL2 on the laptop with Ubuntu's own
  compute runtime (`intel-opencl-icd`, `libze-intel-gpu1`, `libze1`: 23.43; OpenCL and Level Zero both see the Arc
  iGPU through `/dev/dxg`). What it took: the CPU check (`src/cpu_check.h`: `__builtin_cpu_supports` on GCC), the
  power-throttling call (`src/full_speed.h`: nothing on Linux), DXGI's GPU budget (`zrt::video_budget`: false on Linux,
  Sysman later; the memory line shows the engine's allocations and the free RAM), the free RAM (`plat::ram_available`:
  `/proc/meminfo`'s MemAvailable), the working-set trim, 64-bit file seeks, an environment variable, the process and
  thread ids, the NPU runtime's library lookups and page memory (it uses Level Zero's graph extension, which Intel's
  Linux NPU driver has too: compiled, not run), path joins (`PLAT_SEP`), the "other GPU processes" note (`/proc/*/comm`).
  The build, from the repository's folder: `g++ -std=c++20 -O2 -mavx2 -mfma -mf16c -fopenmp -include src/cpu_check.h
  -include src/log.h -I third_party -pthread zero/runz.cpp -o runz` (no warnings with the flags above; CMake to come).
  The 4B: the same 32 greedy tokens as on Windows, 14.7 tok/s (the model read through `/mnt/c`; Windows 16.6-19.1),
  KL 0.00090 against its f32 reference (Windows 0.00100). The logits are not bit-identical to Windows' (another
  driver's kernel compiler: KL 3.5e-4 between the two), as step 2 expected. The Windows build bit-identical to release
  2026.11.13.09 (the 4B and the 35B-A3B). Not tried yet on Linux: MoE models streaming experts (`O_DIRECT` through
  WSL's disk), split models, the NPU, a newer compute runtime; Level Zero reports a 1.1 GB allocation limit there, which
  the 4B's buffers did not hit.
- 2026-11-13: **the compute runtime updated in WSL** to Intel's packages for Ubuntu 24.04 (the `kobuk-team/intel-graphics`
  PPA their GPU docs point to): compute runtime 26.40.40196.15, IGC 2.42, Level Zero loader 1.34 (Ubuntu's 23.43 has no
  Battlemage, B580 / B60 / B70, and likely no Arrow Lake H). The 4B: the same 32 tokens, KL 0.00090 (max 0.0071), 14.3-
  14.8 tok/s, and its logits **bit-identical to those of runtime 23.43** (and from run to run). So the kernel compiler
  is not what makes Linux's logits differ from Windows' (KL 3.5e-4 between them): more likely the CPU side, MSVC's
  `/fp:fast` and its math library against GCC's (an open question; within the limits either way).
- 2026-11-13: **`mgpu_probe` on Linux** (the peer-access probe of docs/multi-gpu.md: `zeDeviceCanAccessPeer` and, for
  every pair of GPUs, a 20 KB / 40 MB copy directly and through host memory, on the compute queue and the copy engine):
  built with the same line as runz (`zero/mgpu_probe.cpp`). Whether peer copies work between Arc cards on Linux can only be
  seen on a native Linux machine with two or more cards (issue #4's owners): WSL has the laptop's one iGPU, and WSL's
  GPU paravirtualization adds ~72 us to every round trip. The same probe on the same laptop: a 4-byte copy and its
  wait 77.0 us in WSL against 4.7 us on Windows; 20 KB GPU -> GPU 74.6 / 5.2 us; 40 MB 35.9 / 40.7 GB/s. A split model
  syncs ~128 times a token, so WSL says nothing about multi-GPU speed. That latency is also why the 4B generates at
  14.7 tok/s in WSL against 16.6-19.1 on Windows (bandwidth is close: pinned host -> GPU 34.9 / 38.0 GB/s).

## The order

1. **A platform layer** (`src/platform.h`): files, mapping, dynamic loading, paths, time, process start and kill,
   sockets, behind one interface, Windows first (the code as it is, moved), then Linux. The Windows build must stay
   bit-identical (`runz --dump-logits` before and after on four models).
2. **`runz` on Linux**: the engine with Level Zero / OpenCL through `dlopen`, model files through `mmap`, the memory
   budget through Sysman, CMake. Checked against the f32 reference (KL, as on Windows) and against the Windows build's
   logits on the same machine (a different driver may change the last bits: the KL limits decide, not bit identity).
3. **`serverz` headless on Linux**: sockets, the file tools, the terminal tool and MCP with process groups, the
   crash report; audio, PDF rendering and `web_view` through the external tools above, or off with a message.
4. **Packaging**: a release archive for Ubuntu 24.04 (x86-64, Intel's compute-runtime packages as the only
   dependency), a start script and a systemd unit; a GitHub Actions job that builds it on every push.

## How it can be tested

- **On this laptop:** WSL2 runs Ubuntu with the Windows GPU driver underneath (GPU paravirtualization; Intel's compute
  runtime has a WSL build), so steps 1-3 can be built and run here without a second machine. It needs ~10-15 GB of
  disk for the distribution, the compute runtime and a model copy (the laptop is short of space: models could be read
  from the Windows side through `/mnt/c`, but that mount is slow for `mmap` and may not honour `O_DIRECT`, which
  the expert stream uses; a model copied into the distribution's own disk is the reliable way).
- **On real Linux:** issue #11's 285H system, and any B-series owner on Ubuntu, for the speed (WSL2 adds a layer and
  says nothing about native speed) and for the Sysman memory figures.

## Open questions

- Native Linux may allow GPU-to-GPU peer copies for split models (Intel's multi-B60 stack on Linux mentions PCIe peer
  to peer), which Windows does not: a later step for `zero/mgpu/`, after `serverz` runs. `mgpu_probe` builds on Linux
  now, so a B60 owner on native Linux can answer it (not WSL: one GPU here, and ~77 us per round trip).
- The iGPU's memory share works differently (no WDDM "shared GPU memory" cap): `--ctx auto` and the expert cache need
  their budget from the driver and the free RAM; to measure on a 285H.
