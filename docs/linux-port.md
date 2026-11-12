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
  to peer), which Windows does not: a later step for `zero/mgpu/`, after `serverz` runs.
- The iGPU's memory share works differently (no WDDM "shared GPU memory" cap): `--ctx auto` and the expert cache need
  their budget from the driver and the free RAM; to measure on a 285H.
