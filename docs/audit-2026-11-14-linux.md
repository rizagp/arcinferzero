# Static audit of all the Linux work, 2026-11-14

**Status (the same night):** every finding checked in the code, the real ones fixed, then the Linux and Windows builds
and tests run again (docs/linux-port.md, Progress, 2026-11-14):
- **F1 fixed:** the Hugging Face cache on Linux, where `hf download` writes it (`plat::hf_home_hub`: `$XDG_CACHE_HOME`,
  else `~/.cache/huggingface/hub`), in the model list (`tools/server_catalog.h`), Whisper (`src/whisper.h`) and
  Qwen3-ASR (`zero/asr_qwen3.h`); Windows' `%USERPROFILE%` lines as they were. Tested: with no `HF_*` variable set, the
  model list finds the 4B in `~/.cache/huggingface/hub`.
- **F2 fixed:** `plat::DirectWriter::write` (Linux) takes a short write as the rest to write; only an error or a write of
  nothing fails.
- **F3 is not a hole:** checked with libcurl itself: its socket callback gets the whole address (`addrlen` 28 for IPv6;
  `::ffff:127.0.0.1` arrives whole, so `public_addr` refuses it). A length check added anyway: an address shorter than
  its family's form is refused, not judged.
- **F4 fixed:** a child (the terminal's shell, an MCP server, a program `plat::run_program` runs) gets only its three
  files (`posix_spawn_file_actions_addclosefrom_np(3)`, glibc 2.34+), as Windows' handle list does. Tested: the shell
  sees 4 open files (its three and the listing's own).
- **F5, F6 fixed:** the terminal state's temporary file has a per-call number; the EXIT trap's two paths are in
  variables, not `$1` / `$2`, which a command's `set --` changed. Tested with `set --`.
- **F7 fixed:** a slot's error number cleared at each read, `EIO` for a file that ends first. **F8 fixed:** at open, one
  aligned read tells whether the filesystem really takes `O_DIRECT`; on `EINVAL` the file is opened cached.
- **F9:** recorded in docs/linux-port.md (the live GPU memory figure is not there on Linux; `--ctx auto` uses Level
  Zero's size). **F10 fixed:** the usual folders as `PATH` when the server's environment has none. **F11 fixed:**
  `-rdynamic` in the documented build line. **F12 fixed:** `src/clrt.h` (the probes' copy) with `PLAT_SEP`. **F13:** the
  table row of docs/linux-port.md now says what the code does (ported, compiled, not run). **F16 fixed:** `runz`'s
  `/proc` walk with the error-code forms only.
- **Not changed:** F14 (the model list's duplicate check folds case, and its "Windows drive first" tie-break does
  nothing on Linux: harmless there); F15 (`kill(pid, 0)` for a recording's process: there is no recorder on Linux).
- **Also found while testing (not from the audit, older than the port, both systems):** the HTTP server does not close
  a connection whose request says `Connection: close`; it keeps it for the next request, as for keep-alive. Clients
  that read by `Content-Length` (all the usual ones) are not affected. Left for a change of its own: it changes the
  Windows server's behaviour.
- **Also:** on Linux the server's log line of switches shows the home folder as `~` (it said `%USERPROFILE%`).

**The audit as written** (Pi with DeepSeek, `deepseek-flash`, reading only; nothing was built or run):

## Static audit: every Linux code path of the port (df12eff .. HEAD)

**Date:** 2026-11-14
**Repository:** `inferzero`, worktree `zen-varahamihira-043816` (HEAD `93de231`)
**Scope:** the whole Linux port, `df12eff^..HEAD`: the Linux halves (`#ifndef _WIN32`, the `#else` of
`#ifdef _WIN32`) and the files written for it — `src/platform.h`, `src/platform_net.h`, `src/platform_proc.h`,
`src/curl_min.h`, `src/pbkdf2.h`, and the Linux halves in `src/log.h`, `src/cpu_check.h`, `src/crash_log.h`,
`src/full_speed.h`, `src/image.h`, `src/audio.h`, `src/whisper.h`, `src/http.h`, `src/gguf.h`, `src/cl_min.h`,
`src/app_dir.h`, `src/kcache.h`, `zero/ze_min.h`, `zero/clrt.h`, `zero/expert_store.h`, `zero/xcache.h`,
`zero/gguf_pack.h`, `zero/engine*.h`, `zero/runz.cpp`, `zero/mgpu_probe.cpp`, `zero/npu/*.h`, `tools/server.cpp`,
`tools/server_*.h`, `tools/web_tools.h`, `tools/web_view.h`, `tools/pdf_render.h`, `webui/index.html`.
Read first, as asked: `AGENTS.md`, `docs/linux-port.md`, `docs/audit-2026-11-13b.md`.

**Method:** reading only. Nothing was compiled, no `runz` / `serverz` / test / GPU program was started, and no
file in the repository was changed; the report is the only file written. Findings that only a build or a run
could settle are marked as such. Everything else is backed by a line of code cited below.

**Port commits read:** `df12eff`, `81d0e92`, `3a14095`, `4fc6580`, `ef9e15e`, `a0c620f`, `c86af9c`, `ed8977b`,
`5b2c9f8`, `9aae7cf`, `258a373`, `6a62147`, `ab06b8c`, `c2016de` (the `--grep=Linux` list; the merges
`73426d3`, `3e81fc6`, `227baa9`, `6d88126`, `0135a9e`, `9d29f26` add nothing of their own). Changes to
`zero/xe2/*`, `zero/mgpu/*`, `zero/engine_qwen.h`, `engine_h8.h`, `engine_setup.h`, `engine_slots.h`,
`zero/kernels/*`, `webui`'s prefill work come from main, not from the port, and were only checked as "not the
port's".

---

## 1. Verdict

* **No Windows behaviour change found.** Every port commit's diff was read: all port hunks in shared files are
  insertions of `#ifdef _WIN32` / `#else` halves around the original lines, or a call swapped for a `plat::`
  function that makes the same Win32 call with the same arguments. `tools/server_mcp.h` and
  `tools/server_terminal.h` gained 427 lines and lost 1 (a `#endif`); no Windows line was edited. Details in §3.
* **The Linux halves are careful** in the hard places: process groups killed *before* the reap (`waitid … WNOWAIT`),
  `O_CLOEXEC` / `SOCK_CLOEXEC` / `accept4`, `MSG_NOSIGNAL`, `EINTR` retried, `O_NONBLOCK` for FIFO-refusing opens,
  bounded output everywhere, `$TMPDIR`, atomic state writes, `sigaltstack` + `write(2)`-only crash output.
  The 16 findings of `docs/audit-2026-11-13b.md` are fixed as described there, and I re-checked each fix (§6).
* **The one finding that hurts a normal Linux user** is a path mapping the port never did: the Hugging Face cache
  is looked for under `%USERPROFILE%` only, so on Linux `serverz` finds no model after the usual
  `hf download …` (§2, F1).
* The rest are two correctness seams (a short write treated as fatal; the libcurl IPv6 address check reading a
  field it does not own), one inheritance leak (children inherit the server's `fopen`'d files), one session-state
  race, and documentation drift.

Counts: 1 medium-high, 2 medium, 3 low, 8 nits. Severity and confidence are given per finding.

---

## 2. Findings

### F1 (Medium-high) The Hugging Face cache is only looked for under `%USERPROFILE%`: on Linux no model is found
**Files:** `tools/server_catalog.h:47` (`hf_hub_dir`), `src/whisper.h:1444` (`whisper::find_model`),
`zero/asr_qwen3.h:423` (`asr_qwen3::find_model`). `src/platform.h:631` shows the port already has
`plat::home_env()` (`$HOME` on Linux) and the rest of the port uses it — these three places still read
`USERPROFILE` directly.

```c
// tools/server_catalog.h:44-48
static std::string hf_hub_dir() {
  if (const char* v = getenv("HF_HUB_CACHE")) return v;
  if (const char* h = getenv("HF_HOME")) return std::string(h) + "/hub";
  if (const char* u = getenv("USERPROFILE")) return std::string(u) + "/.cache/huggingface/hub";
  return "";
}
```

**What is wrong:** on Linux `USERPROFILE` is unset, so `hf_hub_dir()` returns `""` unless the user sets
`HF_HUB_CACHE` or `HF_HOME`. `scan_models()` (`:246`) then skips the hub entirely (`if (!hub.empty()) scan_hub(hub, 0)`),
and the Whisper / Qwen3-ASR finders search only the roots they collected, so they answer "none in the Hugging Face
cache". The layout they would look for is exactly the one `hf download` writes on Linux
(`$HOME/.cache/huggingface/hub/models--org--repo/snapshots/<rev>/…`, or `$XDG_CACHE_HOME/huggingface/hub`), i.e. the
same layout the Windows install has under `%USERPROFILE%\.cache\huggingface\hub`.
**Trigger:** on Ubuntu, `hf download unsloth/Qwen3.5-4B-GGUF …`, then `./serverz` → `/v1/models` is empty, and
`/v1/audio/transcriptions` says Whisper is not installed; the same files load on Windows. A workaround exists
(`ARC_MODEL_DIRS=~/.cache/huggingface/hub`, or the GGUF copied into `~/.inferzero/models`, which *is* scanned) but
nothing in `docs/linux-port.md` says so, and the default path is silent.
**Fix:** use `plat::home_env()` (and `XDG_CACHE_HOME` first) in all three, or better, one `plat`/catalog helper.
**Severity:** Medium-high (a headline feature of the headless server, broken out of the box on the target system).
**Confidence:** High (a grep of `src`, `zero`, `tools` shows no other code path adds a `$HOME`-based cache root).

### F2 (Medium) `plat::DirectWriter::write` treats a short write as a hard failure — the expert-file build dies and deletes its work
**File:** `src/platform.h:321-343`; used by `zero/expert_store.h:412-417` (`put`) and `:491-492`.

```c
// src/platform.h:330-338
      const ssize_t got = ::write(fd_, p, (size_t)c);
      if (got < 0 && errno == EINTR) continue;
      if (got != (ssize_t)c) return false;        // c is up to 1 GiB
```
**What is wrong:** `write(2)` on a regular file may legally return fewer bytes than asked (interrupted after a
partial transfer, a file-system quota/limit, and for `O_DIRECT` a request larger than what the device accepts in
one go — which is why the 1 GiB chunking exists on the Windows side, where `WriteFile` with `FILE_FLAG_NO_BUFFERING`
either writes all of it or fails with an error). Here a short count is reported as a failure and the caller throws
`"write failed: …"`, deletes the `.tmp`, and the whole expert file (15-52 GB, "once per model") has to be written
again. The loop also advances `p` by `c` rather than by `got`, so it cannot recover even if the test were relaxed.
**Trigger:** building an expert file on a filesystem/driver that splits the 1 GiB unbuffered write (or a signal
landing mid-write). Windows keeps its behaviour if the Windows half is left as it is.
**Fix (Linux half only):** `if (got > 0) { p = (const uint8_t*)p + got; n -= (size_t)got; continue; }` and treat
only `got == 0` as an error.
**Severity:** Medium (a long, expensive job fails late; no data corruption, no crash).
**Confidence:** High for the code fact; Medium for how often a real filesystem returns short on a 1 GiB request.

### F3 (Medium, security-relevant) The libcurl address check reads an address field it does not own; an IPv4-mapped IPv6 address is judged public
**Files:** `tools/web_tools.h:338-347` (`curl_open_socket`), `tools/web_tools.h:97-118` (`public_addr`),
`src/curl_min.h:18-24` (`curl_sockaddr`).

```c
// src/curl_min.h:18-24  (curl's own layout)
struct curl_sockaddr { int family, socktype, protocol; unsigned int addrlen; struct sockaddr addr; };
// tools/web_tools.h:342
  if (!public_addr(&a->addr)) { h.refused = true; return curlm::SOCKET_BAD; }
// tools/web_tools.h:100-105
  if (sa->sa_family == AF_INET6) { const uint8_t* b = ((const sockaddr_in6*)sa)->sin6_addr.s6_addr; …
```
**What is wrong:** `curl_sockaddr::addr` is a 16-byte `struct sockaddr`, but `public_addr` reads it as a 28-byte
`sockaddr_in6` and uses bytes 8…23 of `sin6_addr`. libcurl builds that struct on its stack (zeroed) and copies
`min(addrlen, sizeof(addr))` bytes into it, so for an IPv6 address only `sin6_addr[0..7]` is ever set; the
`::ffff:a.b.c.d`, `::/96` and `64:ff9b::/96` comparisons at `:104` need `sin6_addr[8..11]`, which are therefore
always zero. An IPv4-mapped address (`AAAA ::ffff:7f00:1`) fails all three prefix tests and falls through to
`return true` — judged *public*, and the socket is opened.
**Trigger / impact:** `check_host` (the first line, `:139`) resolves the name and checks the *full* addrinfo, so a
name that resolves to `::ffff:127.0.0.1` is still refused there. The gap is the case the callback exists for: a
name that answers with a public address for `check_host` and with an IPv4-mapped local address for libcurl's own
resolution a moment later (DNS rebinding, or a TTL-0 answer). libcurl then connects to what the kernel treats as
IPv4 loopback, and the model reads the machine's own services.
**Fix:** in `curl_open_socket`, use `a->addrlen`: refuse when `addrlen < sizeof(sockaddr_in6)` for `AF_INET6`
(or when `addrlen > sizeof a->addr`), so an incomplete address is never judged; a full re-resolution check (as the
proxy does) would close it completely.
**Severity:** Medium (an SSRF protection weakened for one address form; needs an attacker-controlled DNS answer).
**Confidence:** Medium — the code fact (a 28-byte read of a 16-byte member) is certain; that libcurl fills only 16
bytes is from libcurl's `curl.h` declaration plus its `socket_open()` copy of `min(addrlen, sizeof(addr))`, which I
could not read here (no libcurl source on this machine). If libcurl ever passes a larger object, the checks work
and this is a bound-check hygiene issue only.

### F4 (Low) Children inherit every `fopen`'d file of the server — the log file among them
**Files:** `tools/server.cpp:1041` (`FILE* lf = fopen(f.c_str(), "ab")`, kept for the process's life as
`zlog::cfg().file`), `tools/server_terminal.h:170-181` (`posix_spawnp("bash", …)`), `tools/server_mcp.h:394`
(`posix_spawn(…)`); the sockets and pipes are all `O_CLOEXEC` / `SOCK_CLOEXEC`, but `fopen` is not.
**What is wrong:** on Windows the two children get an explicit handle list (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`),
and `fopen` handles in MSVC are non-inheritable, so a child gets exactly its three pipes. On Linux the children
get every descriptor without `FD_CLOEXEC`: `--log-file`'s fd, and any other `FILE*` open at that moment
(`zero/xcache.h:736`'s trace file, `tools/server_chat.h:166/850` while a write is in flight).
**Trigger:** `serverz --log-file /var/log/inferzero.log`; then any model-run bash command can
`echo 'fake line' > /proc/self/fd/3` and write into the server's log without the log's timestamp prefix (log
spoofing in the file an issue report is pasted from), and a long-lived MCP server keeps the log open after the
server is killed, so the space of a deleted log stays used. Neither the Windows terminal nor the Windows MCP client
gives a child that handle.
**Fix:** `fcntl(fileno(lf), F_SETFD, FD_CLOEXEC)` where the log file is opened (and the same for the xcache trace
file), or `posix_spawn_file_actions_addclosefrom_np(&fa, 3)` in the two Linux spawns (glibc 2.34+).
**Severity:** Low (both children already run with the user's privileges, and the sockets — the ones that matter —
are already `CLOEXEC`).
**Confidence:** High.

### F5 (Low) The terminal session's state temp file is not unique per call: two commands at once corrupt it
**File:** `tools/server_terminal.h:251` (`const std::string tmp_state = state + "." + plat::process_id() + ".tmp";`).
The project's own atomic writers add a per-process sequence number for exactly this reason
(`tools/server_files.h:105/584`, `src/kcache.h:24`), and `plat::write_temp` uses `mkstemp`.
**What is wrong:** two `run_command` requests for the *same* chat session (two tabs, or the terminal panel and a
model tool call) both write the same path with `std::ios::trunc` and then `rename` it. One call's bytes are
truncated by the other's `ofstream` open, or the file is renamed away between write and rename (then unlinked and
no state is written at all). Next call, `Json::try_parse` fails and `term::run` (`:117-126`) silently starts a
fresh shell: the `cd` and the exported variables are lost, with no message.
**Trigger:** `POST /v1/tools/run_command` twice for one `chat_id` with `"session": true`, overlapping.
**Fix:** add the same atomic counter (`static std::atomic<uint64_t> seq`), or use `mkstemp`.
**Severity:** Low (needs concurrency; loses session state, never breaks a file outside it).
**Confidence:** High (the code has no lock around the state; `term::run` is called from an HTTP thread).

### F6 (Low) The bash EXIT trap writes to `"$1"`/`"$2"`, which the command itself can change
**File:** `tools/server_terminal.h:136-139`, arguments added at `:152`.
```c
body = "trap '__az_rc=$?; pwd -P > \"$1\" 2>/dev/null; env -0 > \"$2\" 2>/dev/null; exit $__az_rc' EXIT\n" …
std::vector<std::string> args = {"bash", "--noprofile", "--norc", script};
if (!state.empty()) args.insert(args.end(), {out_pwd, out_env});
```
**What is wrong:** `$1`/`$2` are the *script's* positional parameters at trap time, and the script is the user's
command. A command that runs `set -- x` (or `shift`) redirects the trap: the state file is not written (session
lost, silently) and `pwd -P > "x"` creates a stray file in the chat folder.
**Trigger:** `{"command": "set -- a b; echo $#"}` in a session chat.
**Fix:** pass the paths in variables that the command cannot touch, e.g. put them in the trap as
`printf '%s' "$AZ_PWD"` with `AZ_PWD`/`AZ_ENV` exported in the script header, or use `exec 9>"$out_pwd"` opened
before the command.
**Severity:** Low (a nit with a session-state side effect). **Confidence:** High.

### F7 (Nit) `plat::AsyncFile::error(slot)` can return a stale errno
**File:** `src/platform.h:213-270`. `start()` clears `ok_[s]` but never `err_[s]`; the EOF path (`:225`,
`if (r == 0) return true;`) returns "started" with `ok_` false and `err_` untouched. `zero/expert_store.h:343`
then prints `"expert read failed (0)"` or a number from an earlier failure on that slot.
**Severity:** Nit (a wrong number in a message). **Confidence:** High.

### F8 (Nit) `O_DIRECT` is only fallen back from at `open()`, not at read time
**File:** `src/platform.h:190-196`, `:213-228`. If `open(O_DIRECT)` succeeds but `pread` returns `EINVAL`
(a filesystem whose logical block size is larger than 4096, some FUSE/NFS/`/mnt/c` setups), every expert read
throws `"expert read failed (22)"` and MoE streaming does not work, although the buffered path would.
`docs/linux-port.md` already says MoE streaming was not tried on Linux and that WSL's `/mnt/c` "may not honour
O_DIRECT", so this is a known-untested area, not a surprise; a `EINVAL` on the first read could reopen buffered
once. **Severity:** Nit (documented gap). **Confidence:** Medium (depends on the filesystem).

### F9 (Nit) `zrt::video_budget()` returning false silently drops two safeguards
**Files:** `zero/clrt.h:491-495` (false on Linux), `tools/server_whisper.h:216` and `tools/server.cpp:2042`.
Both call sites are `if (zrt::video_budget(b, u) && …)`, so on Linux the speech service's "is there room for
Whisper before loading it" pre-check never runs (it falls back to the allocation failing mid-load, which is caught
and reported), and `/v1/server/memory` and `/props` simply omit `budget_bytes`. Documented ("Sysman later",
`docs/linux-port.md`), but it is a real behavioural difference from Windows worth knowing for the iGPU, where
`global_mem` (Level Zero's `totalSize`) is the only budget and the engine sizes its expert cache from it
(`zero/engine_stream.h:56`). **Severity:** Nit. **Confidence:** High.

### F10 (Nit) With no `PATH` in the server's environment the terminal's `PATH` is only the app's folders
**File:** `tools/server_terminal.h:100-102`: `base["PATH"] = extra_path + (base.count("PATH") ? ":" + base["PATH"] : "")`.
Under a supervisor that passes no `PATH` (systemd sets one, a bare `env -i` does not) `bash` gets
`<app>/bin:<exe_dir>` only, so `ls`, `grep`, `python3` are "command not found", and the state diff then records
nothing useful. **Fix:** append `:/usr/local/bin:/usr/bin:/bin` (and `/usr/sbin:/sbin`) when `PATH` is absent.
**Severity:** Nit. **Confidence:** High.

### F11 (Nit) The crash report's symbol names need `-rdynamic`, which the documented build line does not pass
**Files:** `src/crash_log.h:80` (the comment: "function names where the binary exports them: link with -rdynamic"),
`docs/linux-port.md:102-104` (the `g++` line for `runz`, "the same build line as runz with `zero/serverz.cpp`").
Without `-rdynamic` the backtrace prints only module offsets, which is still useful but not what the comment
promises, and the doc is the only build recipe there is (no CMake yet). **Severity:** Nit (documentation / feature
quality). **Confidence:** High.

### F12 (Nit) `src/clrt.h` still builds kernel-cache paths with backslashes
**File:** `src/clrt.h:46-47`, `:60` (`app_dir() + "\\kernels"`, `cache_dir + "\\" + hex + ".bin"`).
`zero/clrt.h` (the engine `runz`/`serverz` use) was converted to `PLAT_SEP` in `5b2c9f8`; this older copy is used
only by `zero/grf_probe.cpp`, `zero/page_probe.cpp`, `zero/ze_probe.cpp`, `zero/npu/npu_ffn.cpp` and
`tools/*_ref.cpp`-style probes, none of which are part of the port's build. If one of them is ever built on Linux
the cache lands in one file per driver whose name contains a backslash. **Severity:** Nit. **Confidence:** High.

### F13 (Nit) The NPU is documented "off on Linux" but there is no guard
**Files:** `zero/npu/engine_npu.h:89-105` (`npu_setup()` has no `_WIN32` filter), `zero/npu/npu_rt.h:66,107`
(the loader and cache path are ported on purpose). `docs/linux-port.md`'s table says "The NPU | off (the NPU's
Linux driver is another API)", while the same document's Progress row says the graph extension "Intel's Linux NPU
driver has too: compiled, not run". So `ARC_NPU=1` / `--npu` on Linux will try the NPU path rather than be refused.
Since it is untested, a Linux build with `--npu` is a supported-looking switch with no support behind it either
way; one of the two statements should give. **Severity:** Nit (doc vs code). **Confidence:** High.

### F14 (Nit) Model-catalog details that are Windows words on Linux
**File:** `tools/server_catalog.h:270-290`: the duplicate-file dedupe lower-cases file names (so `Model.gguf` and
`model.gguf` in one folder are listed once on a case-sensitive filesystem) and the "from the Windows drive first"
tie-break reads `getenv("SystemDrive")` → `"C:"`, which no Linux path starts with, so the tie-break degenerates to
"the later scan wins". Neither is harmful today; both are port leftovers.
**Severity:** Nit. **Confidence:** High.

### F15 (Nit) A recording's PID check via `kill(pid, 0)` can be fooled by PID reuse
**File:** `tools/server_whisper.h:696-699`. With no recorder on Linux the pid comes from a state file; a stale pid
that has been reused by any process makes `running` true for ever, so `fix_wav_sizes(w)` never runs and the
recording stays "incomplete". `kill(pid,0)` also fails with `EPERM` for another user's process, which is the safe
direction. **Severity:** Nit. **Confidence:** High.

### F16 (Nit) `warn_gpu_neighbours` on Linux can throw out of a `std::filesystem` iteration
**File:** `zero/runz.cpp:41-56`: `directory_iterator("/proc", ec)` is safe to construct, but the range-for's
`increment` on error throws `std::filesystem_error` (`ec` is not passed to it), and `std::stoul(pid)` can throw
`out_of_range`. Inside `runz`'s start-up this would end the program with an unhandled exception instead of the
intended note. Practically unreachable (`/proc` does not fail mid-walk). **Severity:** Nit. **Confidence:** High
for the code fact, Low for reachability.

---

## 3. Windows behaviour: what I checked and found unchanged

I read the diff of every port commit rather than trusting the messages. The method: for each file, all removed
(`-`) lines in the port commits, and whether each is inside a `_WIN32` half.

| Area | Check | Result |
|---|---|---|
| `zero/ze_min.h`, `src/cl_min.h`, `zero/clrt.h` (`df12eff`) | `LoadLibraryA` / `GetModuleHandleA` / `GetProcAddress` / `FreeLibrary` → `plat::dl_*`; on Windows `plat::dl_*` calls exactly those (`src/platform.h:44-70`) | identical |
| `src/app_dir.h` | `getenv("USERPROFILE")` → `plat::home_env()`; `LOCALAPPDATA` and the move/rewrite logic untouched | identical on Windows |
| `src/gguf.h`, `zero/engine_drafters.h` (`81d0e92`) | `CreateFileA` + `CreateFileMappingA` + `MapViewOfFile` → `FileMap` (same three calls, same order, same messages); `fm_.open()` keeps `FILE_SHARE_READ`, no `_O_`/`FILE_FLAG_` lost | identical; one behavioural nuance: `map()` now also refuses a 0-byte file with return 1 where `CreateFileMappingA` itself failed — same error text path |
| `zero/expert_store.h` (`3a14095`, `4fc6580`) | the `OVERLAPPED`/event/`ReadFile`/`WaitForMultipleObjects`/`GetOverlappedResult`/`CancelIoEx` sequence moved into `AsyncFile` unchanged; slot ids initialised to `i` before the open; the writer's `WriteFile` loop moved into `DirectWriter` with the same 1 GiB chunks; `MoveFileExA(MOVEFILE_REPLACE_EXISTING)`, `DeleteFileA`, `SetProcessWorkingSetSize(…, -1, -1)` → the helpers | identical |
| `src/whisper.h` (`ef9e15e`) | `CreateFileW` (`open_fs`) + mapping; `compressapi` path and `<compressapi.h>` kept, now behind `#ifdef _WIN32`; MSZIP body unchanged | identical |
| `src/http.h` (`a0c620f`) | every `recv`/`send`/`closesocket`/`shutdown`/`setsockopt`/`accept`/`WSAPoll` replaced 1:1; `INVALID_SOCKET` → `plat::kNoSocket` (equal to `INVALID_SOCKET`); `SO_RCVTIMEO`/`SO_SNDTIMEO` DWORDs, `TCP_NODELAY`, `SO_REUSEADDR`, `IPV6_V6ONLY` BOOLs; `send_file`'s `CreateFileW` share mask, `GetFileSizeEx`, `SetFilePointerEx`, `ReadFile` | identical (`sock_send`/`sock_recv` use `::send`/`recv` with 0 flags on Windows, as before) |
| `zero/runz.cpp` (`5b2c9f8`) | `tlhelp32.h` and the snapshot walk behind `#ifdef`, `_fseeki64`/`_ftelli64`/`_putenv_s` → helpers that are those on Windows | identical |
| `src/cpu_check.h`, `src/full_speed.h`, `tools/server_startup.h` (`__cpuid`), `zero/clrt.h` (`dxgi1_4.h`, `#pragma comment(lib)`), `src/audio.h`, `src/image.h`, `src/crash_log.h`, `tools/pdf_render.h`, `tools/web_view.h`, `tools/server_remote.h` (`bcrypt.h`) | Windows halves kept verbatim, includes moved inside `#ifdef _WIN32` | identical |
| `src/http.h`, `tools/server_files.h`, `tools/server_cache.h`, `tools/server_memory.h`, `tools/api_compat.h`, `tools/server_whisper.h`, `tools/xlsx.h`, `tools/server_ocr.h` (LocalTime, GetFileInformationByHandle, GlobalMemoryStatusEx) | `gmtime_s`/`localtime_s` → `plat::utc_time`/`local_time` (which call them); `GlobalMemoryStatusEx` → `ram_available()` (`ullAvailPhys`); `_wfopen` → `fopen_path` (`_wfopen` with the same wide form); `_get_osfhandle` same-file test keeps the `ha != INVALID_HANDLE_VALUE` guard | identical; the one cosmetic difference already recorded (two independently sampled memory figures in `print_head`) remains |
| `tools/server_terminal.h`, `tools/server_mcp.h` (`ab06b8c`) | diff is +427/−1 lines, and the single deletion is a `#endif`; every new helper (`exe_dir`, `env_map`, `shell_own`, `read_some`, `find_program`, `OutKeep`, `stop_`) is inside `#ifndef _WIN32` or new; no shared line uses `w8`/`u8`/`wide` | untouched |
| `tools/server.cpp` | `/props` gains one field (additive), `same_file`, `set_env`, `ignore_sigpipe` (Linux only), `--log-file`'s `fopen` (was `_wfopen`? no: it was and is `fopen(f.c_str(), "ab")` on both) | identical; `--log-file`'s same-file refusal now goes through `fstat`/`GetFileInformationByHandle` with the same result |
| `zero/npu/*` (`5b2c9f8`) | `VirtualAlloc`/`VirtualFree` → `page_alloc`/`page_free` (`VirtualAlloc`/`VirtualFree` with the same `MEM_*`), `GetModuleHandleA`/`GetProcAddress` → `plat::dl_*`, a `std::pair<void*,size_t>` for the blobs so `page_free` gets the length (Windows ignores it) | identical in effect; `npu_shm_bytes_` is set before every `page_free` (also in the failure branch, `:145-151`) |

Two standing notes from the previous audit, still true and deliberately kept: `OutKeep`
(`tools/server_terminal.h:47-67`) is used only by the Linux half while the Windows half keeps its own inline copy
(`:367-380`); and the `8,000`-character command limit's comment is now correct (`tools/server_file_tools.h:943`).

---

## 4. What I checked and found correct

**`src/platform_proc.h` — `run_program` (the piece most likely to hang or leak)**
* All six pipe ends are closed on every exit path (`close_all`, and the spawn-failure branch closes the three
  parent ends after the child's were closed at `:148-150`); `pipe2(O_CLOEXEC)` plus `posix_spawn_file_actions`
  means the child gets exactly stdin/stdout/stderr.
* `O_NONBLOCK` is set in the parent only (`:153`); the child's ends are separate open file descriptions, so its
  stdin/stdout stay blocking. Writing stdin checks `EAGAIN`/`EINTR`, drops `EPIPE` as "it stopped reading".
* The deadline now covers the final wait (`:197-207`): a child that closes both pipes and keeps running is killed
  at it, and a failing `poll` (`k < 0`, `:131`) is treated as "the program is ended", not as "wait for ever" —
  i.e. M2 of the last audit is really fixed, and re-checked here line by line.
* `kill(-pid, SIGKILL)` happens while the leader is a zombie (`waitid … WNOWAIT`), so the pid cannot be reused
  between the test and the kill; the group is then reaped with `EINTR` retry (`:208-210`). The same shape in
  `tools/server_terminal.h:203-221` and `tools/server_mcp.h:531-541`.
* The child gets `SIGPIPE` back to `SIG_DFL` and an empty signal mask (`:157-164`) — the parent's choice to
  ignore it does not leak into `ffmpeg`/`pdftoppm`.
* Output bounds: stdout at `max_out` plus `cut`, stderr at 64 KB, `min(cap, size)` cannot underflow; `on_out` is
  only used by audio, which traps the caller's exception into an `exception_ptr` and rethrows after the child is
  gone (`src/audio.h:649-657`).
* `write_temp` retries `EINTR`, unlinks on a short write, reports `$TMPDIR` in its message; `path_arg` prefixes
  `./` for a name starting with `-`; `last_line` bounds its output.

**Sockets (`src/platform_net.h`)**
* `sock_open`/`sock_accept` add `SOCK_CLOEXEC`/`accept4`, `sock_send` uses `MSG_NOSIGNAL` and retries `EINTR`,
  `sock_recv` retries `EINTR`, timeouts are a `timeval` (`tv_usec` computed from `ms % 1000` — correct), and
  `sock_peer_gone` polls and peeks with `MSG_DONTWAIT`, treating `EAGAIN`/`EINTR` as "not gone" (no false
  positive that would drop a live stream, no blocking peek).
* `sock_wait_read2` on `poll` counts `POLLHUP|POLLERR` as readable, so the proxy's `recv` sees the end instead of
  spinning; `sock_flag` mistakes none.

**The HTTP server on Linux (`src/http.h`)** — no Win32 call left; the Linux compile and `test_http` are already
recorded in `docs/linux-port.md`; `send_file`'s `SeqFile` refuses a folder/FIFO/device at open (`O_NONBLOCK` then
`S_ISREG`, then blocking reads) and a file that shrinks mid-send closes the connection instead of promising bytes
it cannot send.

**Paths and containment (`plat::path_inside`, `src/platform.h:600-614`)** — `/`-boundary, case-sensitive,
trailing slashes trimmed, `or_same` honoured; used by `refuse` (`tools/server_projects.h:93`), `resolve_in`
(`:166`) and `ocr_in_project` (`tools/server_ocr.h:400`). `refuse` refuses the data folder *and* a folder that
holds it, the Linux system folders, the home folder and its `.ssh`/`.gnupg`/`.aws`/`.config`/… subfolders; `f` and
the home folder are `weakly_canonical`ised, so a symlink to `/etc` is caught. `resolve_in` passes `or_same=false`
and a canonicalised path, so `project:<id>/../..` and a symlink inside a project pointing out are refused;
`ocr_in_project` derives `rel` with `lexically_relative` so the OCR output stays inside the project.

**Terminal (`tools/server_terminal.h`, Linux half)** — the command goes into a `mkstemp` file (not `-c`), stdin is
`/dev/null` via `addopen`, the process group is its own, the read end is `O_NONBLOCK` and drained on a 200 ms poll
with `cancelled()` polled on the same cadence as Windows, output is bounded by `OutKeep` (48 KB head + 16 KB
tail) with a UTF-8-safe cut in `server_file_tools.h:995-1001`, `pwd -P`/`env -0` read back with NUL separation, the
diff is computed against a freshly built base map with `PWD`/`OLDPWD`/`SHLVL`/`_` excluded, removed variables
written as JSON `null`, the state validated with `is_directory` before use, a corrupt state file falls back to a
fresh shell, the state is written atomically (temp + `rename`) and not at all after a kill, `LC_ALL=C.UTF-8` set
when the server names no locale, the folder checked before the spawn (so the usual `addchdir_np` 127 cannot
happen), and the venv's `deactivate` redefined for the next call.

**MCP client (`tools/server_mcp.h`, Linux half)** — `pipe2(O_CLOEXEC)` ×3, `posix_spawn` with a process group,
a `stop_` flag the readers poll every 200 ms so `cleanup()` always ends, stdin closed first (so a well-behaved
server exits by itself), 1.5 s grace, then the group killed *before* the reap, both readers joined before their fds
are closed, a spawn failure closes all six ends, `find_program` handles `PATH` unset / an empty element / a command
with a `/`, the environment merge mirrors the Windows `_wcsicmp` merge, `request()` erases `waiting_` and
`replies_` on every exit path, and `handle_line` is called without `mu_` held (no lock-order inversion with
`send`'s `write_mu_`).

**`src/crash_log.h` (Linux)** — `sigaltstack` + `SA_ONSTACK|SA_SIGINFO|SA_RESETHAND`, the signal name, the
`si_addr` for `SIGSEGV`/`SIGBUS`, `backtrace` warmed at install time so libgcc's unwinder is not loaded inside the
handler, `backtrace_symbols_fd` (which writes with `write(2)`) and hand-rolled `put`/`put_num` (no `stdio`, no
allocation) before the non-safe `extra()`; `signal`/`raise` are on POSIX's async-signal-safe list, and a fault
inside the handler ends the process through the default action.

**PBKDF2 (`src/pbkdf2.h`)** — re-read against FIPS 180-4 / RFC 2104 / RFC 8018: the schedule (`rotr 7/18 >> 3`,
`rotr 17/19 >> 10`), the 64 constants, `finish` capturing `len*8` before appending `0x80` (the wrap through a
64-byte block is correct for every length), HMAC's long-key hash and pre-digested pads (so `mac(u, 32, u)` in
place is safe), `U1 = PRF(P, S‖INT(i))` big-endian, truncation to `dk_len`, `dk_len == 0` → empty; `<algorithm>`
is included now. `getrandom` retries `EINTR`, rejects `0` and negatives, and the salt/session tokens cannot be
silently short.

**Media on Linux (`src/image.h`, `src/audio.h`, `tools/pdf_render.h`)** — image: `ffmpeg -frames:v 1 -f image2pipe
-vcodec ppm -pix_fmt rgb24`, a bounded PPM header parse (9 digits per field, `maxval == 255`, `w,h ≤ 16384`,
`w*h ≤ 64 Mi`, a remaining-length test before the copy, `cut`/`timed_out` refused), the empty input refused.
Audio: `ffprobe` channel count, the channels' plain average via `pan=mono|c0<c0+c1…` (renormalised, not ffmpeg's
`-ac 1`), `aresample=async=1:first_pts=0` for the timeline, `-nostdin`, a 24 h cap, `duration_seconds` with
`from_chars` (locale-free), `decode_mono_pcm16` now goes through `ff_audio` (so it throws instead of returning a
silently truncated stream — the last audit's L5 item is fixed), MP3/AAC through ffmpeg with the exit code and the
time-out checked, and temp files unlinked by an RAII guard. PDF: `pdfinfo` twice at open (pages + all page sizes,
not once per page as before), `pdftoppm -f/-l/-png/-singlefile` with `-scale-to 4000` past 4000 px, a size guard
before the `PNG` test (M1 fixed), the body in `try` like the Windows half, and every failure path yields the
"page could not be rendered" note instead of an exception.

**Web tools on Linux (`tools/web_tools.h`)** — libcurl loaded through `dlopen("libcurl.so.4")` with a
`thread_local`-free function-local `static` (one-time, thread-safe init), `OPT_NOSIGNAL`, `OPT_PROXY ""` +
`OPT_NOPROXY "*"` (the environment's proxy ignored), protocols restricted to http/https, redirects by hand with
the port/host checked on every hop, the deadline shared over all hops (`left()` re-read per hop), the body limit
enforced in the write callback (`stopped` → `CURLE_WRITE_ERROR` → `cut`), `too_big` only for whole-or-nothing
files, `timed_out` set for any failure mid-body, and every `easy_setopt` argument typed as libcurl requires
(`long` for `OPT_TIMEOUT_MS`/`OPT_HTTP_VERSION`/`OPT_PROTOCOLS`/`OPT_NOSIGNAL` — correct on x86-64 varargs);
`url_parts` frees every `url_get` result with `curl_free` and cleans the `CURLU` handle; `ascii_host` frees
libidn2's output. (The IPv6-callback caveat is F3.)

**Engine on Linux** — `plat::FileMap` (model, split parts, DFlash2 drafter, Whisper `model.bin`), `plat::AsyncFile`
(slot ids assigned before the open so no read can start with an unset id; short reads loop; `EINTR` retried; a slot
error recorded per slot so the two reading threads do not clobber each other's errno), `plat::DirectWriter` (the
header padded to the first layer's offset, every write a 4096 multiple from page-aligned memory, `.tmp` removed on
failure, `rename` to install), 64-bit `fseek64`/`ftell64`, `PLAT_SEP` in the kernel cache / warm marker / NPU blob
cache / package folders, `kcache_*` on `process_id`, `/proc/meminfo`'s `MemAvailable` for the RAM tier and the
prefix cache, `sysconf` for total RAM, no DXGI (F9), `_Exit(1)` on a fatal GPU error (same as Windows), and the
expert-file/RAM-tier arithmetic localises `integrated()` (so the RAM tier stays off on an iGPU, as on Windows).

**`webui/index.html`** — `serverOs` is a hoisted `var` defaulting to `"windows"` and re-read on every
`refreshModel()` (so a server switch in the same tab changes the prompt, the placeholder and the tool schema both
ways — the last audit's L8 item), `termTools()`/`termSystem()` pick the bash variants, `TERM_SYSTEM_SH`'s
`replace()` target matches `TERM_SYSTEM`'s text exactly (`"Write PowerShell (the first result says which)"`),
`termShellNote` handles the probe's `{sh, os, arch}`, and `TERM_PROBE_SH` reads bash's version, `os-release` and
`uname -m` without a here-doc.

**`docs/linux-port.md` vs the code** — the Progress rows match what the code does, with the exceptions in F11/F13
and the plan rows (Sysman, CMake, headless Chromium, WASAPI) that are explicitly listed under "Not yet". The
documented build line (`:102`) is the one that builds: `-std=c++20 -O2 -mavx2 -mfma -mf16c -fopenmp -include
src/cpu_check.h -include src/log.h -I third_party -pthread` — `_GNU_SOURCE` is not needed (g++ defines it in C++
mode, so `pipe2`, `accept4`, `gettid`, `getrandom` and `posix_spawn_file_actions_addchdir_np` are declared, as the
last audit's M6 concluded), `-ldl` is not needed on glibc ≥ 2.34, and no Windows header remains in the
`runz`/`serverz` include graph (grepped; every `<windows.h>`/`<winsock2.h>`/`<winhttp.h>`/`<bcrypt.h>`/
`<compressapi.h>`/`<dxgi1_4.h>`/`<intrin.h>` in those files is inside `#ifdef _WIN32`). `-rdynamic` is missing
(F11).

**`docs/audit-2026-11-13b.md` fixes, re-checked:** H1 (`pbkdf2.h` committed, `<algorithm>` added) ✓; M1 (PDF size
guard + `try`) ✓; M2 (deadline over the final wait, failing `poll` ends the program) ✓; M3 (`ignore_sigpipe` at
the top of `main`) ✓; M4 (`path_inside` in `refuse`/`resolve_in`/`id_of`/`ocr_in_project`, the `$HOME` arm, the
system folders) ✓; M5 (crash handler: `write(2)` only, `sigaltstack`, `extra()` last) ✓ — `backtrace_symbols_fd`
is not on POSIX's list but writes with `write(2)` and allocates nothing; L1 (folder checked before the spawn —
the `addchdir_np` return value is still ignored, so a folder removed in between still gives 127) mostly ✓; L2
(group killed before the reap) ✓; L3 (`$TMPDIR`, the state written to a temp file and renamed) ✓, with F5's
uniqueness gap; L4 (`SeqFile`: `O_NONBLOCK` + `S_ISREG`) ✓; L5 (time-out named, `EINTR`, `path_arg`,
`from_chars`, empty image refused, the per-page `pdfinfo` gone) ✓; L8 (`LC_ALL`, the page both ways, the comment)
✓. Not changed on purpose and still so: L6 (zlib vs MSZIP, a recorded difference), L7 (`send` under `write_mu_` —
the Linux half could `poll(POLLOUT)` with a deadline, but the Windows half has the same shape), the `OutKeep`
duplication. No CMake, no release archive, no systemd unit yet (step 4).

---

## 5. What I could not settle by reading

* libcurl's exact `opensocket` struct fill (F3): no libcurl source or headers on this machine. The code-level
  bound (28 bytes read from a 16-byte member) is certain; the consequence depends on libcurl zeroing/copying.
* Whether `write()` on the expert file's filesystem ever returns short for a 1 GiB `O_DIRECT` request (F2).
* Whether a real filesystem here rejects `pread` with `EINVAL` after `open(O_DIRECT)` succeeded (F8).
* `pdftoppm -singlefile` with no output prefix writing to stdout: the code assumes it (and
  `docs/linux-port.md`'s "Checked" row says the media functions were compared on both systems, so it must have
  worked in WSL), but I did not run it.
* The exact `/mnt/c` (9p/drvfs) behaviour with `O_DIRECT`: documented as untested for MoE streaming.

---

## 6. Suggested order of work

1. **F1** — one helper (`hf_cache_dir()` using `plat::home_env()` + `XDG_CACHE_HOME`), used by the three files.
   Without it the headless server has no models on a stock Ubuntu, which is the port's whole point.
2. **F2** — three lines in `DirectWriter::write`'s Linux half (advance by `got`, only `got == 0` is an error).
3. **F3** — bound-check `a->addrlen` in `curl_open_socket` (refuse an `AF_INET6` address shorter than
   `sizeof(sockaddr_in6)`), or re-check the address the way `web_view`'s proxy does.
4. **F4** — `FD_CLOEXEC` on the log file (`--log-file`) and the trace file, or `addclosefrom_np(&fa, 3)`.
5. **F5**, **F6** — a per-call suffix for the state temp file; the trap's paths in exported variables.
6. Then the nits: F7 (`err_` reset per start), F9/F13/F14 wording in `docs/linux-port.md`, F11 (`-rdynamic` in the
   documented line), F12 (`PLAT_SEP` in `src/clrt.h` or a note that it is Windows-only), F10 (default `PATH`),
   F8 (buffered fallback on `EINVAL`).
