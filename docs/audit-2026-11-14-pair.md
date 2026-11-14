# Audit of 2026-11-14 (evening): pair review, DeepSeek-V4.1 and Claude

What was reviewed, against main at `7ea1d32`:

1. **The engine and kernels** changed since the morning's audit (`bb67905`..main), with pull requests #21 and #22
   merged in.
2. **The tools, `src/`, tests, scripts, setup and CI** changed in the same range.
3. **The contributor pull requests** #14, #15, #20, #21 and #22, each as submitted and as merged, and the open #24.
4. **The Linux port** (`src/platform*.h`, every non-Windows code path, pull requests #13, #16, #18), and whether the
   Linux audit's fixes ([audit-2026-11-14-linux.md](audit-2026-11-14-linux.md)) held.
5. **The web UI**, `webui/index.html`.
6. **The multi-GPU failsafes** (`zero/mgpu/`, `kernels/tp.cl`, the split parts of `xcache.h` and `clrt.h`), checked
   against every promise in [multi-gpu.md](multi-gpu.md).

**How.** Each part was reviewed twice, by two reviewers who did not see each other's results:

- **DeepSeek-V4.1** through Pi (`deepseek-flash`, high thinking);
- **a Claude reviewer.**

Both had the same brief and read-only tools. Each finding was then checked in the code before anything was changed.
Where a finding could be tested without a GPU it was: the PowerShell dot-source behaviour, `Split-Path` on UNC paths,
the fixes' parse checks.

In the tables:

- **Found by:** D (DeepSeek), C (Claude), or both.
- **Verdict:**
  - **real:** fixed below;
  - **not a bug:** with the reason;
  - **left:** real, but deliberately not fixed here.

No reviewer found anything that changes a model's results: no layout error, no logit change. Neither found an XSS in
the web UI.

## 1. Engine and kernels

| Finding | By | Severity | Verdict, fix |
|---|---|---|---|
| `ARC_TP_ATT_QSPLIT=1` excluded Flash Next but not DeepSeek-V4.1, whose attention cannot be split by q-head: the load failed, or partial outputs would have been added | D | medium | real: `&& !hp.ds41` (engine_tp_shard.h) |
| DeepSeek's `candidate_block_size` read from the file and used as a divisor unchecked: 0 crashed the loader | D | medium | real: `cand_bs >= 1` among the dimension checks (engine_ds41.h) |
| `FnPreciseHold` was process-wide: while an AR4 model was loaded, every other Flash Next engine was precise, the server's dry sizing runs too | C | low | real: precise mode per engine (`Engine::fn_precise`, engine_load.h) |
| engine_ds41.h's header still said "no multi-GPU" | D | low | real: comment |
| DeepSeek with `--mtp` refused before `tp_setup` could clear it | D | low | not a bug: DeepSeek supports no drafter on any GPU count; the message is right |
| The batched pass in precise mode would run int8 attention on f32 query prep | D | medium | not a bug: unreachable, `batch_ok` requires `attn_int8`, which precise mode turns off |

## 2. Tools, scripts, setup

| Finding | By | Severity | Verdict, fix |
|---|---|---|---|
| setup.ps1 dot-sourced get-autoround.ps1, whose `param()` reset setup's own `-Yes` (tested: True, then False): unattended runs stopped at prompts | D, C | medium | real: the functions moved to `scripts/autoround.ps1` (no param block); get-autoround.ps1 is the standalone command |
| Deleting the AutoRound shards resolved a symlink's relative target (`..\..\blobs\x`) against the current folder: the blob stayed, and an unrelated file there could have gone | D, C | medium | real: resolved against the link's folder, deleted only inside that repository's `blobs` folder; the size reported is what was removed |
| The new free-space check threw on UNC and relative cache paths (tested), so every download failed there | D, C | low | real: best effort, in hf-get.ps1 and setup.ps1 (UNC through `Scripting.FileSystemObject`) |
| `HOME="~"` made the server's startup line loop for ever | D, C | low | real: the search goes on after the replacement (server_startup.h) |
| The importer's progress was discarded (`$null = Install-AutoRound`) | C | low | real: `Out-Host` |
| setup's help said flash-ar4 keeps ~96 GB; it keeps ~207 GB | C | low | real: text |
| The importer read its JSON files in the ANSI code page | C | low | real: `encoding='utf-8'` |
| `zpack::Writer::close` ignored `fclose`; the packer left a `.tmp` on failure | D | low | real: `close()` throws on a failed close; the packer removes its `.tmp` |
| A Q5_1 row whose K is not a multiple of 32 leaves its tail unwritten | D | low | not a bug: not a valid GGUF row; every block decoder behaves alike |

## 3. Contributor pull requests

| Finding | By | Severity | Verdict, fix |
|---|---|---|---|
| In precise mode (`--precise`, and every AR4 model) the engine turns int8 attention off, but the server opened its system-prompt cache and saved conversations by its own option: nothing was saved, and `load_state` failing made the server delete the files the same model had saved in its fast mode | D, C | medium | real: the caches follow the engine's mode and are off in precise mode, with one note; the `--ctx auto` key records precise mode (server_load.h, server_catalog.h) |
| `ExpertStore::valid()` compares the layer table with `memcmp`, including 4 padding bytes: on GCC a matching expert file can be refused (the AR4 package on Linux) | D, C | medium | left: pull request #24's fix; both reviewers judge #24 safe to merge |
| `HCK` (10240) is fixed in ops_flashnext.cl and `hc_n × n_embd` was not checked: another shape overran the mixers' local rows | D | medium | real: the shape check requires 10240 (engine_setup.h) |
| `load_state` did not refresh the PLE ids, as #15 made `restore` do | D | low | real (unreachable today: every caller forwards first): `ple_sync()` |
| #24: the Linux one-at-a-time shard teardown stops at a shard whose free throws; the EFAULT staging stays on for that slot | C | low | for #24's author |

Verdicts on the pull requests: #14, #15 and #20 are correct as merged; #21 and #22 only have the findings above; #24
is safe to merge once its CHANGELOG conflict is resolved (both reviewers).

## 4. Linux

| Finding | By | Severity | Verdict, fix |
|---|---|---|---|
| The PDF page count was the first "Pages:" anywhere in pdfinfo's output (a title can hold it) and unbounded before a per-page vector | D, C | medium | real: the last "Pages:" at a line's start, at most 100,000 |
| Terminal shells and MCP servers outlive a killed server (no job object on Linux) | D, C | low | left: comments and docs/linux-port.md say what is left and how a service manager covers it |
| The terminal's `drain()` read while output kept coming: `yes` or `cat /dev/zero` held the request past its timeout and Stop | C | medium | real: at most 1 MB a call, the deadline and cancel checked between calls |
| Project files' and file tools' dates came out around the year 1453 (MSVC's file-clock offset) | C | medium | real: `clock_cast` on Linux, Windows unchanged |
| ffmpeg / ffprobe decoded untrusted bytes with every protocol allowed | C | medium | real: `-protocol_whitelist pipe` / `file` |
| An MCP child that stops reading blocked `send()` holding `write_mu_`, and `cleanup()` took it first: a request thread and `/v1/mcp/stop` hung | D | medium | real on Linux: a non-blocking write end with a stop flag. Windows has the same, left (it needs `CancelSynchronousIo` on the writer) |
| A bad MCP `cwd` gave a misleading message | D | low | real: the folder checked before the spawn |
| A short `pread` under `O_DIRECT` retried unaligned | D | low | real: retried from the block's start |
| `FileMap::open` blocked on a FIFO | D | low | real: `O_NONBLOCK`, regular files only |
| The terminal's temp files were guarded only after all three were made | D | low | real: each guarded when made |
| The expert-file writer has no fallback when `O_DIRECT` fails at write time | C | low | left: filesystem dependent, not seen |
| Two `run_command` calls of one chat lose one's session state | D | low | left: the state file is per chat, last writer wins |

## 5. Web UI

| Finding | By | Severity | Verdict, fix |
|---|---|---|---|
| Deleting a chat with queued messages: the stopped answer sent the next one and saved the chat again | C | medium | real: its queue (and Talk results, a Compact) dropped first |
| A message sent during Compact was queued and never sent | D, C | medium | real: the queue drained after compaction |
| The project folder picker joined paths with `\`: no subfolder on a Linux server | D, C | medium | real: the path's own separator; "Use" off after an error |
| Edit / Delete / compaction treated an image a tool showed the model (`fetched`) as a new turn | C | medium | real: `userTurn()` in the four places |
| `openConv` drew a chat under another's id when a project loaded meanwhile | C | low | real: re-checked before anything on screen changes |
| Enter sent the message during an IME composition | C | low | real |
| Minutes PDF re-enabled Minutes DOCX during a form fill | C | low | real: both buttons, restored in a `finally` |
| "Conversations at once" offered 1-4, the server 1-8 | D | low | real |
| `/props` without `top_p` / `top_k` threw in refreshModel | D | low | real: guarded |
| Editing a voice message drops its recording | D | low | not a bug: the edited text replaces what was said |

## 6. Multi-GPU failsafes

| Finding | By | Severity | Verdict, fix |
|---|---|---|---|
| Unloading after a failure: a shard free over 10 s was given up on and its thread then wrote into the freed `TpEngine`; with no earlier failure a GPU that stopped while freeing hung the unload for good | C (use after free), D (hang) | high | real: the frees through pointers taken out first, one limit (`ARC_TP_TIMEOUT` + 30 s); a thread given up on touches nothing of the model when its job ends (`TpEngine::Life`); the shards' construction too |
| `ARC_TP_OVERLAP`'s sum thread waited for without a limit: in serverz nothing watches shard 0, so a stuck one hung the server | C | medium | real: the same limit; the model is then faulted and left |
| The expert prefetch thread's copy waits had no limit | C | medium | real: its shard's GPU and that GPU's limits |
| The 20-s link test freed its buffers while the timed copy was still queued | C (twice) | medium | real: they and the model are left, as after a GPU fault |
| The direct sums' inboxes were in no memory plan and took the room planned for the first pass's sum buffers | C (twice) | medium | real: the sum buffers made first (`tp_reserve_sums`); the inbox takes what is left or the sums stay in host memory |
| `~XCache` could throw `GpuWaitError` from a destructor | C | medium | real: caught; a busy GPU's memory left |
| The refusal after a leaked fault cleared that model's release flag | C | low | real: kept while a leaked fault exists |
| A leaked model's caches stayed registered, so GPU 0's out-of-memory hook shrank a dead cache | C, D | low | real: a one-GPU engine registers its own for its GPU |
| GPU 0's waits do not leave when another shard fails (docs said all do) | D, C | low | real (docs): GPU 0 is shared with Whisper and Qwen3-ASR |
| `ARC_FAKE_OOM`'s counter raced between two threads | D | low | real: atomic |
| `ARC_TP_PIPE=0` sums through host memory although the direct path is chosen | D | low | not a bug: documented in multi-gpu.md |

## Checks after the fixes

- **Logits bit-identical** to the runs before the fixes, prompt + 8 generated tokens (`runz --dump-logits`, SHA-256):
  - Qwen3.5 4B, Qwen3.8 27B, Qwen3.6 35B-A3B;
  - Flash Next IQ3_S: fast mode, `ARC_FN_PRECISE=1`, and over 2 GPUs (`ARC_TP_SIM`);
  - `ARC_FAKE_OOM=400` on one GPU and over 2: the caches shrink, and the logits are those of the runs without it.
- **The failsafes**, the laptop with `ARC_TP_SIM`, each run under an outer time limit:
  - The 4B over 2: a plain run.
  - `ARC_TP_HANG=1:5:wait` (a GPU wait that gives up): an error, then a clean exit.
  - `ARC_TP_HANG=1:5` (a GPU that stops for good): given up on after `ARC_TP_TIMEOUT=5`, then the exit with the
    stuck thread left.
  - `ARC_TP_OVERLAP=1` with `ARC_TP_HANG=0:20` (shard 0's sum thread stops): an error and an exit, where it waited
    for good before.
  - The 27B over 2 with `ARC_TP_OOM=1` (a shard short of memory at load): the load refused after 25 s with the
    allocation's error, and shard 0 freed through the new path.
- **Builds:**
  - Windows (MSVC): every target of the CI's list, the AR4 tools and tests; no new warning.
  - Linux (WSL Ubuntu 24.04, g++ 13.3, `-Wall -Wextra`): runz, serverz, test_gguf_pack, test_http,
    test_autoround_blocks, import_autoround_pack. The warnings left are older than this audit.
- **Tests:** the CI's unit tests (Windows), test_hc, test_fa, test_route, test_gguf_pack on the tiny DeepSeek model,
  test_autoround_decode on synthetic AR4 data, and test_http plus test_autoround_blocks on Linux.
- **Scripts:** parse checks; the dot-source and free-space behaviour tested in PowerShell 5.1 and 7.

Not run here:

- the web UI in a browser (its script parses with `node --check`);
- the Linux terminal, MCP, ffmpeg and PDF paths at run time (docs/linux-port.md lists the checks);
- the AutoRound download.
