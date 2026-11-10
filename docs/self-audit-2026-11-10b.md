# Self audit 2026-11-10 (evening): after the crash at a meeting's transcription

What was audited: the whole engine and the speech server, for bugs of the class that ended the server at the start
of a meeting's transcription (a kernel argument cached across a free, fixed in commit 97d225b), for code that trusts a
size or index without checking it, and for the server's threads and state machines.

How: three independent read-only reviews, one per area (stale caches and lifetimes; unguarded sizes and shapes; the
server's threads and the transcript / recording states). Every finding was checked in the code before it was fixed.
Then a rebuild, every model on the laptop loaded and run, the 4B's logits compared bit for bit, and the speech paths
run.

## Findings and fixes

| # | Severity | Where | What was wrong | Fix |
|---|---|---|---|---|
| R1 | medium | `tools/server_whisper.h` `live_finish` | A live recording's accurate pass had no record of the draft before it: a pass that was stopped, or whose part failed twice, wrote its partial text over the whole meeting's draft in `transcript.json` | the draft's record is kept as `redo_prev`, as a redo does: a stopped or failed pass keeps the draft |
| R2 | medium | `tools/server_whisper.h` (live audio) | One failed write of a live recording (disk full) left the WAV stream in its error state: every later piece failed, even with room again, and the rest of the meeting never reached the server | the stream's state cleared and its position set back to the end of the whole frames: the page's retry is written once there is room |
| R3 | low | `tools/server_whisper.h` merge | A recording whose MP3 / M4A compression failed, or was cut by a restart, was refused for merging for good ("still being compressed") | refused only while its job runs |
| R4 | low | `tools/server_whisper_asr.h` `chat_run` | A thread that failed to start left the started ones unjoined: `std::terminate` | the started ones joined, then the error passed on |
| L1 | low | `zero/engine_sessions.h` `ensure_stage` | New staging buffers kept the role's "already decoded" mark: a weight decoded before could be read from the new, empty buffers (held off only by the order of the calls); the old pair leaked | the old pair freed, the mark dropped |
| L2 | low | `zero/clrt.h` | The crash log's "launch being appended" pointed at a kernel already released (a model switch): the log could read freed memory | `clReleaseKernel` clears it; the log also says when an argument was set before a later free |
| L3 | low | `zero/engine.h` `read_gbs` | Its kernel was never released, and its module was destroyed before it | released after the measurement |
| S1 | high | `zero/engine_load.h` `load_w_pad` | A tensor with more rows than the model's dimension (Gemma 4's FFN, the Qwen MoE shared expert) was copied into a buffer sized for fewer: heap overflow while loading | refused with the tensor's name |
| S2 | medium | `zero/engine_load.h`, `zero/engine_stream.h` | Dense weights and norms were not checked against the model's dimensions: a matrix wider than the buffer its product writes went past it on the GPU | `expect()` on attention q / k / v / output, DeltaNet's output, the FFN; norms through `load_vec()` with their length |
| S3 | low | `zero/engine_setup.h`, `zero/clrt.h` | `expert_used_count` 0 was accepted (a zero-size launch); a global size past 32 bits was truncated | at least 1 (and at most the experts); a zero-size launch runs nothing, one past 32 bits is refused |
| S4 | low | `zero/engine_setup.h` | `full_attention_interval` indexed the QSA ratios before it was checked | checked first |
| S5 | low | `zero/engine_stream.h`, `zero/expert_store.h`, `zero/asr_qwen3.h` | A tensor's `dims[0]` / `dims[1]` read before its number of dims was checked (GGUF allows 0) | the dims checked first |
| S6 | low | `zero/engine_stream.h` | Gemma 4 26B's router and expert scales read without checking their length | their lengths checked |
| S7 | low | `zero/whisper_engine.h` | Four copies to and from the GPU without a check of their result | `CL_CHECK` |
| T1 | tidy | `zero/clrt.h`, `zero/engine_load.h`, `zero/xcache.h`, `tools/server_whisper.h` | A comment on the wrong function, stale comments ("no bounds checks", "no sub-buffers of it", "with the OpenAI endpoint's lock"), `k * 0 + s` | fixed |

## Checked and clean

- **The argument cache after the fix:** every engine free goes through `zrt::mem_free`; the places that free and
  allocate again while kernels are cached (Whisper's buffers, the f16 XMX GEMM's, the XMX check, the vision and audio
  encoders' `reserve()`) are covered by its count.
- **The group-size cache, `views_` / `from()`, slots (`point_to`, `roles_`), staging, `maps()`, `ARC_BATCH`, the
  kernel caches, the expert cache's `shrink()`, the out-of-memory hook, multi-GPU teardown, thread-local flags.**
- **The GPU from several threads:** every speech path runs under `turn_` and the GPU turn, chat under the engine's
  lock, model switches under `swap_mu`; no path appends to the one immediate command list from two threads, none
  frees an engine in use; no lock-order inversion.
- **Input parsers:** GGUF (counts, strings, dims, sizes, offsets), WAV / MP4 / Ogg, JSON depth and numbers, HTTP
  bodies and ranges, images, the live recording's frames and offsets, device-side bounds of every copy.

## Left as they are

- `free_buf` frees the views of the buffer itself, not views of those views (none exist on a freed buffer today).
- `h8_col_` / `h8_res_` are keyed by the weight's address: an encoder freed and made again within one engine's life
  could find another's H8 scales (encoders live as long as their engine today).
- The crash counter (`attempts`) counts a crash against the job that was running, whatever caused it; a queued job
  whose speech model was unloaded fails instead of taking the other one; an abandoned upload stays "uploading" until
  the next start.
- The Qwen3-ASR encoder checks q, up and down against its dimensions but not k, v and out; `load_w_stack` does not
  go through `mat()`.
- Shutdown code that never runs (`http.run()` does not return), four WAV header writers, duplicated helpers in the
  transcript library.

## Verified after the fixes

- Build: runz, serverz, asr_testz, transcribez without errors or warnings.
- Qwen3.5 4B: logits bit-identical, plain and `--mtp` (SHA-256 FA2AA877... / 716DD3BF..., as before).
- Loaded and answered: Qwen3.8 27B, Qwen3.5 9B, Xiaomi-OCR (BF16), Gemma 4 12B; Gemma 4 E4B KL 0.0039 (20 / 20 top
  tokens), Gemma 4 26B 0.0101 (19 / 20); Qwen3-ASR after a warm-up (120 s); Whisper (163 s, 14.6x real time).
- R1 end to end, through the server's API: 5 minutes of a meeting's track sent as a live recording (a draft of 70
  segments), the recording stopped, its accurate pass cancelled while it ran: `transcript.json` kept the 70
  segments, status done, "transcribing it again was stopped: the earlier text is kept".
- The 35B-A3B and Flash Next files were not on this laptop (their drive was unplugged): their attention and DeltaNet
  outputs write `n_embd` columns as the engine's GEMMs assume, which `expect()` checks.
