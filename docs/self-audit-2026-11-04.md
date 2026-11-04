# Self audit, 2026-11-04 (night): every kernel, the web UI, audio and speech

Six static reviews ran in parallel:
- four over the 15 kernel files (`zero/kernels/*.cl`), each kernel checked against its host launch: sizes, argument
  order, `-D` options and buffer sizes;
- one over `webui/index.html` together with the server routes it calls;
- one over the audio and speech code (`src/audio.h`, `zero/asr_qwen3.h`, `zero/audio_qwen3a.h`,
  `zero/whisper_engine.h`, `tools/server_whisper*.h`).

Each finding was then checked in the code before it was fixed.

**Kernels: no high-severity bug.** Every kernel finding is latent: it needs a shape, a build option or a debug
switch that no shipped model uses today. The fixes turn them into clear errors, or remove the hazard without
changing the arithmetic. Logits are bit-identical to the commit before. The checks are in the table at the end.

## Fixed

### Kernels and their launches

| Where | Finding | Fix |
|---|---|---|
| `decode.cl` `decode_w8_*` | A W8 decode with N not a multiple of 16 writes past the staging buffer. | `decode_w8` / `decode_w8_x` refuse such an N. |
| GEMV views (`engine_load.h`, `engine_stream.h`) | With an odd superblock count, a Q8_0 GEMV reads up to 304 bytes past the last row. The views declared only 256 bytes of slack. | Views carry `kRawSlack` (1 KB), as the parent buffers do. |
| `decode.cl` | The `MT_ACT*` macros and `ARC_MT_LAZY_MIN` were used nowhere. They only made duplicate kernel builds. | Removed. |
| `gemm.cl` `-DMOE_TILES` | The kernel silently assumes TM 16, ACT32 and no mins; it would give wrong tokens with TM 32. | `#error` for any other combination. |
| `moe.cl` `moe_route` | More than 16 experts per token would drop choices 16 and above. | `#error` when NU > 16. |
| `decode_moe.cl` window paths | Masked lanes decoded the next superblock: possibly cache garbage under XCACHE, where a NaN times 0 is still NaN. | Masked lanes decode the current, valid superblock. The sums are unchanged. |
| `decode_moe.cl` `moe_down_q` | Mins were left uninitialized; harmless today. | Initialized to 0. |
| `ops_attn.cl` `attn_q8` / `attn_ps` | A work-group size that does not divide the K/V tile leaves part of the tile unloaded. | `#error`. |
| `engine.h` `ARC_ATTN_DEC_SMAX`, `ARC_QWEN_DS_S` | A value that is not a multiple of 16 gives a fully masked tile, and NaN for the whole head. | Rounded up to 16. |
| `engine_drafters.h` `dfl_attn` | A drafter window longer than the 4096-position ring would read overwritten keys. | Refused with an error. |
| `audio_qwen3a.h` | With a small `--chunk` and two-pass weights, the encoder GEMM divided by zero. | A clear error. |
| `whisper_engine.h` `ARC_WH_ARSG` | On 16-lane-only GPUs this debug switch could make work-groups of half a sub-group. | Rounded to whole sub-groups. |

### Speech and the transcript library

| Where | Finding | Fix |
|---|---|---|
| `server_whisper.h` start / `work()` | A job that ends the whole process (a lost GPU) was queued again at every start: a crash loop. | An `attempts` count on disk. After two, the job is marked failed with a message. |
| `asr_qwen3.h` | A part that throws left the engine bound to slot k. When cancelled, cut parts counted as ended. | `bind(0)` in a scope guard. A cancelled part is marked cut. |
| `server_whisper_asr.h` `chat_run` | An exception while building a part's WAV inside a `std::thread` would end the server. | The whole part is in try/catch. |
| `server_whisper.h` live audio | A lost piece moved the rest of the recording earlier on the timeline. | Silence fills the gap (at most 2 minutes). |
| `server_whisper.h` live audio | An utterance could be queued after the recording's finish task, then write the transcript after the accurate pass. | Queued in the same hold of the lock. |
| `server_whisper.h` merge | A merge went on for hours after its transcript was deleted. | Checked in every decoded block. The partial WAV is removed. |
| `src/audio.h` `decode_mono_pcm16` | The merged audio ignored an MP4's late start and timestamp gaps, so a piece's transcript drifted against it. | Placed on the same timeline as the transcript's decode. |
| `server_whisper.h` `clip_model` | Read `asr_id` without a lock while unload could change it. | Reads the atomic snapshot. |
| `server_whisper.h` `set_loaded` | A lost wakeup: queued jobs could wait up to 60 s after a speech model was loaded again. | `lib_mu_` is taken before the notify, with the engine lock released first. |

### Web UI

| Finding | Fix |
|---|---|
| After "Unload" of the chat model the page still took it as loaded. Effort levels, vision and the help texts stayed wrong until a reload. | The page treats no model as loaded. The next answer reads the capabilities again, as after a switch. `/v1/models` lists the unloaded model's file first, and the page keeps its own model. Found in the browser test: the page took the catalog's first model, the 27B, after the 4B was unloaded. |
| A U+0000 character in an answer could put a kept HTML piece inside a link's `href` (HTML injection). | The marker character is removed from the text first. |
| Changing any transcript option erased the saved recording source and minutes form. | The other preferences are kept. |
| The chat model's button stayed "Unloading…". | Its text is reset. |
| A compaction error showed raw JSON and no unload buttons. | It shows the server's message, and the unload buttons when the model did not fit. |
| An upload whose last answer was lost was reported as failed although the server had the whole file. | The retry asks the server and treats a queued file as sent. |
| A Talk result arriving while the OCR save question was open could start a second job with the same id. | The job counts as running before the question. |
| Summarize and the minutes buttons could be started twice while the model loaded. | Disabled before the first wait. |
| The minutes form threw on a cancelled transcript, on a template without a first table, and on a label listed twice. | Each guarded. The server always returns `fields`. |
| Rename threw an unhandled rejection when the server could not be reached. | It shows "not renamed". |
| Talk could start during a recording, putting its answers into the meeting's microphone track. | Refused while recording. |
| A pause-joined utterance could bring back the message it replaced. | No splice while the stopped answer is still ending. |

## Not changed (noted)

- **Kernels:**
  - the `-DSGN_MASKS` branches in `decode.cl` are never built (an old experiment);
  - `ops_vit.cl`'s `HD` default is never used, because `ops_attn.cl` defines `HD` first; every caller passes `-DHD`;
  - in the MOE_VB builds of `decode_moe.cl`, local arrays are declared but unused (the compiler probably drops them);
  - `moe_rg` stages the codebook in every 16-lane work-group;
  - `wh_self_attn`'s uncoalesced key reads.

  The last three are inefficiencies to measure before changing.
- **Audio:**
  - a merge piece kept as a WAV over 4 GB (3+ hours of 96 kHz, 2 tracks) is read by Media Foundation, which stops
    at 4 GB;
  - `path.string()` with names outside the code page, safe while every program embeds the UTF-8 code page manifest;
  - very short parts between pauses (0.1-0.3 s) are transcribed and sometimes invent text.
- **Web UI:** a long live recording redraws its whole transcript twice a second.
- **The DEVICE_LOST crashes of 2026-11-04.** No review found an out-of-bounds access or a runaway launch in the
  Qwen3-ASR encoder or the attention kernels at 160-s parts. The crash-loop fix above keeps one bad recording from
  bringing the server down again at every start.

## Checks

`runz --dump-logits` (the prompt and 48 generated tokens) from the commit before (9c16e23, built in a worktree)
against this one, compared by hash:

| Model | Plain | `--mtp` |
|---|---|---|
| Qwen3.5 4B Q4_K_XL | bit-identical | (no MTP in this file) |
| Qwen3.5 9B Q4_K_XL | bit-identical | (no MTP in this file) |
| Qwen3.8 27B IQ3_S | bit-identical | bit-identical |
| Gemma 4 E4B | bit-identical | bit-identical (drafter) |
| Gemma 4 12B | bit-identical | bit-identical (drafter) |
| Gemma 4 26B A4B (MoE) | bit-identical | – |

The Gemma drafter builds pass the new `#error` checks.
