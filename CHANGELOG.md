# Changelog

## 2026-10-31: speech to text without a chat model

- **`serverz -m none`** (`arc-server start -Model none`, the launcher's **Speech only** button): only the speech
  service loads (Qwen3-ASR alone, ~3.2 GB of GPU memory; `--whisper auto` adds Whisper), for transcripts through
  the web UI's Transcripts tab or `POST /v1/audio/transcriptions`. `/health`, `/props` and `/v1/models` answer with no
  model loaded (the model started last listed first, none `loaded`), and the first chat request naming a model loads
  it, context auto next to the speech models. A 2:43 voice message: 25 s.
- **The first transcript after a start no longer waits for the encoder**: Qwen3-ASR's audio encoder loaded its
  weights and kernels on its first clip (60 s of audio: encoder 4.1 s instead of 0.5); now when the server starts.
- **Web UI recordings: the MP3 bitrate is a choice** (Transcripts tab, "Keep as MP3 96 / 128 / 160 / 192 / 256 /
  320 kbps", or WAV) instead of a fixed 192. Windows' MP3 encoder takes at most 128 kbps for one track (a
  microphone): higher choices give 128 there; a meeting's two tracks get the bitrate chosen (checked: 96, 192, 320).
- **Recordings kept as M4A (AAC)**: "Keep as M4A (AAC) 96 / 128 / 160 / 192 kbps" next to WAV and MP3 (`POST /v1/live`
  `keep: "m4a"`), with Windows' AAC encoder through Media Foundation as the MP3 (on the CPU: audio encoding is
  sequential and light, and the GPU has no audio unit). AAC takes 192 kbps for one track too (MP3 stops at 128);
  60 s at 96 / 128 / 192 kbps: 0.73 / 0.97 / 1.45 MB, mono and stereo, played back by the page (audio/mp4).
- **Mono or stereo recordings**: the Transcripts tab's new Mono / Stereo choice (default Mono). Stereo keeps left
  and right of the one source as it plays them (a tab's music or video; a stereo microphone, then without the
  browser's echo and noise suppression; `POST /v1/live` `channels: 2` with one track), so the MP3 gets the bitrate
  chosen; the transcript is made from its mono mix (60 s: the same 164 tokens as the mono recording). A source that
  reports one channel stays mono; the Meetings tab keeps one mono track per source.

## 2026-10-30 (night): multi-GPU groundwork

- **bin\mgpu_probe.exe** (zero/mgpu_probe.cpp, in the package): lists the GPUs and measures what moving data between
  them costs: peer access, a 20 KB transfer (a tensor-parallel sync of token generation, ~128 per 27B token) and a
  40 MB one (a prompt chunk), GPU to GPU directly and through host memory, on the compute queue and the copy engine.
  Its numbers decide how a multi-GPU arcinferzero splits the work. On this laptop (one GPU, to itself): 20 KB 5 us
  on the compute queue, 70 us on the copy engine; through host memory 11 us.
- **bench\gemm_sweep.ps1**: sweeps the prompt GEMM's settings (XMX on / off, tokens per sub-group, row groups,
  work-group height, split-K, chunk size) one at a time from the defaults with `runz --bench-pp`, then the best of
  each together, and prints which path ran (XMX, or DP4a when the XMX check failed). For B-series owners to report
  what their GPU prefers: the XMX kernel was written without an XMX GPU at hand (on a B60 the 27B's prompt runs at
  ~12 % of the card's int8 peak). Reading it for this: each DPAS sums only 32 values before the per-token unpack and
  the per-16 / per-32 scales, and the per-16 layouts fill half of every DPAS tile with zeros, so the large gain on
  XMX needs a kernel that accumulates longer runs, beyond what these settings reach.
- **runz compares its GB/s with the GPU it runs on**: it measures the GPU's read bandwidth (Engine::read_gbs, a
  256 MB block-read stream) instead of assuming this laptop's ~75 GB/s (measured here: 70 GB/s).

## 2026-10-30 (evening): faster tool-call turns on the 27B

- **Short prompts of just-in-time models in passes of 8 rows**: a chunk of a model whose weights are decoded just in
  time (the 27B) decoded every matrix first, ~2 s however few tokens it held, so every tool-call turn waited ~2 s
  before its first token. Up to 40 new tokens now go through the generation's multi-row GEMVs (~0.35 s per 8):
  a tool turn's prompt 2.3 -> 1.3-1.4 s. DFlash2's capture writes each pass at its own rows (it overwrote the first
  rows and the drafter fell back to MTP).
- **Gemma 4 12B the same up to 32 tokens** (its two-pass passes cost more; past ~32 the chunk is faster again): 16 /
  22 / 27 new tokens 1.1 / 1.2 / 1.2 -> 0.6 / 0.7 / 0.9 s, replies byte-identical. Qwen 9B (resident weights) and
  the 4B, E4B and MoE models are unaffected.

## 2026-10-30 (afternoon): a progress bar while the model loads

- **arc-server and the launcher show how far the load is**: the server writes `progress: P % <step>` lines to its
  log while it loads (src/load_progress.h): Qwen3-ASR's and the model's weights (the bytes uploaded, without the
  experts streamed from the SSD), Whisper's layers, the expert file written on a first start, the expert cache filled
  from the SSD, and the kernels' warm-up (no percentage: a busy bar). `arc-server start` draws them as a PowerShell
  progress bar with the seconds so far; the launcher has a bar under its status line and the step in the message
  line (a switch reads the log only once the new start has rewritten it, not the stopped server's lines). The
  command-line tools (runz, transcribez, asr_testz) print none.
- Self audit of today's changes, fixed: the web UI no longer follows Gemma 4 E4B when the speech service loads it for
  a transcript (the chat keeps its model); `--asr` without Whisper that cannot load Qwen3-ASR stops the server with
  the reason instead of advertising speech to text with no model behind it; the expert file's per-layer line is
  back for the command-line tools; the weights bar leaves out the tensors kept on the host (E4B's stalled at ~60 %).

## 2026-10-30 (later): Qwen3-ASR in 3.15 GiB

- **Qwen3-ASR's encoder on two int8 passes** (its blocks and projector; the convolutions stay f16): 0.037 % from
  the f32 encoder instead of 0.059 % (the int8 GEMM reproduces the mmproj's Q8_0 blocks exactly, f16 rounded them),
  34 % faster, 0.26 GiB less; the f32 text model's KL over a transcript max 0.000002, the same text. `ARC_ASR_Q2=0`:
  f16. On the way: in two-pass engines the encoder's quantization wrote its residual at the engine's offsets (the
  ops program carries them), so the encoder now uses the engine's layout there.
- **One Qwen3-ASR slot by default** (`--asr-slots N`, new `arc-server -AsrSlots N`, 1-4, kept like `-Asr`): each
  slot past the first ~0.6 GiB, faster only for clips over 4 minutes (a 14-minute file 129 s with one, 80 s with 4).
  Qwen3-ASR alone in the server: 5.37 -> 3.15 GiB; the 35B keeps 55 % of its experts cached (40 % before, 10 % with
  Whisper too).
- `asr_testz --kl-files P` and `ref_forward` `REF_EMB` for the Qwen models: the encoder's effect on the text model's
  KL.
- **The web UI shows the model the server has loaded**: a server started on another model than the one used last
  (arc-server, pi-arc, Pi) still showed the earlier one, because opening a chat took the model saved with it (the
  page reopens the last chat), and the next message then switched the server back to that model. The page now
  follows the loaded model unless a model is picked in its menu (that pick stays until the server loads another
  one), and asks the server again whenever the page is shown again.
- **Whisper large-v3 in 3.94 GiB instead of 4.39** (NPU mode 3.28): the decoder layers' cross keys / values share one
  f32 buffer before they are packed to f16 (it was one per layer); logits and transcripts byte-identical.

## 2026-10-30: Qwen3-ASR without Whisper; the terminal asks once

- **Qwen3-ASR alone**: `--asr auto` (or a GGUF) without `--whisper` (`arc-server start -Asr auto`) starts the speech
  service without Whisper: 5.37 GiB of GPU memory with its 4 parts at a time instead of 9.76 with Whisper, so the 35B
  kept 40 % of its experts cached instead of 10 %. Transcription requests, the Transcripts and Meetings tabs and chat
  audio go to Qwen3-ASR (a 2:43 voice note in 29 s, SRT included). What needs Whisper says so: translations and a
  request naming Whisper get 400; a live recording keeps recording and gets its text from the pass after Stop (none
  while recording; the web UI notes it). The web UI hides and disables "Model: Whisper" then (`/props` "whisper":
  {"whisper": false}). arc-server keeps `-Asr` on a restart with or without `-Whisper`.
- **The web UI's terminal asks once per session**: Settings > Terminal is now **Ask once per terminal session** (the
  default: the first command of a session waits for Run), **Ask before every command** or **Model never**. The old
  default asked before every command, and "Run without asking" in fact asked once: both saved choices become the
  first. The calls of one answer that run at once wait for the first one's Run instead of each asking (three
  `run_command` calls in one answer: one Run, all three ran; the next answer's command ran without asking).

## 2026-10-30: Qwen's reasoning with a presence penalty

- **Qwen models reason with presence_penalty 1.5** (Qwen's recommendation for thinking) when the request sets none:
  the Qwen3.6 35B, asked in the web UI to turn a sprite sheet into a GIF, circled its plan ("Let's run.", "Wait, ...")
  for 7,000 tokens without closing `</think>`, so its `run_command` call never came (17 minutes with Whisper and
  Qwen3-ASR loaded beside it). `</think>` was never held back: at those points it was the most likely token (~50 %),
  and temperature 1.0 drew another. The penalty applies to the reasoning only; the answer and its tool calls keep
  the request's value (0 by default), so code is not penalized for reusing its tokens. A request's own
  `presence_penalty` applies to both, as before. Gemma 4 unchanged.

## 2026-10-29 (evening): a ready-to-use package for people who never open a terminal

- **Downloads in a Windows PowerShell started from PowerShell 7** (GitHub issue #3: "The term 'Get-FileHash' is not
  recognized" at the SHA256 check): the hash is computed with .NET, and setup.cmd, arc-launcher.cmd and the pi-arc /
  arc-server commands clear the inherited PSModulePath, so Windows PowerShell loads its own modules.
- **setup (the installer from the sources)**: a model the server already finds outside the Hugging Face cache (a
  package in `.arcinferzero\zero\models`, a `model-dirs.txt` folder) counts as there (the packaged 35B was offered for
  download again); `setup.cmd` ends with setup's exit code (it said 0 after a failure) and, double-clicked, keeps its
  window open at the end; the last lines point to `arc-launcher.cmd`.
- **Precise mode as an option**: `arc-server start -Precise` / `-NoPrecise`, `pi-arc --precise`, the launcher's
  "Accuracy" box (Flash Next only) and the server's `--precise` (the same as `ARC_FN_PRECISE=1`, which works as
  before); kept on a restart and a model switch like `-Npu`, reported by `arc-server status` and in the server log.
  `arc-server restart` without `-Model` now restarts the running model (it kept the server without restarting it).
- **Flash Next starts with images through arc-server, the launcher and pi-arc**: the model list did not look for a
  projector next to a `qwen4exp` model, so it listed Flash Next without images and arc-server passed --no-mmproj.
  Now it lists `qwen3.8-flash-next-iq4_xs` (images, the default) and `...-std` (none).
- **The launcher sets everything up**: unpack the zip, double-click `arc-launcher.cmd`. The first time, a window
  offers a model in plain words (the one that suits the laptop's RAM ticked: 4B on 16 GB, the 35B on 32 GB), speech
  to text as an add-on and a desktop shortcut; "Download and start" downloads it with a progress bar, starts it and
  opens the web UI. "Get models" adds more later; "Open the web UI when started" opens it after every start. The
  package's files are unblocked once (a zip from the internet marks them), and a missing Intel driver is named with
  Intel's download page (scripts\launcher-firstrun.ps1).
- **No compiler, no Python, no Visual C++ Redistributable**: `scripts\make-package.ps1` makes
  `dist\arcinferzero-<version>-win64.zip` (6 MB: every tracked file, the built exes, `bin\PREBUILT.txt`); setup
  keeps the shipped exes (`setup -Rebuild` builds them). The exes are built with `/MT` (the C++ runtime inside) and
  classic `/openmp`, whose `vcomp140.dll` may be shipped (build.ps1 copies it into bin); `/openmp:llvm`'s
  libomp140 may not. Logits bit-identical to the old build, speeds the same (4B: prompt 840-894 against 857-876 tok/s,
  generation 18.6-18.7 against 18.0-18.4).
- **Model downloads without Python**: `scripts\hf-get.ps1` (curl.exe, part of Windows: resumed after a cut, each
  file's SHA256 checked, the Hugging Face cache layout the server reads; HF_TOKEN for gated repositories) and
  `scripts\get-models.ps1` over one catalog for setup and the launcher (`scripts\model-catalog.ps1`). The 4B, 9B
  and 27B come with their image projectors now; `35b` is Unsloth's UD-IQ4_XS (MTP, images, no conversion) and
  `35b-gsq` the converted 2-bit file (the one step that still needs Python).


## 2026-10-29: the data folder is %USERPROFILE%\.arcinferzero; Qwen3-ASR keeps the language; Flash Next images

- The data folder (kernel cache, model packages and expert files, caches, logs, settings, chats, transcripts,
  recordings) is now `%USERPROFILE%\.arcinferzero`, a dot folder like the other tools' folders in the user's profile
  (`.cache`, `.pi`, `.vscode`). The first exe or script that finds only the old `%USERPROFILE%\arcinferzero` (or an
  older `%LOCALAPPDATA%` one) moves it: a rename on the same drive, instant, nothing is copied. The paths in
  `last-model.txt`, `launcher.json`, `server.json` and `zero\ctx-auto.txt` follow it, in any case (the context
  cache's keys are lower case; until now that file was not rewritten, so an earlier move left its entries behind and
  the contexts were computed again). `ARC_DATA_DIR` still overrides the place.
- The help's file lists line up again after the longer folder name.
- **Option mistakes say what is expected** (serverz, arc-server, pi-arc): a value missing (`--whisper` at the end, or
  followed by the next option), a file where a folder belongs or the other way round, a path with a stray quote (a
  quoted path ending in a backslash), a .gguf that is not one, an unknown option (with the closest one: `--wisper`:
  did you mean `--whisper`?; `-whisper` to serverz: it takes two dashes), each with the kind of value it takes.
  arc-server no longer ignores an option it does not have (`-Whispr on` started the server without speech), checks
  `-Whisper` / `-Asr` paths and drops a path's last backslash; pi-arc passes `--whisper`, `--asr`,
  `--whisper-device` and `--whisper-max-minutes` to arc-server (they went to Pi). The helps of both have WRITING
  OPTIONS (dashes, on / off, paths, what each path is) and WHEN IT FAILS (each message and what to do).
- **`--whisper on`** was taken for a folder named "on" and stopped the server with "cannot open model.bin". on / yes / true / 1
  now mean auto (off / no / false / 0: off), for `--whisper` and `--asr` in serverz and in arc-server, and a folder
  without a model.bin is named in the message. arc-server's help says that options take one dash or two (`-Whisper`,
  `--whisper`).
- **arc-server start without -Model and no console** (a script, a scheduled task, a tool with its input
  redirected, `-NonInteractive`): the model menu asked again on every empty answer, forever, at full CPU. Now a
  running server is kept, else the model used last starts (as Enter and pi-arc do), else it stops with a message.
- **Web UI on a phone**: tapping a chat in the list did not open it: the row's delete button appeared on hover, and a
  touch screen (iOS) takes the first tap that shows something as the hover alone. Touch screens now show the delete
  button (and a message's menu button) all the time; with a mouse they appear on hover as before.
- **Web UI lighter on the browser**: the transcript player's word highlight runs a frame loop only while the media
  plays (it ran 60 times a second all the time, so the page never went idle); a streaming answer is rendered again
  only when its text changed, at a gap that grows with the render's cost (8x its time, at least 120 ms, so a very
  long answer no longer takes the browser's time quadratically); the transcript list's rows are rewritten only
  when what they show changed.
- **Image prompts in one pass** (Qwen images on models without an MTP block or DFlash2 drafter, so Flash Next):
  the text and the image rows go through the layers together instead of a pass per piece, so the streamed experts
  are read once: prompt time -8 to -24 % (290-798 tokens with a picture), the same answers. `ARC_MEDIA_PASSES=1`:
  the old way. Flash Next's image encoder stays f16 (int8 two passes would be 26-28 % faster at KL up to 0.0098).
- **Flash Next reads images right**: an image's rows went into the model as single rows of its width, while Flash
  Next keeps every row as 4 hyper-connection streams, so four image rows shared one row and the PLE n-gram block read
  the previous text's rows. Each row now goes in as its 4 streams with the n-gram rows of `<|image_pad|>` (the model's
  `ple.image_token_id`). A test picture's "PROFIL RISIKO 2026", read as "PFILIKO 2013" before, is now read right.
  Its image projector: `mmproj-Qwen3.8-Flash-Next-f16.gguf` from bartowski/Qwen3.8-Flash-Next-GGUF, next to the
  model (or its package).
- **Qwen3-ASR keeps to the transcript's language**: the request's language (a code, or the first of a list such as
  `id,en`) is now forced (`language Indonesian<asr_text>`), and the Web UI's default is Indonesian. Detection named
  most parts of a 26-min Indonesian meeting Malay, and one part drifted into Malay words; forced Indonesian kept every
  English term the detected text had. A glossary (the Transcripts tab's field) spells the jargon as given and gives
  the punctuation back (docs/qwen3-asr.md).
- **Qwen3-ASR BF16** (`Qwen3-ASR-1.7B-bf16.gguf`) loads: the stacked Q / K / V and gate / up matrices refused BF16 /
  F16 / F32 matrices; they are now requantized to Q8_0 row by row, as a single matrix already was. The transcript is
  word for word the one of the Q8_0 file (the engine holds both as Q8_0), the load takes 17.6 s instead of 5.4 s, so
  the BF16 file brings nothing over Q8_0.

## 2026-10-28: the launcher, setup and the docs brought up to date

- **arc-launcher**: every arc-server option now has its field: speech to text (Speech, Qwen3-ASR, Whisper's encoder,
  the longest clip), NPU, conversations at once, images / audio, no MTP, no streaming, earlier turns' reasoning;
  Record / Stop recording buttons for meetings (when `bin\arc_recorder.exe` is there).
- **setup**: `-Models whisper asr` downloads the speech models (Whisper large-v3, Qwen3-ASR 1.7B with its projector).
- **Transcripts tab**: Accurate (beam 5) / Fast (greedy) shows only for Whisper: Qwen3-ASR and Gemma 4 E4B always
  decode greedily.
- **Docs**: what was written two or three times (setup, Pi, the launcher, troubleshooting, the web UI, NPU, speech)
  now lives in one place with links from the others; out-of-date statements corrected (chats are kept by the server,
  the speech models, files copied between chats); the finished Gemma two-pass plan moved to docs/archive/.

## 2026-10-28: an external audit checked and fixed

A further static audit (23 findings) was checked finding by finding in the code; all held and are fixed:

- **Model and file input checked**: a Whisper model whose tokenizer or alignment heads point outside the model is
  refused at load (it could write past a buffer); a PDF's stream length past the file no longer overflows; JSON
  with a lone surrogate gives U+FFFD, raw control characters (other than tab / newline) are refused; numbers from a
  request are clamped before they are converted (slots, top_k, max_tokens, offsets, live recording rates).
- **Files**: uploads are written to a temporary file and moved into place (a crash left a cut file under the full
  file's id); a file edited to the same size gets a new id (its time is kept); a file reused for an upload is
  compared byte for byte first; an extension longer than 16 characters no longer escapes the name limit.
- **Terminal**: a command never inherits the server's other handles; its output reader cannot end the server.
- **API**: `seed: 0` is a seed (it was taken as none); `top_k: 0` means no top-k; the last streamed token's
  logprobs are sent; web_fetch keeps one deadline over its redirects; `http://[::1]:port` pages are recognised;
  a chunked body with a broken chunk end gets 400; saved conversations with positions outside the context are
  not loaded.

## 2026-10-28: an audit of every source and document

Six reviews (server, file tools, speech, web UI, the GPU engine's host code, scripts and docs); what they found is
fixed ([docs/self-audit-2026-10-28.md](docs/self-audit-2026-10-28.md), "All sources"):

- **Other local pages are kept out**: a page of another server on this computer (another port: a dev server, a
  folder served by Python) could call the terminal tool, the slots, chats, files and transcripts. It now reaches
  only the model endpoints (chat completions, the audio ones, /tokenize); the web UI, Pi and command-line clients
  work as before. Text inside a fetched page, a file or your message that looks like the model's control tokens is
  read as text, no longer as a turn of the conversation.
- **PDFs that hung the server**: a broken font map looped forever, and some files took hours to parse; reading a
  PDF now has a work limit and keeps what it read with a note. Very long paragraphs, a huge word in a written PDF
  and long CSV fields no longer slow to a crawl.
- **Qwen3-ASR keeps the end of long recordings** (parts of 4 minutes, with room for their text; 6-minute parts lost
  their last minute), lets voice messages in between a queued file's parts, falls back to Whisper when it cannot
  load, and keeps the text before a part that failed. Two-track meetings come out in the order they were said.
- **Web UI**: a double click on Talk or the microphone no longer opens it twice (an echo and a microphone that
  stayed on); Compact, Stop, background chats in Talk and a chat's model after switching fixed.
- **A file attached in another chat is copied there**, not linked: a command run in one chat no longer changes the
  other chat's copy.
- `arc-server stop -Port N` stops only that port's server; `setup` downloads E4B's projector (images, audio); the
  launcher offers 1-7 drafts and chunks up to 8192.

## 2026-10-28: Gemma 4 E4B audio encoder checked against f32

- **E4B's GPU conformer matches an f32 reference** (new `tools/gemma4a_ref.cpp`, written from llama.cpp's
  `models/gemma4a.cpp` and `mtmd_audio_preprocessor_gemma4a`: log-mel in double, the two convolutions, 12 conformer
  blocks with every clamp, f32 activations, the mmproj's BF16 weights as f32). Rows: cosine 1.000000, relative error
  max 0.004 % (the 49.8-s test clip, 1,246 tokens; rec1 part_00 / part_03, 722 / 612 tokens). Rounding the weights
  to f16 as the GPU stores them changes nothing measurable. The model's KL, 48 forced reply tokens of the test clip:
  arcinferzero against the full f32 path (`ref_forward` REF_EMB on the reference rows) mean 0.00018, max 0.0054,
  top-1 48 / 48; the encoder alone (the GPU model on GPU rows vs on f32 rows) 0.00019 / 0.0047 (test clip), 0.00058
  / 0.0025 (part_00), 0.00068 / 0.0038 (part_03). Nothing to fix; E4B's noisy-clip mistakes are the model's.
- `vision_testz`: `ARC_AUDIO_EMB=file` runs the model on audio rows from a file (as `ARC_VIS_EMB` for images).

## 2026-10-28: Qwen3-ASR's attention with f16 K

- **f16 K rows in Qwen3's float attention** (Gemma 4 keeps f32): attention -10 %, generation +4 %, the two
  recordings' text 2-3 % sooner, the KV cache a third smaller (server -0.32 GiB); KL < 0.00001, the transcripts
  identical, Gemma 4 E4B / 12B and Qwen3.5 4B bit-identical. `ARC_Q3_KF32=1`: f32 K.

## 2026-10-28: Qwen3-ASR GEMV / GEMM; the speech code in its own files

- **Qwen3-ASR's q / k / v and gate / up each one matrix** (their Q8_0 rows stacked as they are): one token at 2048
  of context 23.8 -> 25.1 tok/s, the 2:34 recording's text 2 % sooner; logits bit-identical. Measured and left: rows
  per sub-group (within 1 %), the prompt GEMMs (11 % of a transcription, ~half the DP4a peak, shared by every
  model); the GEMVs run at 58-70 GB/s of ~75. The larger remaining parts of a token are the float attention (21 %:
  f32 K rows) and the 152K-word output head (12 %).
- **Speech code out of tools/server.cpp**: tools/server_speech.h (starting Whisper / Qwen3-ASR, Gemma 4 E4B for
  transcripts, `/v1/audio/*`), tools/server_whisper_asr.h (Qwen3-ASR / E4B inside the speech service); the Qwen3-ASR
  encoder's kernels in zero/kernels/asr.cl instead of ops.cl. Behaviour unchanged (every route, the queue, the
  encoder's error against f32, the transcript checked).

## 2026-10-28: Qwen3-ASR the default speech model; Gemma 4 E4B and Whisper large-v3 the other choices

- **Transcript models: Qwen3-ASR (default), Gemma 4 E4B, Whisper large-v3** (the user's choice after comparing two
  recordings, [docs/qwen3-asr.md](docs/qwen3-asr.md)): `--whisper auto` loads large-v3 (turbo only without it) and
  Qwen3-ASR with it (`--asr auto` is the default); the Transcripts tab picks the model per file, changeable any
  time; `/v1/audio/transcriptions` without a model (or OpenAI's `whisper-1`) and the chat's voice messages / audio
  go to Qwen3-ASR; `gemma4-e4b` transcribes through the chat model in 2-minute parts; Gemma 4 12B is refused for
  transcription (it garbled a clear recording and summarized a noisy one). Live text while speaking stays Whisper's.
- **Fixed**: a transcription request naming an audio chat model other than the loaded one sent its clip to the
  speech service instead of that model (now `"speech": "model"` in its chat request).

- **Qwen3-ASR 1.7B** ([docs/qwen3-asr.md](docs/qwen3-asr.md)): `serverz --whisper auto --asr auto`, then
  `/v1/audio/transcriptions` with `model=qwen3-asr` or "Model: Qwen3-ASR" in the Transcripts tab (a live
  recording's pass after Stop; its text while spoken stays Whisper's). On the user's 2:34 recording of noisy mixed
  Indonesian and English: every utterance, English kept, punctuation, 3 invented words over the last 40 s of
  background sound (Whisper turbo lost two sentences and wrote Japanese there, large-v3 Thai, Gemma 4 E4B a
  paragraph); 23-25 s for the 2:34. No timestamps (one segment per part of up to 6 minutes); the model calls this
  Indonesian "Malay" (reported as `id` when the request's languages have `id`).
- **Qwen3 dense on the engine** (GGUF arch `qwen3` / `qwen3vl`): float attention and two int8 passes, KL < 0.00001
  against the f32 reference; the int8 attention of the Qwen3.5 path gave 0.236 on it. `tools/ref_forward` takes
  these models; `ARC_Q3_Q2=0`: one pass (KL 0.0064 / 0.0093, prompt +50 %).
- **The qwen3a audio encoder on the GPU** (zero/audio_qwen3a.h; f32 CPU reference tools/ref_qwen3a.cpp: 0.08 %);
  zero/asr_testz.cpp runs and checks it from the command line.
- **1.37 GiB less GPU memory for Qwen3-ASR**: its weights' GEMM form decoded just in time per 512-token chunk
  instead of kept (3.72 GiB with the encoder; the server with Whisper 5.31 instead of 6.68). The same transcript and
  KL, the 2:34 recording 23.2 s either way. The server logs each Qwen3-ASR transcription's times.

## 2026-10-28: a live recording's language preferred, not forced; recordings kept as WAV; a cap for audio to a chat model

- **English in an Indonesian live recording stays English** (src/whisper.h `prefer_language`, tools/server_whisper.h
  `live_lock`): the language a recording locked to was forced on every utterance, and Whisper forced to one language
  translates the others: "How come you translate my words to Indonesian?" came out in Indonesian. It is preferred
  now: an utterance detected in another language with less than 0.7 is decoded in it (noise, short words), one
  clearly in another keeps its own. Checked with the user's 2:34 recording through the live API: its English
  utterances in English in the draft.
- **Recordings kept as WAV** (Transcripts tab: "Keep recordings as WAV / MP3", WAV by default; `mp3_kbps: 0`):
  lossless for a later pass by another model (~5.5 MB a minute a track; MP3 192 kbps ~1.4 MB). Checked: the
  recording's ext stays wav, its header 153.0 s, ranges read back whole.
- **Audio to a chat model** (`/v1/audio/transcriptions` with a Gemma model): at most 8 tokens a second of audio
  (+64) and repeat_penalty 1.1 unless given: greedy, Gemma 4 E4B fell into a loop on a 29-s clip and wrote 16,687
  tokens in 489 s. Compared on that recording (mixed Indonesian and English, a noisy microphone): Whisper turbo 11
  s, large-v3 40 s (no better: repeats, Thai over the closing silence), Gemma 4 E4B 26 s (the most readable, English
  kept, but a whole made-up paragraph over a nearly silent piece), Gemma 4 12B: its audio does not reach the model
  ("the audio is not provided"), to be looked into.

## 2026-10-28: the model's tools audited; create_document is create_file

- **create_file** replaces create_document (it writes any text file now, not only documents): the tool, its texts,
  `/v1/tools/create_file` and the minutes' DOCX / PDF. The old name is gone, from the server too (one tool, one
  name); older chats still show their create_document cards and files.
- **search_chats** replaces list_chats and search_chats (the same job, two tools in every prompt): no query lists
  the other chats newest first, a query finds their exchanges (BM25); `limit` for both. `/v1/tools/list_chats` is
  gone. The six file tools are five.
- **Talk mode**: every tool runs in the background there, but only the web tools' calls were explained, and only
  with Web on: with Files alone a model did not know its read_file results come later, and could say a file was
  written before create_file ran. The text now covers every tool whenever tools are on; a result's label names its
  file ("[Results of #3: read_file laporan.pdf]", before "read_file " alone).
- **The 8-round limit**: calls in the 8th round were dropped without a word, the answer often empty. Now they get
  a "not run" result and their card says so, and the model answers once more without tools from what it has.
- **read_file of a file the model wrote** returns its source within the model's read budget, in parts by offset
  (a 60 KB page overflowed a small context). **web_fetch** keeps to the same budget (12,000 characters a fetch
  whatever the context before).
- **search_file** with nothing to search (an image or audio file, a chat without documents) says so, instead of
  "no passage has these words: try other words", which sent the model off retrying.
- create_file: an empty filename takes the title (it made `file.docx`); md with a title and content that starts
  with its own `# ` heading no longer has the heading twice.
- Earlier exchanges' long call arguments (a file's whole content, a long script) are shortened in what is sent, as
  long results already were; search_chats results too. The card labels: "PDF" / ".html" formats, search_file
  "searching…", read_file without file_id.

## 2026-10-28: a pipe inside a table cell

- **`\|` and a `|` inside a code span stay in their table cell** (webui/index.html `mdCells`, tools/doc_writer.h
  `cells`): "`` `ls | sort` ``" or "a \| b" split the cell in two and moved the row's later cells one column right,
  in the chat and in the docx / pdf / rtf / xlsx files create_document writes. A row with an odd number of backticks
  has no code spans (one would take the rest of the row). Checked: the chat's table in the browser; a docx and an
  xlsx read back cell by cell.

## 2026-10-28: create_document writes any text file

- **The model can save HTML, SVG, code and other text files** (tools/server.cpp `create_document`, webui/index.html
  `FILE_TOOLS`): before, the format had to be docx, pdf, doc, rtf, md or xlsx, so a page the model wrote stayed a code
  block in its answer (Open / Download only) and never became a saved file. Now every other text extension (html,
  svg, css, js, py, json, csv, txt, xml...) is saved exactly as written; no format: the file name's extension.
  Binary formats (images, audio, video, pptx, zip, exe...) are refused with a note. The tool's text says a code
  block is not a file. Source and markup files are served as text/plain with nosniff (never run as this origin);
  an .html / .svg card opens sandboxed (`openPage`).

## 2026-10-28: the prompt cache kept with the terminal; terminal_panel; HTML pages open sandboxed

- **Opening or closing the terminal no longer recomputes the whole chat** (webui/index.html `TERM_SYSTEM`): the
  terminal's system text changed with its state (open / closed, the shell note added once it opened), and the
  system text comes before the tools in the prompt, so each change made the server compute the tools and the whole
  conversation again (the log: "cached 0", 20 s for 2,400 tokens on the 27B instead of 2.7 s). It is one text now;
  the shell's version goes in the session's first result. Checked: the system text the same open and closed.
- **terminal_panel** (open / hide / exit) for the model: asked only to open the terminal, a model made up a command
  to open it with (Get-Location; hostname; python --version) that waited for Run. Checked with the 27B: "open
  terminal" and "buka terminal dong" -> open, "Python versi berapa" -> run_command, "venv vs conda" -> no tool. The
  power button is called Exit now (it was Close, and the model took "tutup" for exit).
- The system text names the server (a model's thinking read "arcinferzero" in its paths as "Arc Infra Zero").
- **HTML and SVG open in a new tab, sandboxed** (`openPage`): a code block's Open (next to Copy and Download) and a
  saved .html / .svg chip. An iframe sandboxed without allow-same-origin and a CSP of https only: checked from inside
  the page, its origin is null, fetch to /props, POST to run_command (also no-cors), an <img> of 127.0.0.1, the web
  UI's localStorage and window are all blocked; its scripts run, an https image loads. The server still sends .html
  as text/plain (a blob: URL or text/html would have run as the web UI's origin, with its API).

## 2026-10-28: the model opens the chat's terminal itself; shorter tool texts

- **The model opens a closed terminal itself** (webui/index.html `runTermTool`, `termOffered`): before, `run_command`
  was there only while the user had opened the chat's terminal, so a model that had to make a `.venv` could not.
  Now it is offered in every chat unless Settings > Terminal is "Model never"; a call in a closed terminal waits for
  Run / Don't run, and the terminal opens once allowed. The first command of every terminal session asks, with
  "Run without asking" too (like Claude Code's approval), so nothing runs on the computer unasked.
- **The tools' texts are shorter, each rule and parameter kept**: what the system text and a tool's description both
  said is said once, the wording is tighter. Measured with the server's tokenizer: all tools 2,320 -> 1,884 tokens
  (-19 %), Files alone with the time 1,732 -> 1,421. The rest is mostly the parameters' JSON (create_document: 20).

## 2026-10-28: Whisper over silence, a live recording's language, replay after the MP3, the Transcripts tab

- **No more text over silence and noise** (src/whisper.h `drop_unvoiced`): a live recording in Indonesian came out
  "Продолжение следует...", "Спасибо.", "Obrigada.", "MBC 뉴스", "Thank you." in its quiet stretches. large-v3-turbo
  writes these with a no-speech probability near 0 (the f32 reference too), so the no-speech check never skipped
  them. Segments with almost no voice under them are left out now, Whisper's stock credit lines and thanks in
  another language when short; a spoken "terima kasih" stays. 12 s of silence: no segment (before: one, Russian with
  auto, "Terima kasih." with id). docs/whisper-native-status.md; tests/test_audio.cpp.
- **A live recording with language auto keeps one language** (tools/server_whisper.h `live_lock`): each utterance
  was detected alone, and quiet ones came out Russian, Spanish, Portuguese; once two clear utterances are in one
  language, the rest is decoded in it ("id,en" for mixed speech is left as it is).
- **Replay stopped mid-way after a recording ended** (~0:55 of 2 minutes): once the recording was kept as MP3, the
  WAV was deleted and the same media URL gave the player, reading the WAV in ranges, the MP3's bytes. The page asks
  for the file by its type (`/media?f=wav`), the server keeps the WAV 30 minutes for it, and the player takes the MP3
  when it is not playing, at the same time.
- Stop in the Live panel now hides the header's Stop at once (it stayed while the last audio was sent, the server
  still saying "live"); the glossary box has 3 lines and grows with its text.

## 2026-10-28: `arc-server restart` keeps -Npu and -Whisper

- `arc-server restart` (e.g. `restart -Model 27b -Whisper auto`) kept the slots but turned the NPU and Whisper off
  unless they were given again: the 27B lost `--npu on` when Whisper was added. Both are now kept like the slots;
  `-Npu off` / `-Whisper off` turn them off, a start after `stop` still begins with both off (scripts/arc-server.ps1).
- A kept `-WhisperDevice npu` read as "npu|think" (the state's options end in "|think N"), so a switch to another
  model moved Whisper's encoder back to the iGPU; the option is now read up to the "|".

## 2026-10-28: Transcripts and Meetings in the web UI

- **A self audit of all of it, and its fixes** ([docs/self-audit-2026-10-28.md](docs/self-audit-2026-10-28.md)):
  among them a 32-byte MP4 that hung the server, an open in the Transcripts tab that was slow or failed (Delete
  then deleted another transcript), recordings past 4 GB that lost their end, a recorder killed or a power cut
  (the recording is now picked up at the server's next start), a headset plugged in mid-meeting (now followed),
  a live recording whose page closed (stopped and transcribed when the page has sent nothing for 5 minutes), subtitle cues of
  two speakers that ended before they started. Recordings up to 6 hours are transcribed (180 minutes before). A
  3-hour unattended recording passed, and showed that recordings' WAV header overwrote their first samples (fixed).
- **A chat reads your other chats and recordings**: type `@` to attach an earlier chat or a recording / meeting
  next to the files, or just ask ("what budget did we settle on for Bali?"): the model searches your other chats
  (list_chats / search_chats) and reads the part it needs. Files now tell the model an image's size and a
  recording's length.
- **Meetings recorded apart from the browser**: `arc-server record -App Zoom.exe` records your microphone and
  the Zoom (or Teams, or any app's) sound as two tracks until `arc-server record-stop`; it goes on if the server
  stops, and the Meetings tab transcribes it as Me / Others when it ends.
- **Every model hears**: an audio file or voice note attached in a chat is written by Whisper, so Qwen and the
  others answer about it (a 10-minute voice note: 39 s the first time, 4 s for each next question). Settings >
  Voice > **Speech** chooses, for Gemma 4 E4B / 12B, whether Whisper writes it or the model hears it itself.
- **Transcripts tab**: drop audio or video files; each is transcribed in the queue with its progress; the
  transcript plays with its media, the word being said lit; click a word to play from it, double-click a line to
  correct it, find and replace a term everywhere, and export SRT, VTT, YouTube, ASS, TTML, text, Markdown, CSV,
  Remotion or JSON.
- **Live**: record the microphone or a tab's / the screen's sound and read the text about 2 s behind; when you
  stop, the whole recording is transcribed again accurately and kept as MP3 (192 kbps).
- **Meetings tab**: "Start a meeting" records your microphone as "Me" and the meeting (a browser tab, or the
  screen's sound) as "Others"; afterwards "Summarize" (the loaded chat model: summary, decisions, tasks, open
  questions with their minute, in the meeting's language) and the minutes as DOCX or PDF.

## 2026-10-28: Whisper takes video files and keeps transcripts

- **Video and voice notes**: the audio of MP4, MOV, MKV, WebM, AVI and WMV files, and Ogg Opus voice notes
  (WhatsApp, Telegram), is transcribed; the times follow the video even when its audio starts late. An MKV with
  16-kHz AAC used to come out as garbage. Long files are read as they go instead of into memory first.
- **Saved transcripts** on the server (`/v1/transcripts`, for the Transcript tab): upload files of any size, a queue
  with progress, cancel, delete, export in every format, edits saved; a server restart continues the queue.
- **Subtitles**: no line longer than 42 characters any more (the second line could be), and two lines evened out;
  WebVTT word timing as the standard asks. Checked in Remotion's caption library and Chrome's subtitle parser.

## 2026-10-28: Whisper for mixed languages and jargon

- **Indonesian and English in one recording**: `--lang id,en` (server `language=id,en`) checks each utterance's
  language and transcribes it in that language, instead of translating the other language's sentences or leaving
  them out. On a test file alternating Indonesian speech and English sentences: 18.5 % / 6.9 % of the words wrong
  (was 45.7 % / 43.5 %; faster-whisper 61.6 % / 19.1 %). About 3x slower than one language (14 minutes in 72 s).
- **A glossary**: `--terms "Qwen, GGUF, Level Zero, ..."` (server `terms`) before every window: the 14-minute
  channel recording's hard terms 14 -> 21 of 24. Tip: the terms inside a casual sentence keep the casual spelling.
- **The previous text is no longer the next window's prompt by default** (`--previous-text` turns it on): with it
  Whisper dropped whole windows (one became "Terima kasih.") and carried errors over.

## 2026-10-28: Whisper faster

- **A 14-minute recording transcribed in 25 s** (greedy, 33.8x realtime; 30.8 s before) and 32 s with beam 5
  (38.2 s): 2.27x OpenVINO's speed on the same iGPU. The decoder reads its weights through the engine's GEMVs, small
  kernels are fused, the cross-attention computes only the rows a pass has, and the word timestamps' host work runs
  on every core. Same accuracy (decoder logits KL 0.0004 against the f32 reference). Whisper takes 0.16 GiB more
  GPU memory (1.55 GiB).

## 2026-10-27: failure paths cleaned up (a static audit by DeepSeek, each finding checked)

- **A model or Whisper that fails to load frees what it already put on the GPU**: a constructor that throws runs
  no destructor, so a failed model switch in the server kept the half-loaded engine's buffers, kernels and programs
  until the process ended (Engine, WhisperEngine); Whisper's `model.bin` is closed when it is malformed.
- **No more process abort from a background save**: the threads writing saved conversations and system-prompt
  states are joined by their owner however it lets go (a `std::thread` still joinable when destroyed ends the process).
- **Request bodies: 1 GiB for all connections at once** (64 connections x 256 MiB could ask for 16 GiB); past it,
  503. The per-request limit stays 256 MiB.
- **Gemma 4 image and audio projectors of another model size are refused** at load, as the Qwen one already was
  (their embeddings would have been read at the wrong width).
- GGUF tensors must hold whole blocks (also the virtual expert tensors, whose unknown type divided by zero); the
  base64 table is built thread-safely; WIC and Media Foundation objects, the terminal tool's pipe / job / process
  and a compiled program are released when an allocation throws; the expert cache's saved state is written whole
  or not at all, never from a throwing destructor; NPU memory import and `OpenCL.dll` cleaned up on failure.
- Checked and not changed: the encoders' GPU buffers when a projector is refused (they load lazily: nothing is
  allocated yet); tokenizing before the context check (memory now bounded by the body budget).
- **Web UI** (DeepSeek's second audit, each finding tried in the browser):
  - Talk while every slot was busy could delete an older message and everything after it (the next utterance
    "joined" a message that was not its own); background web results could overwrite an older message's bubble.
    Both now act only on a message they really posted, and the results wait until a slot is free.
  - Stopping Talk clears its utterance state; something said just before stopping lands in the message box instead
    of being sent.
  - Opening chat A and then chat B quickly could show A (its slower load won); a file upload going on while you
    switch chats stays in the chat it was dropped on.
  - The microphone is released when the browser cannot build its audio graph; closing the terminal ends the command
    still running; deleting an answer not in the history no longer throws; voice conversion errors are shown; pages
    in browsers without speech output keep working.
- **DeepSeek's re-audit of the fixes above** (each finding checked):
  - A model or Whisper failing before the GPU runtime existed (no GPU / driver) gives its error again instead of
    crashing: the cleanup added above waited on a command list that was not there yet.
  - The DFlash2 drafter's file is closed when the model is unloaded, and a damaged one is refused (header length,
    tensor ranges, shapes, codebook sizes checked against the file and the model).
  - Saved-conversation and system-prompt files are closed when loading or saving throws; Whisper decodes an upload
    in its turn (four waiting 3-hour uploads were decoded at once, ~2 GB each); requests with both
    `Transfer-Encoding` and `Content-Length`, or two different lengths, get 400 (request smuggling behind a proxy).
  - Web UI: Compact by hand no longer writes its summary into another chat opened meanwhile (on screen and on
    disk); deleting a file that is attached to the message being written takes it off the message; Talk stopped
    and started again does not send what was said before; a voice message is not sent into a chat opened while it
    was written down, and its file stays in its own chat.
  - Smaller: `OpenCL.dll` let go on every failed start; a model folder that hangs is checked by one thread, not
    one more per model list; the terminal's allocations before its handles; JSON mode's limit on leading
    whitespace; the sampler refuses empty logits.
  - Not changed: tokenizing before the context check (as above).
- **Third round** (each finding checked):
  - A prompt longer than the context can possibly hold (more than context x the longest token's bytes) is refused
    before tokenizing: 30 MB in 0.8 s, with the usual context-overflow error ("at least N tokens"); every prompt that
    might fit is tokenized and checked exactly as before.
  - `/v1/completions` with `n` > 1 returns n completions (the API adapters skipped the n > 1 handling: one came back).
  - Closing the terminal stops the model's running command too (the answer goes on, told the terminal was closed).
  - Web UI: Talk stopped and started again leaves no clip of the old session attached (without Whisper); a file the
    server could not delete stays attached and the failure is shown.
  - Smaller: a half-written cache file is removed when saving throws; DFlash2 dimensions past INT_MAX refused; COM
    released when a server thread that decoded an image ends.
  - Not changed: background web results are not capped (they pile up only while you never pause; capping them
    would drop results the model was promised).
- **Fourth round** (each finding checked): a model's terminal command that is waiting for Run is cancelled when
  the terminal is closed (Run no longer starts it in a reset session), and one whose answer was stopped just
  before it began does not run or ask; a half-written `chat.json.tmp` is removed when saving fails; a model folder
  check that cannot get a thread skips the folder instead of failing the model list. Not changed: re-checking a
  folder once its hanging check has answered (its state may have changed: a drive plugged back in).

## 2026-10-27: web UI audit: every function tried, security and robustness fixes

- **Tried with Qwen3.5 4B (vision, Whisper) in the web UI**, each passing:
  - a voice message through Whisper; DOCX, XLSX (two sheets) and PPTX attachments (speaker notes, the hidden slide);
  - a scanned PDF transcribed by the vision encoder; an XLSX the model wrote, opened in Excel (2 sheets, numbers as numbers);
  - the terminal panel (folder and environment carried, PowerShell variables not, as designed), the model's commands
    with Run / Don't run, and no terminal tool for the model while the panel is closed;
  - web search, the Files panel (@, delete unused keeping an unsent attachment), deleting a message with its file, edit;
  - two chats answering at once after switching to 2 slots in Settings, reopening a running chat, Stop all, Compact.
- **Fixed**:
  - **DNS rebinding through a name that starts like an address** (security): "127.0.0.1.evil.com" passed the Host check
    (it only looked at the "127." start), so a page of that site could run terminal commands and read their output.
    The Host must now be an address literal (or localhost); a server opened to the network (--host) also refuses
    names (open it by its IP). src/http.h; tests/test_http.cpp.
  - **Voice messages and Talk never used Whisper**: the page read /props "speech_to_text", the server sends "whisper".
  - **Run / Don't run took the keyboard focus**: a key typed for the message box could approve a command. A chat whose
    answer waits for that choice in the background now says "needs your OK" in the chat list.
  - **Stop and Esc end a running command** (it ran on to its timeout, up to 10 minutes): the server polls the client and
    ends the command with every process in its job; Esc in the terminal panel stops the command typed there. A command
    whose process left its job no longer blocks the reply (the pipe read ends 2 s after the command).
  - **Command output within the model's context** (the share a read_file part gets), its start and end kept.
  - The terminal session's paths (cwd, VIRTUAL_ENV, PATH) follow the chat folder when it is renamed to its title.
  - A file put into a chat folder (e.g. a download from the terminal) is hashed in parts, not read whole into RAM;
    files over 512 MB are not read as documents.
  - A chat saved in the middle of a tool round gets placeholder results for its unfinished calls (a valid history after
    a closed page); opening a chat waits for a save on its way.
  - XLSX: elapsed times ([h]:mm) as 30:00, not a 1900 date; phone and account numbers (+62..., 12+ digits) stay text;
    slash dates month first in a US table. PPTX: mc:AlternateContent read once. Office files: at most 512 MB unpacked.
  - The terminal tool's description says which programs can outlive a command (COM objects, WMI, scheduled tasks).
  - **Answers render Markdown fully** (webui/index.html `mdBlocks`): tables (column alignment, a header row, scrolling
    sideways in the bubble on a phone; before, they showed as "| a | b |" text), numbered and nested lists, task boxes,
    quotes, rules, links (http(s) and mailto only, in a new tab) and bare URLs, strike; empty lines make one gap.
    Text is escaped first, quotes too (a URL could otherwise leave its href); every prefix of a streamed answer
    renders, a 2,000-row table in 7 ms.
- Not changed, by design: a program on this computer can call run_command without the page's Run (it has the user's
  rights already); the approval guards against the model. Not tried live: a request from another computer to a
  server opened with --host (a firewall prompt), Talk with a real microphone.

## 2026-10-27: scanned PDFs read by the vision model

- **Scanned PDF pages (images without a text layer) are read in the web UI**: `read_file` with `pages` on such a page
  has the loaded vision model (Qwen 3.5 4B / 9B, Gemma 4 E4B / 12B / 26B) transcribe it from its image, up to 4
  pages a call; the text is kept per file and page (the chat folder's `.pages\<id>.json`), so later reads and
  `search_file` find it without the vision model. A model without an image encoder names the scanned pages and says
  a vision model is needed. The server transcribes (a chat request of its own) instead of the web UI attaching the
  page images to the model's next turn: the text can be kept and searched, a page costs its text in the
  conversation rather than ~1,000 image tokens on every later turn, and no chat template has to take images in
  tool results. The cost: with one slot the chat's prompt is processed again at its next turn.
- `tools/pdf_images.h`: the image a page draws (its content stream and form XObjects), out of the PDF on the CPU
  without a library: DCTDecode as the JPEG it is, FlateDecode or unfiltered samples (gray, RGB, CMYK, ICCBased,
  Indexed, image masks, 1-16 bits, PNG predictors) as a BMP, CCITTFaxDecode (Group 4, Group 3) behind a minimal
  TIFF header, all decoded by WIC; JBIG2 and JPEG 2000 a note. Sizes capped as uploads (16384 pixels a side, 64
  megapixels), raw samples only as far as they decoded. `tools/doc_reader.h`: PNG predictors for more than one byte
  a pixel, decode parameters per filter.
- Checks: `tests/test_pdf_scan.cpp` (model-free; in `tests/run_tests.ps1`, CI and setup): PDFs built from known
  images give the same pixels back (raw, CCITT G4 made by WIC) or the same JPEG bytes; scanned pages found, a page
  with text is not one; transcriptions merged and searched; huge sizes, broken streams, a form that draws itself and
  300 randomly damaged PDFs. `test_pdf_scan FILE.pdf` lists any PDF's scanned pages and their images.

## 2026-10-27: batched decoding for Flash Next

- **Flash Next generates for several conversations in one pass** (`--slots 2` or more, without a drafter): the
  PLE n-gram state and history and the QSA blocks run per conversation, everything else once. Bit-identical to each
  conversation alone. With the experts cached 2 slots give 1.43x the tokens per second together; in real chats on
  this laptop only ~1.05x, as each conversation brings its own expert reads from the SSD.
- Batched steps and verification passes of MoE models run their experts on smaller tiles: the 35B 2 / 4 slots
  1.33x -> 1.44x / 1.99x -> 2.19x together, bit-identical.
- `runz --bench-batch N --prof-gen`: the GPU time per kernel of a step alone and batched.
- Prompt MoE on the 35B 2 % faster (an emulated integer multiply in the expert kernel done in float, same results);
  found by reading the kernels' GPU assembly with ocloc ([docs/kernel-audit-2026-10-27.md](docs/kernel-audit-2026-10-27.md)).
- Prompt attention of the 27B 9 % faster (two independent sums per head in the score loop, same results).
- The small kernels that quantize activations between the matrix products read their inputs as vectors: about 1 ms
  less per generated token on the 35B (~2 %), the 27B's prompt normalization a third faster; same results.

## 2026-10-27: Qwen vision encoder faster on the dense models

- **Images 22-30 % faster to encode on Qwen3.5 4B / 9B and Qwen3.8 27B**: the vision encoder's matrix products run on
  the engine's int8 GEMM in two passes, as the Gemma 4 encoder's (27B: a 768-token image 1.76 -> 1.27 s, a photo
  571 -> 450 ms); the model's KL against the f16 encoder 0.0004-0.0024 on 5 test images. The 35B and Flash Next keep
  the f16 encoder (their models are too sensitive to it); `ARC_VIT_Q2=0 / 1` turns it off / on for any model.
- Measured, not changed: the NPU next to the GPU does not pay off on the 35B (-3 to -8 %) or Flash Next (noise),
  whatever share it gets; `--npu on` keeps them off.

## 2026-10-27: the chat's terminal in a right panel, a venv that stays

- **Terminal panel** (webui/index.html, the Terminal button above the chat): the chat's PowerShell on the right of the
  chat, as an editor's terminal pane. You type commands there; the model's `run_command` calls show in the same log
  (marked AI). It belongs to the chat (chat.json `terminal`: open or not, the log, folder, venv) and is there again
  when the chat is opened. **The model gets `run_command` only while the chat's terminal is open**; Close ends it
  (the model loses it, the next one starts fresh in the chat's folder), × only hides the panel. Settings > Terminal is
  now the model's access to an open terminal: Ask each time (the default), Run without asking, Model never.
- **The model knows which shell it writes for**: when the terminal opens, it is asked once for PowerShell's version
  and edition and Windows' name, version and build (kept with the chat), and the system prompt says it ("Windows
  PowerShell 5.1.26100 (Desktop) on Microsoft Windows 11 Home Single Language 25H2 (build 26200), AMD64") with the
  syntax that differs: PowerShell, not bash or cmd; curl.exe / where.exe; and for 5.1 no `&&` / `||`, no ternary or
  `??`, Set-Content's encoding.
- **Folder and environment kept between commands** (tools/server_terminal.h, `run_command` with `session: true`):
  each command still runs in its own PowerShell, and the variables that differ from a fresh one plus the folder are
  written to the chat folder's `.terminal.json` after it and set again before the next. `cd`, `$env:X`, an activated
  `.venv` (and its `deactivate`) stay; exit codes are as before. `reset: true` ends the session. Checked:
  test_chat_store (folder, variable set and removed, exit codes, PATH not doubled, a venv and deactivate) and a real
  `python -m venv`: the next command's `python` and `pip` were the venv's, after `deactivate` the system Python.
- **Thinking card follows the newest text** while it streams (unless scrolled up to read), and opens at its end.

## 2026-10-27: web UI sends earlier tool results shortened

- **Long tool results of earlier exchanges go to the server as a short stub** (webui/index.html `trimTools`): a
  `read_file` part, a `web_fetch` page, `search_file` passages or `run_command` output (1,500+ characters) from
  before the previous user message becomes one line with what was read and how to read it again (file id and name,
  section and title, pages, size, next_offset; the query and the passages' sections; the URL), e.g. `[read earlier:
  read_file doc1 (kontrak.docx), section 3 "PASAL 3" pages 3, 12,000 characters of 20,000; shortened here: read it
  again with read_file if needed]`. Whole: the last exchange, every round of the answer running now, and the last 2
  results before them; `list_files` and `create_document` results always. Only in the request: the saved chat and
  the screen keep every result. The cut moves only when the user sends a message, so the server's prompt cache holds
  through the rounds of an answer. A few reads of a long document no longer fill the context, and compaction comes
  later (the context estimate counts the shortened form). Compaction's summary request gets the same shortened
  messages, so it reads less and starts as the last request did (the server's cache still fits). Checked in the browser with a scripted fake server: three
  12,000-character reads, a search and a page over four questions; the fourth request had the two oldest reads as
  stubs, the requests within one answer kept the same start, the summary request matched the last request's start,
  and the saved history stayed whole.

## 2026-10-27: Excel spreadsheets in the web UI (read and written), PowerPoint read

- **Uploaded .xlsx / .xlsm are read** (tools/xlsx.h, no library; `read_file` / `search_file` and attachments): each
  sheet is a heading "Sheet: <name>" and each row a line "A | B | C"; with a header row every line repeats the
  column names ("Nama: Budi | Nilai: 80"), so a passage `search_file` finds still says what each value is. Shared and
  inline strings (rich text joined), numbers, booleans, errors, formula cells as their cached value, dates and
  times from the cell's number format (built-in and custom) as yyyy-mm-dd / hh:mm, percentages, a currency prefix
  ("Rp 1234000"), merged cells (the value in the first cell), hidden sheets marked; empty rows and columns left out,
  at most 200,000 cells (a note says so). A binary .xls gets a note (save it as XLSX). **CSV / TSV** now read the same
  way (delimiter , ; or tab, quotes, line breaks inside quotes) instead of as plain text.
- **`create_document` format "xlsx"**: from the same Markdown, each table becomes a sheet named by the heading before
  it (31 characters, no `[ ] : * ? / \`, a repeated name gets "(2)"); the text around the tables goes to a first
  "Notes" sheet (with a line where each table was), and the workbook opens on the first table. Numbers become real
  numbers with a number format ("1.234,5", "1,234.5", "12.500", "Rp 1.234.000", "$12.50", "15 %", "12,5%"; Indonesian
  or US style per table: "1.500" / "1,500" alone cannot tell, so each table goes by its numbers that do, "1.234,5" or
  "2,5" against "1,234.5" or "3.14", else the whole document's, else Indonesian; `number_format: "id" | "us"` fixes
  it), dates
  real dates ("2026-10-27", "27/10/2026", with a time); a leading zero ("0812...") or a text starting with "=" stays
  text. The header row is bold on a fill, frozen, with an autofilter; column widths from the content. Checked with
  Excel itself (COM): the file opens without repair (a copy with one element out of order does not open), values,
  number formats, frozen pane and filter as written, and Excel's own re-save reads back the same.
- **Uploaded PowerPoint .pptx / .pptm / .ppsx / .potx are read** (tools/pptx.h): each slide is a heading "Slide N:
  <title>" on page N (the outline lists the slides, `read_file` pages are slides), then its text in the slide's
  order: bullet points with their level, tables as rows ("Column: value" with a header row), SmartArt text, a
  picture's alt text as "[image: ...]", the speaker notes as "Notes: ..."; footer, date and slide number left out,
  hidden slides marked. A binary .ppt gets a note (save it as PPTX). Not read: text inside charts and embedded objects.
- Checks: `tests/test_office.cpp` (a workbook made in Excel, `tests/data/xlsx_sample.xlsx`, and a deck made in
  PowerPoint, `tests/data/pptx_sample.pptx`, metadata scrubbed: dates, percent, currency, formulas, an error, rich
  text, merged cells; titles, bullet levels, a table, notes, alt text, a hidden slide; Markdown -> xlsx -> text; the
  element order; hostile input: huge references, bad shared string indexes, 200,000-deep nesting, 20,000 merges over
  the whole sheet, 260,000 cells, 100,000 slide ids, a part whose notes point at itself; CSV), in
  `tests/run_tests.ps1`, CI and setup. It runs in 0.5 s.

## 2026-10-26: Whisper native (speech to text on the iGPU or the NPU)

- **`arc-transcribe` / `bin\transcribez.exe`**: Whisper large-v3-turbo (faster-whisper's `model.bin` from the
  Hugging Face cache) on Level Zero: files, folders or wildcards to json, fw / captions (Remotion), srt, vtt, sbv,
  srv3, ass, ttml, txt, md, csv, tsv in `%USERPROFILE%\arcinferzero\transcripts`; language auto per window, fixed or
  a list (`id,en`); greedy or beam 5; word timestamps; a progress bar or `--progress` JSON lines. `--device npu`: the
  encoder on the NPU, the decoder on the iGPU. 14-minute file: iGPU greedy 31.8 s (26.7x realtime, 1.8x OpenVINO's
  iGPU), beam 5 39.3 s; NPU mode 81 s (OpenVINO NPU 144 s); decoder logits KL 0.00039 (iGPU) / 0.00052 (NPU) against
  the f32 reference (`tools\ref_whisper`).
- **Server**: `serverz --whisper auto [--whisper-device npu]` (`arc-server -Whisper auto`) answers
  `/v1/audio/transcriptions` and `/translations` as OpenAI's API (json, text, srt, vtt, verbose_json with words,
  `stream=true`) plus the formats above; its GPU memory comes off `--ctx auto`'s context; it takes turns with the
  model on the GPU; limits: 180 minutes, one at a time and 4 waiting. Details: [docs/whisper-native.md](docs/whisper-native.md).
- **MP3 / M4A / video audio 13x faster to decode**: the resampler to 16 kHz computes its filter weights once per
  phase (a 14-minute 48-kHz MP3: 21.4 s -> 1.6 s; bit-identical samples). Checked: MP3 and M4A through
  `arc-transcribe` and `/v1/audio/transcriptions`.

## 2026-10-27: PDF outlines from bookmarks, fewer false headings

- **PDF bookmarks are the outline** when a PDF has them (tools/doc_reader.h: /Outlines, /Dest or a GoTo action,
  explicit or named destinations, PDFDocEncoding or UTF-16 titles): each marks the first paragraph on its page that
  starts with its title (the rest split off as body text); a title not in the page's text (PowerPoint's "Slide 5")
  marks the page's top heading by font instead, or is put in when the page has none; font headings before the first
  bookmark's page stay (bookmarks for only a part of the document).
- **Fewer false headings by font**: list items (a bullet, "a.", "1)") and lines ending in ':' ',' ';', table of
  contents lines (dot leaders), numbers ("39,84%") and sentences are never headings; a bold line counts only when it
  stands apart (a gap before it larger than a line's), not as a table's cell (the next cell a little up the page) and
  not when the same heading comes three times or more ("Done", a table's labels); a larger font's line of 16 words or
  more is a sentence. Numbered lines in capitals ("II. PIHAK KEDUA") and short numbered titles in a larger font
  ("1. Peserta") stay headings. On real documents: a 215-page memo 817 -> 108 headings (its sections and risks
  R1-R10, not its list items), a 70-page slide deck 523 -> 0 level-4+ headings (one slide title per slide; 138 by font
  alone), a 18-page agreement keeps all its PASAL headings.
- **Heading levels with many font sizes**: with more than five sizes above the body text's, the levels go to the five
  on the most pages in short lines, the others take the level of the nearest of them below. A 142-page requirements
  document with a cover and slides attached: its chapters, sections and subsections (I, II.1, II.3.1, IV.1.1.1) at
  four levels instead of all at one (the cover's and the slides' sizes had taken the five levels).
- **Long documents within a small context**: a long section without headings over several pages (a five-page risk
  table, a chapter without subheadings) shows in the outline with its pages as parts one level below it (about 3,600+
  characters each), so the model reads it a piece at a time: a 215-page memo's largest section 34,900 -> 7,300
  characters. One `read_file` gives at most a fifth of the loaded model's context (3 characters a token; 4,900
  characters at 8K, the old 12,000 from 20K up) whatever `max_chars` asks, cut at a paragraph's end, the rest by
  `next_offset` with a note; `search_file`'s passages together stay within it too (the best one always). A document
  in a message goes in whole only when it takes at most a quarter of the context (else its outline, as above
  24,000 characters before). The context is read without waiting during a model switch (then the old limits).
- Checks: `tests/test_doc_read.cpp` with a Word-made bookmarked fixture (`tests/data/doc_bookmarks.pdf`: headings
  1-3, bold list items; also read with its bookmarks taken out), a hand-made PDF (a named destination, a GoTo action,
  a UTF-16 title, a title not in the text), PDF text strings and a long section's page parts;
  `tests/test_chat_store.cpp`: a document in a message whole or as its outline by the context, the read budget.

## 2026-10-27: web UI answers in the background, conversations at once from Settings

- **An answer goes on when another chat is opened** (webui/index.html: one job per chat): its chat in the list shows
  "answering…", opening it again shows the answer as it runs (with Stop), and it is saved when it ends; the question
  is saved at once and each tool round too, so a closed page keeps them. At most as many answers run as the server
  has slots: with every slot answering, New chat is off and a message in another chat stays in its box (no queue)
  until one ends or is stopped. Failsafes: a bar over the chat list
  ("2 chats answering, 1 waiting", **Stop all**), a watchdog that stops an answer after 10 minutes without a byte from
  the server, deleting or clearing a chat stops its answer (and nothing is saved again), closing the page ends every
  request (the server cancels a request whose connection closes).
- **Conversations at once** in Settings (`POST /v1/server/slots {"slots": N}`, this computer only): the loaded model
  again with N slots (5-40 s, after the running requests), the context shared out (`--ctx auto`); later model switches
  keep N, a server restart goes back to its `--slots`. A chat that no longer fits (more slots, a smaller model, a chat
  opened again) is compacted before its next answer, from an estimate of its size against the target model's
  context; when the server still says it does not fit, every tool round compacts and tries once more. The session
  cache skips (keeps) saved conversations longer than the context instead of failing to load and dropping them.
- Fixes from a review: Talk background results stay in their chat (the old chat's abort is checked, not the new
  one's); Stop while a command waits for Run / Don't run stops the answer instead of saving a refusal; the overflow
  retry compacts everything before the question (it kept the last exchange); two commands at once no longer wait for
  each other (only the command's own handles go to it: PROC_THREAD_ATTRIBUTE_HANDLE_LIST); "delete unused" leaves the
  files of a chat whose first answer runs, of the message box and of running answers; editing a message keeps images
  and audio sent inside it; Talk results never take the message box's attachments; an upload that ends after another
  chat was opened stays out of that chat; a streamed answer's redraw no longer waits for an animation frame (one that
  never came, in a hidden tab, stopped every later redraw).
- Small screens: the gear in the chat header too; a written document's card also under the final answer.

## 2026-10-26: long DOCX / PDF without a long prompt

- **Uploaded DOCX and PDF are read** (tools/doc_reader.h, tools/inflate.h: DEFLATE, ZIP, a PDF object and content
  stream reader, no libraries): paragraphs with their page and heading level. DOCX: headings from the styles
  ("heading N", outline levels), tables as `a | b | c` rows, lists, pages from Word's page breaks or, when it marked
  few, from `docProps/app.xml` spread over the text. PDF: fonts with ToUnicode maps, simple encodings and their
  Differences, glyph widths for the spaces, form XObjects, object streams; lines grouped into paragraphs by their
  spacing, headings by font size and by short bold lines, table rows by column gaps, running headers and footers
  (the same line on most pages) left out; a scan without a text layer, an encrypted PDF or a binary .doc gets a note
  instead. A 227-page PDF reads in 0.25 s, the same document as DOCX in 0.04 s.
- **A long document is not put into the prompt whole**: above ~6K tokens the model gets its outline (the top heading
  levels, at most ~60-120 sections, with pages and sizes) and its beginning, and reads what the question needs:
  `read_file` with `section` (a chapter with its subsections) or `pages`, or the new `search_file` (BM25 over chunks
  of ~2,400 characters within sections; Indonesian particles -nya / -lah / -kah / -pun folded, words of 5+ letters
  also find longer words; one document or every document of the chat), each passage with its section and pages.
  Tried with Qwen3.5 4B on a 227-page PDF (~219K tokens whole): the question about a fact on page 151 took a 6,357-token
  prompt (8.1 s) and one search, and the answer named the page; a chapter summary came from its pages 8-10.
- Checks: `tests/test_doc_read.cpp` (inflate, Word-made DOCX and PDF fixtures in `tests/data/`, doc_writer output,
  search, damaged and hostile files: delimiter runs, self-referencing forms, a 512 MB decode budget per PDF), in
  `tests/run_tests.ps1`, CI and setup. `test_doc_read FILE [QUERY]` shows any document's outline or search hits.

## 2026-10-26: web UI documents, one folder per chat, terminal tool

- **One folder per chat** (tools/server_files.h): `chats\<date> <first words> (<id>)\` with `chat.json` and the
  chat's files under their own names (a name taken: `name (2).ext`), `.files.json` (id, name, time) and `.src\` (the
  sources of the documents the model wrote). The folder gets its name at the chat's first save. A file attached from
  another chat (@) becomes a hard link, so its bytes are kept once; deleting a chat deletes its folder (a file another
  chat holds stays there); files put into a folder or deleted by hand show up in the Files panel at its next listing.
  Uploads take a `chat_id` (form field or query). The first version's `uploads\` and `chats\<id>.json` are still read.
- **Documents the model writes** (tools/doc_writer.h, `POST /v1/tools/create_document`): DOCX (every part in the
  schema's element order, checked in Word: no repair prompt), PDF 1.4 (standard fonts, real text, justified lines,
  tables that continue on the next page with their header row, page numbers), DOC (RTF, which Word opens as a
  document), RTF and Markdown, from Markdown with inline HTML for underline, colors and sizes. Options: page size and
  orientation, margins, font, size, line spacing, heading / table / text colors, header, footer, page numbers. The
  file shows as a card in the answer. `POST /v1/tools/list_files` and `/v1/tools/read_file` (a text file in parts; a
  document the model wrote: its source, so a correction makes a new version; a file's name works as well as its id).
  The **Files** setting (on) gives the model the three tools.
- **Terminal tool** (tools/server_terminal.h, `POST /v1/tools/run_command`): one PowerShell command in the chat's
  folder, with arcinferzero's folder on PATH, 2 minutes by default (up to 10), ended with every process it started
  (a job object), output up to 64 KB (the start and the end of a longer one). The **Terminal** setting: off (default)
  / ask each time (Run / Don't run in the command's card) / run without asking. Refused to every connection that does
  not come from this computer (`--host` too), to foreign pages (Origin check), and with `--no-terminal`.
- Per-message **Edit** and **Delete** (with the message's files), and **Data** in the settings (clear the chats, or
  the chats and files). Saved chats show their tool cards again (documents as file cards, commands with their output).
- Checks: `tests/test_chat_store.cpp` (28 checks: folders, renames, hard links, files changed by hand, deleting; the
  terminal's output, exit code, timeout and long output), in `tests/run_tests.ps1`; `tools/doc_test.cpp` writes a
  sample in every format. Tried with Qwen3.5 4B in the web UI: notes to DOCX, a corrected version (Arial, dark red
  headings; Word opens it), a command allowed and one refused (the model asked instead of trying again), @ a file into
  a new chat, the old chat deleted.

## 2026-10-25: the NPU next to the GPU (`--npu on`); Flint support removed; the model list without duplicates

- **Web UI rebuilt, laid out like WhatsApp Web**: the chats kept in the browser (searchable), the model picker in the
  chat header (the server switches on the next message), message bubbles with times, the settings behind a gear.
  No max-tokens field any more and nothing sent for it: an answer may use the rest of the context (before, the UI's
  4096 counted the reasoning too, so a long thinking pass could end the reply before its answer). No temperature
  sent unless set: the server samples as the model file recommends. Thinking offered as the model has it (Qwen3.8
  off / low / medium / high, others on / off), from `/props` (new: `thinking`, `reasoning_effort`, `sampling`,
  `think_budget`). Voice messages (record, then send) next to Talk, both ready for Whisper speech to text
  (`speech_to_text` in `/props`). Streamed answers are drawn once per frame instead of per token.
- **Chats and files on disk** (tools/server_files.h): the web UI's chats in `%USERPROFILE%\arcinferzero\chats\`, attached
  files uploaded once into `uploads\<yyyy-mm>\` (content-addressed: the same file kept once) and sent as references
  (`{"type": "file", "file": {"file_id"}}` parts; the server reads documents as text with their name, type, size and
  upload time, images and audio from disk). `@` in the message attaches an earlier file again; a Files panel lists
  every file with the chats using it and deletes files (one, selected, or all unused). New endpoints: `/v1/files`
  (POST upload, GET list, DELETE `/v1/files/<id>`, POST `/v1/files/delete_unused`), `/files/<id>`, `/v1/chats`
  (GET, PUT, DELETE per chat). Every user message carries the time it was sent, so the model knows the day and time.

- **The data folder moved to `%USERPROFILE%\arcinferzero`** (`C:\Users\<name>\arcinferzero`, visible in Explorer;
  `ARC_DATA_DIR` overrides it): kernel cache, NPU graphs, model packages and expert files, sessions, prefix cache,
  logs, settings, and later transcripts and recordings. The exes and scripts move `%LOCALAPPDATA%\arcinferzero` there
  once on the first start (a rename on the same drive: instant) and fix the paths in `last-model.txt`,
  `launcher.json` and `ctx-auto.txt`. Results unchanged (27B with the NPU, 35B, 4B bit-identical to the previous
  build).

- **Cleanup** (no change in results: 4B, E4B, Gemma 4 12B, 35B, 27B with and without MTP / the NPU bit-identical
  before and after; Qwen3.5 4B and Gemma 4 26B images checked): unused code removed (`zero/kernels/gemm8.cl`,
  `weight_ext` / `gemm_ext`, the `attn_decode` kernel built at load and never used, the next-layer expert guess and
  its always-zero counters, the `scale_rows` / `vit_ln_quant` / `vit_quant` kernels, a few unused helpers); the
  JSON / tool-call constraint checkers share one copy of their type / literal helpers; outdated help texts and docs
  fixed (Gemma 4 26B prompt chunks 2048, no Windows RAM reserve since 10-22, Gemma 4 26B's default of 3 drafts, the
  model folders, the NPU), QUEUE's open list tidied.

- **The NPU next to the GPU for prompts** (`--npu on|off|force` for serverz / runz, `arc-server -Npu on`, `pi-arc
  --npu on`; off by default; docs/npu-experiment.md; built on the branch npu-experiment, merged): the Meteor Lake NPU
  computes part of the Qwen3.8 27B's FFN and DeltaNet projections in each prompt chunk through Level Zero's graph
  extension alone (no OpenVINO): 2K prompt +15 %, 11K +15 %, 34K +13 % (time to first token 292 -> 258 s), KL vs
  f32 0.0028-0.0031; off bit-identical to before, MTP / DFlash2 still bit-identical to plain decoding. The setting
  is the server's: switching models keeps it, a restart turns it off. Models with resident W8 (4B, 9B, 35B) gain
  nothing and stay off. Every model can use it (`--npu force`): Gemma 4 (its FFN in f16; the 12B gains +10 % and is
  on with `--npu on`), the Q8_0-form matrices (Gemma QAT Q4_0, the 35B's attn_qkv), Flash Next's attn_qkv. RAM: 0.37 GiB (the NPU's weight rows decoded per layer by the GPU into the shared block; a
  resident copy took 6.1 GiB, same speed).
- **Flint is no longer supported** (the user's decision): the `flint` architecture in the engine and the f32
  reference, its pre-tokenizer (and `src/unicode_ps.h`), the GLM-style tool calls, the server's Flint template and
  thinking handling, the catalog entry and the `flint` shortcut, and the kernel options only it used (attention
  sinks, f32 V rows, attention prep without int8 copies, sigmoid routing with an ungated shared FFN) are gone. Kept,
  as Flash Next's precise mode uses them: BF16 router rows, two passes for the experts, `moe_gelu_mul -DSILU`. The
  other models' logits are bit-identical before and after.
- **The model list shows each model once**: a copy with the same file name and size in another folder (e.g. a
  backup on an external drive) and the GGUF an arcinferzero package was made from are no longer listed as extra
  `-2` entries; the package, then the Hugging Face cache, then the Windows drive keep the entry (35B and Gemma 4 26B
  on D: next to their C: packages: 39 -> 33 entries).
- A plain folder of `.gguf` files inside the Hugging Face cache or a model folder (a copied model, not the
  `models--org--repo\snapshots\...` layout) is listed too, one level deep (e.g. a copied Qwen3.8-27B-AP-IQ4_XS).

## 2026-10-24: renamed to arcinferzero; Flint runs; Gemma sliding attention, /W4 and CI

- **Self-audit fixes** (all source files read for bugs, each finding checked against the code):
  - Server: the Anthropic / Ollama / Responses adapter no longer outlives its request (a cancel or an engine error
    called into freed memory); their streams end once (the finish chunk and the usage chunk both ended them: Ollama
    sent its tool calls twice); a Responses stream with two tool calls closes each with its own data; a stop string
    split over tokens is held back in streams (it reached the client before the cut); a request naming a model no
    catalog has ("gpt-4o", "claude-...") no longer takes the switch lock and rescans every time (a minute's memory);
    `-m model.gguf` without a folder loads.
  - Kernels: `gemv_f32` (Gemma 4 26B's router) and `vit_gemm_f16` (every vision / audio encoder) had lost their
    16-lane sub-group requirement to a kernel inserted before them (right today only because the compiler chose 16).
  - Engine: the vision / audio encoders free their smaller buffers when they grow (a server getting larger and
    larger images kept every set); a Gemma image chunk that throws no longer leaves its non-causal span set;
    DFlash2's head rows fit `logits_` with `--mtp-drafts` below its block; `selector_rank` is bounded (a stack
    array); the MTP layer is prefetched only when the MTP block runs; a second engine reuses the OpenCL compiler
    context; a damaged kernel-cache file is rebuilt from source instead of ending every start, and a short write is
    never installed; the expert cache's shrink also unlinks slots a failed prefetch freed.
  - Tools: GGUF files that fail to parse close their handle and mapping; very thin images stay within the image
    token budget; `runz` checks option values and `--bench-batch` prompts, reads `--kl-ref` files over 2 GB;
    `hf-download.py` keeps `HF_TOKEN` on huggingface.co (urllib copied it to the CDN); `setup.ps1` downloads the 4B
    before the 35B conversion that reads it; `arc-server -NoStream` restarts a running server that streams.
- **Flint** (unsloth/flint-GGUF UD-Q4_K_XL, arch `flint`) runs: Gemma 4's float attention with attention sinks,
  1/sqrt(d), no RoPE on the global layers and a 513-token window on the sliding ones; two leading dense SwiGLU
  layers, then 256 experts (sigmoid routing, streamed from the SSD) and a shared FFN without a gate. Server: its
  template through minja, its stop tokens, `</think>`, tool calls in its GLM form (`<tool_call>NAME<arg_key>..`),
  model id / shortcut `flint`. 4K prompt 248 tok/s, generation 13-17 tok/s.
- **Flint's accuracy, KL 0.029 -> 0.0043** (prompt; token by token 0.067 -> 0.0027) against the f32 reference: two
  int8 passes as Gemma (the experts too), the **V rows in f32** (f16 V alone, simulated in the f32 reference,
  moved it as far as the whole engine and flipped the same near-tie expert choice) and the **router on the file's
  BF16 rows with f32 inputs** (its sigmoid scores have near-ties 0.005 apart). `ARC_FL_Q1=1`: one pass.
- **From reading the GPU assembly (ocloc)**: Flint's attn_prep_g 8,855 -> 481 instructions (no unused int8 copies;
  the in-kernel RoPE angles only where no table is passed, the Gemma drafter); the two-pass expert down kernels no
  longer spill (256 registers, as gate / up); Flint's router GEMV 4 rows x 4 tokens per sub-group (744 -> 229 ms per
  4K prompt). **Gemma 4 26B** gets the same router tiling on its f32 rows: prompt +4.6 %, logits bit-identical.
- **Flash Next precise mode** (`ARC_FN_PRECISE=1`, opt-in): BF16 mixer and router rows, two passes for every
  matmul input: KL 0.0143 / 0.0147 -> 0.0117 / 0.0096 at about half the prompt speed. On a 3,584-token prompt with
  QSA the default measures 0.0186 / 0.0177, precise 0.0127 / 0.0132; the user keeps the fast default (a float
  attention is not needed).
- **KL on the tail of long prompts**: `REF_ALL_FROM=N` (ref_forward) writes the reference from position N on,
  `ARC_KL_FROM=N` (runz) compares from there.
- **Tests**: 27B long-prompt MTP / DFlash2 logits equal plain decoding bit for bit; Flint's tokenizer against
  llama-tokenize; Flash Next auto-detect takes the packaged single file first (its expert file exists; the GGUF parts
  would write a 56 GB one).
- `tools/server.cpp` split into `server_cache.h`, `server_chat.h`, `server_catalog.h` and `server_load.h` (3,337 ->
  1,316 lines, the same code); setup rebuilds serverz when `tools\*.h` change.
- (The 2026-10-24 entries IQ1_S weights, the Qwen verification rows bit-identical again and the Gemma ring slot per
  16-key block are listed under 2026-10-23 below.)
- **Gemma 4 E2B / E4B images fixed** (wrong since the Level Zero engine): the vision attention kernel was built for
  256-dim heads (an earlier default in the kernel file won), so it read the wrong rows; now 3.1 s instead of 51.5 s
  for a 1,064-token image, and the model sees what is in it.
- **Gemma 4 26B images 12.7 -> 7.4 s**: the encoder's matrix products on the engine's int8 GEMM with two passes, and
  its attention with 8-wide block reads and tiles of 8 keys; the model's KL against an exact encoder stays ~0.001-0.003
  (4 test images). An int8 attention was measured too: accurate enough, but no faster, not taken. Tools: `tools/gemma4v_ref.cpp` (the f32 encoder reference in
  C++, 3x faster than the numpy one) and `vision_testz` `ARC_VIS_LM` (the model's KL on an image).
- Qwen3.5 9B / Qwen3.8 27B images: the vision attention in tiles of 8 keys, 3-7 % faster (KL <= 0.0006). `vision_testz`
  now also runs Qwen3-VL mmprojs (timing, `ARC_VIS_LM`).
- **Setup / build** find Visual Studio in more places (scripts/vcvars.ps1: vswhere anywhere, every drive's usual
  folders, a Developer prompt's own compiler).

- **The project is now called arcinferzero** (the repository `rizagp/arcinferzero`; GitHub forwards the old
  `rizagp/arcinfer` address). Entries below this one keep the old name: there "arcinfer" is the project, and before
  2026-10-20 also its first, OpenCL engine. What moved with the name:
  - the data folder is `%LOCALAPPDATA%\arcinferzero` (kernel cache, model packages and expert files, context /
    prefix / session caches, logs, `model-dirs.txt`): an existing `%LOCALAPPDATA%\arcinfer` is renamed once by the
    first exe or script that runs (`src/app_dir.h`, `scripts/app-dir.ps1`), on the same drive, so nothing is copied;
  - Pi's provider is `arcinferzero` (models `arcinferzero/<id>`): the next `pi-arc`, `arc-server` or `setup` replaces
    the old `arcinfer` entry in `~/.pi/agent/models.json`; `pi-arc --model arcinfer/<id>` is still understood;
  - models converted with `tools/convert_gsq.py` carry `arcinferzero.expert_format`; files with the old
    `arcinfer.expert_format` load as before. The web UI keeps its settings (the old storage key is read once).
- **Gemma 4 26B / 12B long prompts**: the sliding layers' attention scores two blocks of 16 keys per step, each
  query load used for both: 11K-token prompt, sliding attention 26B 8.6 -> 7.1 s, 12B 16.5-20.1 -> 13.4-14.0 s,
  bit-identical. E4B and the global layers keep one block (slower with two).
- **Flint groundwork**: the float attention kernels take 128-dim heads and
  attention sinks, attn_prep_g has V without a norm and layers without RoPE, the MoE routing has sigmoid weights
  with an ungated shared FFN. New GPU tests against double / CPU computations: `tests/test_fa.cpp` (every
  attention form, heads 128 / 256, global / sliding ring, with and without sinks) and `tests/test_route.cpp`.
- **Build**: `/W4 /permissive-` (our sources warning-free; third_party as external headers) and a GitHub Actions
  workflow that builds every exe and runs the model-free unit tests on Windows.
- Measured and dropped: Flash Next gate / up on 32-slot tiles (bit-identical, 6.7 -> 8.4 s); larger key blocks or
  other shapes for the head-512 attention (all slower).

## 2026-10-23: Xe2 / Xe3 GPUs (no 8-lane sub-groups); Unsloth Studio statistics and context

- Every exe embeds a UTF-8 code page manifest: model and data paths with any characters work.
- **Gemma 4 26B long prompts ~2 % faster**: the attention's ring slot computed once per 16-key block (global
  layers 16.4 -> 15.2 s at 13,921 tokens), bit-identical.
- **IQ1_S weights** (CPU and GPU decoders; IQ1_M's codebook with one scale per 32 values): agentionai's
  Qwen3.8-27B-AP quants (two IQ1_S `ffn_up` matrices in the IQ3_S file) load. Against ISTA's IQ3_S on this laptop:
  AP-IQ3_S same prompt speed (116 tok/s), generation 4.9 -> 5.05 tok/s, with MTP 11.6-11.9 -> 12.4-12.6 (drafts
  accepted 76 -> 81 %); AP-IQ4_XS generation 4.44, with MTP 12.25 (83.5 % accepted), 18.3 GiB GPU with MTP + DFlash2;
  auto context 40,960 for all three. KL of AP-IQ3_S against its own f32 reference 0.00071 (64 tokens, top token 64 / 64).
- **Qwen 27B MTP / DFlash2 output equals plain decoding again** (verification rows' attention one by one, as single
  tokens: they had been summed differently once the context needed more than one key split); MTP -3 %.
- **Gemma 4 E4B prompts +12-33 %**: its global layers' prompt attention reads an int8 copy of K / V again
  (`attn_ps`), the sliding layers and generation stay float: 13,921 tokens 30.2 -> 22.6 s, 4K 648 -> 724 tok/s;
  KL 0.00075 -> 0.00088 (the limit is 0.015); +0.34 GiB GPU. `ARC_G4_ATT8=0` turns it off.
- **Fixes from DeepSeek's second audit** (static reading, confirmed in the code):
  - a model switch whose model and then the previous model both fail to load no longer leaves the server to crash on
    the next request: it answers 503 until a request names a model that loads;
  - a stop string is cut where it matched at the end of the text, not at an earlier copy of it;
  - a server bound to this machine (the default) answers only requests whose Host names it (localhost, 127.x,
    [::1]): a web page under a name that resolves to 127.0.0.1 can no longer drive it (DNS rebinding);
  - web fetch connects directly, never through a system proxy (through one the checked address was the proxy's);
  - GGUF files: every bound checked without overflow, counts and alignment validated, tensor ranges checked before
    use; invalid head counts / layer counts refused before they divide anything;
  - images larger than 16384 pixels a side or 64 megapixels refused before decoding; `/tokenize` answers 400 to
    invalid JSON; `--ctx`, `--chunk` and the cache sizes clamped (no nan / inf).
- **Flash Next prompt**: the per-tile expert gate / up kernel reads the activation scales token by token (a small
  transpose first): gate / up 8.1 -> 7.0 s per 4K prompt, bit-identical.
- **XMX safer** (from DeepSeek's audit): a GPU module or kernel that cannot be created during the start-up check no
  longer ends the process (it keeps DP4a), and the check now launches the kernels as the engine does (split-K pieces,
  accumulation, the fused SwiGLU, several work-group shapes) and covers the mixture-of-experts tile kernel, which now
  takes XMX too when the check passes (it had been left on DP4a). The check itself is tested here with a right and a
  wrong DP4a variant in the XMX slot (`ARC_XMX_TRY=2` / `=3`); the XMX tile kernels compile for Lunar Lake,
  Battlemage and Panther Lake.
- **Web fetch hardened** (same audit): the address WinHTTP actually connected to is checked before anything is read
  (a host that resolves to a public address first and a local one at connection, DNS rebinding, gets nothing back);
  more reserved ranges refused (192.0.0.0/24, documentation and benchmark ranges, 240/4, IPv4 inside IPv6 judged
  as IPv4: compatible, NAT64, 6to4; Teredo, site-local, 2001:db8::/32, 100::/64); a read never grows the body past its
  limit.
- **Qwen3.8 Flash Next with the DFlash2 drafter** (`--dflash DIR`, incoai/Qwen3.8-Flash-Next-DFlash2): the drafter's
  features are the means of the 4 residual streams, a verification pass of up to 8 rows computes every row as a single
  token is (the HC mixers' single-token kernels for several rows), and rejected drafts roll back the PLE history and
  token ids and the QSA tails. Greedy logits bit-identical to plain decoding (1,097-token prompt, 128 tokens;
  3,291-token prompt with QSA, 96 tokens); 79 % of the drafts accepted, 3.66 tokens per pass, but 5.8-6.6 tok/s
  against 6.4-6.8 plain: decoding here waits on the SSD, and a pass reads the experts of all its rows. Not picked
  automatically.

- **GPUs without 8-lane sub-groups** (Xe2 / Xe3: Lunar Lake, Battlemage, Panther Lake's Arc B390; GitHub issue #1):
  the engine reads the sub-group sizes the GPU supports and builds the two 8-lane kernels (`gdn_recur`, Qwen
  DeltaNet layers; `vit_attn_rows`, Qwen image encoder) with 16 lanes there. Same arithmetic in the same order:
  forced on this Xe-LPG laptop (`ARC_NO_SG8=1`) the 35B's logits are bit-identical and Qwen3.5 4B / 9B describe
  images as before (the 16-lane DeltaNet kernel spills here, Xe-LPG's registers are half as wide; not on Xe2 / Xe3).
- **Gemma 4 E4B: matmuls in one int8 pass** (the float attention stays): KL against f32 0.00075 / 0.00093
  (prompt / generation, 480 tokens), within the user's new limit of 0.015; 13,921-token prompt 45.4 -> 30.1 s,
  generation with the drafter 29.3 -> 32.2 tok/s. 26B and 12B keep both passes (their one-pass KL was 0.098 /
  0.033). Saved E4B states of the two-pass mode are not loaded (state flag 16). `ARC_G4_Q2MM=1`: two passes.
- **XMX for prompt GEMMs** (Xe2 / Xe3 GPUs with XMX, e.g. Arc B390; not this laptop's Xe-LPG): the prompt GEMMs'
  int8 dot products on DPAS, picked automatically and only after a start-up check that the XMX kernels give the DP4a
  kernels' results bit for bit (else DP4a stays). `ARC_NO_XMX=1` turns it off. Untested on XMX hardware so far
  (compiled for Lunar Lake / Battlemage / Panther Lake with ocloc); waiting for a report from issue #1.
- **Unsloth Studio** (arcinfer as a llama.cpp endpoint): the last streamed chunk carries `usage` and `timings`
  together, as llama-server's does (Studio keeps the timings only from a chunk with usage, so it showed no token
  statistics), and `/v1/models` gives the served window as llama-server's `meta.n_ctx` (plus `max_model_len`), which
  Studio reads for the context size.


- **Talk** (web UI, Gemma 4 E4B / 12B): an always-open microphone with speech detection, each utterance sent as
  audio to the model, the answer read aloud sentence by sentence while it streams (browser speech synthesis, voice
  and speed settings), speaking interrupts the AI, a pause inside a sentence joins the two parts. Tested with
  synthesized speech fed to the detector (E4B: speech starts 2.3 s after the user stops talking).
- **Web search while talking** (Talk + Web): the tools run in the background; the AI says it is looking it up
  and talks about the results when they arrive (a message of their own), as the user designed it (F22.4 / F24.5).

## 2026-10-22 (night): web search in the web UI

- **Web search and page reading in the web UI** (the **Web** checkbox): the model gets `web_search` and
  `web_fetch`; the server runs them (`tools/web_tools.h`, WinHTTP: Brave Search's result page, Bing as the
  fallback; HTML to plain text with headings and lists) at `POST /v1/tools/web_search` / `web_fetch`, also while a
  model loads or generates. Each call shows as a card; the answer continues with the results (8 tool turns at
  most); Stop keeps the conversation valid. Only public http(s) addresses are fetched (every host and redirect is
  resolved first; loopback, private, link-local, multicast and reserved addresses are refused, and the connection's
  address is checked again). `--no-web-tools` turns the
  endpoints off. An independent implementation (no code from the AGPL `pi-unsloth-webtools`).

## 2026-10-22: Gemma 4 runs only in its accurate mode; faster sampled tool calls

- **Gemma 4's two-pass mode is its only mode** (the user's decision: "buat apa mode tidak akurat"; Qwen models are
  unchanged). KL against an f32 reference, 480-token code prompt: 26B 0.0042, 12B 0.0002, E4B 0.00003 (one-pass:
  0.0985 / 0.033 / 0.0010). The day's speed work for it: one-token and verification GEMVs and the experts read each
  weight once for both passes, head-512 prompt attention reads each key row once per row, flash-decoding for
  generation, 2048-token chunks with streamed experts. 26B with its drafter ~25 tok/s (one-pass 26.8), after a 14K
  prompt 13.0 (14.9), 4K prompt ~220 tok/s (526); E4B with its drafter 38.1 (42.5). `--q2`, `-Q2` and `ARC_Q2` are
  gone; saved one-pass Gemma conversations are not loaded.

- **The two-pass mode's saved state** (first as the option `--q2`). It is part of the server's context key, its saved conversations and system-prompt cache get their own folders
  (`-q2`), and the saved state carries a flag and the float K / V rows: a conversation saved, a second one held,
  the first one loaded again gives the same greedy answer as a fresh server (26B, 1,511 tokens); a plain server
  refuses the file. Saved states of the other modes are unchanged.
- **GPU memory = the GPU share Windows reports - 1.2 GB** (the user's rule): the share is the user's own setting, so
  the 14 GB Windows RAM reserve and `--windows-reserve-gb` are gone (with a 22 GB share: 17.7 -> 19.2 GiB for
  arcinfer). Gemma 4 26B prompts of the head-512 layers keep `attn_fa_gh` only up to 8 q heads per kv head (the 12B's
  16 spilled and were slower: 4K prompt global attention 4.4-4.9 -> 3.5 s; DeepSeek's audit).
- **`--v4` removed** (the 4-bit V copy for Qwen generation at very long contexts; the user does not use it): the
  option, `-V4`, `ARC_V4`, `ARC_V4_SINK` / `ARC_V4_RECENT`, its kernel paths and buffers (-179 lines). Default
  output unchanged bit for bit (Qwen 35B / 27B with and without drafting, Gemma 26B).
- **Gemma 4 one-pass code removed** (-227 lines): its int8 attention paths, `gemv_glu` and its buffers, the fused
  GELU epilogues, `ARC_Q2_FUSE` / `ARC_Q2_ATT` / `ARC_Q2_KF32`; output bit-identical.
- **Server hardening (audit by DeepSeek, reviewed)**: a request with an `Origin` from another site gets 403 (a web
  page elsewhere could otherwise drive this no-authentication server; local pages and the server's own web UI, also
  opened from another device with `--host`, keep working; command-line clients send no Origin), CORS only for those,
  at most 64 connections, request heads up to 1 MB, an unknown `--host` is an error instead of every interface (IPv6
  literals and names accepted), and a bad option value (`--port abc`) is a message instead of a crash.
- **Sampled tool calls 4x faster**: constrained sampling with top_k > 1 sorted and tested the whole 248K vocabulary
  whenever the 512 most likely tokens held too few allowed ones (after every JSON key); it now stops when the rest
  could not be drawn (e^-20 below the best allowed token). 35B, as Pi samples: 4.8 -> 20.4 tok/s; a 1,240-token
  `write` call 24.3 (greedy 24.4). The same tokens as before in the seeded tests.

## 2026-10-20: the OpenCL engine removed; robustness fixes

- **Two-pass mode for Gemma 4 (`ARC_Q2=1`, opt-in, slow for now)**: every activation in two int8 passes (the value,
  then its rounding residual), the experts too, and the attention in float (f32 query and K, f16 V). KL against the
  f32 reference, 480-token code prompt: 26B 0.0985 -> 0.0045 (decode path 0.104 -> 0.0045), 12B 0.033 -> 0.0003,
  E4B 0.0010 -> 0.00003; drafters stay bit-identical, slots too. Speed work comes next (QUEUE F21.11).
  `tools/ref_forward` now has Gemma 4's per-layer embeddings (E2B / E4B).
- **Gemma 4 26B-A4B with its drafter** (Unsloth's `mtp-gemma-4-26B-A4B-it.gguf`, next to the model): 3 drafts per
  step; server answers of 400 tokens, code 18.8 -> 27.0 tok/s, Indonesian 19.4 -> 21.7.
- **Gemma 4 drafts verified bit for bit**: a verification pass ran several rows per attention launch with the keys
  split by the last row's position, so its logits were not those of decoding each token alone (26B: greedy text
  changed after 7 tokens; 12B / E4B logits up to 1.3 / 0.85 off). One row per launch now: bit-identical over 128
  tokens on the 26B, 12B and E4B, as fast or faster (12B 21.3 -> 23.2 tok/s with the drafter).
- Gemma 4 26B prompts: the experts' gate / up through `moe_rg` with a GELU epilogue (they were per tile because of
  the 704 -> 768 padding): 4K 451 -> 506 tok/s, 8K 405 -> 457; KL unchanged (0.0985 -> 0.0993).
- `tools/ref_forward` reads the 26B's package (its experts from the expert file) and simulates two int8 passes
  (`REF_Q2`, `REF_ATTQ2`): with the matrix inputs and the attention's q / k / v in two passes the 26B's KL would fall
  from 0.0985 to 0.0053 (QUEUE F21: the plan). `runz --dump-logits` also with `--mtp`; `vision_testz` with
  `ARC_PROF=1` prints the encoder's kernels.

- **The OpenCL engine is removed** (`bin\server.exe`, `bin\run.exe`, `src/engine.h`, `kernels/`, the OpenCL copies of
  the vision / audio encoders, `tools/run.cpp`, `bitcheck`, `test_gemm`, `vision_test`, `bench/gemm_bench`, the old
  `stable/` server). arcinferzero (`bin\serverz.exe`, `bin\runz.exe`), which matched its output bit for bit on every
  model, is the only engine: `arc-server -OpenCL` / `pi-arc --opencl` stop with a message, `-Zero` / `--zero` are
  still accepted, the launcher has no Engine field, `setup` builds serverz / runz / clinfo, `tests\run_tests.ps1`
  has no `-Engine opencl|both`. The micro-benchmarks read `zero\kernels`. `runz` now warns about a running serverz /
  runz (it looked for the old names).
- Slots and batches refuse misuse instead of corrupting state: a slot out of range or given twice, a batch over
  drafts not yet settled. A batch pass that throws binds the slot it started from again and changes no slot's
  position or pending drafts (they change once the pass is done); the server fails every request of that pass
  (requests waiting for it to lead their step no longer wait for good) and starts their conversations over. The
  estimated memory per slot equals the allocation exactly (`ARC_SLOT_BYTES=1` prints both; `runz --slots-check`
  and `tests\run_tests.ps1` fail when it counts less).
- MTP block + DFlash2: a step drafted by DFlash2 now also writes its token's row of the MTP block's KV cache; the
  MTP drafts after DFlash2 steps read stale memory there (output was always right; drafts depended on what the
  slot held before). Speed unchanged (35B / 27B, prose and code).
- A failed generation step (alone or batched) fails its request(s) and starts the slot's conversation over; a
  streamed request ends with an error event (`data: {"error": ...}`) instead of a cut stream, and a 500's message is
  escaped as JSON. Tested by `tests/fail_inject.ps1` (in `run_tests`): `ARC_FAIL_PASS=N` makes the engine's N-th
  verification or batched pass throw after its layers.
- QUASAR-QAT's Q4_0 files load (ids from their file names): the Qwen3.5 4B is about as accurate as Unsloth's
  UD-Q4_K_XL (perplexity 14.93 vs 14.83), its Gemma 4 E4B clearly worse than Google's QAT file (48.0 vs 30.4;
  llama.cpp the same). `serverz --list-models` marks mixture-of-experts files (`moe`), which `arc-server` uses for
  its engine line (Gemma 4 26B was described as dense).
- The expert cache's prefetch thread passes its errors to the caller instead of ending the process; the expert
  file is written to a `.tmp` and renamed only when complete.
- JSON nested more than 256 deep is a 400 (it overflowed the connection thread's stack); the build uses `/utf-8`.
- Saved conversations remember the MTP draft count and the slot layout (an older save is not reused).

## 2026-10-19: two conversations at once, Gemma 4 26B-A4B, Flash Next accuracy knob

- **Gemma 4 26B-A4B** (Google QAT Q4_0; arcinferzero, `pi-arc g26`): the dense FFN and 128 routed experts side by
  side, experts streamed from the SSD like the 35B's (gate / up from one tensor; GELU; the router in f32 on its own
  input norm; a scale per expert). Prompt 450 tok/s (4K), generation ~20 tok/s; KL 0.07 against the CPU reference
  (`tools/ref_forward` now runs its experts), the size of int8 activations' own effect. Images too: its vision
  encoder's 72-dim heads (`g4v_qkv_hd`) and std_bias / std_scale; the embeddings match `tools/gemma4v_ref.py` (cosine
  >= 0.999997; `bin\vision_testz.exe` writes them for arcinferzero). `--ctx auto` keeps no memory for an encoder it
  cannot run.
- The server prints why a model cannot load instead of stopping silently.

- **Conversation slots** (arcinferzero; `serverz --slots N`, `arc-server -Slots N`, `pi-arc --slots N`): the server
  holds N conversations, each with its own KV caches, DeltaNet / conv states, QSA keys, MTP / DFlash2 state and
  snapshots (`Engine::add_slots` / `bind`: the state buffers are re-pointed to the slot's memory, no copies).
  Requests in different slots take turns step by step by GPU time used, so two Pi sessions answer at the same time;
  a short question during another slot's 46K-token prompt: 49 -> 9.6 s. Logits bit-identical alone and interleaved
  (`runz --slots-check`: 27B, 35B with MTP and DFlash2, Flash Next with QSA); one slot unchanged.
- **Batched decoding for Gemma 4** too (26B-A4B: two answers at once 19.8 -> 26.2 tok/s); `--ctx auto` with slots on
  streamed MoE models aims at 128K in all and keeps at least 32K per slot (26B 2 x 32,768 instead of 2 x 8,192).
- **Batched decoding over slots** (Qwen dense and MoE): requests generating at the same time run their next tokens
  and drafts in one engine pass (`Engine::forward_batch` / `verify_batch`; every slot drafts in its own state, each
  settles its own drafts). Two answers at once: 27B 7.3 -> 10.5 tok/s together (prose), 11.6 -> 15.4 (code); 35B
  20.2 -> 24.7, 29.0 -> 30.7. Logits bit-identical to each slot alone; `runz --bench-batch N` times it (27B, no
  drafts: 2 / 3 / 4 slots 1.78 / 2.43 / 3.04x).

- `ARC_HC_F32` (arcinferzero, Flash Next prompts; `4` the default since 20 Oct, `0` the old single int8 pass): the hyper-connection mixers of prompt chunks with more
  precision. `4` takes the up projection's input as two int8 passes (values, then their rounding residual): KL
  against the f32 reference 0.0647 -> 0.0483, top token 324 -> 331 of 363, prompts 5-7 % slower. `1` f32 for both
  projections (KL 0.0369, about 22 % slower), `2` / `3` the down / up projection only. The default (int8) is unchanged.
- `tools/ref_forward` reads arcinferzero packages; `runz --kl-ref` reports top-token agreement.

## 2026-10-18: SSD expert prefetch on the copy engine, 35B chunks of 8192, drafter per model, long contexts

- **Deep expert prefetch** (arcinferzero, prompts of >= 2048 tokens): while the GPU runs a layer, a thread reads every
  missing expert of the next layer from the SSD and the GPU's copy engine copies them into their cache slots; the
  kernels wait on an event. Flash Next prompts 4096 tokens 107 -> 166 tok/s, 9000 116.9 -> 144.5, 32K 131.7 -> 162.6;
  35B 32K at ctx 131K 488.6 -> 557. Logits bit-identical; 56 regression runs (small caches, simulated GPU
  out-of-memory, 40 prompts in a row) identical. `ARC_XCOPY=0` turns it off.
- **Qwen3.6 35B prompts in 8192-token chunks** (streamed MoE): 32K prompt 533 -> 604 tok/s, 128K 306 -> 324,
  generation about the same (ctx 262K; expert cache 81 -> 77 %).
- **Drafter per model** in Pi's `/model` and `pi-arc`: `<id>-nodf` drafts with the MTP block only, `<id>-df` with
  DFlash2 only (the MTP block is not loaded: 27B context 69,632 -> 94,208). 35B: Python code fastest with `-df`
  (49.9 tok/s), Indonesian with `-nodf` (21.9). `--draft auto|mtp|dflash` for the server and `runz`.
- `--v4`: a 4-bit copy of V for generation, opt-in (35B at 250K depth 10.6 -> 11.5 tok/s, KV +25 %); the
  automatic context gives the 35B and Flash Next 131,072 tokens; Flash Next's QSA scores 8 rows per work-group
  (125K prompt 78.6 -> 123.4 tok/s).
- Fixes: a server switching models now keeps one Level Zero context (the new model used the old context's copy
  list, events and staging buffer); the expert prefetch thread's read errors no longer end the process; GPU
  out-of-memory during a prefetch is handled; numeric `ARC_*` knobs that divide are kept >= 1; `/fp:precise` for the
  accuracy tools.
- `scripts/hf-download.py` keeps trying until a download is complete (a router that drops out is waited out), with
  no speed limit by default and several files per command.
- Docs: [docs/before-after.md](docs/before-after.md) (every change with its numbers before and after, accuracy and
  consistency), a new README overview.

## 2026-10-16 (night): DFlash2 block drafter for Qwen3.8 27B

- `--dflash DIR` (runz, serverz) loads a z-lab DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2): 5 layers fed by the
  target's hidden states after layers 5 / 19 / 33 / 47 / 61 (fc + norm into a context K/V ring of 4096 positions,
  attention window 2048), a block of 8 rows [token, mask x 7] per draft with two-tap grouped dynamic convolutions
  around attention and MLP, the target's output head and the candidate selector (top 16, greedy bilinear path).
  Weights BF16 -> Q8_0 on the GPU (~1.9 GB), selector codebooks read from the mapped file.
- With the model's MTP block also loaded (the default with `--dflash`), each step picks the drafter: DFlash2 while
  its recent accepted drafts per pass stay >= 2 (`ARC_DFL_MIN`), else the MTP block (3 drafts), with a DFlash2 step
  every 8th to re-check. `ARC_DFL_P` (default 0.3) drops a DFlash2 path's tail below that probability.
- 27B, greedy, answers identical to plain decoding: Python code 4.85 -> 11.5 tok/s (MTP alone ~8.9; 89 % of drafts
  taken, 6.9 tokens per main-model pass); Indonesian prose 4.67 -> 5.6 (MTP alone 5.8-6.3; DFlash2 takes ~11 % of
  its drafts there: its training data is mostly English and code).

## 2026-10-16 (evening): Qwen3.8 Flash Next on arcinferzero

- New architecture `qwen4exp` (Qwen3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_S, split GGUF of 2 parts) in arcinferzero:
  4 hyper-connection streams with their mixers before every block, Gated DeltaNet with a sigmoid output gate, gated
  attention, 512-expert MoE (10 active) streamed from the SSD with 640-wide experts (down rows padded to 768), the
  PLE n-gram block (rows read from the 26.8 GiB table in part 2 per token, its conv history kept with the recurrent
  state) and QSA sparse attention (indexer, pooled block keys, top 512 blocks per query; dense below ~2K tokens).
- Accuracy against the CPU reference (`tools/ref_forward`, which now covers QSA and dumps per-block residuals):
  short chat prompt KL 0.0014 ("Jakarta" 0.92 vs 0.93). On a 337-token raw text the KL is 0.15, the same as the
  reference with int8 activations (0.13): this model is more sensitive to int8 activation quantization than the
  others (every matrix group adds 0.02-0.10).
- Speed (first numbers, 32 GB laptop, expert cache 7.9 GiB = 12 %): generation ~4 tok/s, short prompts 3-6 tok/s
  while the cache is cold. The expert file (56 GB) is written once on the first start (~9.5 minutes).
- Expert cache: with more than two record sizes (Flash Next has 5 type mixes) one pool of the largest record serves
  every layer; separate pools each needed a whole layer and ran out of GPU memory.
- `src/quants.h`: Q8_0 rows whose length is not a multiple of 256 dequantize fully (the CPU reference read only the
  first 512 of 640 values of Flash Next's last shared expert).
- 27B and 35B logits unchanged (bit-identical).

## 2026-10-16 (afternoon): extra model folders, kernel timings on Level Zero

- Models can live on another drive: `%LOCALAPPDATA%\arcinfer\model-dirs.txt` (one folder per line) or
  `ARC_MODEL_DIRS` (`;`-separated) adds folders in the Hugging Face cache layout or with `.gguf` files directly in
  them. The Hugging Face cache is searched first, so a model kept in both places still loads from there.
- `runz --prof-gen` on arcinferzero now reports per-kernel times (Level Zero kernel timestamps; before: 0 ms).
- `zero/page_probe.cpp`: GPU page size and random-read speed by span (2 MiB pages, ~4 MiB GPU cache, random 256 B
  reads slow down above ~1 GiB; the Level Zero virtual-memory API is refused on this Windows driver).

## 2026-10-16: arcinferzero is the default engine; MTP up to 7 drafts

- `arc-server`, `pi-arc` and the launcher start arcinferzero (Level Zero) by default; `--opencl` / `-OpenCL` (launcher:
  Engine "arcinfer") runs the OpenCL server as before. `--zero` is still accepted. Tested the night before: identical
  logits to OpenCL for every model (35B with the same prompt chunking), images and audio identical (E4B, 12B, 9B),
  every API format, Pi, tool calls, MTP identical to plain decoding.
- MTP drafts up to 7 on arcinferzero (`--mtp-drafts 1..7`): a verification pass writes its DeltaNet states to a second
  buffer and keeps its inputs; rejected drafts are undone by replaying the recurrence over the accepted tokens (one
  state copy instead of one per draft, no 150 MB copies per step). Default drafts: 35B 1 -> 3 (23.9 -> 26.1 tok/s),
  4B 2 -> 3 (22.9 -> 24.8); 9B 2, 27B 3 (5 and 7 are not faster yet: each draft runs the MTP block with the full
  output head).
- Multi-token GEMVs (verification of 3..7 drafts) load each token's activations where they are used instead of
  holding all of them in registers: 4-7 drafts no longer spill (27B, 5 drafts: 3.8 -> 9.2 tok/s).
- Qwen3.6 35B decode attention: attn_dsplit (4 q heads per sub-group, dims split over 4), 1.16 -> 0.94 ms per layer at
  64K; verification rows stay bit-identical to single-token decoding (fixed key splits, per-row P bound).
- Gemma kernels without register spills: attention prep with the large register file and its norm / RoPE work in
  pieces of 8 (8.8 KB -> 0 per thread), the post-attention / post-FFN norm sized to the model (3.7 KB -> 64 B);
  identical logits. E4B prompt 2K 733 -> 827 tok/s, 16K 571 -> 625; 12B 271 -> 289, 218 -> 228.
- Fixes: constrained JSON / tool calls no longer allow a comma after the last allowed key (a `json_schema` answer
  dead-ended as `{"name": "x", "age": 34,`); `/api/tags` lists the loaded model with `--no-swap`.
- `arc-server stop` waits for the process to end plus 5 s: a server started while Windows was still releasing the
  previous one's memory sometimes ran at half speed for its whole life. The server and runz opt out of Windows power
  throttling (EcoQoS) as a precaution (no effect measured).
- Diagnostics: `ARC_ZE_KPROPS=1` prints each kernel's spill and its build options; `zero/grf_probe.cpp` (also
  `grf_probe FILE.cl KERNEL "OPTIONS"`: one kernel's spill); `runz --bench-pp` prints the time of each generation step
  (min / median / max) and takes `--prof-gen`; `ARC_CONV_GRF=256` (experiment: no spill, not faster).
- `ARC_DRAFT_VOCAB=N` (off): the drafts' output head over the first N token ids only; faster for English code, slower
  for Indonesian (its tokens lie beyond 65536).

## 2026-10-15 (night): Level Zero tests, Qwen tool-call constraint, saving the held conversation

- Constrained tool calls also for Qwen's XML format (`<tool_call><function=..><parameter=..>`): the function and
  parameter names are the tool's own, every required parameter is there before `</function>`, non-string values are
  JSON of their schema type, string values cannot contain `</parameter>` / `<parameter=` (the parser would cut them
  there). Qwen: only after the reasoning (it never calls tools inside it). Gemma stress test (E4B, temperature 1.3, a
  long edit forced through tool_choice): 2/10 well-formed calls without the constraint, 9/10 with it.
- `POST /sessions/save`: the held conversation goes to the session cache and the call returns once it is on disk;
  `arc-server stop` / `restart` call it first, and the server does it before a model switch, so continuing that
  conversation afterwards loads it instead of processing it again.
- `ARC_LOG_TOOLCALLS=file`: every answer that writes a tool call is appended as a JSON line (raw text, parsed calls,
  tokens the constraint corrected), to see what a model wrote when a call fails.
- Fix: `logprobs` sent tokens that are part of a UTF-8 character as raw bytes, making the JSON invalid; the text is
  now valid UTF-8 (U+FFFD for a partial character), `bytes` keeps the exact bytes.
- Web UI compaction: the summary request asks for a list that copies every specific item exactly (temperature 0);
  with the first wording Qwen 4B dropped one of seven code words, now none.

## 2026-10-15: web UI compaction

- `--preserve-thinking` (server; `pi-arc --preserve-thinking`, `arc-server -PreserveThinking`): Qwen3.6's template
  option, earlier turns keep their reasoning in the prompt (requests can set `chat_template_kwargs.preserve_thinking`).
  The prompt then only grows, so a new message reuses the whole cache; the context fills sooner. Only for models
  whose template has the option (Qwen3.6 35B); Qwen3.8 27B always keeps the reasoning.

- Qwen: a third recurrent-state snapshot at the start of the assistant turn that follows the user's last message.
  Qwen's chat template drops the reasoning of earlier turns once the user writes again, so every new message in a
  Pi session diverged right after the previous user message and was processed again from the end of the system
  prompt; now only the last turn is processed again. In `server-new.exe` until tested on the GPU.

- Constrained tool calls (Gemma 4; `tools/tool_constraint.h`): while the model writes a tool call, only tokens that
  keep the call well formed and fitting the tool's parameter schema are drawn: required arguments cannot be left
  out (Pi's edit without `path`), keys are the tool's own (no code as a key, no `edits` inside an edit), strings use
  `<|"|>`, enum values and integers are respected, and the call ends with `<tool_call|>`. A call that is already well
  formed is sampled exactly as before. The request log shows "tool call: N of M tokens corrected" when the
  constraint changed something; `ARC_TOOL_CONSTRAINT=0` turns it off. In `server-new.exe` until tested on the GPU.

- The disk session cache is on by default (16 GB; `pi-arc --session-cache 0` turns it off): a Pi conversation that
  another session or the web UI pushed out of the GPU is loaded from disk in seconds instead of processed again.

- The web UI shows the context the conversation uses ("context 12.3K / 40.9K") and compacts it: when the next
  request would nearly fill the context, or the server answers `exceed_context_size_error`, the earlier messages are
  summarized by the model (thinking off, the cached prefix reused) and the conversation goes on with that summary
  plus the latest exchange. A history too long to summarize at once is summarized in parts. The Compact button does
  it on demand; Stop cancels it and keeps the conversation. The old messages stay on screen, dimmed, with the
  summary under "Earlier messages compacted".

## 2026-10-15: the full API (OpenAI, Anthropic, Ollama formats), structured output, one server source

- One server source: `zero/serverz.cpp` is `tools/server.cpp` built with `ARC_ZERO` (Level Zero engine, its own
  data folder `%LOCALAPPDATA%\arcinfer\zero`); `setup.ps1` builds both, and `runz`.
- Sampling: `frequency_penalty`, `repeat_penalty` / `repeat_last_n`, `logit_bias`; `logprobs` / `top_logprobs`;
  `n` > 1 (without streaming).
- Structured output: `response_format` `json_object` / `json_schema` constrains sampling with a JSON-schema prefix
  checker (`tools/json_constraint.h`), so the answer always parses and fits; `tool_choice` `required` / a named
  function starts the answer with a tool call.
- Other formats on the same handler (`tools/api_compat.h`): OpenAI `/v1/completions` (raw prompt) and
  `/v1/responses`; Anthropic `/v1/messages` (+ `count_tokens`); Ollama `/api/chat`, `/api/generate`, `/api/tags`,
  `/api/show`, `/api/ps`, `/api/version`; `/v1/audio/transcriptions` and `/translations` (Gemma 4 audio).
- `tests/test_api_compat.cpp`: 44 CPU-only checks of the JSON checker and the request translation (all pass).
- Launcher: the API address for each format with a Copy button. `pi-arc --zero` / `--no-stream`, `arc-server -Zero`
  / `-NoStream`; startup messages name the engine.
- Not yet: `/v1/embeddings` (needs an embedding model architecture the engine does not have).

## 2026-10-14: arcinferzero: Qwen3.6 35B with its experts streamed from the SSD (Level Zero)

- New engine in `zero/` (arcinfer itself is unchanged): `zero/clrt.h` runs arcinfer's OpenCL kernels under Level
  Zero (native binaries from the shared kernel cache, USM buffers). Same results bit for bit, faster: 35B TTFT 6.45 ->
  4.07 s, generation 16.9 -> 18.3 tok/s; load 4.3 -> 2.3 s (4B).
- Routed experts live in `%LOCALAPPDATA%\arcinfer\zero\<model>-<id>.experts` (GPU block form, one 4 KB-aligned record
  per expert; made once from the GGUF, 15.1 GB / 144 s for 35B; remade when the GGUF changes; may be deleted) and are
  read with unbuffered I/O (no Windows file cache) into an LRU cache on the GPU. The expert kernels take each
  expert's address from a table (`-DXCACHE`); prompts use the back of the LRU list as their transient area and read
  the next layer's missing experts on a thread while the GPU works; the cache's contents are saved (`.lru`) and loaded
  again at start (10.6 GB in ~6 s).
- Cache size: automatic so the process stays within RAM - 14 GiB (Windows) and the GPU share - 1.4 GiB;
  `--expert-cache-gb`, `--expert-cache-pct`, `--windows-reserve-gb`, `--no-stream`. When the GPU runs out of memory
  anyway (the share is shared with every program), the cache gives back 1 GiB and the launch is retried.
- Qwen3.6 35B at 262K context, 4096-token chunks: MTP 22.2 tok/s (cache 79 %, hit 97.6 %), without MTP 19.3 tok/s,
  time to first token 3.3-3.5 s for 2,216 tokens (arcinfer: 4.1-6.5 s at 40K context); output identical to arcinfer.
  With images and MTP together: 262K context, cache ~70 %.
- `bin\runz.exe` / `bin\serverz.exe` (build `zero\runz.cpp` / `zero\serverz.cpp`); `arc-server -Zero`,
  `pi-arc --zero` and the launcher's Engine field choose it; without them the usual arcinfer runs.

## 2026-10-14: Qwen3.6 35B images; projectors that leave too little context are opt-in

- Qwen3.6 35B-A3B reads images with Unsloth's `mmproj-BF16.gguf` (Qwen3-VL, 27 layers, 72-dim heads): text in a
  1920x1080 screenshot quoted exactly; 1,008 image tokens encoded in ~3 s (~60 s the first time: kernel compile).
- `--ctx auto` now counts the image encoder of MoE models too (it reserved nothing). For 35B the encoder (~1.1-1.3 GB)
  leaves only 4,096 tokens at 18.3 GiB, so a projector that cuts a model's auto context below 16K (to less than
  half of it) is no longer part of the default variant: `35b` stays MTP without images (40,960 tokens), images go
  through `35b-vision`. Other models keep their variants.

## 2026-10-14: launcher window, GPU memory by part, monitoring

- `arc-launcher.cmd` / `scripts/arc-launcher.ps1`: a window with every model and variant (MTP, images, automatic
  context, GPU memory at that context, compatibility), how the automatic context was counted for the selected one,
  a field for every server option, and the running server's state, GPU memory by part, requests and speeds. Start /
  Stop go through `arc-server`; the server keeps running when the window closes. Every field starts at the selected
  model's best value (`--list-models` "defaults": chunk, MTP drafts, image tokens, ...); changed fields are marked
  and are the only ones passed on; "Set as default" saves them and the default model, "Reset options" clears them.
- `arc-server.ps1`: every server option (`-MtpDrafts`, `-ThinkBudget`, `-ImageMinTokens` / `-ImageMaxTokens`,
  `-BudgetGb`, `-AttnF32`, `-NoPrefixCache`, `-PrefixCacheGb`, `-NoSwap`, `-ListenHost`, `-SessionDir`); a running
  server is restarted when one given differs (a plain `start`, as pi-arc does, keeps it).
- The engine counts its GPU memory by part (weights, MTP block / drafter, working buffers, context); `server.exe
  --list-models` gives each variant's counts as `--ctx auto` used them ("memory": device, kept free, snapshots,
  encoders, weights, mtp, buffers, context, context per 1K tokens), `/props` the loaded model's ("memory") and
  monitoring ("stats": requests, busy, filled context, the last request's timings). `/props` reports audio when the
  model has an audio encoder (it said false). The ctx-auto cache key carries layout version m9.

## 2026-10-13: Qwen3.6 35B: prompt experts through the int8 GEMM, MTP, more context; Qwen prompt attention

- MoE prompt chunks of >= 1152 tokens run the experts through the int8 GEMM (kernels/gemm.cl `MOE_TILES`): per group
  of 64 experts the raw rows are decoded to W8, then one GEMM over that group's slot tiles. Qwen3.6 35B UD-Q3_K_M,
  2048-token prompt: 476 -> 642 tok/s; per-layer outputs equal to the per-tile kernels (differences ~1e-7).
- `decode_w8_sl`: whole rows (up to 16 superblocks) are copied to local memory with coalesced loads, then decoded
  from there (types without mins). IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms per 2048-token chunk; Q8_0 / Q4_0
  use their vector block decoder (`decode_w8_vb`): Q8_0 211 -> 72 ms.
- MoE buffers share memory with buffers only used in the attention / DeltaNet half of a layer (expert activations
  over z_, expert outputs over conv_out_ / o_ / qrope_): 167 MB. Automatic context: `35b` 86K -> 102K (94K before
  the GEMM staging), `35b-s` 217K.
- Single tokens: DeltaNet qkv + z (+ alpha / beta when Q8_0) stacked into one GEMV (views, nothing stored twice).
  35B 19.2 -> 19.8 tok/s, 27B 4.75 -> 4.8-5.0, output identical; 9B unchanged (qkv / z types differ in most layers).
- MTP for Qwen3.6 35B (Unsloth's MTP repository, `35b-mtp`): MoE in the MTP block, Q3_K on the GPU (decode32), expert
  types with mins (Q4_K / Q2_K / Q5_K) in the expert kernels (activation sums computed in the kernel). Output
  bit-identical to plain decoding; 19.7 -> 22.3 tok/s (83 % of drafts accepted), prompts -3.4 %, automatic context
  40,960 (`35b-mtp-std`: 98,304). MoE models default to 1 draft per step (1 and 2 equally fast; one DeltaNet state
  copy less). Verification rows of models whose single token takes `attn_decode` (35B: 2 kv heads) now take it as
  well, row by row: `attn_split` split the keys differently and MTP output drifted after ~40 tokens of a 2K prompt.
- Qwen prompt attention (`attn_q8`): 4 q heads per sub-group where the GQA ratio allows. 21,590-token prompt:
  35B 9.4 -> 8.4 s, 9B 8.2 -> 7.0 s of attention, same output; 27B (6 q heads per kv head) unchanged.
- Server: Qwen3-VL projectors also load for MoE models (a projector whose output width does not match the model is
  refused); no 35B mmproj was available to test with.
- tools/bitcheck: residual stream per layer and logits of single-token vs 2-token passes and of the MTP flow
  (draft, verify, reject / accept); tools/ref_forward reads standard GGUF experts.
- Tried and left off: Q8_0 GEMV through `gemv_v` (slower), down + combine fused for single tokens (212 vs 211 ms),
  `attn_ps` for Qwen shapes (slower), Q4_0 requantization of the MTP block's experts (replaced by native Q3_K / Q4_K).
- The ctx-auto cache key carries layout version m8.

## 2026-10-13: Qwen3.6 35B: both Unsloth files selectable, router + shared expert in one GEMV

- `35b` = UD-Q3_K_M (IQ4_XS down experts; automatic context 94,208), `35b-s` = UD-Q3_K_S (204,800); both in Pi's
  /model (several files of one repository are named with their file), and the server switches between them.
- MoE layers keep the router rows (padded to 384) and the shared expert's gate and up rows in one Q8_0 buffer:
  a single token runs one GEMV over the 1,408 rows instead of three small ones (each paid a launch's fixed cost);
  the three matrices are sub-buffers of it for prompts, so nothing is stored twice. Qwen3.6 35B UD-Q3_K_M
  generation 17.4 / 17.8 / 18.5 -> 18.2 / 18.8 / 18.8 tok/s (alternating runs), output identical.
- Tried and left off: split-K GEMV for matrices with few rows (`ARC_SPLITK`): 512-row matrices 10-15 % faster,
  larger ones slower, no gain overall; per-type GEMV rows / kernels tuned on single repeated matrices: no gain in
  the model (Qwen3.8 27B). Qwen3.8 27B with its MTP block measured 4.5 -> 7.9 tok/s on code (2 drafts, 91 %
  accepted).
- The ctx-auto cache key carries layout version m4.

## 2026-10-13: memory: Q8_0 stored once, encoders reserved by --ctx auto; `35b` alias for Unsloth's GGUF

- Q8_0 matrices keep their repacked GEMM copy only with resident weights of a dense model; MoE models and
  just-in-time models decode the raw rows into the same form per prompt chunk (decode_w8, PER32). Qwen3.6-35B-A3B
  UD-Q3_K_S: GPU 16.47 -> 15.04 GiB, load 62 -> 28 s, automatic context 61,440 -> 208,896 tokens (a 24K-token prompt
  at 356 tok/s); prompts ~4 % slower (decoding per chunk), results identical.
- `--ctx auto` keeps the GPU encoders' memory free (weights + buffers for the largest image / a 30-second clip; they
  upload on first use and were not counted): Qwen3.5 9B 217,088 -> 143,360 tokens (~1.4 GB for the vision encoder at
  2048 image tokens), Gemma 4 E4B unchanged at 131,072 with 1.6 GB kept. The ctx-auto cache key carries an engine
  memory-layout version ("m2"), so stale contexts are computed again.
- `qwen3.6-35b-a3b` / `35b` (`pi-arc 35b`) also name Unsloth's `Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`.
- Qwen3.5 9B vision attention (72-dim heads): q read again per key tile instead of held in registers (spills
  295 -> 164); speed unchanged at ~1.9 TFLOPS (64-dim heads: 2.8). The compiler keeps 128 registers per thread for
  this kernel despite -cl-intel-256-GRF-per-thread; going further needs another kernel layout.

## 2026-10-12: GEMV bandwidth, Qwen3.5 9B images, Qwen3.6-35B-A3B from standard GGUFs

- GPU raw forms regrouped for aligned vector loads: Q8_0 (the 8 scales first; GEMV 37 -> 77 GB/s), IQ2_XXS /
  IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S (the f16 scale moved behind the quants; IQ2_XXS 27 -> 60, IQ3_XXS 41 -> 62,
  IQ3_S 61 -> 66 GB/s); `gemv_v` vector decoders for the codebook types where faster. Token generation, share of
  the measured ~75 GB/s: Qwen3.5 4B 77 -> 79 %, 9B 83 -> 86 %, Gemma 4 E4B 76 -> 81 %, 12B 80 -> 82 %, Qwen3.8 27B
  65 -> 72 % (4.3 -> 4.7 tok/s). All regression tests pass.
- Qwen3.5 9B images (head size 72, width 1152): `vit_attn_rows` / `vit_qkv_rope` take the head size as a build
  option (72-dim heads read K / V with interleaved sub-group block reads: 6.0 -> 1.6 s of attention for a 1222-token
  image), a LayerNorm for any width; 0.38 % from the f32 reference (tools/vision_test.cpp, generalized).
- Qwen3.6-35B-A3B from standard GGUFs (Unsloth UD-Q3_K_S: IQ3_XXS / IQ3_S / IQ4_XS experts): `moe_gateup_q` /
  `moe_down_q` (kernels/decode.cl) on the experts' raw rows, each decoded weight serving a tile of 16 slots;
  generation 17-18 tok/s, 2048-token prompts 473 tok/s (150 with 8-slot tiles re-decoding per tile), automatic
  context 61K at 17 GiB. Prefill and token-by-token agree (top-1 0.9718 / 0.9712).

## 2026-10-12: Google's Gemma 4 QAT models (Q4_0), E4B images and audio on the GPU

- **Q4_0** weights (Google's QAT GGUFs): the GPU raw form regroups each 8-block superblock (8 scales, then the
  16-byte nibble groups, same 144 bytes) so every block is one aligned vector load. Token generation reads them in
  `gemv_v` / `gemv_mt_v` / the fused `gemv_glu` at 66-73 GB/s (bit-identical to `gemv`, ~1e-7 from the CPU
  reference); prompts decode them losslessly into the GEMM's native Q8_0 form (per-32 f32 scales; there is no such
  W8 variant: code 104 was read as per-16 + mins). Google Gemma 4 12B: 9.1 tok/s (85 % of the measured bandwidth),
  17-20 with the drafter; KL 0.0206 vs its f32 reference, greedy = llama.cpp's. E4B: 20.9 tok/s, 26.8 with the
  drafter, prompts 788 tok/s.
- Google's GGUFs omit `attn_k_norm` on shared-KV layers (they compute no K): optional there.
- Google's chat template (2026-07-09) concatenates adjacent string literals (`"a " "b"`), which minja cannot parse:
  the server crashed at start. Adjacent literals inside `{{ }}` / `{% %}` are now joined first; renders identical
  to jinja2 (tests/test_chat_jinja.py, new tests/data/gemma4_google_template.jinja).
- **E2B / E4B images** (mmproj gemma4v, src/vision_gemma4v.h): a 16-layer ViT on the GPU (f16 weights, f32
  activations, clamped linears, RMS sandwich norms, NEOX 2-D rotary, unscaled attention, 3x3 pooling), 0.11-0.17 %
  from an f32 numpy reference (tools/gemma4v_ref.py); a 1120-token image in 3.2 s. The FFN uses gelu_tanh as
  Google's config says (llama.cpp falls back to gelu_quick for these files). In the language model, image rows take
  the per-layer embedding of token 0 and stay causal (llama.cpp: E2B / E4B are always causal).
- **E2B / E4B audio** (mmproj gemma4a, src/audio_gemma4a.h): log-mel and the subsampling convolutions on the CPU, the
  12-layer conformer on the GPU; 7.2 s of speech encoded in 226 ms and transcribed word for word.
- `--image-min-tokens` / `--image-max-tokens` for Gemma 4 12B (default 1120..2048: small screenshots were read
  wrongly at ~140 tokens); Gemma models with an image projector keep one image per prompt step (`min_chunk`).
- Model list: projectors named `*-mmproj.gguf` (Google) are no longer listed as models; Qwen3.6-35B-A3B GGUFs
  without GSQ experts are listed as unsupported (they failed to load). setup.ps1 / docs: g12 and e4b now download
  Google's QAT files plus Unsloth's drafters (copied next to the model).
- `run.exe` prints the weight bytes the GEMVs read per token and the resulting GB/s.

## 2026-10-12: Gemma 4 12B reads small text in images (1120..2048 tokens per image)

- Gemma 4 12B images now become 1120 tokens by default (`--image-min-tokens` default 1120, was 70 as in
  llama.cpp): its encoder turns each 48x48-pixel patch into one token with a single linear layer, so a ~550-pixel
  screenshot became ~140 tokens with ~10-pixel text per patch, and the model misread words ("BORENK", "AEROWR") and
  invented numbers in a spec table. Scaled up to 1089 tokens it read both test screenshots exactly (five 5x3
  tables, part numbers, prices). Large images may now use up to 2048 tokens (`--image-max-tokens` default 2048,
  was 1120, the most Gemma 4 was trained with); at 2025 tokens it still read the logo and the 4x-upscaled table
  screenshot exactly. Cost: 1120 tokens ~0.7 s encoder (CPU) + ~7 s prefill; 2025 tokens ~1.5 s + ~11 s.
- `--image-min-tokens` / `--image-max-tokens` apply to Gemma 4 12B too (before: Qwen only); scaling up no longer
  rounds past the maximum.
- Image placeholder ids in the prefix cache include the token count, so a saved conversation whose image was
  encoded with other limits is not matched against the new image tokens.

## 2026-10-12: Qwen3.5 9B keeps its decoded weights (short prompts ~4x sooner)

- `--ctx auto` now also chooses whether the decoded int8 (W8) weights stay on the GPU: they do when that still
  leaves at least min(128K, trained) tokens of context; otherwise they are decoded just in time, with the largest
  context, as before. Before, a fixed 9 GB budget decided, so the 9B (raw + W8 12.4 GB) decoded every matrix again
  on every request. Qwen3.5 9B profile (11,240-token prompt, 27.4 s): GEMMs 76 % at ~7.7 TOPS (the kernel's
  practical ceiling; its Q6_K per-16 matrices cost only ~4 % more than the per-32 ones, not worth requantizing),
  decoding 1.4 s. Resident: 83-token prompt 1.05 -> 0.27 s to the first token, 11K prompt 27.4 -> 26.6 s; context
  262,144 -> 217,088 at 18.3 GiB (17.0 GiB used). The other models keep their mode and context (4B, E4B resident;
  12B, 27B just in time: resident would not fit). An explicit `--budget` keeps the old rule.
- The `ctx-auto.txt` cache key now includes this choice (before, a `--budget` run reused a context computed for
  just-in-time weights: the 9B with resident weights then took 17.8 GiB).

## 2026-10-12: image input for Qwen3.5 4B (in Pi, the web UI and the API)

- Qwen3.5 4B reads images: its `mmproj-F16.gguf` (Qwen3-VL projector) is found next to the model and runs on the GPU
  (`src/vision_qwen3vl.h`). `--list-models` and Pi's model list mark it `text, image`; Pi's `read` tool, the web
  UI's Media button and OpenAI `image_url` parts work. Checked in real Pi: it read the text of a test image exactly
  ("HELLO 42") and described a photo correctly (ship, terminal and crane names), 2,320-token prompt with two images
  in 2.2 s.
- Encoder precision: with the language model's int8 GEMMs the encoder drifted 10-23 % from an f32 CPU reference on
  photos. A CPU emulation of each rounding (`vision_test`, `VT_EMU`) showed every layer's int8 activations
  contributing, and int8 attention a further 7-9 %. The encoder now runs f16 weights with f32 activations
  (`vit_gemm_f16`, 2.4 TFLOPS) and f32 attention: 0.2-0.5 % error, worst token cosine 0.999. A typical image
  (~250 tokens) encodes in ~0.4 s, a full-size 1080p screenshot (1008 tokens) in ~2.3 s; the first image after a
  start also uploads the encoder (~3 s).
- Language model: M-RoPE (Qwen3-VL interleaved sections 11/11/10): image tokens get (t, h, w) = start + (0, row,
  column) of the merged-token grid, and the text after an image runs max(grid w, h) positions on instead of the
  token count (`Engine::rope_shift`, per-row positions in `attn_prep`). Text-only output is bitwise unchanged
  (regression tests: same KL to 5 decimals).
- `/v1/models` also carries llama-server's own list: a `models` array whose `capabilities` hold `multimodal` for
  the models with an image encoder (next to OpenAI's `data` with `capabilities`, `input_modalities`, `type: vlm`),
  for clients that treat arcinfer as a llama.cpp server.
- Encoder attention: one query row per lane (SIMD8) with q and the accumulators in registers, K / V rows broadcast,
  4-key tiles, native_exp (`vit_attn_rows`): 1008-token image 1118 -> 588 ms (2.7 TFLOPS), 4096 tokens 36.1 ->
  15.3 s, 2048 ~11 -> 5.0 s; error against the f32 reference unchanged (0.2-0.6 %).
- Images are scaled to 256-2048 tokens (32x32 pixels each; `--image-min-tokens`, `--image-max-tokens`): at 1024 a
  6750-pixel logo was misread ("Logan" for "Logam", llama.cpp at 1024 too), at 2048 read right (~8 s to encode). Not yet:
  Qwen3.5 9B (its encoder is 1152 wide with 72-dim heads), Gemma 4 E4B, the 12B's encoder on the GPU.

## 2026-10-12: just-in-time weight decoding 2.7x faster (27B prompts); GEMM measured against its ceilings

- `decode_w8_sb` (the W8 conversion of models whose weights do not stay decoded, e.g. Qwen3.8 27B) loads each
  superblock into registers with vector loads and decodes it from there, and is now used for every type. The
  per-32-block kernel used for IQ3_S / IQ3_XXS / IQ4_XS read global memory byte by byte. 27B, 11,240-token prompt:
  those three types 5.8 -> 2.2 s (all decoding ~6.3 -> ~2.6 s, ~3.5 % of the time to first token). The prompt
  logits are bitwise identical (6,615-token prompt; `ARC_W8_SB=0` selects the old kernel). It also shortens model
  loads that decode at load time (E4B).
- `run --dump-logits` writes the prompt's logits too; `ARC_PROF_W8TYPE=1` profiles decoding per weight type.
- GEMM, measured in isolation (new `bench/gemm_w8_bench`, 27B shapes, bitwise comparison with the original):
  W8 with per-32 scales 8.3-8.6 TOPS, per-16 7.1; no-load ablation 9.7, no loads and no scale epilogue 11.0, against
  13.2 TOPS for signed DP4a with a broadcast operand (`peak_ss_bc_a`; the 7.1 TOPS `peak_dp4a_bcast` is the slower
  signed x unsigned form). Tried and not kept: a software-pipelined chunk loop (identical results, 7.0 TOPS: it
  loses the unrolled superblock), 32 tokens per sub-group (2.6 TOPS), 24-bit multiplies in the scale step (no
  change). In the model the 27B's GEMMs run at ~6.8 TOPS over a 100 s prompt: sustained heat, the per-16-scale
  types (about a fifth of its weights) and the kernels in between, not the kernel's shapes (all 8.3-8.4 alone).
  Experiment switches: `ARC_GEMM_DEFS`, `ARC_GEMM_TM`, `ARC_GEMM_WGY`, `ARC_GEMM_RG`.

## 2026-10-12: web UI follows the reply; Copy / Download on code blocks

- Every code block in a reply has **Copy** and **Download** buttons, above and below it (no scrolling to either end
  of a long block), and shows its language. Download names the file
  from the block's language (```` ```html ```` -> `code.html`, `py`, `js`, `ps1`, ...; `code.txt` when none). The
  copied / saved text is the code exactly as written (HTML special characters included).

- The chat view stays at the newest output whenever the thread grows (streamed text, the thinking box appearing or
  opened, images loading, tool calls, the stats line). Before, it only scrolled when a new piece of text arrived,
  and it pulled the view back down on every piece even when you had scrolled up to read. Now scrolling up (wheel,
  touch, scrollbar, PageUp / Home / arrow keys outside text boxes) pauses following at once and shows a "↓ Latest"
  button; scrolling back to the bottom, the button, or sending a message resumes it. Checked in the browser against
  the server: 0 px behind while following (also with the thinking box opened mid-stream), the view stays put while
  paused.

## 2026-10-12: Gemma 4 E4B generation 17 -> 21 tok/s (31 with the drafter), faster long prompts

Profile of one generated token (E4B IQ4_XS, short context): 56.5 ms of GPU time; the matrices (2.74 GB read per
token) would take ~37 ms at the 75 GB/s the iGPU reads. The IQ4_XS GEMVs ran at 44-65 GB/s, and ablations showed
why: without its memory loads the kernel still took 75-90 % of its time, i.e. decoding the 4-bit codebook indices
(byte-wise reads from local memory, a table lookup and shift / or packing per weight) cost as much as reading them.

- **`gemv_v` / `gemv_mt_v`** (`kernels/decode.cl`): each lane vector-loads its block's header and indices straight
  from global memory (no local-memory window, no barriers), and IQ4_XS's codebook comes from four local tables
  pre-shifted into byte 0..3, so a packed int8x4 word is four lookups ORed together (~8 -> ~4 instructions per
  weight). Used for IQ4_XS and for Q6_K output heads; Q4_K / Q5_K stay on `gemv` (their decoding is cheap, and there
  `gemv` was as fast or faster). gemv_bench, weights streaming from memory: E4B FFN gate 65 -> 70 GB/s, down 57 ->
  69, attention q 44 -> 67, output 48 -> 69; Q6_K output heads (E4B, Qwen3.5 4B) 66-67 -> 72 GB/s. Results are
  bit-identical to `gemv` / `gemv_mt` (new `GB_CMP` mode of `bench/gemv_bench`: every matrix compared bitwise).
- **`gemv_glu`**: Gemma's FFN for 1-8 tokens in one launch: gate and up rows of each 32-value block in the same
  work-group, then `gelu_tanh(gate) * up` quantized per 32 values in the epilogue (the same formulas as
  `swiglu_quant`), into a second activation buffer that the down projection reads. Replaces gemm_gate + gemm_up +
  swiglu_quant (two launches per layer fewer).
- `attn_prep_g` / `add_norm_quant`: vector loads for each lane's run of values (same arithmetic; ~5 % faster).
- Result, E4B on `bench/prompts/gemma-code160` (128 tokens, same session): 17.2 -> 21.2 tok/s without drafting,
  ~23 -> 30.7 tok/s with the assistant drafter (`run --mtp`; the drafter itself is unchanged); through the server
  with sampling (temperature 0.7) 30.9 tok/s. GPU time per token 56.5 -> 46.6 ms. The logits of 64 generation steps
  are bitwise identical to before (new `run --dump-logits F`; `ARC_NO_GEMV_V=1` / `ARC_NO_GLU=1` select the old
  path for such comparisons).
- **Long prompts**: Gemma 4 E4B's global layers (head size 512, 4 q heads per kv head) now use `attn_ps` (head dims
  split over sub-groups) like the 12B's, with 8 tokens per work-group: attn_q8 spilled 2 KB of registers there.
  `bench/attn_bench` at 13,921 tokens: 870 -> 619 ms per layer (same error against the CPU reference). E4B, 13,921-
  token prompt: time to first token 23.5 -> 21.8 s (591 -> 638 t/s). Not bit-identical (partial scores summed in
  another order): perplexity over 6 chunks of 4,096 tokens 88.44 -> 88.34.
- Prompt speed in context: E4B does ~27 % more arithmetic per token than Qwen3.5 4B (wider FFN, head-512 global
  attention); pp2048 789 t/s (Qwen 4B: 880).
- `tests/run_tests.ps1` (Qwen3.5 4B, Gemma 4 12B): all checks pass (Gemma KL 0.0188, greedy = llama.cpp, drafter
  and MTP output identical).

## 2026-10-12: pi-arc opens the model it started

- Fixed: pi-arc started Swift but Pi opened Gemma 4 E4B, shown with Swift's context (86,016). `last-model.txt`
  holds the path as `server.exe --list-models` wrote it, with mixed separators (`C:\Users\me/.cache/huggingface/
  hub\models--...`); arc-server compared it to the resolved path as text, found no model and started the file under
  a name made from the file name (`swift-...-iq3_s-mtp`). That name was not in `/v1/models`, whose first entry was
  then `gemma4-e4b`; pi-arc and arc-server took the first entry as the running model, so Pi was given
  `gemma4-e4b`, its list got the running context, and Pi's first request made the server switch to the E4B.
  - server: the scanned paths have one kind of separator; model files are compared as files
    (`std::filesystem::equivalent`), so the loaded model is listed first and marked `loaded` however its path was
    spelled, and a request for its catalog id does not load it again. A model file outside the scanned folders is
    listed first under the alias it was started with.
  - arc-server / pi-arc: model paths compared as full paths; the running model is the entry marked `loaded`.
- web_search on networks where DuckDuckGo is blocked (Indonesia): pi-unsloth-webtools searches DuckDuckGo and
  Startpage by default, and from here DuckDuckGo times out while Startpage returns nothing, so every search ended
  in "No results found" (or was cancelled while waiting). Yandex answers (5 results in ~1 s), so Pi's
  `settings.json` now has `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (Pi's `/search-config`
  changes it). Not an arcinfer change: the model's `web_search` calls were well formed.

## 2026-10-11 (late night): image input published per model

- `/v1/models` says per model whether it takes images, in the fields clients read: `capabilities` (llama-server /
  Ollama style: `completion`, `tools`, `vision`, `multimodal`), `architecture.input_modalities` (OpenRouter style),
  `type` (`vlm` / `llm`, LM Studio style), `modalities`; `/props` adds llama-server's `modalities` block. Before,
  only `/props` had a `vision` flag, for the loaded model only, so a client such as Unsloth Studio could not see
  that `gemma4-12b` reads images. Only models whose image encoder arcinfer runs are marked: Gemma 4 12B
  (`gemma4uv`). Qwen3.5 4B / 9B (`qwen3vl_merger`) and Gemma 4 E4B (`gemma4v`) ship an mmproj whose encoder is
  not implemented yet, so they stay text-only.

## 2026-10-11 (late night): Pi's edit streams; Pi's model list follows the disk

- Tool calls: array and object arguments now stream while they are generated, as the model's JSON text kept as
  written (`chat::json_prefix` cuts it where strict JSON can continue: never inside an escape, a number, a literal
  or a UTF-8 character). Before, only string arguments streamed, and Pi's `edit` (its whole change is the `edits`
  array) arrived in one piece at the end. Measured on Swift 1.5 Qwen3.8 27B: a 282-character edit arrived in 95
  pieces over 14 s instead of at once. If the model's JSON is not strict (e.g. a raw line break inside a string),
  the part already sent is closed and the parameter is repeated with the whole-response value; JSON readers take the
  last one, so Pi sees exactly what it saw before. `tools/tool_stream_test`: new cases (streams before the array is
  closed, non-strict JSON, numbers and literals), 20,000 random responses; regression tests pass.
- All of Pi's tools checked (pi-coding-agent 1.1 with pi-unsloth-webtools, pi-browser-control and MCP `codemode`:
  17 tools, their parameter types captured from Pi's own request): `tools/tool_stream_test` now calls each of them
  as Qwen writes it and checks the final types against the schema and that every long string / array value
  streams before its parameter closes (bash command, edit edits, write content, browser_type text, codemode
  code, ...); numbers and booleans arrive when their parameter closes. Boolean parameters also accept Python-style
  `True` / `False` (before: passed on as the string "True", which Pi's validation rejects).
- Live, Swift 1.5 Qwen3.8 27B with Pi's real tool list (one streamed request per tool, nothing executed): all 17
  tools give arguments that match their schema; long values stream (write: 282 pieces over 54 s, edit: 99 over
  19 s, codemode: 34, bash: 22). Before the boolean fix 5 of them failed: this model writes `True` for
  browser_navigate, browser_screenshot, browser_click, browser_type and browser_console_logs.
- `arc-server start` / `restart` bring Pi's model list up to date (they rescan the Hugging Face cache), also when
  the server is already running, so a deleted model leaves Pi's `/model` without running `pi-arc` (which no longer
  writes the list itself). Checked with a model deleted from a temporary cache.

## 2026-10-11 (late night): help for every option, README brought up to date

- Every command documents all of its options: `pi-arc --help` (aliases, `-c` / `-r`, setup, what happens to the
  caches when you continue a session, all server options and files), `arc-server help`, `bin\server.exe --help`,
  `bin\run.exe --help` (all 24 options; before: a one-line usage) and `setup.cmd --help` (new: steps, models,
  what it does not install). `--help` exits with 0 everywhere.
- `pi-arc --model <text>` (or `--model arcinfer/<id>`, as in Pi) chooses the model like `-Model`; before, it went
  to Pi next to the `--model` pi-arc sets.
- README: "Using Pi" rewritten (model choice and `/model`, contexts per GPU memory setting, `--ctx auto`, a table of
  what the server processes for each way of starting or continuing a session, Esc), the Pi commands, the server's
  model switching and `--list-models`, the prompt cache under "How it works", troubleshooting (compiler search,
  Windows slow after raising the GPU memory, the first session in a project), layout.

## 2026-10-11 (late night): system prompts kept per project folder

- Pi's saved system prompts are grouped by project folder, read from the `<cwd>` section of Pi's system prompt
  (`<folder>-<folder hash>-<prompt hash>-<tokens>.arcp`; the folder's spelling normalized, so `C:/X` and `c:\x` are
  one folder). Each folder keeps its 2 latest versions (an `AGENTS.md` edit or a Pi update replaces the oldest;
  two variants used in turn both stay), and other folders do not replace them; only the size limit below removes a
  folder's files, least recently used first. Before: the 8 used last per model,
  across all folders. Prompts without a folder (other clients): the 8 used last, as before. All models' files
  together stay under `--prefix-cache-gb` (default 8; least recently used first). Files saved before this are
  renamed into their folder's group the first time they are used (no reprocessing).
- Checked on an isolated 4B server: per-folder names, 2 versions per folder, other folders untouched, load after a
  restart (0.07 s), a pre-change file adopted, the size cap; and in real Pi (file named after its folder).
- Not changed: the session cache (whole conversations, `--session-cache`) stays separate and off by default. Only a
  request that starts with exactly the saved tokens can reuse a saved state, so a folder's cache can cover the
  system prompt its sessions share but not one session's conversation for another.

## 2026-10-11 (late night): one setup command

- `setup.cmd` (`setup.ps1`): checks the requirements (Windows, Intel Arc GPU, Visual Studio C++ compiler, Python,
  Pi; says where to get what is missing), builds `server.exe` / `run.exe` / `clinfo.exe` when missing or older than
  the sources (compiler output in `%LOCALAPPDATA%\arcinfer\build.log`, shown on failure), checks the GPU with
  `clinfo` (OpenCL 3.0, DP4a, sub-groups, GPU memory), downloads the chosen models (`-Models 27b,4b`, `all`, `none`,
  or asked; Gemma with drafter and projector; the 35B downloaded and converted, with the 4B for its tokenizer),
  installs `huggingface_hub` / `numpy` with pip after asking, connects Pi, and runs the regression tests with
  `-Tests`. Safe to run again: done steps are skipped.
- `build.ps1` finds the compiler with `vswhere` (any Visual Studio / Build Tools with `vcvars64.bat`;
  `$env:ARC_VCVARS` overrides) instead of a fixed path, and builds with `_CRT_SECURE_NO_WARNINGS` (no
  getenv / fopen deprecation notes).
- `install-pi.ps1` writes Pi's model list from `server.exe --list-models` (every model found, each with its
  context) instead of a fixed list with fixed contexts; `scripts\pi-models.ps1` holds that code for it and `pi-arc`.

## 2026-10-11 (night): Esc in Pi stops the server right away

- A client that closes its connection (Pi's Esc) cancels the request: the server polls the socket (no write needed)
  after every layer of prompt processing and before every generated token. Before, a prompt kept being processed to
  its end (a 7K-token prompt on the 27B kept the GPU ~73 s after Esc; 50K tokens: ~10 minutes) and the next request
  waited for it.
- Prompt processing stops within one layer (`Engine::cancel`): the block it was in (layer-major segment or chunk) is
  undone (DeltaNet states from a host copy taken at the block start, 27B: 157 MB, ~20 ms per block; Gemma moves back
  in its sliding-window rings, else resets), complete blocks stay cached.
- Pi drops an aborted (or failed) reply and sends the conversation without it, so the next request diverges inside
  the `<|im_start|>assistant` header. The resume snapshot was right after that header: the whole conversation after
  the system prompt was processed again (2.5K-token conversation: 21.5 s). It is now taken at the header's start:
  5.0 s.
- Checked in real Pi on the 27B (a scratch project with Pi's normal tools and extensions plus a local MCP server):
  parallel tool calls, edits, files with backslashes, quotes, braces and Unicode, bash with long output and failing
  commands, MCP through codemode, git, follow-ups: 27 replies, 21 tool calls, all arguments valid, no errors from the
  server or Pi; files on disk exactly as asked. The regression tests' server runs with `--no-prefix-cache`.
- `pi-arc --ctx auto` / `arc-server -Ctx auto` work (pi-arc failed on a non-number): auto, and a server running a
  fixed context is restarted with auto; without the option a running server keeps its context. Bad values give a
  message instead of a PowerShell error.
- Help: `server.exe --help` (exit code 0) lists every option with its default; `pi-arc --help` has a WORKING IN PI
  section (Esc, new sessions, project folders); `arc-server help` says where saved system prompts are.
- `docs/review.html` and `docs/review_notes.md` (now docs/archive/) updated: stages 12–14, contexts at 17.4 GiB, Pi on the 27B with
  today's measurements and the use-case run, two new limits.

## 2026-10-11 (evening): system prompt kept on disk

- The state after a client's system prompt is saved to `%LOCALAPPDATA%\arcinfer\prefix\<model>` the first time it is
  processed and loaded when a request starts with it while the server holds something else (server start, model
  switch, a conversation with another system prompt). Pi's system prompt contains the project folder and its
  `AGENTS.md`, so each project has its own file; the 8 used last per model are kept. `--no-prefix-cache` turns it
  off. Measured in Pi, Qwen3.8 27B, 4146-token system prompt: first new session after a server restart 40.2 s ->
  5.3 s to the first token (file loaded in 0.53 s); the file (290 MB) is written in the background.
- The state is saved without the duplicate host snapshot at the same position (157 MB less per file; the snapshot
  is taken again after loading). Works for Gemma 4 too (its system turn is recognised the same way).
- `--ctx auto` measures with a dry load that only adds up the GPU allocations (seconds, nothing uploaded), so
  `--list-models` reports every model's context and Pi's `/model` list gets each model's own context.

## 2026-10-11 (later): one command for Pi, models switched from inside Pi

- `pi-arc` starts Pi right away on the model the server runs, else the one used last (`--pick`: choose from the
  list of every supported GGUF model in the Hugging Face cache, by full name). It registers all those models in
  Pi's `models.json`, so Pi's `/model` lists them.
- The server switches models itself: a request whose `model` is another model of `server.exe --list-models` (its
  id, `org/repo` name or file) unloads the current one and loads that one (`--no-swap` turns this off).
  `/v1/models` lists them all, the loaded one first. E4B <-> 4B: 4-9 s; GPU memory is the same after any number of
  switches (the engine now releases its buffers, kernels and context: the `--ctx auto` measuring load used to stay
  in memory).
- `--ctx auto` for every model by default (the context follows the GPU memory setting); `--list-models` reports each
  model's id, shortcut, image support (a `gemma4uv` projector: Gemma 4 12B, not the E4B's) and computed context.
- Disk session cache off by default (`pi-arc --session-cache 16`); `pi-arc --help` documents every option.

## 2026-10-11: session cache (switching between conversations), 27B back to 4096-token chunks

### Saved conversations

The server held one conversation: a new Pi session, or going back to an older one, processed the whole prompt again
(Qwen also its ~4.2K-token system prompt: the DeltaNet state can only be resumed where a snapshot was taken). Pi on
the 27B, two sessions taking turns: every first request had 0 tokens cached, 36 s for a new session, 55–64 s to go
back to a 6.6–7.5K-token one.

- `Engine::save_state` / `load_state`: the KV cache (int8 blocks and scales of every layer with its own cache; for
  sliding-window layers only the window before the resume point), the MTP block's cache and hidden state, the DeltaNet
  states and snapshots; copied in batches of up to 256 MB.
- Server: when a request does not continue the held conversation's last prompt, that conversation is saved to
  `%LOCALAPPDATA%\arcinfer\sessions\<model>-<size>\` first; a saved conversation that shares at least 512 more
  reusable tokens with the request is loaded. `--session-cache-gb` (default 0 = off; least recently used first),
  `--session-dir`; `GET /sessions` (size), `DELETE /sessions` (clear); `arc-server status` / `arc-server clear-cache`.
- Qwen: a recurrent-state snapshot at the end of the system prompt, kept in host memory (no GPU memory), so a new
  conversation with the same system prompt processes only its own messages.
- Answers are identical with and without other conversations in between (greedy; Qwen3.5 4B, Gemma 4 E4B, Gemma 4
  12B). Saved size for ~4.5K tokens: 4B 225 MB, E4B 49 MB, 12B 215 MB; saving 0.1–0.9 s, loading 0.05–0.3 s.

### Qwen3.8 27B: 86,016 tokens with 4096-token chunks

Default changed from 118,784 tokens with 1024-token chunks to 86,016 with 4096 (the largest that fits with the same
~1 GiB left for other apps). `arc-server start -Model 27b -Ctx 118784 -Chunk 1024` (or `pi-arc --ctx 118784
--chunk 1024`) gives the larger context; Pi's configured window stays 86,016.

## 2026-10-10 (evening): prompts at 90 % of the context

GPU memory is now 18.34 GiB (`CL_DEVICE_GLOBAL_MEM_SIZE`, was 16.5); the desktop and other apps hold 0.7–1.0 GiB of
it, and the driver ~0.22 GiB on top of the server's own buffers (Windows counter `GPU Process Memory(*)\Shared Usage`).

### Measured: a prompt of 90 % of the default context (arcinfer's own sources, `arc-server` settings)

| model | context | prompt | prompt processing | time to first token | generation after it |
|---|---|---|---|---|---|
| Qwen3.8 27B (before) | 90,112 | 81,100 | 82.5 t/s | 16.4 min | 4.92 t/s (MTP) |
| Qwen3.8 27B (2048-token chunks) | 106,496 | 95,846 | 78.6 t/s | 20.3 min | 4.55 t/s (MTP) |
| Qwen3.8 27B (now, 1024-token chunks) | 118,784 | 106,906 | 72.8 t/s | 24.5 min | 4.59 t/s (MTP) |
| Gemma 4 E4B | 131,072 | 117,965 | 165.5 t/s | 11.9 min | 11.6 t/s (drafter) |
| Qwen3.5 4B | 262,144 | 235,929 | 144 t/s | 27.3 min | 9.0 t/s (MTP) |

(35B-A3B, 9B and Gemma 4 12B were not run to the end.) At that depth attention takes a large share: ~38 % of the
27B's time at 81K, about half of the 4B's at 236K, most of the Gemma 4 12B's.

### Prompt attention (`bench/attn_bench`, new `ATT_P0`: only the queries from a position on, i.e. one chunk at depth)

- `attn_q8` at depth: 1.9–2.2 T MAC/s for the Qwen shapes (head 256), 0.87–0.91 for Gemma's head-512 global layers,
  which spill 1984 bytes of registers. Ablations on the 27B shape (compile-time `ABL_NOQK` / `ABL_NOSM` / `ABL_NOPV`):
  tile loads + barriers 19 %, QK 24 %, P·V with P packing 33 %, online softmax 19 %.
- **Gemma 4 12B global layers now use `attn_ps`** (new, `kernels/ops.cl`): the head dims are split over sub-groups
  (4 groups of 128 dims, 4 q heads each, 2 tokens per work-group) and partial scores are summed through SLM, so no
  registers spill: 1.25–1.4x faster at every prompt length (0.96 → 1.34 T MAC/s at 4K, 0.87 → 1.10 at 102K). KL vs
  the f32 reference on 160 code positions 0.0174 (0.0180 before); greedy output and drafter tests unchanged.
  `ARC_NO_PS=1` switches back. With one scale per row (Qwen) `attn_ps` is bit-identical to `attn_q8`
  (`ATT_CMP=1` checks it), but it is slower there (1.0–1.6 T MAC/s): splitting the dims doubles the per-row softmax
  and P packing work, which already take ~40 %. E4B's global layers (4 q heads) gain nothing either.
- Tried and removed: 32-key tiles (one max reduction and barrier pair per 32 keys): the extra live state spills
  (2–3 KB) and is 14–20 % slower for Qwen; 4 heads per sub-group: +19 % on the 4B / 35B shapes but spills.
  `attn_q8` is at the register limit of the Qwen shapes.

### Qwen3.8 27B: context 90,112 → 118,784 tokens

- The Qwen KV scale buffers were sized for one scale per 64 dims while Qwen uses one per row: 1,632 bytes per token
  less (36,992 → 35,360; −0.14 GiB at 90K).
- `arc-server` runs the 27B with 1024-token prompt chunks (`--chunk 1024`), which frees 1.1 GiB of activation
  buffers (GPU buffers at 106,496 tokens: 4096-token chunks ~17.5, 2048 16.78, 1024 16.41 GiB). Same outputs. Time
  to first token, 11,240 tokens: 90.1 s (4096), 91.9 s (2048), 94.0 s (1024); 33,937 tokens: 321.6 s (2048),
  322.2 s (1024); at 96K–107K the per-16K-segment times equal those of 4096-token chunks.
- At 118,784 the server holds 17.29 GiB; with 0.88 GiB of other apps that is 18.17 of 18.34 GiB. The previous
  90,112 with 4096-token chunks needs 17.62 GiB: over the budget once other apps hold more than ~0.7 GiB (paging).
- Not done: switching the chunk size during a session (2048 while the context is small). The KV cache is
  allocated for the whole context at load, so the memory for larger chunk buffers is never free; it would need a
  growing KV cache, for ~2 s on an 11K-token first prompt and nothing on long prompts or ordinary agent turns.

### Streaming

- A character whose UTF-8 bytes are split over two tokens (e.g. emoji) was streamed in two invalid pieces (the
  client shows two replacement characters). `emit` now holds back an incomplete character at the end of the text
  (Qwen and Gemma parsers); `tools/tool_stream_test` checks every delta byte by byte.

## 2026-10-10 (later): streamed tool calls

- **Tool calls are streamed while they are generated** (`tools/server.cpp`, `chat::partial_tool_calls`,
  `gemma4::partial_tool_calls`). Before, the server held a tool call back until the end of the response and sent it
  in one piece, so Pi showed nothing while the model wrote a file through its `write` tool (one call: 2,368 tokens,
  6 minutes without output). Now the first delta of a call carries its id and name, and the following ones the
  arguments JSON piece by piece, as OpenAI and llama-server send them. String parameters grow token by token;
  Qwen's non-string parameters (e.g. the `edits` array of Pi's `edit` tool) appear once the parameter is closed,
  because the model's JSON may still turn out invalid (it then becomes a string, as before). Gemma 4 delimits its
  strings, so all of its arguments stream. The pieces of a call only grow and end as the arguments of the
  whole-response parser, so non-streaming responses are unchanged.
- `tools/tool_stream_test.cpp` checks this for every prefix of hand-written and 3,000 random responses (both
  formats); part of `tests/run_tests.ps1`. The web UI and `tests/test_tools.py` join the pieces by index.
- Gemma 4 tool-call parser: an array closed by a stray `}` (`call:x{a:[1}`) made it loop forever while allocating
  memory (the server hung). A stray `}` now ends the array.
- `run.exe` warns when other GPU processes (`server.exe`, `run.exe`, `llama-*`) are running. This explains the
  "`run.exe --mtp` on E4B at 3–5 tok/s" below: alone, `run.exe --mtp` gives 23 tok/s on E4B (17 without drafting;
  20.5 after a 13.9K-token prompt), but with a 27B server processing a prompt at the same time it gave 8 tok/s
  (5–6 without drafting), and with an idle 27B server loaded the E4B load took 13–43 s instead of 2 s. The iGPU
  and its 16.5 GiB are shared between processes.

## 2026-10-10: token generation for Gemma 4 (12B and E4B)

Goal: faster generation on Gemma 4 12B and E4B, at short and long context, without changing the model's output.
The Gemma 4 assistant drafter (speculative decoding) was deliberately left as it was.

### Where the time went (Gemma 4 12B, before)

Profile of a single-token step after a short prompt: 134 ms per token (7.45 tok/s), 139–144 ms of GPU time when
profiled. The 7.4 GB of weights would take ~100 ms at the 72–77 GB/s the iGPU reads from memory, so ~30 % was
overhead:

| part | ms / token | note |
|---|---|---|
| FFN GEMVs (gate, up, down) | 78 | 61–71 GB/s, close to the limit |
| lm_head (262,144 × 3840, Q6_K, 826 MB) | 21–28 | **30 GB/s**, half the speed of the same kernel in isolation |
| attention GEMVs (q, k, v, o) | 24 | 50–60 GB/s |
| small kernels (attn_prep, swiglu, norms, attention, combine) | ~11 | attn_prep alone 55 µs per layer |

At 13.9K context the step grew to 157 ms: decode attention of the 8 global layers (head size 512, 16 q heads on
one kv head) ran at only ~20 GB/s, 0.7 ms per layer.

### Changes

1. **Output head allocated first** (`src/engine.h`, `load_weights`). The GPU read the 826 MB head buffer at about
   half speed when it was allocated after gigabytes of other weights (the same kernel on the same data took
   13–14 ms in isolation, 21–28 ms in the engine; loading the head first gives 14 ms in the engine). Applies to
   every architecture.
2. **Gemma's final logit soft-capping on the GPU.** `c·tanh(x/c)` is now the epilogue of the head's GEMV
   (`-DSOFTCAP`, `kernels/decode.cl`), instead of a host loop over 262K logits per token whose OpenMP worker
   threads kept spinning between tokens.
3. **RoPE table once per pass** (`rope_table` kernel). `attn_prep_g` used to compute `pow`, `sin` and `cos` for
   every element in all 48 layers; the angles are now computed once per pass for the sliding and the global layer
   type and read from a table (bit-identical values). attn_prep: 3.4 → 1.9 ms per token.
4. **Fused residual + norm + quantization** (`add_norm_quant`). For single-token and verification passes, the
   post-attention / post-FFN sandwich norm with the residual add and the next block's input norm and int8
   quantization run as one kernel (2 launches fewer per layer). Output is identical to the two separate kernels.
5. **Quantization of the SwiGLU and attention outputs** for single-token passes: one sub-group per 256-value block
   of the valid rows instead of one work-group per padded row (swiglu_quant 2.4 → 1.2 ms per token).
6. **Decode attention**:
   - New kernel `attn_dsplit` (`kernels/ops.cl`): a work-group covers one key split of one kv head; its sub-groups
     split the head dimensions (64 dims each) and the q heads, so every byte of the K/V cache is read once for all
     heads and up to 4 query rows. Partial scores of the dim groups are summed through local memory; softmax and
     P·V use the same int8 math as before. Used for Gemma 4 (not for the drafter's own layers, not for Qwen).
   - K/V tiles in local memory are single-buffered (`attn_q8`, `attn_split`): twice the work-groups per Xe-core.
     Decode with head size 512: 1.5× faster; prompt processing unchanged or up to 4 % faster.
   - New `attn_combine`: the lanes of a sub-group take the per-split maxima 16 at a time, no data-dependent
     branch; 3–5× faster (26 → 5 µs for 33 splits).

   Attention + combine per layer, `bench/attn_dec_bench` at 14K context:

   | layer type | rows | before | after |
   |---|---|---|---|
   | 12B global (16 q / 1 kv, head 512) | 1 | 742 µs | 310 µs |
   | 12B global, 36K context | 1 | 1,818 µs | 803 µs |
   | 12B sliding (window 1024) | 1 | 125–220 µs | 73 µs |
   | E4B global (8 q / 2 kv, head 512) | 1 | 1,008 µs | 420 µs |
   | E4B sliding (window 512) | 1 | 56 µs | 17 µs |
   | 12B global, verification | 2 | 779 µs | 520 µs |
   | E4B global, verification | 2 | 944 µs | 465 µs |
   | 12B sliding, verification | 2 | 142 µs | 78 µs |

   At 14K context this saves about 6–8 ms per token on the 12B (8 global + 40 sliding layers, plus attn_prep) and
   ~5.5 ms on the E4B (7 global + 35 sliding layers); at 36K the 12B's global layers alone save 8 ms.
7. **Server sampler** (`tools/server.cpp`): the top-k candidates are taken from the tokens within 40 temperatures
   of the best logit (one pass) instead of building and partially sorting a pair for each of the 262K vocabulary
   entries per token; when fewer than k candidates are left after penalties it falls back to the full vocabulary,
   so the sampled distribution is exactly the same.

### Results

- Gemma 4 12B, short context, 128 tokens, same session A/B: 7.79–7.83 → 8.09–8.16 tok/s without drafting
  (a later run on a warm laptop: 7.66); with the assistant drafter 8.74 tok/s (50 % of drafts accepted,
  1.94 tokens per main-model pass). GPU time per token 139–144 → 123 ms (profiled).
- Gemma 4 E4B, short context: 16.9 tok/s without drafting; with the drafter through the server 17.4–18.1 tok/s on a
  story prompt (40 % of drafts accepted; code accepts more).
- Long-context end-to-end generation after these changes was not measured; the per-layer attention savings above
  were. Before them, the 12B generated 6.4 tok/s after a 13.9K-token prompt (llama.cpp Vulkan 5.5).
- At short context the end-to-end gain is within the laptop's run-to-run variation (±4 %); the gains are at long
  context, where attention grows.

### Accuracy

- `tests/run_tests.ps1` with Qwen3.5-4B and Gemma 4 12B: all 10 checks pass (Gemma KL vs the f32 reference
  0.02927, greedy equal to llama.cpp, drafter output identical, Qwen MTP output identical).
- New `run --kl-decode`: feeds a prompt token by token through the generation path and reports the KL against the
  f32 reference at every position. Gemma 4 12B, 160 code positions: 0.013 (0.020 before the changes); 120 raw-text
  positions: 0.029 (0.034 before). A 20-position test swung between 0.009 and 0.043 between builds whose
  intermediate values differ by at most 1e-6: one near-tied position flips. Use the longer references.

### Tried and not kept

- IQ4_XS (E4B) weight decoding: a 256-entry byte-pair lookup table was slower (local-memory bank conflicts,
  66 → 55 GB/s), and register shuffles with a variable lane index halved the speed. The 16-byte table stays.
- `attn_dsplit` for Qwen: up to 2× faster for 2-row verification, but the verification rows are then no longer
  bit-identical to single-token decoding, which the MTP regression test requires. Qwen keeps the previous kernels.
- Drafter changes (native Q4_0 GEMV for the drafter's weights, which halves their bytes; draft argmax on the GPU
  instead of reading 1 MB of logits per draft): written and tested, then reverted on request. The drafter runs as
  before.

### New tools

- `bench/gemv_bench.cpp`: the single-token GEMV on real tensors of a GGUF (all layers in turn, so weights stream
  from memory), checked against the CPU dequantization; variants via build defines or another kernel file.
- `bench/attn_dec_bench.cpp`: decode attention (attn_dsplit / attn_split / attn_decode + combine) for 1–4 query
  rows at any head shape and window, checked against a CPU reference.
- `run --kl-decode` (with `--kl-ref` of every position) and `ARC_PROF_LAYERS=n` (profile per layer group).

### Known issues

- ~~`bin\run.exe --mtp` on Gemma 4 E4B generates at only 3–5 tok/s~~: not reproducible on an otherwise idle GPU
  (23 tok/s); the slow runs shared the GPU with other processes (see the entry above).
- E4B still has headroom: its q/o projections (IQ4_XS, K = 2560) run at 38–49 GB/s, the per-layer-embedding
  projections (256 rows, or K = 256) at 11–22 GB/s, and its small kernels take ~7 ms per token.
- Gemma 4 audio input (12B): the code compiles and is loaded when the mmproj has the audio projector; not yet tested.

## Earlier in this round (before 2026-10-10)

- Qwen3.6-35B-A3B, ISTA-DASLab GSQ 2-bit: `tools/convert_gsq.py` (safetensors → GGUF with the exact 2-bit expert
  codes) and `kernels/moe.cl` (routing, counting sort, DP4a GEMMs on the 2-bit codes). 11K-token prompt in ~14 s,
  ~15 tok/s generation, KL vs f32 0.0008.
- `gelu_tanh` attribute fix in `kernels/ops.cl`.
- Gemma 4 at long context: attention for head size 512 and windows, the assistant drafters, a persistent kernel
  binary cache (`%LOCALAPPDATA%\arcinfer\kernels`), lm_head tuning.
- Gemma 4 accuracy: one int8 scale per 64 dims for q/K/V (was one per row): KL on code 0.23 → 0.018.
- Segmented layer-major prefill (16K tokens) and the largest contexts that fit 16.5 GiB for every model
  (`scripts/arc-server.ps1`, `scripts/install-pi.ps1`).
- Review page `docs/review.html` (EN/ID) with `bench/review_data.py`; prompt benchmark `bench/prompt_bench.py`.
