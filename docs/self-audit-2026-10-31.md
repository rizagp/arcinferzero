# Self audit, 2026-10-31: multi-GPU (tensor parallel) and the day's server / web UI changes

Scope:
- **Multi-GPU:** the per-GPU runtime (`zero/clrt.h`), the shards and their sums (`zero/mgpu/`), and `runz --gpus`.
- **The day's other commits:** `serverz -m none`, the Qwen3-ASR warm-up, the recording formats (MP3 / M4A bitrates,
  mono / stereo).

Focus: memory leaks, missing guards, error paths. Method: the code read again line by line, every suspicion run
before it was called a bug, every fix checked the same way.

## Found and fixed

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| T1 | high | `TpEngine`'s constructor threw (e.g. `--gpus 3` on a model that did not divide) with its worker threads still running. A joinable `std::thread` destroyed ends the process: no message, exit code 127 | the constructor catches, frees the shards made so far (each on its own GPU's thread), stops the threads, frees the shared memory, rethrows | `--gpus 3` before the uneven split: "this model cannot be split over 3 GPUs (DeltaNet heads)", exit 1 |
| T2 | high | the host copies of the cut tensors (`tp_keep_`, 1/N of the model per shard) were kept for the engine's lifetime: the 27B over 2 GPUs held ~7.5 GB of RAM per shard, the model a second time | freed once `load_weights` has uploaded them (`tp_loaded`); nothing reads them later (raw rows and W8 come from the GPU copies) | logits unchanged (4B 2 / 3 / 4 shards, same KL as before) |
| T3 | medium | a shard that failed mid-forward left the others' barrier count and parts out of step; a later call would hang or add wrong parts | the engine marks itself broken and refuses further calls ("load the model again") | code path |
| T4 | medium | the error reported was rank 0's, which after another shard failed is "another GPU's shard failed", not the cause | the first shard to throw is recorded; its error is the one rethrown | code path |
| T5 | medium | the 27B over 4 GPUs (the B60 owner's set-up) could not load prompts: a shard's DeltaNet alpha / beta projection has 24 rows (2 x 12 v-heads), the GEMM takes multiples of 16 | split models pad those rows to 16 (zero rows) into a scratch buffer, packed back by `tp_pack` (`zero/kernels/tp.cl`); one GPU unchanged | 27B over 4 shards: greedy identical |
| T6 | low | GPUs of another kind than GPU 0 (an iGPU next to a B60) were opened, and their modules of GPU 0's kernels would fail to load later with a less clear message | `open_devices` takes GPUs of GPU 0's device id only and says how many of another kind it left out | code path |
| T7 | low | the barrier spun at 100 % of a core per waiting shard, also through a prompt chunk's long GEMMs | it yields the core after 4,096 spins (a token's sums come microseconds apart and still spin) | 4B / 27B shards: sums unchanged |
| T8 | low | `tp_setup` divided by the DeltaNet head count before checking it (a hostile GGUF with 0) | checked first | code path |
| T9 | — | `gdn_conv` launched work-groups of 64 (4 head groups); the 27B over 3 GPUs has 30 groups on GPU 0: error -54 | a split shard whose group count 4 does not divide runs one group per work-group (the kernel needs no more), through the hook `tp_conv_lws` | 27B over 3 shards (below) |

New at the user's request during the audit:
- **Uneven splits:** a count that does not divide by N is split as evenly as its unit allows: k-heads in pairs,
  kv-heads one by one, FFN columns 256 at a time, vocabulary rows 16 at a time. The 27B over 3 GPUs holds 6 / 6 / 4
  k-heads and 2 / 1 / 1 kv-heads.
- **Multi-GPU code kept apart:**
  - `zero/mgpu/` holds the code, `zero/kernels/tp.cl` the kernels;
  - the one-GPU code only calls hooks that do what it did before when the model is not split;
  - the one-GPU logits of the 4B and 27B stay bit-identical.

## Checked and left as is

- **`zrt::Dev` objects of GPUs 1..N-1** and each GPU's `maps` vector are kept until the process ends, as device 0's
  state always was. A later engine reuses them (`open_devices` returns early), so repeated loads do not add more.
- **The shared buffer of the sums** is pinned host memory of 2 x N x max_chunk x n_embd floats: 671 MB for the 27B
  over 4 GPUs at chunk 4096. It is freed with the engine (`shutdown`), also after a failed construction.
- **The reuse of a part's host memory** is safe with two halves:
  - a shard passes the barrier of sum k+1 only after every shard has waited for its own GPU at sum k+1;
  - that wait covers its reads of sum k;
  - so a shard writing a half at sum k+2 never overwrites one still being read.
- **Buffers made on one GPU and freed by another thread** (an engine destroyed after a failed construction) are freed
  on their own GPU's list: each `ZMem` remembers its GPU.
- **`serverz -m none`:**
  - `/v1/models` never dereferences the missing model (`listed` starts true);
  - `/props` and the web UI page answer without one;
  - `-m none --no-swap` is refused at start;
  - `-m none` with no speech model is refused.
- **Recordings:**
  - `channels: 2` is taken only with one track (any other value: mono);
  - the stereo mix reads the right channel at byte 2 of a 4-byte frame;
  - the bitrate is clamped (MP3 96-320, AAC 96-192);
  - a failed encode keeps the WAV and records why (`mp3_error`).

## Results after the fixes (one GPU here; `ARC_TP_SIM=1` for the splits)

| | 4B: KL vs f32, 200 positions (top-1) | 27B: greedy 16 tokens, KL vs one GPU |
|---|---|---|
| 1 GPU | 0.00113 (196/200), logits bit-identical to before | bit-identical to before |
| 2 | 0.00096 (196/200) | identical, KL 0.00087 (top-1 17/17) |
| 3 | 0.00110 (198/200) | identical, KL 0.00078 (top-1 17/17) |
| 4 | 0.00109 (197/200) | identical, KL 0.00068 (top-1 17/17) |

## Evening: OCR mode, PDFs, the launcher, the NPU

Scope: `tools/server_ocr.h`, `tools/pdf_render.h`, the web UI's OCR view and save dialog, `scripts/arc-launcher.ps1`,
`zero/npu/npu_rt.h`.

### Found and fixed

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| O1 | high | A page the model could not read (an OCR model with no image encoder, an error) went into the answer, and the answer still ended with "N more page(s)". The web UI then sent "lanjut" again, which hit the same error each turn, page after page to the PDF's end | the pages stop at the first failed one; the answer ends with `[OCR: stopped at page P: why]`, which the web UI does not go on from | read through: the note does not match the web UI's `OCR_MORE` |
| O2 | high | The first PDF of a new chat failed with "An item cannot be found at the specified path": the chat's folder is renamed at its first save, while the PDF was being opened (user report) | the PDF read into memory once, every page rendered from it; the file looked up again if it moved | the 13-page PDF that failed reads |
| O3 | medium | A PDF answer carried no `timings`, so the web UI's statistics line said "stopped" | the pages' timings added up, sent with the last chunk and in a non-stream answer; the web UI's context meter does not take them (each page has a context of its own) | stream and non-stream: prompt / generated tokens and t/s |
| O4 | medium | An `.md` that could not be written (a read-only folder, a full disk) failed silently: the pages were read into no file | the file's first line checked; HTTP 400 "cannot write NAME there" before any page | `C:\Windows\System32` as the folder: 400 |
| O5 | low | A device name as the `.md`'s name (`CON`, `NUL`, `COM1`) opened the device, not a file | such a name gets "ocr " in front | `CON` saved as `ocr CON.md` |
| L1 | high | The launcher popped up an index error every 2 s (issue #5): commas inside `$L.Add("..." -f a, b)` split the method's arguments | each such line has its own parentheses | PowerShell 5.1 and 7: the old form fails, the new one works; the script parses |
| N1 | low | "no NPU" said nothing of why; a driver with a newer graph extension only was not taken | graph extension 1.8 to 1.20 tried; the message names what is missing | the NPU here still runs (pp1024 ~4,900 tok/s) |

### Checked and left as is

- The `.md`'s name is a name only (no path, the reserved characters replaced); a folder only from a request of this
  computer, and only one that exists.
- The PDF's bytes stay in memory for one request only (the file's size, freed when the answer ends).
- WinRT on the request threads: each thread joins the multi-threaded apartment once.
- A chat with an OCR model is not compacted (the server reads only its last turn); a PDF's pages stay in its history
  and go with every "lanjut" (the server reads only the highest `## Page N` from them).
- `halaman` / `page` in the words also pick the document task (Markdown, OTSL tables): kept, a PDF's pages read
  better that way.

## Late night: Gemma ring snapshots, digest_file, the tool-call fixes

Scope: commits c5fb88c to e6e998a (the OCR notes, read_file whole and by pages, digest_file, the Qwen tool-call
constraint fixes, tool_choice "none", Gemma 4's history and ring snapshots). Read by a second reader with no part in
writing them; every finding checked in the code and, where it could be run, with `runz --ring-snap-test`.

### Found and fixed

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| G1 | high | After a ring snapshot was restored, the rings held only the window before it, but `rewind()` checked `written_hi_` alone: a later rewind to just before the snapshot (an edited message, a moved tool-result cut) read the abandoned conversation's keys, silently | `restore()` raises `swa_lo_` to the restored window's start; `rewind`, `rewind_floor`, the saves and `ring_segs` already honour it | `--ring-snap-test`: a rewind 16 below the snapshot refused (E4B, 12B) |
| G2 | medium | A ring copy at every snapshot: 12B 472 MB and 233 ms each, up to three per request (0.7 s a turn, 1.4 GB of RAM) | the copy is taken only when the rings are about to overwrite that window (`ring_guard` before every pass: forward, layer-major, embeddings, verify, both batched paths), and not for the prompt's end (slot 1) | `--ring-snap-test`: 0 bytes at the snapshot, the copy on the way (E4B 30 MB, 12B 480 MB), logits bit-identical |
| G3 | low | After a loaded conversation, snapshots without a copy were still chosen, failed, and reset the cache | `snapshot_usable()`: a copy taken or still to take | read through |
| G4 | low | `restore()` could queue some writes before finding a size that did not match | every size checked first | read through |
| O1 | medium | OCR: the page heading goes out before the page is read; stopped there, the heading stayed without text and the next "lanjut" skipped that page | the server counts a heading only with text after it; the web UI drops a trailing empty heading | read through |
| D1 | low | digest_file: a combining level that did not shrink the notes still replaced them; a cut could fall inside a UTF-8 character | the notes before it kept; the cut moved off continuation bytes | read through |

### Checked and left as is

- The ring block ranges (16-key blocks, the wrap), the drop rules on rewind and restore, `--slots` (snapshots move
  with their slot), the Gemma drafter (no state of its own), Qwen models (every ring path gated on `swa_rings`).
- `gemma_history` in both chat() and count_tokens; `tool_choice: "none"` masks one token on the sampled path only.
- The Qwen enum and integer checks, the list_files / search_chats caps, read_file's whole / outline rules.
- digest_file's part size in characters (bytes): Latin and CJK text alike are ~3-4 bytes a token, so a part is
  about half the context.
- The web UI's deep trim after an overflow moves its cut each round (the prompt processed again from there): a
  fallback for an answer that would otherwise fail, kept.
