# Changelog

## 2026-11-11: DeepSeek-V4.1-Flash, a first engine path (tested on a small synthetic model)

- **`deepseek41` loads and runs** (`zero/engine_ds41.h`, `zero/kernels/ops_ds41.cl`), written from
  [docs/deepseek41-spec.md](docs/deepseek41-spec.md): the 4 hyper-connection streams with their Sinkhorn mixes, the
  Engram n-gram rows (read by rows from the GGUF, as Flash Next's PLE table), one 512-wide KV head over a sliding
  window plus up to top-k compressed rows picked by the indexer (ratio 2 and 1 compressors, the candidate block pool,
  attention sinks), the grouped output LoRA, and a MoE with 384 experts streamed from the SSD through the existing
  expert path: its router in f32, a sqrt(softplus) choice with the selection bias (`moe.cl -DROUTE_DS`), SwiGLU
  clamped at 10 (`-DSWI_CLAMP`).
- The real model is ~231 GB (ISTA-DASLab IQ1_M + its Engram table): it cannot run on the laptop. It was built and
  checked on `tools/make_tiny_deepseek41.py`, a 39 MB GGUF with the real keys, tensor names and layer roles and random
  weights: KL against the f32 reference (`tools/ref_forward`) 0.0032-0.0039 for 100 / 300-token prompts, in one
  chunk, in chunks of 128 and token by token. Not yet for this model: server prompt-cache snapshots, saved
  conversations, slots, drafters, multi-GPU; the caches are f32 and the kernels plain (speed comes after the real
  model's first run).
- Other models: their logits are bit-identical (Qwen3.6 35B, Gemma 4 26B, Qwen3.5 4B checked against the commit
  before); the MoE kernels' new switches are off for them.

## 2026-11-11: Gemma 4's attention keys in f16 (more context); a faster attention tried and dropped

- **Gemma 4's float attention keeps its K rows in f16** instead of f32, as Qwen3 dense does (`ARC_G4_KF16=0`: f32 as
  before). Half the key memory, so a larger automatic context: Gemma 4 26B 102,400 -> 131,072 tokens (with images
  110,592 -> 131,072; `-std` 131,072 -> 163,840); the 12B and E4B were at their caps already. The same speed (8K
  prompts within 1 %: the laptop's prompt attention is bound by its instructions, not by bytes). Accuracy as f32:
  20-token KL 12B 0.00000, E4B 0.00395 (0.00389 with f32), 26B 0.01013 (0.01014); after an 8,839-token prompt the
  12B's next 33 tokens KL 0.00007 against f32 K, every top token the same. Gemma conversations saved to disk with f32
  keys are processed again once (another layout).
- The automatic context is estimated again once (its cache's layout version is new, and the K row switches
  `ARC_G4_KF16` / `ARC_Q3_KF32` are part of its key: a context sized for f16 K is never taken for f32 K). The
  two-pass fallback attention (`ARC_FA_GH=0`) takes one row a work-group with f16 K (its multi-row form copies f32
  words), the same results. From a self audit of the change.
- **Tried and dropped:** Gemma's prompt attention with the softmax computed once instead of by every sub-group and a
  P.V without a check per key (ocloc: ~2 -> ~1.45 instructions per FMA where it matters). Bit-identical, but no
  faster: 12B 8K prompt 129.6 / 127.5 tok/s against 126.1 / 127.3 (twice as slow with the probabilities in a second
  local array: half the work-groups per core). The scores' loop over the keys, unchanged, is what bounds it.

## 2026-11-11: a GPU benchmark, in the launcher too

- **`bin\gpu_bench.exe`** (and the launcher's More > Benchmark this GPU) measures what the GPU really does: its
  matrix engines (XMX: int8, f16, bf16 DPAS, in the sub-group size its DPAS takes), its vector engines (FP32, FP16,
  int8 DP4a) and memory read bandwidth, each next to the GPU's figure on paper (EUs x maximum clock x the per-EU
  rate of its kind: the rates behind Intel's numbers, a multiply-add counted as two). The kind and generation come
  from the driver's IP version (Meteor Lake 12.70 / 12.71, Arrow Lake H 12.74, Battlemage 20.1, Lunar Lake 20.4,
  Panther Lake 30.x). It writes a result file and a page in `%USERPROFILE%\.inferzero\bench\gpu`: a comparison table
  of every result file in that folder (results from other PCs dropped there too), then a card per GPU in the layout
  of a spec database's page, with its notes.
- The laptop (Core Ultra 5 125H): FP32 3.85 of 3.94 TFLOPS on paper, FP16 7.69 of 7.88, int8 DP4a 12.98 of 15.77
  TOPS, memory 76.2 GB/s. To check the vendors' matrix figures (the 140T: Intel 77 int8 TOPS, a spec database 154;
  a Pro B60 ~197) on the cards themselves.
- `bench/peak_bench.cpp` removed: the laptop-only DP4a / FMA probe it was, with the 125H's EU count fixed in it.

## 2026-11-11: the 140T's H8 GEMM with coalesced loads and a wider tile, opt-in (QUEUE 33.4b)

- **`ARC_GEMM_XW=1` on a GPU with 8-lane XMX** (Arrow Lake H's Arc 140T): the H8 prompt GEMMs, the 140T preset's
  chunks of 512+ tokens included, run through `zero/kernels/gemm_xw8.cl` instead of `gemm_x8.cl`. A 16-column
  group's weights come in 4 block reads instead of 32 single-word ones, an activation row in one instead of two, and a
  sub-group holds 16 tokens x 16 columns (both 8-column halves). ocloc for the 140T (`arl-h`): 90 instructions and
  6.3 memory reads per DPAS instead of 110 and 13.3, no spills at the default register file. Larger shapes
  (`ARC_GEMM_XW_TM=32`, `_NC=2`) need `ARC_GEMM_XW_GRF=256`.
- Bit for bit gemm_x8's results: checked on the laptop with the 140T preset emulated (the 27B at four shapes) and
  at start against gemm.cl. Off until a 140T measures it; `bench\xw_sweep.ps1` finds the 8-lane shapes by itself.

## 2026-11-11: self audit of gemm_xw and the fused GEMV; the 140T's XMX described

- **gemm_xw's start-up check** built its gemm.cl reference with `ARC_GEMM_TM`'s tokens: with `ARC_GEMM_TM=32` and
  `ARC_GEMM_XW_TM=16` it always failed (the reference left rows unwritten) and turned gemm_xw off. It takes 16 now.
- **`ARC_GEMM_XW_TRY=1` on an 8-lane XMX GPU** (the 140T, Alchemist) ran the 16-lane DPAS instead of emulating it.
- **The split-K pieces of an H8 GEMM** come from one helper (`h8_splits`) for gemm.cl and gemm_xw: the pieces decide
  the result's bits, so two copies of the rule could drift apart; the one-token GEMV's kernel and rows likewise
  (`gemv1_rows`, matmul1 and the fused gate + up).
- **`bench\xw_sweep.ps1`** marks a shape "xw not used" when gemm_xw did not run (H8 off, its check failed) instead of
  "identical"; **`decode_try.ps1`** compares `GEMV_GU=1`'s logits with plain's bit for bit (32 steps), not only tokens.
- Checked: the default path's logits as before (4B, 48 steps), gemm_xw = gemm.cl (emulated), GEMV_GU = two launches.
- **The Arc 140T** in docs/xe2-tuning.md section 12: an Alchemist-kind XMX (16 engines of 1024 bits a core, ~77 int8
  TOPS), where the 27B's prompt reaches ~12; its GEMM's loads per DPAS read with ocloc, and QUEUE 33.4b.

## 2026-11-11: gate + up in one GEMV launch (opt-in), website_bench warmed up

- **`ARC_GEMV_GU=1`** (QUEUE 33.5): for one token, the FFN's gate and up GEMVs run as one launch of both matrices'
  rows, each row as in its own launch (bit-identical: every step's logits of the 4B, 9B and 27B). On the laptop no
  faster (27B about 3 % slower, 4B within noise: its iGPU already reads at ~84 % of its bandwidth), so it is off; a
  B60, at 57 % of its bandwidth with 837 launches a token, measures it with `decode_try.ps1`'s new `GEMV_GU=1` row.
- **`bench\website_bench.ps1`** makes one run per model that is not counted before the sizes: a model's first run
  compiles its kernels, which landed in the first size's generation (a B60: the 4B at 128 tokens 12.8 tok/s, 61 at
  512).

## 2026-11-11: the H8 prompt GEMMs with wider XMX tiles, opt-in (QUEUE 33.4)

- **`ARC_GEMM_XW=1`** (GPUs with 16-lane XMX; off until a B60 measures it): the H8 prompt GEMMs (per-column and
  per-token scales, the Qwen dense models' default on discrete cards) run through a new kernel,
  `zero/kernels/gemm_xw.cl`, whose sub-group holds 32 tokens x 64 columns instead of 16 or 32 x 16. A 64-value chunk
  of activations is then loaded once for 64 columns and a column group's weights once for 32 tokens: twice the
  matrix work per byte loaded. On a B60 the 27B's FFN GEMMs ran at ~33 of ~197 int8 TOPS.
- Bit for bit the results of `gemm.cl`: the same integer sums, split-K pieces and epilogue; checked at start bit for
  bit (a difference keeps gemm.cl and says so). `ARC_GEMM_XW_TM` / `_NC` / `_GRF` change the tile;
  `bench\xw_sweep.ps1` measures them against the default and compares the logits.
- Checked on the laptop with DPAS emulated (`ARC_GEMM_XW_TRY=1`): the start-up check passes at four tile shapes and
  the 4B's logits are identical to gemm.cl's H8; without the switch the default path and H8 are bit-identical to
  the build before. ocloc for the B60 (`bmg`): 32 x 64 without spills at 256 registers. For Panther Lake (`ptl-h`)
  the compiler sizes the register file itself when 256 is not asked for (192 at 32 x 64, 96 at the small shapes).

## 2026-11-11: packing a MoE model from the launcher, and deleting its original

- **More > Pack the selected MoE model** in the launcher (or `scripts\zero-server.ps1 pack -Model <name or path>`)
  makes the package `serverz --pack` makes (the GGUF's experts are kept once, not twice) and copies the model's own
  `mmproj` / `mtp-` files into it. It refuses a dense model, a package, a model the running server has loaded (or
  a server that does not say which it has), and a disk without room. The default model (`last-model.txt`) moves to
  the package. Run again on a packed model, it only copies the side files that are missing.
- Then it offers to **delete the original GGUF** (`zero-server remove-original`), with a warning: deleted for good,
  and the package runs in inferzero only. It deletes nothing unless the package's GGUF and expert file are both
  there and the package was made from that very file: a package now holds `package-source.txt`, its source's full path
  (two folders' models of one file name make one package folder). Every part of a split model goes, and in a
  Hugging Face cache the file the snapshot links to; a link whose file is missing stops it before anything goes.

### Self audit of the 2026-11-11 changes

- **`"cache_prompt": false` could make a later answer wrong:** the system prompt's saved state (the hybrid Qwen
  models: the DeltaNet states at its end) stayed after the request dropped the rest, and a later request that shared
  the start of the `cache_prompt: false` prompt got it restored under its own tokens. It is dropped now, and the
  conversation the server held is saved to the session cache first (it was lost).
- **Projector names** match at a `-` / `_` only: `mmproj-qwen3` is no longer Qwen3.5-4B's in a shared folder. The
  launcher's copy uses the same rules, independent of Windows' language.
- **`runz --kl-ref`** takes the prompt in one chunk only with a reference of every position: a last-position
  reference (and `--kl-decode`) checks the passes the server takes, as before 2026-11-11.

### Second self audit (pack and remove-original)

- **A running server picked the original after a pack:** its model list was from its start, so switching to the
  packed model loaded the original GGUF and wrote its whole expert file again next to the package (or failed once
  the original was gone). A model that has a package since, or whose file is gone, now makes it read the list again.
- **The package's record of its source is a file of its own** (`package-source.txt`): the folder's `source.txt` is
  the expert store's, rewritten whenever a same-named GGUF from another folder runs, so it could name a file the
  package was not made from. Packing again a package of another same-named file is refused too.
- **remove-original deletes nothing when any part is open** (a server, another program) and moves the default model
  first: before, it could stop halfway with the default model pointing at deleted files. The launcher's own model
  list being read, or a pack running, no longer counts as a running server; the launcher passes its port; it reads
  the list again only after the delete question; it does not start the model being packed. The free-space check
  leaves out an expert file the engine wrote before (it moves into the package).

## 2026-11-11: Gemma 4 12B's every-position KL check fixed (it was the check, not the model)

- **Why `tests\run_tests.ps1`'s Gemma 4 12B KL read 10.75:** with weights decoded just in time, a prompt of up to
  32 tokens (two-pass models; 40 otherwise) goes through the layers in passes of 8 tokens. After the 20-token test
  prompt the residual stream held the last pass's 4 rows only, and `runz --kl-ref` read rows 0..19 of it: every
  position compared garbage, while the model's own next token and text were right. E4B and 26B (weights resident:
  one chunk) passed.
- **Fix:** `Engine::keep_rows` (set by runz for `--kl-ref` and `--perplexity`) takes the prompt in one chunk, and
  `logits_at()` refuses a row the last pass did not leave instead of reading it. The server is unchanged.
- Checked: Gemma 4 12B KL 0.00000 (max 0.00001), top token 20 / 20; E4B 0.00389 and 26B 0.01014 as before; the
  4B's logits bit-identical; `--perplexity` runs (4B, 2 x 512: 12.11).

## 2026-11-11: the update notice, old packages explained, `cache_prompt`, projectors in a shared folder

- **The launcher said a newer package was out when it was this one** (issue #8): a check made by an older package
  was kept for 20 hours in the data folder every package shares. A check now holds for the package that made it
  (its folder and serverz.exe's time).
- **A package or GSQ conversion from before 2026-11-10 says why it cannot load:** the model list and the launcher
  show what to do (delete the package's folder, then load the original GGUF or `--pack` it again; convert a GSQ
  model again) instead of "NOT supported". The launcher's update bar no longer promises more than it keeps.
- **`"cache_prompt": false`** in a chat request evaluates the whole prompt (llama.cpp's option); it was ignored, and
  a repeated prompt came back from the cache (a user's report from a Panther Lake B390). Checked: 376 tokens
  evaluated again with it, 371 cached without.
- **A projector in a folder of several models** is taken only by the model it is named after (`mmproj-<name>...`);
  in a Hugging Face snapshot folder, or a folder with one model, as before. The first `mmproj*.gguf` of a folder
  went with every model in it (the same report: an unrelated one on a Qwen3.8 27B fine-tune). The log says when it
  found others; `--mmproj FILE` takes any.
- **Measured on a B60 (issue #4):** `ARC_WIDE_ROWS` +2.3 % on the 27B's generation (swiglu_quant -72 %),
  `ARC_GEMV_SMALL_WG` no gain (kept, bit-identical); docs/before-after.md.

## 2026-11-10 (evening): self audit after the transcription crash ([docs/self-audit-2026-11-10b.md](docs/self-audit-2026-11-10b.md))

- **A live recording's draft is kept** when its accurate pass is stopped or a part of it fails: before, the partial
  text was written over the whole meeting's draft. Checked through the API: a 5-minute live recording's 70-segment
  draft survived a cancelled pass.
- **A live recording goes on after a failed write** (disk full): the WAV stream's error state is cleared, so the
  page's retry is written once there is room; before, every later piece failed.
- **A recording whose compression failed can be merged** (it was refused for good as "still being compressed").
- **Model files checked against their dimensions when loaded:** a tensor with more rows than the model's dimension
  overflowed a host buffer (`load_w_pad`); attention, DeltaNet and FFN matrices and the norms are now checked
  (`expect`, `load_vec`); `expert_used_count` 0, `full_attention_interval` out of range, tensors with too few
  dims and short scale vectors are refused with the tensor's name.
- **Launches:** a zero-size one runs nothing, one past 32 bits is refused. Staging buffers made again drop their
  "already decoded" mark; the crash log no longer reads a released kernel; Whisper's copies are checked.
- Checked: every model on the laptop loads and answers, the 4B's logits bit-identical (plain and MTP), Gemma 4
  E4B / 26B KL as before, Qwen3-ASR after a warm-up, Whisper.

## 2026-11-10: prompt attention on XMX by default on discrete cards; website_bench in a package

- **`ARC_ATTN_XMX` is on by default on discrete cards** (Xe2 / Xe3 with 16-lane DPAS; iGPUs unchanged, `=1`
  forces it, `=0` turns it off): a Pro B60's 64K-token prompts 4B +11.7 %, 9B +8.4 %, 27B +2.1 %, 35B-A3B +10.2 %,
  4K prompts +0.2-1 %, logits bit-identical (issue #4, package bb8952f; docs/before-after.md).
- **`bench\website_bench.ps1` ran nowhere but in a git folder:** it asked git for the commit, and in a package
  (no `.git`) Windows PowerShell stopped the whole benchmark at git's error (issue #4). The commit now comes from the
  package's `bin\PREBUILT.txt`, else from git when there is one.
- Found while testing: `tests\run_tests.ps1`'s Gemma 4 12B KL fails (10.75, every position) with the last commit's
  runz too, so not from today. Not its reference (made again: the same KL, and its last position equals the engine's):
  the every-position logits `--kl-ref` takes for the dense 12B are wrong; generation is not affected (QUEUE).

## 2026-11-10: the server's crash at the start of a meeting's transcription, found and fixed

- **What happened:** an 83-minute meeting (two tracks) recorded fine with its live text, then the server ended at
  the start of its accurate pass, every time it was started again (three times), with nothing in its log: an access
  violation in `ze_intel_gpu64.dll` (0xC0000005, offset 0x386975; the same as a crash on 2026-11-04).
- **Why:** the engine's argument cache (`zero/clrt.h`) skipped setting a kernel argument whose value was the same
  as at its last launch. Level Zero keeps the *allocation* an argument was set with, not only its address. The
  server warms Qwen3-ASR's encoder up on 1 s of silence at start, so a recording's first long part grows the
  encoder's buffers (freed, made again); a new buffer at the address of a freed one kept the kernel's argument on
  the freed allocation, and the next launch (`q3a_gather`) read it inside `zeCommandListAppendLaunchKernel`.
  Reproduced outside the server with `asr_testz --warm` (a 1-s clip, then 5 minutes): it crashed with the cache on
  and ran with `ARC_ARG_CACHE=0`.
- **Fix:** every free of Level Zero memory the engine makes goes through `zrt::mem_free`, which counts it; a cached
  argument counts as unchanged only within the same count, so after any free each kernel sets its arguments again
  once. The case that crashed runs (twice, and with its text: the same as without the warm-up); the 4B's logits are
  bit-identical, plain and with MTP; the meeting was transcribed to its end.
- **A crash now names itself in the log** (`src/crash_log.h`, the server): the exception and the address it read,
  the launch the thread was appending (kernel, sizes, arguments, any argument in no live buffer), and the stack
  with module!function (file:line with a .pdb next to the exe).
- `asr_testz --warm` encodes 1 s of silence first, as the server; `tests\run_tests.ps1` runs it on a generated
  300-s clip (section 7).

## 2026-11-10: small kernels and matrices of few rows in a generation step

- **Measured first:** a Pro B60's 27B GEMVs read 68 % of its bandwidth over all; the Q4_K output head 99 %, the
  codebook FFN 64-77 %, the 1,024-row K / V projections 45 % and DeltaNet's 96-row alpha / beta 14 % (issue #4's
  per-kernel profile against each matrix's bytes in the GGUF; docs/xe2-tuning.md section 10). `swiglu_quant` (15 us
  x 64 a token) and `rmsnorm_quant` (7 us x 129) ran one work-group per row.
- **`ARC_WIDE_ROWS`** (on, every GPU): a 1-8 token pass's `swiglu_quant` (Qwen and MoE paths), `gated_norm_quant`
  and `attn_gate_quant` run one sub-group per superblock (as Gemma's `swiglu_quant` did), `rmsnorm_quant` one
  work-group per 16 superblocks; the same arithmetic. `=0`: as before.
- **`ARC_GEMV_SMALL_WG`** (on, discrete cards, `zero/xe2/`): the single-token GEMV of a matrix with at most 512
  sub-groups in work-groups of 64 instead of 256; the same sums. `=0`: as before.
- `bench\decode_try.ps1` has rows for both off, their tokens checked against plain. Not measured yet: the laptop's
  check (bit-identical logits, speed) waits for a free GPU; the B60's speed for its owner.

## 2026-11-10: the GGUF keys are inferzero.*

- **The metadata keys inferzero writes into a GGUF** are `inferzero.package`, `inferzero.experts_file`,
  `inferzero.source`, `inferzero.virtual.names` / `.types` / `.dims` (a package, `zero/gguf_pack.h`) and
  `inferzero.expert_format` (a GSQ conversion, `tools/convert_gsq.py`). The old names `arcinferzero.*` (and
  `arcinfer.expert_format`) are no longer read: a package or a GSQ model made before this is refused when opened,
  with the way out in the message. A package: delete its folder under `%USERPROFILE%\.inferzero\zero\models`,
  then load the original GGUF (its first start writes the expert file again: 15 GB for the 35B, 52 GB for Flash
  Next) or make the package again with `serverz --pack <the original GGUF>`. A GSQ model: convert it again
  (`setup -Models 35b-gsq`). The server's model list shows such a file as unsupported and the log says why, once
  per file; setup no longer counts it as downloaded. The expert file format (`.experts`) is unchanged.
- **The desktop shortcut from before the rename** (`arcinferzero.lnk`, which opened the removed
  `arc-launcher.cmd`) counts as a shortcut only while its target exists, and a new `inferzero.lnk` replaces it.
- `tools/gguf_dump` prints why a file is refused (an old package, a truncated file) instead of aborting.
- Metadata names only: the logits are bit-identical (Qwen3.5 4B, `runz --dump-logits` before and after).

## 2026-11-10: the old command names are gone

- **Removed:** `arc-launcher.cmd`, `arc-server.cmd`, `scripts\arc-launcher.ps1`, `scripts\arc-server.ps1` and
  `scripts\pi-arc.ps1` (the aliases kept at the rename earlier today), zero-pi's `--arc-model`, and the `pi-arc` /
  `arc-server` shims `install-pi` wrote into Pi's bin folder (`install-pi` now deletes them). The commands are
  `zero-launcher`, `zero-server`, `zero-pi` and `zero-recorder` only; scripts and shortcuts that used the old names
  need the new ones.
- **Pi's command shims hold the folder's path:** after the repository folder is moved or renamed (this laptop's was,
  with the project), `zero-pi` and `zero-server` in a terminal fail with a missing-file error until `install-pi` (or
  `setup`) runs again; `install-pi` says so now.
- Also: three stale git worktrees from the old folder pruned; `.gitignore`'s comment names the data folder
  `%USERPROFILE%\.inferzero`.

## 2026-11-10: self audit of the website, logo and renames ([docs/self-audit-2026-11-10.md](docs/self-audit-2026-11-10.md))

- **The package's key filter** (`zero/gguf_pack.h`) is `arcinferzero.` again: the rename had made it `inferzero.`,
  so a re-pack doubled the package keys and a GSQ model's `expert_format` was copied into new packages.
- **Recordings of an older recorder** are recognised by the server (not imported while still written) and block a
  second `zero_recorder start`.
- **`zero/kernels/decode.cl`** is back to its text before the rename (a comment): the kernel cache stays valid.
- **Old names still found:** a model asked for as `arcinferzero/<name>`; `%TEMP%\arcinferzero-view-*` cleaned up; bench
  results recorded as `arcinferzero`; `bin\arc_recorder.exe` of an older build; Pi `settings.json` entries.
- **The alias scripts** return the script's exit code; the web UI's prompt explains the name as infer + zero.
- **Website and docs:** the grid animation, the speed-up claim (2.7-7.4x, as the bars show), numbers that disagreed,
  the code tabs' accessibility; `zero_recorder.exe` spelled right; the recording commands; links to the archived log;
  a note for packages from before the rename.
- Checked: serverz, runz and zero_recorder build; Qwen3.5 4B logits bit-identical to the package from before the rename
  (4ee6693, SHA-256); every changed script parses; the recorder and alias commands run.

## 2026-11-10: the commands start with zero-, as the logo

- `arc-launcher` -> **`zero-launcher`**, `arc-server` -> **`zero-server`**, `pi-arc` -> **`zero-pi`**, `arc-recorder` ->
  **`zero-recorder`** (`bin\zero_recorder.exe`); the files moved (`zero-launcher.cmd`, `zero-server.cmd`,
  `scripts\zero-launcher.ps1`, `scripts\zero-server.ps1`, `scripts\zero-pi.ps1`, `tools\zero_recorder.cpp`) and every
  living doc and help text with them (560 uses in 38 files; the planned `arc-transcribe` is `zero-transcribe`, zero-pi's
  `--arc-model` is `--zero-model`).
- **The old names still work:** `arc-launcher.cmd`, `arc-server.cmd`, `scripts\arc-server.ps1`, `scripts\pi-arc.ps1` and
  `scripts\arc-launcher.ps1` hand over to the new ones; `install-pi` puts `zero-pi`, `zero-server`, `pi-arc` and
  `arc-server` on PATH (shims installed before point at the old scripts, which hand over); `--arc-model` is still taken.
- **The recorder's stop event** became `inferzero-recorder-stop` with the project's rename; `bin\zero_recorder.exe
  stop` / `status` (what `zero-server record-stop` / `record-status` run) now also find a recording an older exe
  started (`arcinferzero-recorder-stop`).
- Checked: `zero-recorder` builds and reports "not recording"; every changed script parses; `zero-server help` and
  `status`, `zero-pi --help`, and the same through `arc-server.cmd`, `scripts\arc-server.ps1` and `scripts\pi-arc.ps1`;
  `zero-launcher -SelfTest`; `install-pi` on this laptop (the four shims; Pi's provider "inferzero" alone).

## 2026-11-10: arcinferzero is now inferzero

- **Why:** Intel's trademark guidelines ask third parties not to put an Intel mark, or any part of one, into a product
  name, and "Arc" is Intel's graphics brand. The new name keeps the rest; the 4 x 4 "Z" logo stays.
- **Renamed:** the code, the scripts, the web UI, the launcher, the website and the living docs (672 uses in 120
  files); packages are now `inferzero-<version>-win64.zip`; the README and the website say "formerly arcinferzero"
  and carry Intel's trademark notice.
- **Carried over by themselves:**
  - the data folder `%USERPROFILE%\.arcinferzero` moves once to `%USERPROFILE%\.inferzero` (the same drive: a rename,
    instant), and the paths in `last-model.txt`, `launcher.json`, `server.json` and `zero\ctx-auto.txt` follow it;
    if a file is held open (an older server running) the old folder is used as it is (`src/app_dir.h`,
    `scripts/app-dir.ps1`); the bench scripts and the launcher's Downloads dialog now ask `Get-AppDir` too;
  - Pi: the provider becomes "inferzero" (the "arcinferzero" and "arcinfer" entries are replaced), and `pi-arc`
    takes `arcinferzero/<id>` and `arcinfer/<id>` as before;
  - the web UI's settings in the browser (`arcinferzero-prefs` is read when `inferzero-prefs` is not there yet);
  - a desktop shortcut `arcinferzero.lnk` counts as the launcher's shortcut;
  - `arc-server` reads a package's `VERSION.txt` and the server log with either name.
- **Kept on purpose:** the GGUF keys packages already carry (`arcinferzero.package`, `.virtual.*`, `.experts_file`,
  `.expert_format`, `.source`: an older engine must still read a newer package); the `ARC_*` switches (settings
  people have in their scripts); the command names `arc-launcher`, `arc-server`, `pi-arc`, `arc-recorder` (renamed
  later the same day: the entry above); history
  (older CHANGELOG entries, `docs/archive`, dated reports, `docs/before-after.md`, `BUGS.md`).
- **Checked:** the ten package exes build; every changed PowerShell script parses; the data-folder move in a test
  profile for the scripts and an exe (the three state files rewritten, JSON's doubled backslashes and the lower-case
  context file included; a second start uses the new folder); `arc-launcher -SelfTest` in the real profile (it moved
  this laptop's data folder; the state files followed).

## 2026-11-10: the logo: a 4 x 4 grid drawing a Z, everywhere

- The "az" circle (two letters of a system font) becomes a drawn mark: 4 x 4 rounded tiles, ten lit, drawing a Z
  for zero; the tiles are the engine's int8 GEMM tiles and its mixture-of-experts slots, only some of them in memory.
  The launcher's accent colours stay (Windows' #0078D4, on a dark theme #59A7E3); the dim tiles are #C9D3DE /
  #2B3440. No font, so it looks the same on every system.
- Where: the launcher's splash, window and tray icon (`Draw-ArcMark` in scripts/arc-launcher.ps1, snapped to whole
  pixels at 16 / 32 px; the tray is accent when a model runs with a smaller green dot, grey when stopped), the web
  UI's `/icon.svg`, and the website's favicon, nav and footer (`site/icon.svg`, byte for byte the server's).
- Checked: `arc-launcher -SelfTest` in light and dark (the splash), the tray icon at 32 and 16 px on a light and a
  dark taskbar, the website in the browser; serverz rebuilt. The concepts compared (the grid, a slashed zero, a gauge,
  a drawn monogram) are in a local research page, not in the repository.

## 2026-11-10: one logo: the launcher's "az" mark on the website and the web UI

- The web UI's icon (`/icon.svg`: the browser tab, and the app icon when installed) was a green chat bubble, and the
  website had an arc of its own. Both are now the launcher's splash mark: a circle in Windows' default accent
  (#0078D4) with a bold white "az" in Segoe UI; on a dark browser theme the accent 35 % lighter (#59A7E3) with a black
  "az", as the launcher draws it on a dark window.
- `site/icon.svg` is the website's favicon and the same SVG the server sends; the website's wordmark is Segoe UI
  semibold, as in the splash. The app manifest lists the icon as `any` (a round mark, not a full-bleed one).
- Checked: the server's SVG is byte for byte `site/icon.svg`; both themes rendered in the browser next to
  `arc-launcher -SelfTest`'s splash picture.

## 2026-11-10: Qwen3.8 Flash Next MTP from its head file (QUEUE F34, GitHub #7)

- **MTP for Flash Next**: its MTP head as a file of its own, llama.cpp's form (`mtp-Qwen3.8-Flash-Next-Q4_0.gguf`,
  base Flash Next's head, in SC117's Swift-1.5 repository), found next to the model or given with `--mtp-model FILE`
  (runz; the server and the catalog list such a model with MTP). Its `blk.48.*` tensors join the model file's tensor
  list (`GgufFile::adopt`), so the existing loaders read the layer; the model's expert file is not touched (the
  layer is left out of it).
- The head as llama.cpp's `qwen4exp` `graph_mtp`: per row and residual stream `eh_proj([rmsnorm(e) * enorm |
  rmsnorm(h_s) * hnorm_s])` (new kernel `mtp_concat_hc`; the (row, stream) pairs as rows of one matmul), then a
  Flash Next layer (mixers, QSA attention with its own KV and QSA state, 512-expert MoE), its next h the layer's
  residual streams, logits through its own head mixer and the model's lm_head. The model's hidden rows for it are
  the residual streams before the head mixer (hc x E: `mtp_hw()`, also in slots, snapshots and sessions).
- Its 512 Q4_0 experts **pinned** in the expert cache: built in the expert file's record form into one device buffer
  (1.41 GiB, counted before the cache is sized; a dry load counts it too), the cache's table row fixed
  (`XCache::pin`, kept when the table is reset), no `ensure()` for that layer. Q4_0 experts with K 640 accepted.
- Laptop, code prompt, 128 tokens: plain 6.65 tok/s, MTP 1 / 2 / **3** / 5 drafts 7.02 / 7.57 / **7.84** / 7.06
  (96 % accepted at 3, 3.88 tokens a pass; more drafts read more experts from the SSD per pass): **3 by default**.
  Greedy text equal to plain on every run; the default path bit-identical to the build before (Flash Next, 35B,
  4B). serverz with DFlash2 and the head: both picked per step.

## 2026-11-10: website benchmark's KL for Gemma with BOS

- **`run-tests.cmd`** (bench\run_tests.ps1) for testers: double-click, leave it running, drag the zip it puts on the
  Desktop into the GitHub issue. It finds the models (Hugging Face cache, packages, and every folder the server
  reads), runs `f33_sweep` (Qwen models) and `website_bench`, keeps the PC awake, writes the user's folder in paths as
  %USERPROFILE%, and packs the logs. `run-tests.cmd quick` (about an hour), `run-tests.cmd quick 27B` (one model).
- **`website_bench.ps1` found no models outside the Hugging Face cache** (issue #4: models in `C:\hf`) and then stopped
  at its summary: it now also asks `serverz --list-models` (model-dirs.txt folders), takes DFlash2 drafters from
  there too, and ends with a plain message when nothing is found. Flash Next's split first part is found too.
- **`mgpu_probe`: host memory <-> GPU one way** (pinned / pageable, compute queue / copy engine, 40 / 256 MB) and every
  GPU at once: the link that MoE experts streamed from RAM come over. The 6.6 GB/s measured on the B60s before was
  GPU -> host -> GPU (two copies), not this. Laptop iGPU: pinned 36.8 GB/s host -> GPU, pageable 12.5, copy engine
  20.5.
- B60 results of 2026-11-10 (issue #4, `f33_sweep`, one GPU): `ARC_ATTN_XMX=1` bit-identical and faster where
  attention weighs most (35B 64K prompt 1,808 -> 1,993 tok/s, +10 %; 27B +2 %; attention kernel time -5 to -15 %);
  the chunked DeltaNet on XMX slower than `gdn_recur` (27B 4K: 900 vs 290 ms), so it stays off.

- `runz --dump-ids` (used by `bench\website_bench.ps1` for its KL against f32) now puts BOS first when the model
  asks for it (`tokenizer.ggml.add_bos_token`, the Gemma models); Qwen ids are unchanged. Without it the KL measured
  a prompt the model never sees: Gemma 4 E4B 0.01644 -> **0.00273**, 26B-A4B 0.09272 -> **0.00479** (480 tokens of
  USAGE.md, laptop). Only the measurement changed; the engine did not.
- runz warns when the model and its context need more GPU memory than the share less the driver's part
  ("does not fit: ... expect paging"); `--must-fit` stops there with exit code 3. `website_bench.ps1` passes it and
  marks such a size "does not fit" instead of measuring hours of paging: Qwen3.8 27B at 131K needs 17.13 GiB, over this
  laptop's 16.5 GiB share (back to its default after a driver update), and had measured 8.6 tok/s and a first token
  after 4.2 hours (at 32K: 14.09 GiB, 125 tok/s). The server's `--ctx auto` already stays inside the share.
- **Qwen3.8 Flash Next in setup's model list** (`setup -Models flash`, the launcher's first run; asked for in issue #4):
  the two IQ3_S parts from ISTA-DASLab (84.5 GB) and bartowski's f16 image projector, copied next to them. The disk
  check counts 141 GB (the first start writes the 56 GB expert file; `serverz --pack` then makes a package and the
  parts can go to a backup disk). `-Models all` leaves it out. A model kept as a package counts as present.
- **Launcher: a Downloads dialog** (Get models, and More > Download models; `scripts\launcher-downloads.ps1`): several
  models of the list at once, or a Hugging Face link (a repository, a folder or a file; the other parts of a split file
  and an image projector ticked with it, the projector copied next to a model in another folder); Pause / Continue
  (curl stopped, the `.part` files kept, resumed with `-C -`); a speed limit (`ARC_HF_RATE`, curl `--limit-rate`),
  changed while it runs; a cut connection tried again without end (5 s doubling to 60 s, back to 5 s once data flows;
  a token or a missing file shown at once instead); models kept as packages count as here. `hf-get.ps1` ends a
  stalled connection after 2 minutes under 1 byte/s (`--speed-time`), so it is tried again. Tested in Windows
  PowerShell 5.1: 5 MB/s measured 4.5-5.0, 1 MB/s 1.0; pause, continue and two cut connections on a 0.67 GB file; the
  first-run welcome dialog is unchanged.
- **Launcher splash screen**: shown as soon as the theme's colours are read, while the fonts and the window are made
  (before, a few seconds with nothing on the screen); the "az" mark in the Windows accent colour, a status line, the
  theme's light / dark colours and rounded corners. Nothing new to load: WinForms and System.Drawing as before, the
  mark drawn in code. It closes just before the window shows (closed in the window's Shown event it was the window's
  owner and took the window with it). `-SelfTest x.png` also saves it as `x-splash.png`.

## 2026-11-09: DeltaNet in chunks and prompt attention on XMX, opt-in (QUEUE F33.1 / F33.2)

For the Arc Pro B60 (issue #4) and the Arc 140T (issue #5); every switch is off by default, the default path is
bit-identical to before (laptop, `--dump-logits` of a ~1,500-token prompt and 24 tokens: Qwen3.5 4B, Qwen3.8 27B,
Qwen3.6 35B-A3B against the build before tonight's work, bb8fd33; `bench\f33_sweep.ps1`'s check, USAGE.md as the
prompt and 16 tokens, all seven models of the laptop: Gemma 4 E4B, Qwen3.5 4B and 9B, Gemma 4 26B-A4B, 27B, 35B,
Qwen3.8 Flash Next).

- **Chunked DeltaNet, `ARC_GDN_CHUNK`** (prompt passes of 32+ tokens; MTP / DFlash2 verification keeps `gdn_recur`):
  - The rule in chunks of 32 tokens: per chunk lg = the cumulative log decay, A = beta exp(lg_l - lg_m) k_l.k_m and
    P = exp(lg_i - lg_l) q_i.k_l; the tokens' deltas by one forward substitution, the outputs from P and the state at
    the chunk's start, the state moved once a chunk.
  - `=3`: on XMX. `gdn_chunk_x` (zero/kernels/gdn_x.cl, 16-lane DPAS: B580 / B60) holds the state as DPAS
    accumulators and does every product of the chunk as f16 DPAS (k.k and q.k, state x k, state x q, the deltas
    through (I + A)^-1, the outputs, the state update): 192 DPAS per sub-group and chunk, no spills (ocloc,
    Battlemage). `gdn_chunk_x8`: the same at 8 lanes for the 140T (264 DPAS, ocloc Arrow Lake H). Every f16
    operand is scaled by a power of two and split into high and low halves (~22 bits): the low halves stay out of
    f16's subnormal range, which a DPAS may flush (found by an independent review of the kernels). Checked once per
    device at start against `gdn_chunk`; a difference above 1e-4 keeps `gdn_recur`. Without XMX `=3` runs
    `gdn_recur`.
  - `=1` / `=2`: the vector form `gdn_chunk` (any GPU; 4 work-items per state row, 4x `gdn_recur`'s). **No gain on
    the laptop** (4K prompt, DeltaNet kernel time): 4B `gdn_recur` 241-272 ms, `=1` 3,506-3,662 ms, `=2` 2,367-2,409 ms;
    9B 235 / 3,507 / 2,358 ms (512 -> 353 / 388 tok/s); 27B 782-806 / 12,420 / 8,832 ms (prompt 136 -> 95 / 110
    tok/s); 35B 538-542 / 5,802 / 4,448 ms (782 -> 381 / 454 tok/s); Flash Next 435 / 7,492 / 5,090 ms (207 -> 156 /
    172 tok/s). Kept for a B60 to measure.
  - Its own kernel part, zero/kernels/ops_gdn.cl (the last ops part), so ops.cl and the other ops kernels' cache are
    as before.
- **Prompt attention on XMX, `ARC_ATTN_XMX=1`** (zero/kernels/attn_x.cl, put in front of the ops parts for
  `attn_q8`; 16-lane DPAS, Qwen): attn_q8's int8 q.k sums (8 DPAS per 16-key tile at 256 dims) and its u8 P x int8 V
  sums (16 DPAS) replace 2 x 256 DP4a per tile; everything else is attn_q8's in the same order. DPAS sums integers
  exactly: bit-identical, checked at start bit for bit (else attn_q8). Not for the 140T: its attention kernels run
  16-lane sub-groups, its DPAS takes 8 (QUEUE F33.2b).
- **Checks without XMX:**
  - The chunked math in numpy against the sequential rule (f32: 1e-7 relative, chunks of 16 and 64, decays to 1e-6).
  - On the CPU's OpenCL (oneAPI), against a double-precision reference, 100 tokens (three chunks and a short one),
    gated and fused gates: `gdn_chunk` output 5.5e-7, `gdn_chunk_x` / `gdn_chunk_x8` with DPAS emulated 6.8e-7
    (state 1.4e-7); `gdn_recur` 1.4e-7.
  - Laptop GPU with DPAS emulated (`ARC_GDN_XEMU=16` / `8`, `ARC_ATTN_XEMU=1`): both DeltaNet forms pass their start
    check (4.5e-7 of the result); the attention's start check and the logits of the 4B, 27B and 35B (~1,500-token
    prompt, 24 tokens) bit-identical to attn_q8, five shapes of the kernel test too (33 to 100 tokens, 3 and 4
    heads per sub-group).
  - The laptop GPU compiler got a byte of a shuffled word wrong in an unrolled loop (the emulation's first form):
    every attention output of the check apart, the CPU's OpenCL exact. The emulation now splits bytes before the
    shuffle (both agree). Only the emulation was affected; real DPAS does not shuffle.
- **KL against `gdn_recur`** (4 perplexity chunks of USAGE.md): 4B `=1` / `=2` 0.00084 (perplexity 13.2847 ->
  13.2987), `=3` emulated 0.00087 (16 lanes) / 0.00091 (8 lanes); 27B `=1` / `=2` 0.00063 (6.3926 -> 6.3941); 35B
  `=1` / `=2` 0.0052 (9.9937 -> 9.9532). Not bit-identical sums change later int8 roundings (and in the 35B the
  experts some tokens pick, as a prompt chunk size of 512 instead of 8,192 does: 0.0026-0.017): that is this size of
  KL, below 0.015. 9B `=1` / `=2` 0.00071 (10.9342 -> 10.9305), `=3` emulated (8 lanes) 0.00078; 27B and 35B `=3`
  emulated 0.00062 / 0.0053.
- **Flash Next** moves more: `=1` against `gdn_recur` 0.025 (perplexity 7.2929 -> 7.3449; precise mode 0.0125), `=3`
  emulated 0.026. Against the f32 reference (`ref_forward`, 480 tokens of USAGE.md) the chunked kernel is as close as
  `gdn_recur`: fast mode 0.0134 against 0.0117 (top token 456 / 480 against 451), precise 0.00879 against 0.00875
  (463 against 464): within its limits (~0.02 fast, 0.015 precise). The two runs sit ~0.012 from f32 each, on
  different sides, so they are 0.025 apart.
- `bench\f33_sweep.ps1` for the B60 / 140T owners: per model, prompt speed at 512 to 65,536 tokens for every mode,
  the kernels' own time, KL, and the attention's logits against attn_q8. `ref_forward.exe` is in the package now,
  for `bench\website_bench.ps1`'s KL against f32.

## 2026-11-09: MoE expert files kept with their model

- A MoE GGUF's expert file went loose into `%USERPROFILE%\.arcinferzero\zero\`, so a deleted GGUF left it there
  unnoticed (a Gemma 4 26B's 12.3 GB, found and deleted by hand). It is now written next to the GGUF when the GGUF is
  inside the data folder (`models\`, or a package in `zero\models\<name>\` as before), else into a folder of its
  own, `zero\models\<model name>\`, with a `source.txt` naming the GGUF (`zero/expert_store.h`, `place`).
- One written to `zero\` before moves to its place (a rename, with its `.lru` state) the next time its model is
  loaded: not written again. A package made from a GGUF still takes that file over as before.
- At each MoE load the log names an expert folder whose GGUF no longer exists (on a drive that is there: an unplugged
  drive's GGUF is not "gone") and any expert file still loose in `zero\`. Nothing is deleted automatically.
- Checked with a test in a temporary data folder (no model, no GPU): the outside and inside places, `source.txt`, an
  old file and its `.lru` moved, nothing moved while a package is made, and both log notes.
- Self-audit the same day: a package made from a GGUF lands in that GGUF's folder, and the original GGUF is usually
  deleted afterwards; the log then called the package's folder deletable. A folder holding `<folder>.gguf` (a
  package) is never named now. A split GGUF's folder is named without its `-00001-of-N`, as its package would be,
  and `source.txt` holds the GGUF's full path. Both cases added to the test.

## 2026-11-09: earlier server logs kept

- `arc-server` (and so the launcher) wrote `server.log` anew at each start, so the log of the server before a restart
  or a crash was lost. That log now moves to `%USERPROFILE%\.arcinferzero\logs\`, named by the local date and time
  of its last line, `server_2026-11-09_14-13-48.log` (readable, sorted by name = by time; two in the same second:
  `server_... (2).log`), and the newest 10 are kept. `server.log` stays the running server's. Checked over 14 starts
  in a test folder: 10 files, one from the day before removed as the oldest, a same-second pair kept apart; a log
  still held open is left as it is without an error.

## 2026-11-09: discrete cards, MoE generation defaults from a B60's run

- `ARC_XALL` is on by default on discrete cards: when every expert of a MoE model is in the GPU's cache, a token's
  layers no longer read their experts' ids on the host. A Pro B60's 35B: 78.77 -> 88.05 tok/s, host waits 43 -> 3 a
  token, the same tokens. `ARC_XALL=0` turns it off.
- The single-token GEMV prefetch (`ARC_GEMV_PF`) is off for MoE models on discrete cards (the same B60's 35B 78.77
  with it, 85.15 without); dense models keep it (27B 23.21 with it, 22.45 without).
- iGPUs are unchanged: both switches stay off there.
- `bench\decode_try.ps1`: for MoE models the rows now try the old settings (`XALL=0`, `XALL=0 GEMV_PF=1`, `mtp XALL=0`)
  against the new defaults; dense models keep `GEMV_PF=0`.
- `docs/xe2-tuning.md`: the B60's 2026-11-09 generation and prompt numbers.

## 2026-11-09: web UI, queue and steer messages while an answer runs

- Send (or Enter) while the model answers no longer does nothing: the message goes into a queue above the message
  box and is sent when the answer ends, the queue in order; the draft in the box is kept. A queue belongs to its chat
  and goes on when the chat is opened again.
- Each queued message has **Steer**, **Edit** (back into the box) and **×**. Steer puts it into the answer under
  way: while tools run it is added to the conversation before the model's next round (the tools are not stopped);
  while the model writes, the answer stops where it is (kept) and the message is sent at once.
- Checked in the browser with the server's replies simulated in the page (no request reached the model, no chat
  saved): two queued messages sent in order after the answer, the draft kept; Steer while writing stopped the answer
  after 11 of 30 words and sent the message within 0.6 s; Steer while a 2.5 s web search ran left it running and the
  next request held assistant (call), tool, user (the steered message).

## 2026-11-09: web UI, the menu button on your own messages visible

- The ▾ button of a message (Edit, Copy, Delete) was drawn in muted grey on the user's green bubble, nearly
  invisible, so deleting one's own message looked impossible. It now takes the bubble's text colour. Checked: the
  user's menu holds Edit, Copy and "Delete (and its files)", an answer's Copy and Delete.
- Near the bottom of the chat the menu opened under the message box (its place assumed a 140 px menu; yours has three
  rows), so its Delete could not be clicked. It is now measured and opens above the button when there is no room
  below. Checked at 1280x800: the last message's menu at y 518-636 above its button (647), every row clickable (it
  was placed down to y 769, under the box at 742); the first message's still opens downwards.

## 2026-11-09: Qwen's reasoning: the line-break exemption undone

- The change below ("no stray ctlsc") is undone: every token the reasoning used is penalized again, line breaks
  included. With line breaks free of the penalty the 35B circled twice in one chat within hours, writing short lines
  ("(Done).", "(End).", "<br>", indented) without closing its reasoning (18,038 characters once, 82 `<br>` lines;
  the second stopped by hand). The presence penalty is there against such loops, and these lines are mostly line
  breaks and indents. A stray "ctlsc" now and then is the smaller harm. `Sampler::set_presence_penalty` keeps its
  optional exempt list (unused, tested in `test_sampler`).

## 2026-11-09: web UI, the thinking box rendered as Markdown

- The thinking box showed the reasoning as plain text, so a model's `**bold**`, tables and `<br>` appeared as
  written (the 35B uses them often). It is now rendered with the answer's Markdown renderer (HTML escaped as there;
  `<br>` the only tag kept), while streaming, when the box is opened and in a reopened chat. Checked: bold, a table
  with `<br>`, a list, and a `<script>` line shown as text.

## 2026-11-09: web UI, several files attached at once

- Choosing several files with the attach button (or dropping them on the chat) attached only the first. The page
  cleared the file input right away, which also empties the browser's live list of the chosen files while the first
  upload was still running. The list is now copied first (checked: 3 files chosen, the live list 0 after clearing,
  the copy 3). Every model was affected, OCR ones included; the transcript upload already copied its list.

## 2026-11-09: Qwen's reasoning, no stray "ctlsc"

- The 35B sometimes ended its reasoning with a stray "ctlsc" (as in "Let's investigate.ctlsc") and then copied it
  in later turns of the same chat. Cause: the presence penalty of 1.5 the server uses while a Qwen model thinks
  also hit "\n" and "\n\n", which every reasoning uses, so the line break before `</think>` lost to the model's
  4.4 % "ctl" (raised to ~17 %). Tokens of spaces and line breaks alone are now free of that penalty; words are
  still penalized. A request's own presence_penalty and Gemma are unchanged. DFlash2 / MTP were not involved:
  their drafts never change which token is drawn. A chat that already holds "ctlsc" may keep copying it: start a
  new one.

## 2026-11-09: web UI, the time in an answer's bubble

- The time under a model's answer sat on the bubble's bottom edge (0 px): the floated time fell below the answer's
  blocks with the message bubble's -4 px offset. It is now its own right-aligned line with 8 px below it, as in the
  user's bubble (checked: user, short and long answers all 8 px).

## 2026-11-09: self-audit: web UI headers, OCR Stop and saves, model file checks, CET

- **Web UI page:** `X-Frame-Options: DENY` and a CSP (`frame-ancestors 'none'; object-src 'none'; base-uri 'none'`):
  no other site can frame it and trick a click. **Every reply:** `X-Content-Type-Options: nosniff`.
- **Web UI:** a chat file's link only to this server's files, `data:` or `blob:`; a stream's error event is shown
  (it was skipped: the answer just stopped); a summary answer with no text, a damaged saved tool call and a minutes
  answer that is not an object give a clear error instead of a TypeError.
- **OCR:** Stop ends the page being read (it ran to the end of the page); a `.md` that cannot be written stops the
  reading with "could not write ... (is the disk full?)"; saving into the chat's folder is refused (403) from a page
  another local program serves, as a folder of one's own was; a file name with bad UTF-8 is cleaned (it threw); a
  project's image up to 64 MB and sent with its own type; a PDF page at most 4000 px on its long side.
- **Model files:** the embedding tables' width is checked against `n_embd` (and Gemma's per-layer table against
  `n_ple x n_layer`), every matrix is checked to be 2-D with int-sized dimensions and bytes, a GGUF's array counts
  against their element width, tensors without data only for a package's experts, and the package's expert file
  name must be a plain name. A damaged or hostile GGUF now stops with an error at load instead of reading past a
  tensor.
- **Fatal errors:** `CL_CHECK`, a failed device allocation and a failed kernel module end the process with
  `_Exit(1)` like `ZE_CHECK` (no destructors run under the server's threads); a module build log is no longer left
  behind on a soft failure.
- **Build:** linked with `/CETCOMPAT` (the CPU's shadow stack may be used).
- Checked: `runz --dump-logits` bit-identical before / after for Qwen3.5 4B, Gemma 4 E4B and the Qwen3.6 35B-A3B
  package; serverz headers read back; Xiaomi-OCR-0: two images into one `both.md`, the same request with a foreign
  Origin 403, a bad UTF-8 name answered; `node --check` of the web UI's script. Stop during a page was not tried
  by hand. Findings left as they are, and why: BUGS.md "Audit 2026-11-09".

## 2026-11-09: the session cache's size holds for all models together

- **`--session-cache-gb` (default 8 GB in arc-server, pi-arc and the launcher) is one limit for every model now.**
  Before, each model file (and each layout of it: `-std`, `-nomtp`, `-q2`) had its own 8 GB folder, so trying many
  models filled the disk: 49 GB of saved conversations on the laptop, 20 folders, most of models no longer there.
  When the server starts and after each save, other models' saved conversations go first, oldest first, until all
  of them fit; then this model's least recently used ones, as before (`prune_other_sessions`, `tools/server_cache.h`).
  Only finished `.arcs` files are removed (a `.tmp` being written is left). The size is set in the launcher's
  "Session cache (GB on disk)" field ("Set as default" keeps it), `arc-server start -SessionCacheGb N`, a preset, or
  `serverz --session-cache-gb N`; 0 turns it off.
- Checked: a sessions folder with another model's five 3 MB files and a 10 MB limit: the server removed the two
  oldest at start (9 MB left), left the `.tmp` and its own folder alone. Its own folder is found by file identity
  (a model renamed only in case), and only files that start with the conversation header are removed.

## 2026-11-09: launcher and CLI, stages 0 and 1 of docs/launcher-research.md

- **Launcher, look:** follows Windows' light / dark mode and accent colour; Segoe UI Variable; on Windows 11 a dark
  or light title bar, rounded corners and Mica behind the title bar (`DwmSetWindowAttribute`); sharp at 125-200 %
  scaling (DPI-aware, sizes scaled). Main actions in one row (Start, Stop, Get models), the rest under **More**; the
  running server, the selected model and the **Log** (live) as tabs; a GPU memory bar.
- **Launcher, behaviour:** the server state is read by a background runspace, so a busy or hung server no longer
  freezes the window; errors and notices go to a bar under the title ("Copy details", "Show log") instead of boxes, and
  an error in a WinForms event too (GitHub #5: boxes repeated under the ticking timer); a failed start says what its
  exit code means.
- **Launcher, new:** whether the options fit the GPU memory, under the context field; **presets** (save, pick, delete;
  the same files `arc-server start -Preset` reads); a **tray icon** (on / off, Start the default model, Stop, web UI;
  minimized, the window waits there); a **doctor report** with Copy; an **update check** once a day against the latest
  GitHub release (a link only; off in More).
- **No terminal window next to the launcher:** `arc-launcher.cmd` runs PowerShell in `conhost --headless`; with Windows
  Terminal as the default terminal, `-WindowStyle Hidden` still opened one (GitHub #5).
- **`arc-server`, new commands:** `list`, `logs [-Follow]`, `doctor`, `version [-Check]`, `pull <keys>`, `presets`,
  `save-preset -Preset <name>`; `-Json` on `status`, `list`, `doctor`, `version`, `presets`.
- **`arc-server start`, new options:** `-Preset <name>` (the command line wins), `-DryRun` (checks everything, prints
  the `serverz.exe` command and a GPU memory estimate, starts nothing), `-Drafter auto|mtp|dflash|off|<folder>` (one
  name for `-NoMtp`, `-DFlash` and the `-nodf` / `-df` ids). A start whose estimate does not fit prints a warning.
- **Exit codes:** 1 a wrong option, 2 the server stopped while loading, 3 not ready within 10 minutes, 4 the port is
  used by another program (checked before the server starts).
- **`arc-server help`** is a short page now; `help options`, `help writing`, `help errors`, `help all` show the rest.
- **`arc-server.cmd` in the package root**, so the command works without Pi; setup offers to put the folder on PATH
  when Pi is not installed.
- Checked: both scripts parse on PowerShell 5.1 and 7; every new command run on both (the dry runs, presets, `-Drafter`,
  exit 4 with a port held by another program); the launcher's `-SelfTest` (opens, lists 26 models, loads a preset,
  saves a picture, closes) on 5.1 and 7, light and dark. No server was started for these checks (other sessions'
  servers were running).

## 2026-11-08: the launcher's start command (issue #6); self-audit of ARC_BATCH / ARC_XALL

- **Launcher, "Copy command" (GitHub issue #6):** copies the command that starts the server the way Start would,
  with the selected model and the options as set, for a script or shortcut of one's own:
  `powershell -NoProfile -ExecutionPolicy Bypass -File "...\scripts\arc-server.ps1" start -Model <id> ... -Port N`
  (also shown as `arc-server ...` when Pi's commands are installed).
- **`arc-server` prints the server's own command line** (`bin\serverz.exe -m ... --ctx ...`) when it starts one and
  in `arc-server status`; it is kept in `server.json`, and the launcher's "Running server" panel shows it.
- **Self-audit fix, ARC_BATCH on a split model:** a recorded list that had to wait for an earlier one (64 lists
  behind) waited without the multi-GPU time limit and abort; it now waits through the guarded wait. One GPU and
  ARC_BATCH off: unchanged. Checked: 4B tokens the same plain / ARC_BATCH=32 / 2 GPUs simulated / both.
- Audit notes, no change needed: ARC_XALL refuses while an expert prefetch thread exists (its copies finish before
  it ends), counts the MTP layer (n_layer + 1 layers), and drops to the usual path after a shrink(); the recorded
  lists and their events live as long as the process (no growth on model switches); the server's engine lock keeps
  Whisper and the chat model from recording at the same time.

## 2026-11-08: batch OCR: many images and PDFs into Markdown at once

- **Several images and PDFs in one message** to an OCR model (Xiaomi-OCR-0) are read one after another, in their
  order, each under `# File K / F: name`, a PDF's pages under `## Page N / M` (`ocr_batch`, `tools/server_ocr.h`).
  Before, two images went into one prompt together, and of a PDF with other files only the PDF was read.
- **Saving (`ocr_save` gets `per_file`):** one merged `.md` with every file under its own `# name` heading, or one
  `.md` per file named after it. The web UI asks with a dialog when 2+ such files are sent: one Markdown per file,
  one merged Markdown, or just show it, and a folder.
- The single PDF's rules hold: no existing file is written over (`name (2).md`), 50 pages a request over all the
  files (an image is one), and "lanjut" goes on from the last file and page with text. The bar shows the file and
  page under way.
- **OCR from a project's folder, no upload:** file parts `project:<id>/<path>` are read in place (`ocr_file`), with
  read_file's rule: only the chat's own project (`project_id`, which the web UI now sends in OCR mode), from this
  computer and the server's own page. A project's chat saves its `.md` into the project's `ocr` folder by default
  (made when needed; the `name (2).md` rule holds), with no copy in the chat's files: every chat of the project lists
  it. A folder named inside the project is treated the same. The web UI's new folder button next to the paperclip
  (an OCR model in a project's chat) lists the project's PDFs and images, subfolders too, to tick several or all.
  A chat model later in the same chat sees such a part as a pointer to `read_file`.
- **Self audit fixes** (single PDF too, where the cause was older):
  - A heading is a whole line `# File K / F: ` / `## Page N / M`. Before, the model's own "## File menu" or
    "## Page 9" in a page's text moved the place "lanjut" starts from, so pages were read twice or skipped.
  - A page or file that failed is read again by the next "lanjut". Before, its error counted as its text; the web
    UI's kept note now names the page before it.
  - Per-file saving makes the first `.md` before the stream begins, so a folder that cannot be written is an HTTP
    error. A later file's error stops the answer with a note.
  - "lanjut" after Stop appends only to a `.md` named after the file in progress, never into an earlier file's.
  - A refused project file stops only its own turn, not every later turn of the chat.
  - The project's `ocr` folder is made only when the `.md` goes there.
- `ocr_pdf`'s file handling is now shared helpers (names, folder check, a new `.md`, appending, one image read);
  its output is the same. A save error (bad folder, every name taken) now comes before the stream begins, as a
  proper HTTP error.

## 2026-11-08: generation switches for the B60: recorded command lists (`ARC_BATCH`), MoE tokens without host waits (`ARC_XALL`)

Both are off by default and bit-identical; they wait for a Pro B60's numbers (issue #4).

- **`ARC_BATCH=n`:** kernels are recorded into regular in-order command lists of up to n kernels. Each list is
  appended to the immediate list when full (`zeCommandListImmediateAppendCommandListsExp`, Level Zero 1.9), instead
  of every kernel being appended there on its own.
  - On a B60, `replay_probe` measured 2.56 us a kernel appended one by one against 0.66 replayed. The 27B's
    generation step has ~3 ms between kernels (46.6 ms against 43.6 of kernel time).
  - The lists start at 4 kernels after every host wait and double up to n.
  - Copies, fills, barriers and waits first append the list being recorded (`zrt::imm()`, `zrt::sync()`).
  - Kernels with an event (profiling, `ARC_ZE_WATCH`) and `ARC_ZE_SYNC_EACH` go to the immediate list as before.
  - A driver without the call says so and runs as before.
- **`ARC_XALL=1`:** a MoE model whose experts are all in the GPU's expert cache (the 35B-A3B on a 24 GB card) no
  longer reads each layer's expert ids on the host. That read is ~40 host waits a 35B token, and each one leaves the
  GPU idle.
  - The check runs once a token (`XCache::all_in`) and sets every expert's address, as `ensure()` does for its hits.
  - It is cleared when a launch that ran out of memory takes slots away.
  - Not used for multi-GPU or prompt chunks.
- **`ARC_LAUNCH_STATS=1`** also prints the host waits per token.
- **`bench\replay_probe.exe`** also times lists appended to the immediate list and lists recorded again each step
  in chunks of 8 / 32 / 128 / all.
- **`bench\decode_try.ps1`:**
  - new rows `BATCH=16/32/128`, `mtp BATCH=32`, and for MoE models `XALL=1` and `XALL=1 BATCH=32`; these rows say
    whether the tokens match plain;
  - launches and host waits per token;
  - the prefetch row is now `GEMV_PF=0`, since the prefetch is the default on discrete cards.
- **Laptop (no gain expected there):**
  - Logits bit-identical with `ARC_BATCH` 8 / 32 / 256 on the 4B and the 35B, with MTP, and over a 2,161-token
    prompt.
  - Speed: the 4B the same; the 35B -5 %. Its cache holds 83-87 % of the experts, so a token waits 43 times and the
    lists stay short.
- **Tried and removed: a dense prompt GEMM straight from the raw rows (DP4a)**, meant for short prompts.
  - On the laptop `decode_w8` takes 1.1 s of a 4.6 s 512-token 27B prompt.
  - The fused kernel was bit-identical but not faster (512 tokens 106-110 against 110 tok/s): the GEMM and the
    decoding both run on the vector units.
  - An XMX version could still pay on B-series cards; see `docs/before-after.md`.

## 2026-11-08: B60 defaults: H8 + 4 column groups for Qwen dense models, GEMV prefetch; arc-server -DFlash

- **H8 + `ARC_GEMM_CG=4` by default** on discrete cards with 16-lane XMX, for the Qwen dense models. A Pro B60
  (issue #4, package d980970) measured the 27B at 512 tokens 615 -> 677 tok/s, 4K 766 -> 1,024 and 16K 706 -> 937.
  The fused H8 decode made H8 win at short prompts too. KL against the plain path 0.00154 (max 0.064).
  - Not for MoE models (the 35B: H8 -1 %), Gemma, Flash Next or two-pass models.
  - Off again if XMX does not pass its start-time check, which now runs the H8 kernel with 4 column groups.
  - `ARC_H8=0` turns it off.
- **`ARC_GEMV_PF=1` by default on discrete cards** (bit-identical). The B60's 27B generated 22.33 -> 23.08 tok/s, as
  in an earlier run (22.39 -> 23.11). iGPUs keep it off. `ARC_GEMV_PF=0` turns it off.
- **`scripts\arc-server.ps1 -DFlash <folder>|off`:** a DFlash2 drafter of your own, e.g. for a fine-tuned variant the
  model list does not pair with one (asked for on issue #5).
- **`bench\b60_sweep.ps1` / `gemm_sweep.ps1`:** their "no H8" rows set `ARC_H8=0`; the KL base is the plain path.
- **Also measured on the B60:** `ARC_GEMM_WPF=2` +3.6 % without H8 (794 tok/s) but slower with it; f16 XMX 400-550,
  so f16 is out on the B60 too. Two cards with H8: 1,292 tok/s at 4K, four cards 1,624. MTP 47.7 tok/s.
- The laptop's iGPU path is unchanged: logits bit-identical (4B, ~6,000-token prompt in 1,024-token chunks).
- **The GPU memory left to the driver is 0.5 GiB instead of 1.2**, on iGPUs and discrete cards alike (the user's
  choice). On four B60s that frees 2.8 GiB for context and the expert cache. The auto context and the expert cache use
  the share less 0.5 GiB. It is a setting, not fixed: `arc-server -DriverReserveGb N`, `serverz` / `runz
  --driver-reserve-gb N`, or `ARC_DRIVER_RESERVE_GB=N` (0 to 8 GiB). The laptop: the 35B's expert cache 83 % of its
  experts.
- **Self audit of these changes:**
  - The `--ctx auto` estimate now counts H8's buffers (~0.35 GB on the 27B).
  - The H8 kernels are checked apart from the rest of XMX, and per kind of check, not once per process. A failing H8
    kernel turns H8 off, not XMX.
  - The 140T check covers split-K.
  - The 140T preset keeps to Qwen dense models and stays off with `ARC_H8_SB=1`.
  - The per-256 H8 path tracks its staged form.
  - `-DFlash` takes an absolute path and refuses `-NoMtp`.
  - The B60 sweep's FUSE rows and its CG labels for H8 (4 column groups now by default).

## 2026-11-08: web UI: bold that holds italic

- `**not *clean* yet**` was shown with its asterisks: the bold pattern allowed no `*` inside, so italic inside bold, as
  models often write it, broke the whole pair. Bold may now hold single asterisks, and the italic inside it is
  formatted too. `***both***` stays `<em><strong>`. Checked with the web UI's own functions on the answer where it
  showed (22 bold pairs, none left raw) and on edge cases (two bolds on a line, an unclosed `**`).

## 2026-11-08: the Arc 140T preset (XMX + H8 for long prompt chunks), on by default

- **Measured on a 140T** (issue #5, package 65e9e81, 27B, two runs): XMX + H8 + 32 tokens per sub-group + 2 column
  groups + 128 registers, against DP4a: 512 tokens 154-159 / 154 tok/s, 2K 221 / 180-181, 8K 230-232 / 184-186.
  The fused H8 decode is part of the gain: without it 123 / 201 / 226. KL against DP4a 0.00154.
- **On iGPUs with 8-lane DPAS and weights decoded just in time, it is now the default, per GEMM.** Chunks of 512+
  tokens take XMX + H8 with those tiles; shorter chunks and resident weights keep the DP4a default (16 tokens, 256
  registers). TM 32 needs a multiple of 32 tokens, and DP4a spills with 128 registers.
- `gemm_kernel` takes its tokens per sub-group and register file per call. The staging buffer remembers whether it
  holds the W8 or the rotated H8 form. The start-time check runs the preset's kernels against DP4a bit for bit.
  `ARC_140T=0` turns it off.
- **Laptop 125H, the preset forced with DP4a standing in for DPAS** (`ARC_140T=1 ARC_XMX_TRY=2`, 27B):
  - a short prompt is bit-identical to the default;
  - 2,048 tokens in 1,024-token chunks are bit-identical to H8;
  - 2,148 tokens (H8, H8, then a W8 chunk) run with the same greedy text as the default.

## 2026-11-08: generation switches on every GPU (for the 140T)

- `ARC_GEMV_GRF`, `ARC_GEMV_PF` and `ARC_GEMV_MT_GRF=128` now act on any GPU when set. They had needed the xe2
  switches, which the 140T no longer has since it keeps DP4a for prompts, so its generation runs ignored them.
  Unset, nothing changes.
- `bench\decode_try.ps1`: a row for MTP with 128 registers (iGPUs default to 256).
- Docs: the 140T's generation is within ~15 % of its bandwidth bound; MTP / DFlash2 drafting is the lever.

## 2026-11-08: generation switches for the B60, and a generation profile in decode_try

- `ARC_GV_IQ3S=1`: single-token IQ3_S weights (the 27B) through `gemv_v`, loaded straight into registers, instead of
  `gemv`'s local window with two barriers a step. Bit-identical (27B logits). The 125H's iGPU is 6 % slower with it
  (27B step 210 -> 223 ms), so it is off until a B60 measures it.
- `bench\decode_try.ps1`: rows for it (with 2 and 4 rows per sub-group) and, for MoE models, the experts' GEMV rows per
  sub-group (`ARC_MOE_GR_GU` / `ARC_MOE_GR_DN`). It ends with a `--prof-gen` table per model.
- Docs: where a B60's generation time goes, and command-list replay as the largest lever left (35B ~10 %, from
  `replay_probe`: 2.56 against 0.66 us a kernel).

## 2026-11-08: H8's weight rotation in the decode pass; a weight prefetch switch for the B60

- H8 with weights decoded just in time (the 27B on a B60): `decode.cl h8r_decode` decodes the raw rows and rotates
  them in one pass. Before, `decode_w8` wrote a W8 copy and `h8r_weights` read it back. It runs from a weight's second
  decode on, once its column scales are known. Q2_K / Q4_K / Q5_K and the per-32-float types keep the two passes.
  `ARC_H8_FUSE=0` turns it off.
- Laptop 125H, 27B, 2K prompt in 1K chunks, H8: logits bit-identical; decode + rotation 4,569 -> 1,566 ms of GPU time;
  101.5 -> 114.3 tok/s. A B60 spent 561 ms there per 4K prompt.
- `ARC_GEMM_WPF=n`: the XMX prompt GEMM prefetches its weights n 64-value chunks ahead. Off by default, a B60 to
  measure; bit-identical.
- `bench\b60_sweep.ps1`: rows WPF and FUSE, and a phase E over several GPUs (`-Gpus 2,4`).
- Self audit (`docs/self-audit-2026-11-08.md`):
  - XMX is off again only on iGPUs with 8-lane DPAS. Discrete Alchemist cards had been caught too; they now go
    through the model-sized speed check.
  - The sweeps' FUSE rows and f16 labels fixed.
  - `ARC_MOE_CG` x `ARC_MOE_WY` bounded.

## 2026-11-08: f16 XMX at 8 lanes for the Arc 140T (opt-in)

- `ARC_X16=1` now also works on iGPUs with 8-lane DPAS (`gemm_x16.cl -DX16_SG8`). The weights and activations are f16,
  so one DPAS sums the whole row with no per-32 rescaling between DPAS, where the 140T's int8 XMX kernel lost to DP4a.
  Strata chose f16 XMX for the A770, the same 8-lane XMX generation.
- ocloc for Arrow Lake H: real DPAS, no spills at 8 / 16 / 32 tokens and 128 / 256 registers.
- On the 125H with DPAS emulated: the start-time check passes (largest error 2.9e-4). KL against DP4a: 4B 0.00087, 27B
  0.00069.
- `bench\igpu_grf_sweep.ps1` measures it with its own settings, and its KL.
- Off by default; the 140T has to measure it.

## 2026-11-08: a 140T sweep with 128 registers and H8; its last results recorded

- The 140T's `igpu_sweep` (issue #5): DP4a still ahead, 172-185 tok/s on the 27B's prompt. The best XMX variant was 128
  registers at 162-168 tok/s; `ARC_X8_W` did not help. GEMV switches stayed within noise.
- `bench\igpu_grf_sweep.ps1` sweeps the tiles of XMX with 128 registers and of DP4a, each without and with H8. Then the
  H8 forms, and KL / perplexity against DP4a.
- `ARC_GEMM_GRF=128` and `ARC_GEMM_CG` also act on the DP4a GEMM of iGPUs when set. Bit-identical on the 125H (4B,
  27B, H8). Defaults unchanged.
- ocloc: DP4a with 128 registers spills on Arrow Lake H as on the 125H, so the sweep measures it once only.

## 2026-11-07: variants of the 8-lane XMX GEMM for the 140T, and a sweep script

- `ARC_X8_W=1`: `gemm_x8.cl` with one 8-lane sub-group for both halves of a 16-column group, so 8 tokens' activations
  are loaded once for 16 columns. ocloc (`arl-h`): 28 % fewer memory messages and 17 % fewer instructions per DPAS, no
  spills.
- `ARC_GEMM_GRF=128` now also applies on 8-lane GPUs.
- `bench\igpu_sweep.ps1`: prompt GEMM settings with XMX forced; prompts of real use (512 / 2048 / 8192 tokens, chunk
  sizes); the GEMV switches for generation.
- `bench\b60_sweep.ps1` for the B60: every tile, register and 2D-load setting changed alone from the default (764 tok/s)
  and from H8 (918), the winners put together; then prompts of real use (512 / 4K / 16K, 16K in 4K chunks), KL and
  perplexity against the default, and the 35B's expert GEMM shapes.
- Bit-identical on the 125H with DP4a standing in for DPAS (4B, 27B, H8). Defaults unchanged: XMX stays off on 8-lane
  GPUs until a 140T measures a variant faster than DP4a.

## 2026-11-07: a model-sized XMX speed check on iGPUs; Panther Lake's XMX code read

- iGPUs with 16-lane XMX (Lunar Lake, Panther Lake such as the B390) now time a 512 x 17,408 x 5,120 GEMM at start, XMX
  against DP4a, and keep XMX only if it is at least 3 % faster. DP4a is timed with the iGPU's own settings. If XMX loses,
  the discrete cards' settings are dropped too, so the GPU runs as one without XMX.
- Discrete cards skip the check (`ARC_XMX_SPEED=1` runs it). `ARC_XMX_FORCE=1` skips it anywhere.
- ocloc for Panther Lake (`ptl-h`): real DPAS in all 19 builds, 2D loads, no spills. The 140T's `gemm_x8` (`arl-h`, the
  options of its log): no spills either, but about 2.5x the vector work per multiply.
- Tested on the 125H only with DP4a standing in for XMX: DP4a kept, the same text. No iGPU with XMX has run it yet.

## 2026-11-07: XMX off again on 8-lane GPUs (Arc 140T)

- A 140T ran package 911b6fa (issue #5): `gemm_x8.cl` passed the start-time check, and the 27B's logits were
  bit-identical to DP4a on real DPAS. But its 2K prompt fell from 178-182 to 113-120 tok/s (GEMMs 1.6x slower).
- The start-time speed check had picked XMX: its 64 x 64 x 1024 GEMM stays in the caches. It is removed.
- 8-lane GPUs run as with `ARC_NO_XMX=1` again; `ARC_XMX_FORCE=1` still takes XMX after the bit-for-bit check. Other GPUs
  are unchanged (the 125H: no XMX; `ARC_XMX_TRY=2` still runs).

## 2026-11-07: MoE expert GEMM work-group shapes (for the B60)

- A B60's profile of the 35B-A3B (16K prompt, issue #4) showed the expert GEMMs at about 21 TOPS, against about 50 for
  the dense GEMMs: each work-group held one sub-group, 16 columns x 16 slots, so nothing was shared through L1.
- `ARC_MOE_CG` / `ARC_MOE_WY`: several sub-groups per work-group, along a tile's columns (activations shared) and along
  the tiles, mostly one expert's (weights shared). Default 1 x 1, as before.
- Logits bit-identical at every shape (35B on the laptop, five shapes). Laptop speed within noise (745-766 tok/s), so
  the default stays until a B60 measures it: `bench\gemm_sweep.ps1` has the rows MOECG and MOEWY.

## 2026-11-07: an XMX kernel for GPUs with 8-lane sub-groups (Arc 140T)

- A 140T's log (issue #5) showed the 16-lane XMX GEMM differing from DP4a in every layout. The extension requires an 8-lane
  sub-group on such GPUs, so that form was undefined there, and the earlier note that its DPAS looked emulated was wrong.
- `zero/kernels/gemm_x8.cl`: the W8 and H8 layouts at 8 lanes, with real `dpas` for Arrow Lake H in ocloc and no spills.
  Native types and MoE tiles stay on DP4a on these GPUs. It is checked against DP4a at start, then taken only if at least
  5 % faster.
- Not run on real DPAS yet. On the 125H with DP4a in its place (`ARC_XMX_TRY=2`): logits bit-identical (4B, 27B, H8).
  GPUs without XMX, as the 125H, are unchanged.

## 2026-11-07: self audit of remote access, speech, the web UI and H8

Full list: [docs/self-audit-2026-11-07.md](docs/self-audit-2026-11-07.md). The ones that matter most:

- **Security:** with `--host`, a peer on the network could set the remote password through `/v1/remote/config`: it is from
  this computer only now. A sign-in being checked can no longer outlive a new password or "sign everyone out". Bodies
  through a tunnel with access off are refused before they are read. `/sessions` is fenced, and plain http moves to https.
- **Crash:** the first load of an offered speech model could end the server when GPU memory was short. It is an error now.
- **Lost work:** the transcript model, accuracy, kept format and glossary were not being saved (a stray comment in
  `trSavePrefs`). A pass again could replace a good transcript with an empty one, or an open edit could overwrite the new
  text.
- **Audio without speech models:** a chat with audio and `/v1/audio/*` reach a model that hears again.
- The tool loop guard looks at results too: a cycle stops, and polling a job whose output changes goes on.
- The default path's logits are bit-identical (4B, 27B). H8's KL is unchanged. Tests: 53 passed, 5 skipped.

## 2026-11-07: a Pro B60 measures the new H8 (no code changed)

- 27B, 4K prompt, one card: default with 2D loads 764 tok/s, `ARC_H8=1` 918 tok/s (+20 %). KL against the default path
  0.00144. Wider tiles (750) and 256 registers (783) do not help H8.
- Over 2 and 4 cards: default 1,097 and 1,501 tok/s, H8 1,570 over 4. The sums limit it there.
- 2D loads gave bit-identical logits on a short prompt. H8 stays opt-in until the 35B and a model with resident weights
  have run on it.

## 2026-11-07: iGPUs with XMX use it, with the discrete cards' defaults

- Lunar Lake, Panther Lake and Arrow Lake H now go through the XMX check and take the defaults found on the B60: XMX
  GEMMs, 2D block loads with a prefetch where the driver has them, and 128 registers on GPUs with 64-byte registers.
  None of this has been measured on such an iGPU yet.
- Arrow Lake H (Arc 140T) was refused XMX until now because it has 8-lane sub-groups. XMX is taken there only if a
  start-time speed check finds it at least 5 % faster than DP4a; the log gives both times. `ARC_XMX_FORCE=1` takes it
  anyway. It keeps 256 registers (128 spills). (Its kernel: see the entry above.)
- A GPU without XMX, as the 125H, is unchanged: logits bit-identical (4B, 27B).

## 2026-11-06: H8 with one scale per column and token, the rotation in registers

- `ARC_H8=1` (opt-in) now works as urakozz's int8 prefill does: one scale per column, computed once per weight, and one
  per token, so a GEMM's dot products over K are one integer sum; the rotation in registers instead of a barrier per
  stage through local memory; random signs before it; on XMX the sums stay in the DPAS accumulators. `ARC_H8_SB=1`
  keeps the per-256 form, `ARC_H8_NOD=1` leaves out the signs.
- Laptop iGPU, 27B 4K prompt: H8 111.9 -> 122.5 tok/s, rotations 6.64 -> 4.3 s, and KL against the default path 4B
  0.00403 -> 0.00243 (largest row 2.41 -> 0.072), 27B 0.00100 -> 0.00148. **The default path stays faster on the iGPU
  (140 tok/s) and stays the default:** DP4a's GEMMs gained only 1-4 % from the scales.
- Not run on an XMX GPU yet; a B60 has to say what it gains (docs/xe2-tuning.md section 8 has the commands and the
  comparison with urakozz's code).
- The start-time XMX check tests an H8 kernel only when H8 is asked for, so one cannot take XMX away from other runs.
- The default path's logits are bit-identical (4B, 27B).
- Against f32, 480 tokens: 27B default KL 0.00054, H8 0.00141, urakozz's method simulated on our model 0.00080; 4B
  0.00100, 0.00274, 0.00150. `runz --kl-ref` also prints the logits' relative L2, worst cosine and top-5 overlap;
  `ref_forward REF_HAD_D=1` simulates the signs. The signs show no clear gain.
- H8 on Qwen models now rotates the activations from f32 instead of their int8 form, without the signs (`ARC_H8_INT=1`:
  from int8; `ARC_H8_D=1`: with signs). KL against f32: 27B 0.00141 -> 0.00113, 4B 0.00274 -> 0.00195. The iGPU's 27B
  H8 prompt goes 122 -> 116-117 tok/s, +285 MB. The default path's logits stay bit-identical (4B, 27B); its memory and
  kernels do not change.

## 2026-11-06: 2D block loads by default on discrete cards; the B60 and 140T results

- **Prompts on discrete cards:** the XMX GEMM now loads its activations with 2D block loads and a prefetch. A Pro B60's
  27B prompt goes 608 -> 763 tok/s (+25 %, the owner's runs with the switches on). `ARC_GEMM_2D=0` and
  `ARC_GEMM_2D_PF=0` turn them off. If they do not build or differ from DP4a at start, XMX goes on without them. iGPUs
  are unchanged (laptop: 4B logits bit-identical). Logits on a B60 and runs over several GPUs are still to be checked.
- `bench\gemm_sweep.ps1`: the X2D row is pf (default), on, off.
- **Measured and recorded, nothing changed** (docs/xe2-tuning.md section 8, docs/multi-gpu.md):
  - H8 stays off: 570 tok/s on one B60 against 608, because the rotations cost more than the GEMMs gain. KL 0.0012
    against the default path.
  - H8 over 2 and 4 B60s: 863 and 1,232 tok/s (27B).
  - `ARC_TP_ATT_QSPLIT=1`, 35B over 4 B60s: 3,441 -> 4,290 tok/s. It stays opt-in.
  - The 140T has not run XMX or H8 yet: the engine's XMX GEMM needs a GPU without 8-lane sub-groups, and its H8 runs
    came from a package older than H8.

## 2026-11-06: the speech models load from Settings whatever the server was started with

- Settings > GPU memory said "not on this server" for Whisper when the server had started without `--whisper`, with
  the model in the Hugging Face cache, and a server started with a chat model alone offered no speech models at all.
  Now Whisper and Qwen3-ASR found in the cache can be loaded and unloaded there, and by Transcribe again, whatever the
  start said. They are not loaded at the start, so they take no GPU memory until asked. Only an explicit
  `--whisper off` or `--asr off` keeps one away.
- The Transcripts tab is therefore there on every server, with its saved transcripts.

## 2026-11-05 (night): remote access with a password; motion in the web UI; expert cache policies measured

- **Remote access** (docs/remote-access.md): the web UI from a phone or anywhere through a Cloudflare Tunnel, with a
  password.
  - Settings > Remote access (on this computer only): on / off, the tunnel's host names (`*.trycloudflare.com` for
    a quick tunnel), the password, sign everyone out. `scripts\remote-tunnel.ps1` starts the tunnel.
  - The password is kept as PBKDF2-HMAC-SHA256 (200,000 rounds, random salt) in `remote.json`. A session cookie is
    `HttpOnly`, `Secure` and `SameSite=Strict`, valid for 30 days.
  - Wrong passwords: one second's wait each. 8 within 15 minutes lock that visitor out; 40 from everyone lock every
    sign-in. Parallel guesses count before they finish.
  - Before signing in, a remote request may only fetch the sign-in page or send a small sign-in. Anything else is
    refused before its body is read.
  - A remote page may only send its own origin. A request through a tunnel or proxy for any other host is refused,
    so a tunnel that rewrites the Host header cannot pass as local.
  - Unless "Full access remotely" is on, a remote session is not "this computer". Every local-only check refuses
    it: the terminal, projects and their folders, memory, MCP servers (their tokens too), slots, loading models by
    hand, and the OCR save into a folder.
  - Local use is unchanged: requests that name localhost go through the same checks as before.
- **Files the model wrote (HTML, SVG, XML) open sandboxed at their own address** (`/files/<id>`): a page with
  scripts cannot act as the web UI. The web UI already showed them in a sandboxed frame.
- A self audit (an independent review of this change) found the gaps fixed above before they shipped:
  - `local_peer` was true for every tunnel request;
  - `mcp.json` was readable remotely;
  - bodies were read before the password check;
  - parallel guesses passed the lockout;
  - the app manifest needed the cookie;
  - sessions outlived turning access off;
  - forwarding headers of other proxies were not checked.
- **The web UI installs as an app** (`/manifest.webmanifest`, `/icon.svg`): "Add to home screen" on a phone.
- **Motion** (docs/webui-plan.md):
  - new messages slide in from their side;
  - three dots wait in the answer until the first token;
  - the prompt bar shows a sheen until the server reports progress;
  - menus and dropdowns pop in, buttons press, and on a phone a chat slides in.
  - Opening a chat draws its messages at once. Everything uses transform and opacity, and nothing animates with
    reduced motion on.
- **Security fix:** a page of another local server (`http://localhost:<another port>`) could reach `/v1/mcp`, which
  starts MCP servers (programs). `/v1/mcp` is now only for the server's own page or a client without an Origin, as
  the tools and files already were.
- **Measured, not changed: the expert cache by use instead of recency** (QUEUE). Every expert route was recorded
  (`ARC_XTRACE=file`, zero/xcache.h) and replayed with the cache's own slots (`bench/xtrace_sim.py`).
  - Flash Next, 19 % cached: a protected part for experts used again saves at most 2.8 % of the generation misses;
    choosing the least-used victim costs 0.5-19 % more.
  - Qwen3.6 35B, 98.4 % hits: -1.1 to +1.1 %.
  - The bound with the future known (Belady) is -52 % / -76 %: what would help is predicting the next layer's
    experts, not counting the past. The LRU cache stays.
- **Measured, not changed: Flash Next's prompt gate / up** (QUEUE 21.15). The 4K prompt runs at 186-194 tok/s, and
  gate / up take 6.2-6.3 s of its 20.6 s of GPU time. Two variants were slower, so both were left out:
  - through moe_rg, as the down projection: 8.6 s, 170-174 tok/s;
  - the tile's activations shared through local memory: bit-identical, but 9.6 s, 166-170 tok/s.
- **Tests:** `test_hc` and `test_fa` (hyper-connection and float attention kernels) failed in `run_tests.ps1` since
  `ops.cl` was split into parts on 2026-11-03. They built from `ops.cl` alone, so their kernels were not found.
  They now build from every part in order and pass. The whole suite: 53 passed, 5 skipped (4B, E4B, 27B, 35B, Flash
  Next, DFlash2, server, error paths).

## 2026-11-05 (evening): measured, not changed: Vulkan, fewer kernels, GEMVs inside the model

- **Vulkan instead of Level Zero: not worth a port** (`bench/vk_probe`). Bandwidth 74.1 against 72-75 GB/s; a
  launch 1.07 against 1.50 µs. llama.cpp's Vulkan backend on the 4B: prompt 254 against 855 tok/s, generation
  15.0 against 16.9.
- **Fewer kernels per token: not worth it.** 426-852 empty kernels added to every 4B token cost nothing. Fusing would
  save only the small kernels' own work (~1.7 ms a token, 3 %). Gate and up as one stacked GEMV were 1-2 ms a token
  slower.
- **The GEMVs are not slower inside the model.** Every GEMV of the 4B back to back, with no other kernel, took
  43.7 ms, and ~45 ms in the model (`--prof-gen`, measured right after). The earlier 48-60 GB/s profile was taken
  while other programs used the memory: one CPU thread copying memory slows the GEMVs from 67.5 to 55 GB/s, while CPU
  arithmetic changes nothing. The size of the data read and the weights' alignment change nothing either. This
  corrects the 2026-11-04 note below that put the lower profile numbers on its events.
- Details: `docs/lessons-intel-arc-igpu.md` section 5, QUEUE 31.13.

## 2026-11-05 (afternoon): transcribe again with another model; summaries kept per model; minutes not cut short

- **↻ Transcribe again** (a transcript's header): a finished, failed or stopped transcript is transcribed again with
  the speech model, language, accuracy and glossary chosen in the list's options (`POST /v1/transcripts/<id>/redo`).
  - The text there now stays until the new pass is done.
  - If the new pass is stopped or fails, the record and the text are put back as they were.
  - A two-track recording (Me / Others) kept as M4A or MP3 is read one channel per track. Before, only its WAV could
    be read, and a second pass failed with "not a WAV file".
- **Summarize again keeps the summary before it.** The button reads "↻ Summarize again" once there is a summary.
  - Each summary is saved with the chat model that wrote it.
  - A list above the summary shows up to 10 earlier ones (model and time); picking one shows it and puts it in the
    minutes, and the shown one is kept in its place (`summaries.json`, `PUT {summary_use: k}`).
- **Transcribe again loads the speech model it needs.**
  - Whisper or Qwen3-ASR are offered also while they are unloaded, marked "(unloaded)". The page loads the chosen
    one first, for Transcribe again and for new uploads; before, a new upload waited in the queue forever.
  - If the GPU memory is short, the page unloads the chat model first. The next chat loads it again. It does not
    unload it while this page is writing an answer.
  - Gemma 4 E4B transcribes as the chat model, so Whisper and Qwen3-ASR are unloaded to give it room, after a
    question.
  - Gemma 4 E4B is offered while Whisper and Qwen3-ASR are unloaded too (`/v1/server/memory` names it). Before, it
    was missing from the list just when it was the only speech model there.
- **Failsafes of Transcribe again:**
  - stopped while it waits in the queue, the transcript is put back as it was (before, it was marked stopped);
  - if the server stops twice during it, the record is put back with a note, not marked failed;
  - a pass again that failed is noted apart from the record's error (`redo_error`), shown after its details. The
    first build wrote it as the record's error, and the page then said "could not open the transcript".
- **No length limit on summaries and minutes.** Each answer goes on until the model ends it or the context is full.
  - The notes of each part of a long meeting had 1200 tokens. On a 2-hour meeting, part 2 reached that limit and its
    notes stopped mid-way, so the rest of that part was missing from the minutes.
  - The part summaries (700 tokens, at most 15 points), the merges (900, at most 20) and the form (6000) had limits too.

## 2026-11-05: a recording survives a muted or switched microphone; a tab's sound without the speakers

- **A live recording no longer stops when the microphone goes away.** Muting it, Bluetooth earphones connected or
  disconnected, or another default input all ended the microphone's track, and the page took that as Stop: a
  meeting was cut into three recordings. Now the page opens the microphone again (the new default) and the same
  recording goes on.
  - The gap is silence on that track, so the timeline stays right.
  - If the device is still switching, the page tries again after 1, 2, 3 ... s.
  - If the microphone cannot be opened, the recording continues with silence on its track.
  - Stopping a share from the browser's bar still ends the recording.
- New source **"A tab's sound, silent here"**: a tab's sound (a video) is recorded and transcribed without being
  played on the speakers (`suppressLocalAudioPlayback`, Chrome / Edge). It works for browser tabs only; a desktop app
  such as Zoom is captured through "Entire screen" with "Share system audio".
- Every dropdown takes the page's colours. The minutes form's dropdown was the browser's white, and open lists were
  light in the dark theme (`color-scheme` now follows the theme).
- Every dropdown has one look:
  - rounded corners, an arrow, a ring on hover and on keyboard focus;
  - in Chrome / Edge with `appearance: base-select` (checked in Chrome 152), the open list is a rounded card with
    rounded rows, and the arrow turns while it is open;
  - a choice longer than its room wraps instead of running under the arrow;
  - the minutes form's dropdown looks like the buttons beside it.
- Self audit of these changes:
  - a microphone retry still pending after Stop could join the next recording and replace its tab sound; each
    retry now belongs to its own recording;
  - "silent here" with a window or the screen picked says to pick a tab, and the dialog opens on the Tab list;
  - the arrow sits at the right edge of a wide dropdown;
  - buttons, dropdowns and Run / Don't run share an 8 px corner;
  - neutral buttons hover in the accent colour, and only the delete buttons turn red;
  - disabled buttons look disabled.
- Talk and voice messages keep their microphone as a live recording does. When its track ends (Bluetooth
  earphones switched, connected or gone), the new default input is opened and feeds the same capture: Talk keeps
  listening, and a voice message keeps recording. Retries come after 1, 2, 3 ... s, only for the session they
  started in. Checked in the page with stand-in microphones: replaced twice, and nothing after a stop.
- **A shared sound that drops no longer stops a live recording.** Chrome ends the shared sound's track (a tab's, or
  the system's with the screen) when the sound output changes, for example Bluetooth earphones on or off, while the
  share itself goes on. The recording now goes on with that track silent (the microphone keeps recording in a Me /
  Others recording), and a "↻ Share sound again" button takes it up again. The browser's dialog needs a click, so
  this cannot be done for you. Stop sharing in the browser's bar still ends the recording. Checked in the page with
  stand-in shares: kept on, shared again, and the old share's late end ignored.
- **Transcripts stays open with speech to text unloaded.** Unloading Whisper and Qwen3-ASR to make room for
  Qwen3.6 35B locked the whole menu, saved transcripts, summaries and minutes forms included, though none of them
  needs a speech model. The menu is now open while the server has its transcript library, and says that new
  transcriptions need speech to text loaded again. `@` on a transcript in a chat works then too. Checked with the 35B
  loaded and both speech models unloaded: the menu opens, a transcript shows Summarize and the forms.
- **No limit on tool rounds in the web UI.** An answer stopped after 8 rounds of tool calls (the model said it had
  run out), whereas Pi has no such limit. Now it goes on until the model answers. Stop ends an answer at any time,
  and a full context is compacted as before. Only a loop ends it: the very same calls (names and arguments) a third
  time in a row are not run, and the model is asked to answer from what it has.

## 2026-11-04 (night): MCP servers for the model, a self audit of every kernel, the web UI and speech

- **MCP servers** (docs/mcp.md): the web UI's model can use the tools of other programs on this computer, Blender
  through blender-mcp for example. The server is the MCP client:
  - stdio servers come from `%USERPROFILE%\.arcinferzero\mcp.json` in the usual `mcpServers` form;
  - each server is switched on per project, or for the chats outside projects;
  - a server starts at its first use and runs in a job object;
  - its tools reach the model as `<server>__<tool>`;
  - a call that may change something asks Run / Always for this tool / Don't run;
  - returned images go into the chat's files.

  Settings has a section for the servers, their switches and the file. The endpoints are `/v1/mcp*`, from this
  computer only. Checked against a test server: 22 API checks, covering tools over two pages, a missing program,
  the instructions, text, image and error results, an unknown tool, a timeout, parallel calls, a crash and its
  restart, stop, and a config change. In the browser with Qwen3.5 4B: a read-only tool ran without asking, a tool
  that changes something asked for Run, and its image was saved and described by the model.
- **Self audit** (docs/self-audit-2026-11-04.md). Six static reviews covered the 15 kernel files against their
  launches, the web UI against the server routes, and the audio and speech code. Each finding was checked before
  it was fixed.
  - The kernel findings were all latent. They are now errors or harmless, with logits bit-identical to the commit
    before (4B, 9B, 27B, E4B, 12B, 26B).
  - Speech:
    - a job that ends the process (a lost GPU) is no longer restarted at every start: after two attempts it is
      marked failed;
    - live audio keeps its timeline when a piece is lost;
    - utterances can no longer be queued after the recording's finish;
    - a deleted merge stops;
    - merged audio follows an MP4's start delay and gaps, as the pieces' transcripts did;
    - the engine is rebound to slot 0 after a failed part;
    - a lost wakeup after loading a speech model is fixed.
  - Web UI:
    - unloading the chat model no longer leaves the page thinking it is loaded, and the page keeps that model
      instead of the catalog's first (`/v1/models` lists the unloaded model first);
    - a U+0000 HTML injection is closed;
    - transcript preferences are no longer reset;
    - Summarize and Minutes cannot run twice;
    - an upload whose last answer was lost is no longer reported as failed;
    - Talk is refused during a recording;
    - plus smaller fixes.
- QUEUE 31.11 (1), measured and not taken: the driver's wait for the GPU does not spin a CPU core.
  - Default: Qwen3.8 27B, prompt 2048 + 32 tokens, used 16-25 s of user CPU time in ~70 s, most of it loading.
  - Polling with 200-µs sleeps: no faster (prompt 132-137 tok/s either way, generation step 204-206 against 208-212
    ms).

## 2026-11-04 (afternoon): unload models, a model that does not fit refused

- **A chat model that does not fit is refused instead of loaded**: `--ctx auto` raised "no context fits" to 4,096
  tokens and loaded anyway. After a restart Windows gave the GPU 16.55 GiB (20.42 before), Gemma 4 12B needed 8.1 GiB
  next to Whisper and Qwen3-ASR's 7.7, ran out of GPU memory at layer 40 and ended the server. Now the switch fails
  with what it needs and what is free, and the server stays up (`ctx-auto.txt` entries made again: m24).
- **Unload / load** (Settings, GPU memory; `GET /v1/server/memory`, `POST /v1/server/unload {"which": "model" |
  "whisper" | "asr"}`, `POST /v1/server/load`): the chat model, Whisper or Qwen3-ASR freed, the speech models loaded
  again as at start (Whisper 16 s, Qwen3-ASR 10 s), each only while nothing of it runs (no recording; a running
  transcription makes it say busy). The next chat model's `--ctx auto` counts the speech memory as it is then:
  Gemma 4 12B with both speech models unloaded loaded with 106,496 tokens of context.
- **Minutes in your own Word forms**: any `.docx` in `%USERPROFILE%\.arcinferzero\templates` (a company's minutes form
  with its letterhead and signature block) is offered next to "Minutes DOCX" in a transcript. The loaded chat model
  reads the transcript (a long one through notes of its parts) and answers the form's content as JSON held to the
  form's shape (`response_format` json_schema); the server fills the form's tables (tools/doc_template.h: the first
  table's "Label : value" rows by label, every other numbered table rebuilt row by row from its first data row; the
  signature block, header, footer and styles kept) and saves it as a file. Today's 2-hour meeting in the user's
  event form with Qwen3.6 35B: agenda, date and time, 11 participants, 6 agenda items, the Q&A with units, in ~2 min
  of generation; Word opens it. New endpoints: `POST /v1/tools/minutes_templates`, `POST /v1/tools/minutes_docx`.
- A data row that repeats a table's header ("No | Acara", "No. | Acara", "Nomor | Catatan") is left out by the
  server too: the 35B began each table with it and the minutes showed the header twice.
- A form is written whole: a numbered table the model left empty gets one row of "-", a field "-", so a form's own
  sample text (a template made from an earlier meeting's minutes) never stays in new minutes. Keep the templates
  clean anyway: fields empty, one empty row per numbered table, the signature block as it should appear.
- Summarize and the forms load the chat model first when none is loaded, so the transcript is cut to its context
  (it was cut to 8K characters: a 2-hour meeting in 12 parts on a 100K model).
- create_file: a model asked for a 2-hour meeting's minutes wrote six files (one per part, and versions). The tool's
  description now says one call per file, a long document whole in one call, never one file per part; its reply says
  the file is done instead of inviting "a new version".
- A `<br>` in an answer (models write it in table cells) is a line break, not the tag's text; any other HTML stays
  shown as text, and a `<br>` inside a code span as written.
- **Select** in the Transcripts list: a circle on each row; the picked ones are numbered in the order Merge joins
  them (the order they were made), Merge N then joins them, Cancel drops the picks (Ctrl+click and a long press start
  it too).
- **Transcripts and Meetings are one menu** (they were the same list and tools; a recording is not always a meeting):
  one list of files, recordings and meetings; **Record** takes the microphone, a tab's or the screen's sound, or both
  as two tracks (Me / Others, named "Meeting ..."); **upload files**; Summarize and the DOCX / PDF minutes for every
  finished transcript, not only meetings.
- After a bug hunt of the above: loading Whisper / Qwen3-ASR back checks the free GPU memory first (next to Gemma 4 12B
  it is refused: "unload the chat model first"; it would run out of memory); a split model (`--gpus`) keeps the old
  4,096 floor; the speech engines are read under their lock (a request during an unload could read a freed one) and a
  job whose engine was unloaded takes the other one; with both unloaded `/props` has no speech to text, queued
  transcripts wait instead of failing one by one, recordings and transcriptions are refused with the reason; the chat
  model cannot be unloaded where a request could not load it again (`--no-swap`); a model that does not fit at start
  leaves the server running without it (models switchable); the image variant is not the default when it fits no
  context. Web UI: the page is always asked again (no-cache), a stray 22-px SVG that let the page scroll and cut the
  header off is gone, and "GPU memory settings" scrolls the Settings list only.
- A failed answer in a chat is a box with a title and the server's message as text (it was the raw JSON); a model
  that does not fit offers "Unload speech to text" and "GPU memory settings".

## 2026-11-04: transcripts through E4B keep a timeline, warm-up after an update

- A transcript or meeting transcribed by Gemma 4 E4B was one segment per 2 minutes (the model gives no times), so a
  meeting whose live draft showed its utterances with times finished as one block. Its parts are now cut at the
  first pause after 10 s, at most 30 s (the quietest 100 ms near it in speech without pauses), a segment each: an
  89-s meeting 1 -> 5 segments, a 163-s voice message 2 -> 6. Words against Qwen3-ASR's text: 0.129 -> 0.133 on
  the voice message (one word of 241), 0.436 -> 0.487 on the meeting (four words; casual speech the reference also
  got wrong). `/v1/audio/transcriptions` keeps its 2-minute parts.
- **Merge**: recordings picked in the Transcripts or Meetings list (Ctrl+click or a long press, then Merge;
  `POST /v1/transcripts/merge {"ids": [...]}`) become one transcript: their audio one after another as one mono
  48-kHz M4A (a two-track meeting's Me / Others mixed; each segment keeps its speaker), their segments on that
  timeline, each piece's start a mark, `pieces` naming them with their clock times; the pieces stay. Today's meeting,
  four pieces after two server crashes (8 + 1 + 67 + 45 min), merged in 175 s into 2:01:03, 33 segments, 118 MB.
- Merge after a self audit: a merge the server stopped in (or a crash) is "failed" at the next start with its partial
  audio removed (it stayed "merging" for good); a failed merge removes its WAV (gigabytes, a full disk); the server
  waits for merges at its stop; pieces still being compressed are refused for a minute; the 4-GB WAV limit is checked
  as it writes; a merging transcript cannot be edited; a read error inside a piece's audio fails the merge instead of
  cutting the piece short (every later piece's times shifted). Checked: refused requests, a 2-piece merge, a server
  killed mid-merge.
- Web UI after the same audit: a failing part of a meeting summary stops the others and its error stays shown; picks
  for Merge stay in their tab; a voice message keeps the name of what wrote it at recording time; a long press on a
  row no longer selects text.
- The Transcripts tab records mono only (the Mono / Stereo choice is gone; the text was made from the mono mix
  anyway).
- Debug switches `ARC_ZE_SYNC_EACH`, `ARC_LAUNCH_STATS`, `ARC_ZE_KPROPS`: `=0` now means off (any value turned
  them on).
- `ARC_ZE_WATCH=ms` (diagnostics, after docs/device-lost-recording.md §5): every launch signals an event of its own
  and is waited for on it, timed; a launch over `ms` prints as "slow kernel" with its sizes and arguments (device
  addresses), one still running after 1.5 s is printed while it runs (Windows resets the GPU at 2 s), and an error
  prints the last 64 launches before the process ends. For finding the kernels behind the GPU timeouts (TDR,
  LiveKernelEvent 141) Windows logged dozens of times on 2026-11-01..04; slow (4B generation 1.2 tok/s with it on).
- A long meeting's summary asks for its parts' summaries (and each merging round's groups) at once, as many as the
  server has slots (`--slots`, or Slots in the web UI's Settings), so they run together instead of one after
  another; with one slot nothing changes. Not measured yet.
- The web UI's Speech setting picks what writes voice messages and Talk as text: Qwen3-ASR, Whisper, or Gemma 4 E4B
  (while the chat is with E4B). A model no longer hears the clip itself when the server has speech to text (Gemma 4
  12B's own hearing was less exact).
- The first start after an update warms Gemma 4's drafter (MTP) with the generation kernels, and the audio encoder
  with a second of silence: a first meeting after this morning's update took 116 s for 89 s of audio (E4B
  generating at 3.7 tok/s and its audio encoder 21.6 s while their kernels compiled), the next one 3.4 s.

## 2026-11-04: codebook GEMVs, DeltaNet with fewer launches

- One generated token of a DeltaNet layer launches two kernels fewer: `gdn_conv` stores the conv history itself (`conv_state_update` was a launch of its own) and `gdn_recur` turns the raw alpha / beta into decay and beta (`gdn_gates` was one). The same expressions, so the logits are bit-identical on every Qwen3.5 / 3.6 / 3.8 model, with MTP and on split models. Verification passes keep the separate kernels (their replay needs the gated values and the history before the pass). `ARC_GDN_FUSE=0` turns it off. On the laptop's iGPU the speed is unchanged (4B 48.8 -> 48.7 ms a token, 27B within noise): there the GPU's work, not the gaps between launches, sets the time; it is for GPUs where the host's launches show, as between a split model's sums.
- Generation through the codebook types (IQ1_M / IQ1_S / IQ2_* / IQ3_XXS) reads 4 rows per sub-group where it read 2 (`ARC_GV4=0` for 2): Qwen3.8 27B, whose FFN is mostly these types, 181.8 -> 179.2 ms a token (1.4 %), bit-identical; the other models have no such matrices.
- Measured, not changed (QUEUE 31.13): the GEMVs on their own (`bench/gemv_bench`, weights streaming) read 68-76 GB/s on the 4B's Q5_K, Q4_K and Q6_K matrices; the lower numbers of `--prof-gen` come from its events. Q6_K FFN down projections through `gemv` instead of `gemv_v` were 4 % faster alone but no faster in the models (4B 49.0 -> 48.8, 27B slower), so the choice stays.

## 2026-11-03: kernel audit, files split, lighter launches

- **Kernel self audit (every `.cl` file, read against the code that launches it; each finding checked):**
  - Qwen3 dense with 5 q heads per kv head (Qwen3-14B): the split decode attention (`attn_fa_g -DFA_SPLIT`) took 4 q
    heads per sub-group and read the first one's kv head for all four, so heads 5-7 of each group attended with the
    wrong K / V at every generated token. Now the largest divisor of the ratio (`Engine::gq_div`); Gemma 4 (2, 4, 8)
    and the Qwen3 models with 2 or 4 are unchanged.
  - MoE routing: a router row of NaN logits picked no expert, and the id NE then wrote past the counts and read past
    the experts (device lost). The id is clamped to the last expert (`moe_route`, `moe_route1`); finite logits are
    unchanged.
  - `ARC_H8=1` with `ARC_NPU`: the NPU shares copy a weight over the same buffers, so the in-place H8 conversion ran
    twice on shared rows and the NPU read H8 rows as W8. H8 is now off with the NPU, with a line saying so.
  - Guards for shapes no shipped model has (each one silently wrong before): Qwen3.5 attention with a q / kv ratio
    that leaves key tile words unloaded (5, 7, 10 ...) and a RoPE width other than 64; Flash Next with hc_n not
    dividing 8 or inject logits past the padded low rank; QSA's local block kept within 32 KB; Whisper text contexts
    over 448; a DFlash2 window below 1. Gemma 4 vision K / V buffers got the 64-byte slack the other encoders had
    (head 72 read 16 bytes past). Multi-GPU f16 sums keep 1/1024 below the f16 limit (parts that each rounded up could
    add to inf).
- **Files split** (code moved, not changed: logits bit-identical on every model):
  - `zero/kernels/ops.cl` (3,516 lines) into `ops.cl`, `ops_attn.cl`, `ops_attn_g.cl`, `ops_vit.cl`,
    `ops_flashnext.cl`, `ops_dflash.cl`; `decode.cl` (2,047) into `decode.cl` and `decode_moe.cl`. A kernel builds from
    its part and the ones before it (`KernelParts`, zero/clrt.h), the same text it had in the one file, so a norm's
    or a GEMV's build no longer compiles every attention, encoder, Flash Next and MoE kernel: shorter first-run
    kernel builds. The cache of compiled kernels is rebuilt once.
  - `zero/engine.h` (4,262 lines -> 1,378) into `engine_setup.h`, `engine_load.h`, `engine_load_gemma.h`,
    `engine_alloc.h`, `engine_stream.h`, `engine_slots.h`, `engine_sessions.h`; `tools/server.cpp` (3,308 -> 2,185)
    gives its file tools to `server_file_tools.h`. Each part is included where it was cut, so the compiled class is
    the same text.
- **Lighter kernel launches:** an argument given the same value as at the kernel's last launch (as passed to Level
  Zero: a buffer as its device pointer) is not set again; most arguments of a decode step repeat. `ARC_ARG_CACHE=0`
  sets every argument as before.
- **Web UI tool and system prompts tightened**: 3,013 -> 2,520 prompt tokens with every tool on (-16 %); on Gemma 4
  E4B the same tool chosen with the same arguments in 8 of 8 test requests (web, page view, memory, Word / Excel
  file, terminal, file list, no tool).

## 2026-11-03: launcher without serverz.exe

- **2D block loads for the XMX prompt GEMM (QUEUE 31.8; built, not yet run, off by default: `ARC_GEMM_2D=1`).** On
  Arc B-series cards a step's activations, 8 token rows x 64 bytes, come in one 2D block load
  (`cl_intel_subgroup_2d_block_io`) shaped as the DPAS A operand, instead of 16 one-row reads; `ARC_GEMM_2D_PF=1`
  prefetches the next 64 bytes. Checked against DP4a at start with the other XMX layouts; when it fails, XMX stays
  on without it. GPUs without the extension (the Core Ultra 125H's Xe-LPG) are unchanged: logits bit-identical.
  `bench/gemm_sweep.ps1` row X2D.
- **H8 prompt GEMMs (QUEUE 31.12; built, not yet run, off by default: `ARC_H8=1`).** Weights and activations rotated
  by a Walsh-Hadamard transform in blocks of up to 1024 values along K, both int8 with one scale per 256 values, so
  the GEMM is pure int8 dot products with one multiply-add per 256 values instead of a rescale every 16 or 32
  (zero/engine_h8.h, zero/kernels/h8.cl, gemm.cl QTYPE 110). One kernel for the iGPU (DP4a), Arc B-series (int8 DPAS,
  checked against DP4a at start like every XMX layout) and split models (each shard rotates within its own K).
  Accuracy simulated first on the CPU (`tools/ref_forward` `REF_HAD`): Qwen3.8 27B KL 0.00057 against f32 (plain
  per-row int8 0.038), Gemma 4 E4B 0.00198. Two-pass models (Gemma 4 26B / 12B) and expert GEMMs keep their path.
  `bench/gemm_sweep.ps1` has an H8 row for B-series owners.
- **Second self audit of the web tools and multi-GPU** (a static review, each finding checked in the code; built, not
  yet run: the GPU was busy):
  - web tools: audio from a URL decoded with a 30-minute cap (~115 MB of samples instead of up to 700 MB) and 60 MB
    at most; each kind's size decided from the headers before the body (a refused type or a too-large Content-Length
    reads nothing); a body cut at its size or the deadline says so; a read of 0 bytes ends the loop; any web port
    (browsers' restricted service ports refused) instead of 80 / 443 only, on every redirect too; one URL parser.
  - web_view: Edge must run inside its job (else not at all), a timed-out run's read is cancelled so a request
    cannot hang; the job holds at most 3 GB and 32 processes; the proxy serves only Edge's processes of a render
    running now (no other program can tunnel through it), counts connections atomically and stops a tunnel at
    256 MB; the URL is percent-encoded on Edge's command line; the DOM read from its first 4 MB; profile folders
    swept at every render; WIC's buffer lock checked.
  - multi-GPU: a pass with no barrier completing for `ARC_TP_TIMEOUT` + 30 s is given up on (a shard stuck inside
    a driver call waited forever), with a log line 30 s before; after a fault left a split model unfreed, no new split
    model in that process; a job is never published over one still running, the stuck list read under its lock;
    `ARC_TP_TIMEOUT` 5-3600 s; odd row lengths, a bad layer number and a vocabulary too small for the GPU count
    refused at load.
  - the re-audit of these fixes: the size check from Content-Length applies only to files (whole or nothing; a page,
    a search page or a GitHub API reply is read up to its limit as before, which the first fix had broken); a file too
    large is 413 and one cut by the deadline 504 (both had come back as 415); a page or text that is not cut no longer
    falls into the file path (a slip of the first fix, found in the re-check); internationalized hosts (münchen.de) are
    checked and opened as punycode; the proxy looks a connection up again before refusing it and logs refusals; a page
    stopped by Edge's 3 GB limit says so; a 304 is not reported as a redirect; `WSAStartup` once in `check_host` too.
    tests/test_web_tools.cpp checks the kept links, the web ports, the URL parser, IDN hosts and web_view's
    command-line URL.
  - the third round: a body that stalls or drops mid-way marks itself not whole, so a cut file is refused (504)
    instead of saved as whole; web_fetch connects to an internationalized host by the punycode name it checked; the
    proxy's refusal log never goes silent (every 100th after 20); Edge's memory note kept next to a timeout's.
- **web_view** (new tool, tools/web_view.h): a page as a browser shows it. Microsoft Edge, headless, takes a
  screenshot of one screen (shown to a model that sees images) and gives the page's text after its scripts ran, so
  pages built by scripts are read too (a GitHub issue page: its comments). Edge's requests go through a proxy in the
  server that reaches public addresses on ports 80 / 443 only; a new empty profile each run, removed after.
- **Audio from a URL**: web_fetch saves MP3, M4A, WAV, FLAC, Ogg Opus... (up to 60 MB, 30 minutes transcribed) and returns its speech as
  text from Qwen3-ASR or Whisper (whisper.cpp's jfk.wav: 5.7 s, word for word).
- Self audit of the web tools: Edge inherited every inheritable handle of the server (other clients' sockets): now
  only its output pipe; a page or text is read from its first 4 MB inside the request (100 MB was for any URL), and at
  most 4 fetches or views run at once (64 at once could hold 6 GB); the proxy's listener cannot spin a core on errors;
  COM set up and taken down in pairs; Edge's profile folders removed with retries and old ones swept; the Latin-1 to
  UTF-8 conversion only for text (it could change a PDF's bytes).
- Every exe is built with Control Flow Guard (`/guard:cf`; build.ps1), as the server now parses files from the
  internet. Logits bit-identical; Gemma 4 E4B generation 14.06 / 15.00 without against 13.46 / 13.56 / 15.28 tok/s
  with, run-to-run noise (docs/before-after.md).
- **web_fetch reads what a person reads** (tools/web_tools.h, tools/web_github.h; docs/server.md):
  - links to files and to other issues / pull requests keep their URLs (`[text](url)`), so the model can open them;
    the other links show their text, and `"links": true` gives every URL to explore a site (the URLs of every link made
    pages 30-75 % longer; this way 0-22 %);
  - a GitHub issue or pull request comes from GitHub's API with every comment, the files attached in the thread listed
    at the top (its page, built by scripts, often had no comments);
  - PDF, Word, Excel and PowerPoint files (up to 20 MB) are read and saved in the chat's files; images are saved and
    shown to a model that sees images; text attachments (`.txt`, `.log`) read as before; archives and programs refused.
  - Checked: Gemma 4 E4B asked about the event_probe log attached to issue #4 opened the issue, picked that attachment
    from the list and answered from it; it read a screenshot fetched from issue #5.
- `bin\membw_probe.exe` (new, in the package): memory read bandwidth of the CPU (1..all threads), the GPU, and both
  at once. On the 125H the iGPU reads 73-74.5 GB/s, the CPU at most 72.5 (all 18 threads), both together ~84: decoding on
  the CPU (xenolith's choice on Raptor Lake) would not be faster here (docs/speed-progress.md).
- Discrete GPUs: the 2-8 token GEMV of MTP / DFlash2 verification now uses 128 registers per thread by default (a
  Pro B60: 27B MTP 43.3 -> 47.1 tok/s). `ARC_GEMV_MT_GRF=256` goes back; `decode_try.ps1`'s row is now "mtp
  MT_GRF=256". iGPUs unchanged.
- When `bin\serverz.exe` is missing (a git clone has no exes), the launcher says so in the model panel: download the
  ready package from the releases page, or run `setup.cmd` to build it. It also offers to run `setup.cmd` once per
  window. Before, only the line under the empty list said it, which was easy to miss (issue #5).
- `ARC_GEMV_MT_GRF=128` tried on the 125H iGPU: 27B with MTP 11.78 -> 9.57 tok/s, so the iGPUs keep 256 registers
  (docs/xe2-tuning.md). Nothing changed in the engine.

## 2026-11-02 (night): Gemma 4 31B, memory and projects in the web UI, B60 switches

**Gemma 4 31B QAT** (`g31` in setup and the catalog; Google's Q4_0, Unsloth's drafter)
- KL against the f32 reference on a 536-token prompt: 0.00000 on the prompt path and the decode path (max 7e-5),
  the same top token at every position.
- A tight-memory mode for a dense model whose weights fill more than 85 % of an iGPU's share
  (`Engine::plan_tight`): part of the weights in host memory (bit-identical logits), prompt chunks of 1024,
  sliding-window rings with a margin of one chunk, `--ctx auto` from 1024 tokens. Only the 31B here; the other
  models plan as before. The host share is bounded by the RAM free at the start and is not cached; the server's load
  places the same amount its context probe did.
- On this laptop: generation 3.6 tok/s, 7.8 with the drafter; context 2048 (1024 with vision).

**Web UI memory and projects**
- `MEMORY.md` in the data folder: the model's `remember` / `forget` tools and Settings > Memory. A chat keeps the
  notes it started with; chats from before memory keep their prompt.
- Projects: a folder of yours, with its `AGENTS.md` (or `CLAUDE.md`) as instructions, a memory of its own, its
  files for the file tools (`project:<id>/<path>`) and the terminal opening there.
- Self audit: a chat reads only its own project's files; memory and projects only on this computer; `forget`
  removes at most 3 notes; the user folder, AppData and key folders cannot be projects; project files have a
  document cache of their own; a removed project's chats show under "No project"; a project file's digest notes
  get their own name.
- `AGENTS.md` for coding agents (Claude Code reads it through `CLAUDE.md`).

**Discrete GPUs and multi-GPU** (off by default; no change on the iGPU)
- `ARC_GEMV_PF=1`: the single-token GEMV issues all its rows' reads first (bit-identical; a row in
  `bench/decode_try.ps1`).
- `ARC_TP_ATT_QSPLIT=1`: the 35B's attention split by q-head over 4 GPUs (KL to one GPU 0.011 vs 0.008 whole).

## 2026-11-02 (late night): self audit of the day's B60 work

Three independent reviews (f16 XMX GEMM, multi-GPU changes, event probe and scripts). Each finding was checked in
the code before it was fixed.

**`event_probe`**
- Events were reused every 2 sums. A GPU that ran ahead could pass its wait on a peer's signal from 2 sums earlier
  and read stale data. The probe now uses a ring of 3 events and checks the data of 200 sums. QUEUE F31.3's design
  is corrected to match.
- A wait that never ends now releases every GPU and stops the probe instead of hanging it.
- Every call's result is checked.
- Output is unbuffered, so nothing is lost when it goes to a file.
- On real GPUs, one thread per GPU appends the work, as the engine does.

**f16 XMX GEMM**
- Weights are decoded times 16, not 256. With 256, a weight of 256 or more became inf and its column NaN.
- The f16 activations' buffer is reserved with the others, so the memory plan counts it (143 MB on the 27B).
- Chunks of 41-127 tokens whose size is not a multiple of 32 now use 16 or 8 tokens per sub-group instead of
  falling back to DP4a. A 48-token prompt: KL against DP4a 0.00019.
- The start-up check covers more cases and goes through the engine's own launch.
- The check's verdict is kept per device. A GPU wait that gives up during the check (this applies to
  `xmx_check` too) is not taken as a verdict.
- `act_half` is built without the GEMM.
- Checked: CG 1 / 2 / 4 bit-identical; 27B KL against DP4a 0.00069, unchanged by the new scale.

**Multi-GPU**
- A faulted split model now also sets every GPU's abort, so a given-up thread stops polling its wait.
- The wait-limit fields are atomic.
- The weight upload's staging buffer comes from the throwing allocation, so a shard short of host memory falls back
  to one GPU.
- Freeing a half-built shard keeps the first error.

**Also**
- A fused SwiGLU is never cut into split-K pieces (gemm.cl too; no model reached it).
- `gemm_sweep.ps1` / `decode_try.ps1`:
  - environment variables are removed even after Ctrl+C;
  - stderr no longer stops a sweep;
  - X16 rows are skipped where the f16 GEMM does not run;
  - X16 is never combined with DP4a only.

**Regression:** one GPU bit-identical; split 27B as before; split 35B as before with `ARC_TP_RG_GU=0`, and equal to
b3da0bf by default; f16 fallback equal to f32; the server's wait fault, OOM fallback and E4B tests pass.

## 2026-11-02 (night, 2): B60 queue F31.1 and F31.2

- **`ARC_X16_CG=2 / 4`**: the f16 XMX GEMM covers 2 or 4 column groups per sub-group, so each activation read feeds
  that many DPAS. With 1 group, every 16 columns re-read the activations from L2, which likely caps it far below
  XMX's rate. Logits are bit-identical for CG 1 / 2 / 4 (27B, DPAS emulated). `gemm_sweep.ps1` has an X16CG row.
- **`bin\event_probe.exe`** (in the package): can GPUs wait on each other's events through host memory, without
  peer access, and what does a sum cost that way against today's host wait and CPU barrier? It also checks the
  failsafe (the host releases a GPU waiting on a dead peer) and the driver's memory-wait extension. Laptop
  (`--sim 2`, one iGPU, a test of the tool): parts intact, 22.4 -> 16.7 us per sum, failsafe works.

## 2026-11-02 (late): B60 checklist items

- **Split MoE prompts: gate / up through moe_rg** as the down projection already was. Before, gate / up ran the
  per-tile kernel. Laptop (simulated 2-GPU split, 35B, 1K prompt): 498.7 -> 511.9 tok/s; KL to one GPU 0.0113 ->
  0.0066. `ARC_TP_RG_GU=0`: as before.
- `ARC_GEMV_MT_GRF=128` (the 2-8 token GEMV's register file, discrete GPUs) and `ARC_TP_RS_MIN=N` (the f16
  reduce-scatter's row threshold), for measuring on B60s.
- The XMX checks run once per process, not once per GPU of a split model.
- `tp_half` no longer writes a GPU's own rows (only range-checks them). Bit-identical.
- One GPU: logits bit-identical; split 27B and split 35B with gate / up per tile: bit-identical to before.

## 2026-11-02 (night): an f16 XMX prompt GEMM for discrete GPUs (`ARC_X16=1`)

- **`kernels/gemm_x16.cl`**: the prompt GEMM for W8 weights (the 27B's IQ3_S) in f16 on the XMX engines. Weights are
  decoded to f16 once per 16 columns with their scales applied, activations are made f16 once per GEMM, and one
  DPAS accumulator per 8 tokens runs over the whole row. The int8 XMX kernel rescaled every 32-value block on the
  vector unit, so it was only ~10 % faster than DP4a on a B60.
- Off until a B60 measures it: `ARC_X16=1` on a discrete GPU with XMX, checked against DP4a at start.
  `bench/gemm_sweep.ps1` has rows for it (X16, X16TM, X16GRF). Checked on the laptop with DPAS emulated
  (`ARC_X16_TRY=1`): KL against DP4a 0.00069 on the 27B. iGPUs: unchanged, logits bit-identical.

## 2026-11-02 (night): fixes from a second audit (DeepSeek, static)

- **A split shard out of memory ended the server** instead of loading the model on one GPU: its allocations now
  throw while it is made (`ARC_TP_OOM=r` tests it).
- **Whisper / Qwen3-ASR on GPU 0 shared the split model's wait limit**: a long wait of theirs could mark the split
  model faulty. The limit now applies only to the split model's own work.
- **A faulted split model left its wait limit on the GPUs** for the next model loaded; it is now taken back.
- `verify_batch` reads this engine's own logits rows (a split shard has a part of the vocabulary; not reachable yet,
  a split model has no MTP); the RAM tier's size is read under its lock. One GPU: logits bit-identical.

## 2026-11-02 (evening): multi-GPU self audit and failsafes

- **f16 sums out of range**: a prompt chunk's partial sum beyond f16 (65504 / GPUs) on any GPU makes every GPU sum
  that projection in f32 instead of turning it into inf / NaN (`ARC_TP_F16_LIMIT=0`, tests: bit-identical to f32).
- **A GPU wait that gave up was taken as "in step"** (from a review): with 2 GPUs a hung GPU 1 could leave the
  barrier count off by one and later sums silently wrong. Now any barrier or wait give-up breaks the split model
  for good (`/health` 503); a clean failure needs every GPU to have stopped alike, the same sums and no barrier in
  progress.
- **Unloading after a GPU fault** could end the process (a wait throwing in a destructor) or free memory a GPU or
  thread still used: the waits go back to no limit first, and after a fault nothing is freed until a restart.
- Shards with another weight layout / chunk / context than GPU 0 fail at load (runz too); `serverz --gpus` loads a
  model that cannot be split, or a split that fails, on one GPU; the three new sum kernels are released; a broken
  model's mirror no longer writes to the other shards first. One GPU: logits bit-identical.

## 2026-11-02: `serverz --gpus`, with a failsafe for a GPU that hangs

- **`serverz --gpus N`**: the server runs a split model (Qwen3.5 / 3.6 / 3.8 dense, the 35B-A3B) with `--slots` and
  batched steps. Shard 0 is the server's engine; its state-changing calls are mirrored on the other shards
  (`zero/mgpu/tp_mirror.h`) and their logits joined; a request's cancel is decided once for all. Text chat only for
  now (no MTP / DFlash2, images, prompt or session caches on disk). Over HTTP on the laptop (simulated split): the
  same greedy answers as one GPU, two chats at once the same as each alone, cancel and the next request fine.
- **Failsafe**: a shard waiting at a sum longer than `ARC_TP_TIMEOUT` s (default 120), or a split model's GPU wait that
  long, stops every shard with an error naming the GPUs that did not come; a thread stuck in the driver is given up
  on. The request fails, the server stays up, `/health` answers 503 until it is restarted. One GPU: no limit as
  before, logits bit-identical.

## 2026-11-02: batched decoding over a split model

- **Slots and batched decoding with `--gpus`**: a split model keeps its conversation slots (each GPU its part of
  every slot's state) and `TpEngine::forward_batch` runs several slots' generation steps in one pass, the sums
  carrying all their rows at once. `runz --gpus N --bench-batch M` checks the batched logits against each slot alone
  (bit-identical for the 27B over 2 and the 35B over 4, simulated on the laptop) and times 1..M slots. Not yet in
  `serverz`. One GPU: logits bit-identical.

## 2026-11-02: multi-GPU prompt sums in f16, reduce-scatter

- **A split model's prompt chunks sum through a reduce-scatter in f16** (`zero/mgpu`, `tp.cl`): each GPU sums one
  block of rows over all parts and shares that sum, instead of every GPU reading every part in f32. About 4.4x fewer
  bytes through host memory per sum on 4 GPUs, 4x on 2: on 4x B60 the split 35B's prompt spent 65-80 % of its time
  in the sums and ran slower than one GPU. Not yet measured there. Generation keeps the one-step f32 sum;
  `ARC_TP_SUM=f32` for the old sum everywhere. KL against one GPU (laptop, split simulated): 27B 0.0016-0.0018 mean
  (f32 sums 0.0009-0.0012), the same greedy text; one GPU unchanged (bit-identical).

## 2026-11-02: 128 registers for the prompt GEMM on discrete GPUs

- **Arc B-series (one card or several): the dense prompt GEMM now uses 128 registers per thread**, from the B60
  owner's sweep (issue #4: Qwen3.8 27B 450 -> 617 tok/s). `ARC_GEMM_GRF=256` for the old build; the MoE expert
  GEMMs keep 256 until measured (`ARC_MOE_GRF=128`).
- The `zero/xe2` switches act on discrete GPUs only: iGPUs with XMX (Lunar Lake, Arrow Lake) keep the iGPU kernels,
  where 128 registers made the 35B's prompt about 3x slower on the 125H. iGPUs: the same kernel builds as before,
  logits bit-identical.
- `bench\gemm_sweep.ps1`: GRF 128 is the default it sweeps from; new `MOEGRF` row for MoE models.

## 2026-11-02: launcher "Failed" pop-ups (issue #5)

- **Every start and stop showed "Failed:" over and over in Windows PowerShell 5.1**: there `Start-Process
  -PassThru` gives a `$null` ExitCode once the process ends, unless its handle was taken while it ran, so a start
  that went well counted as failed. The launcher now takes the handle (start / stop and recording), and clears
  the finished process before showing a message box: the timer goes on ticking while one is open, which showed
  the same box again every 0.5 s. The box now also gives the exit code. PowerShell 7 was not affected.

## 2026-11-01 (night): switches for Arc B-series tuning

- **Switches to measure on GPUs with XMX** (B580 / B60), in `zero/xe2/engine_xe2.h`: `ARC_GEMM_GRF=128` (the prompt
  GEMMs without 256 registers per thread), `ARC_GEMM_CG=2|4` (column groups per work-group), `ARC_GEMV_GRF=256`.
  Ignored on the iGPUs, where every kernel is built as before (checked: the same kernel builds, bit-identical logits,
  same speed); with them on, the XMX check passes and the logits are bit-identical.
- `serverz --act256` (as in `runz`): one activation scale per 256 values; less accurate (35B: KL 0.009-0.026 against
  the default), a switch to measure, not a default.
- `bench\gemm_sweep.ps1` sweeps `GRF`, `CG` and `ACT` too; new `bench\decode_try.ps1` (generation plain, MTP,
  DFlash2, GEMV switches); what to run on a B-series card: [docs/xe2-tuning.md](docs/xe2-tuning.md).

## 2026-11-01 (evening): XMX check at 32 tokens per sub-group

- **`ARC_GEMM_TM=32` kept DP4a on XMX GPUs**: the XMX check's 64 test tokens gave its split-K case an empty grid
  at 32 tokens per sub-group, so the check failed and `bench/gemm_sweep.ps1`'s TM 32 rows measured DP4a. It now
  tests 4 tiles of whatever TM is set (64 tokens at the default 16, unchanged). Found in a DeepSeek review.
- `ARC_GEMM_TM` takes only 16 or 32 (others warn and use 16; `gemm.cl` stops at build time on any other TM).

## 2026-11-01 (later): multi-GPU MoE, experts in RAM

- **Qwen3.6 35B-A3B over several GPUs** (`runz --gpus N`, and its fine-tunes such as Ornith 1.5 35B): the routed
  experts divided over the GPUs, each streaming and caching only its own; the attention layers stay whole on every
  GPU over 3-4 GPUs (the 35B has 2 kv-heads). Checked on one GPU with `ARC_TP_SIM` over 2, 3 and 4 shards: KL to
  one GPU no larger than one GPU's own between two chunk sizes. Details: [docs/multi-gpu.md](docs/multi-gpu.md).
- **Experts in RAM** on GPUs with memory of their own (B580 / B60): each expert read from the SSD stays in pinned
  RAM (up to the free RAM less 16 GiB, `serverz --expert-ram-gb N`, 0: off) and is copied from there by the GPU
  the next time. Never on an iGPU. Logits bit-identical.

## 2026-11-01: Ornith 1.5 35B-A3B checked

- **Ornith 1.5 35B-A3B** (a Qwen 3.5 MoE fine-tune for coding agents, MIT) runs as it is: the official Q4_K_M GGUF
  with its MTP block and `mmproj`, listed with MTP and images; generation 15.9 tok/s plain, ~23-24 with drafts (its
  MTP block 71 % accepted, Qwen3.6's DFlash2 83 %). Its Q4_K_M experts are 28 % larger than Qwen3.6's UD-IQ4_XS, so
  less of them fit the cache. Details: [docs/models.md](docs/models.md).

## 2026-10-31 (late): fixes from a static review ([BUGS.md](BUGS.md))

- **OCR save to a folder of your own** only from the server's own page or a client without an Origin: a page
  another local program serves could call the chat API and write a `.md` into any folder (checked: 403).
- An OCR save never replaces a file after 999 taken names (409); `chat.json.tmp` and `CHAT.JSON` are reserved names
  in a chat's folder; chat file references found in pretty-printed chat JSON too (they could be deleted as unused).
- Numbers from requests and prompts read without overflow (`read_file` offset / pages, OCR page numbers, transcript
  export options); `read_file` never starts inside a UTF-8 character; error replies say `Connection: close`;
  `ARC_QWEN_PS` checked; the web UI skips a non-JSON stream event; smaller hardening from `/analyze`.

## 2026-10-31 (last): multi-GPU measured on 4 Arc Pro B60

- **First numbers from real GPUs** (issue #4): Qwen3.8 27B generation 1.63x on 2 B60s, 1.97x on 3, 2.44x on 4
  (45.7 -> 18.7 ms a token, ~53 tok/s); pp512 1.28-1.38x. Table and what limits it: [docs/multi-gpu.md](docs/multi-gpu.md).
- **Fixed**: `serverz --npu on` found no NPU when started without Whisper / Qwen3-ASR (Gemma 4 12B: "no NPU device
  in Level Zero"): Level Zero was initialized for every driver type only in the speech service's start, after its
  early return, and `--ctx auto` started the GPU runtime (GPUs only) before the engine could. The server now does it
  first, right after reading its options.
- **Fixed**: 3 and 4 GPUs sometimes ended while loading in the driver's "Abort was called at 252 line in file:":
  the shards after the first were made at the same time on one Level Zero context; now one after another
  (`ARC_TP_PARALLEL_LOAD=1` for the old way).

## 2026-10-31 (late night): self audit fixes, the README as a landing page

- **Gemma ring snapshots, after an audit** ([docs/self-audit-2026-10-31.md](docs/self-audit-2026-10-31.md)):
  - after a restore no rewind reaches below the restored window (it could read an abandoned conversation's keys);
  - the rings' copy is taken only when the rings are about to overwrite it, not at every snapshot: a turn that never
    goes ~4K positions past its start costs nothing (a copy: E4B 30 MB, 12B 480 MB of RAM, 12B 233 ms);
  - `runz --ring-snap-test` sizes its run to the model's rings and also checks the rewind below; E4B and 12B pass.
- **OCR**: a page stopped while being read is read again by the next "lanjut" (its heading had counted it as read).
- **digest_file**: a combining step that does not shrink the notes keeps the ones before it.
- **Documentation**: the README is a landing page (what it is, why, how to try it, results, models, an index of the
  docs); its long sections moved word for word to [docs/install.md](docs/install.md),
  [docs/server.md](docs/server.md), [docs/models.md](docs/models.md) and [docs/architecture.md](docs/architecture.md);
  [docs/features.md](docs/features.md) tours the best features.

## 2026-10-31 (night): Gemma 4 conversations keep their cache after long answers and tool results

- **The same prompt every turn**: Gemma 4's template keeps an assistant tool call's reasoning while the turn's tool
  rounds go on and drops it once the user writes again, so the next turn's prompt differed right before the tool
  results. The server now renders a tool call without its reasoning every time (`gemma_history`). E4B in the web UI,
  "Ringkas README.md" (a 20K-token read) then "Model apa saja yang didukung?": the follow-up's prompt 22,092 tokens,
  **21,846 from the cache, 0.8 s** (it was 2,234 from the cache, 38.4 s).
- **Ring snapshots**: Gemma's sliding-window layers keep a ring of ~4K positions beyond their window, so the cache
  could go back only that far (`rewind`). The engine's snapshots (end of the prompt, start of the model's turn) now
  also keep a host copy of the rings' last window (13.7 ms to take, 6.4 ms to restore on E4B); when a request
  differs further back than the rings reach, the latest snapshot before it is restored instead of processing the
  whole conversation again. `runz --ring-snap-test` (a Gemma model, a prompt of 7,600+ tokens): restored after 6,000
  more tokens, the next 600 tokens' logits bit-identical to a fresh run.
- `digest_file`'s outline note says "do not answer from it": E4B, asked about all of a 159 KB file, had answered from
  the outline alone; now it called `digest_file` (4 parts, 306 s) and answered from the notes.
- `ARC_DUMP_PROMPT=file`: every rendered prompt appended to the file (where two turns' prompts differ).

## 2026-10-31 (late night): digest_file, tool-call fixes from an audit, a shorter prompt

- **`digest_file {file_id, focus}`** (Files on): a document longer than one read, for all of it (a summary, a review,
  its weak points). The server reads it part by part with the loaded model, each part a request of its own (no
  compaction, no tool rounds used), and returns notes of every part with their pages.
  - Parts end before a section's heading (a function's start in source files), else a page's heading, else a
    paragraph's end; a paragraph longer than a part is cut at its lines; a part that starts inside a section is
    told which one it goes on with.
  - The notes are saved as `<name> (notes).md` in the chat's files: a later call reads them at once (whatever its
    focus), an unfinished one goes on from its last part (Stop keeps the notes so far).
  - Notes longer than one read are combined again until they fit, and never come back longer than one read.
  - Checked with Qwen3.5 4B at 16K context: README.md (57 KB) 3 parts in 177 s; `tools/server_ocr.h` (30 KB) 3 parts in
    85-92 s; the same call again 0 s.
- **Fixed** (an audit of the tool-call path):
  - Qwen: a `<tool_call>` written in the reasoning was taken as an open call, so after `</think>` every token failed
    the constraint (a scan of the whole vocabulary each) and the real call went unchecked.
  - Qwen: string parameters with an `enum` (list_files' `scope` / `kind`, create_file's `page`, terminal_panel's
    `action`) took any text; now one of their values.
  - `tool_choice: "none"` keeps the tools in the prompt and only stops the call's open marker, so the web UI's last
    answer after 8 tool rounds no longer processes the whole chat again (the prompt is the same 2,680 tokens; three
    tries answered in words).
  - A chat answer that no longer fits in the middle of its tool rounds shortens its own earlier rounds' results
    before it gives up (a compaction keeps them).
  - `list_files` 20 files by default and `search_chats` without a query: at most one read of the context.
  - `read_file`: the outline for any text one read cannot hold (it was only over 24,000 characters), and its parts
    point to `digest_file`.
- **Shorter tool texts:** what a system text and a tool's description both said is said once (the sheet and slide
  layout, the terminal's session rules, digest_file's use), and two texts that disagreed on who allows a command now
  agree. The prompt with every tool on: 2,870 -> 2,680 tokens (Files alone 2,132 -> 1,995).

## 2026-10-31 (night): reading an OCR .md with a chat model

- **`read_file` reads a document whole** when the model's context takes it in one read (`files::read_budget`) and
  no `max_chars` is asked: without arguments, and from an `offset` too (the rest at once); a longer one still gets
  its outline, and parts. Qwen3.5 4B (262K context) read the 56-page OCR `.md` (98 KB, 25K tokens) a page-section at
  a time, then with `offset: 0` in 12,000-character parts, and ran out of its 8 tool rounds at 80K of 98K. Now, in
  the web UI, "Baca seluruh file ... lalu berikan kekurangannya": one call, the whole file (23K-token prompt,
  670 t/s), the answer from all of it.
- **`pages: "A-B"` in a Markdown file with `## Page N / M` headings** (an OCR model's `.md`): those pages' sections
  in one call ("26-30" said "no text" before). The web UI's `read_file` description says so.
- **Qwen tool calls with a number argument** (`section: 12`): every token after its `</parameter` was refused by the
  call's constraint, each one a scan of the whole vocabulary, so tool calls ran at ~13.5 tok/s instead of ~18.
  Fixed in `tools/tool_constraint.h` (a closed value is checked as one that has ended); details in
  [docs/before-after.md](docs/before-after.md). `ARC_DEBUG_TOOLFIX=1` logs each refused first choice.

## 2026-10-31 (late evening): saved OCR text stays in the file

- **Saved to a .md**, the OCR answer shows only the bar of the pages done ("reading page N / M…": the server now
  sends each page's heading before it reads the page), not the text.
- The chat keeps a note instead of the pages: the last page's heading (the next "lanjut" goes on after it) and
  "the text of pages A-B is saved in the file NAME in this chat's files (read_file it)". A chat model switched to in
  the same chat reads the file from there.
- A folder of the user's gets a copy in the chat's files under the same name (a chat model's `read_file` reads no
  folder of this computer). Checked in the web UI on a 2-page PDF: the bar only; the external `.md` and its copy the
  same 1,881 bytes; `list_files` / `read_file doc_sample.md` of that chat find it.
- Not saved: every page shows, as before.

## 2026-10-31 (evening): launcher errors, PDFs of a new chat, NPU messages

- **Launcher** (issue #5): the server monitor and the model details popped up "Index (zero based) must be greater
  than or equal to zero..." every 2 s. In `$L.Add("... {0} {1}" -f a, b)` the commas split the method's arguments,
  so `-f` got one value; each such line now has its own parentheses (`scripts/arc-launcher.ps1`).
- **PDFs for OCR models**: the first PDF of a new chat could fail with "An item cannot be found at the specified
  path". The chat's folder is renamed when the chat is first saved, which came while the PDF was being opened. The
  PDF is now read into memory once and rendered from there (`tools/pdf_render.h`).
- **Self audit** of the OCR mode ([docs/self-audit-2026-10-31.md](docs/self-audit-2026-10-31.md), evening):
  - a page the model could not read stops the pages (`[OCR: stopped at page P: why]`), so the web UI no longer
    sends "lanjut" into the same error to the PDF's end;
  - PDF answers carry the pages' `timings` (the statistics line said "stopped");
  - an `.md` that cannot be written is refused before any page; `CON` / `NUL` / `COM1` as its name become
    `ocr CON.md`.
- **NPU**: a driver's newer graph extension (1.9 and up) is taken too, and "no NPU" now says what is missing: no NPU
  device in Level Zero (the Intel NPU driver), or a driver too old for graph extension 1.8.

## 2026-10-31 (late night): long PDFs for OCR models

- **Every page, by itself**: a PDF is read 50 pages a turn. An answer with pages left ends with
  `[OCR: N more page(s); "lanjut" reads pages A-B]`, and the web UI sends "lanjut" itself until the last page
  (Stop ends it). The words may also name pages (`halaman 51-100`, `page 7`). A "lanjut" goes on with the same
  PDF and task after the highest `## Page N` of the answers so far.
- **Saved to a file**: when a PDF is sent to an OCR model, the web UI offers to save its text to
  `<the PDF's name>.md` in the chat's files (`ocr_save`, `chat_id`).
  - The question comes in a modal of the page, asked again with every PDF sent, not the browser's confirm box:
    - the file's name (`<the PDF's name>.md`) and folder (empty: the chat's files; or a folder of this computer,
      taken from requests of this computer only);
    - Save to .md / Just show it (Esc: just show it).
  - An existing file is never written over: `name (2).md` then. The "lanjut" turns go on with the file the server
    named at the end of its answer.
  - Each page is appended once read, and the file shows in the Files panel.
  - Saved or not, the answer shows every page under a bar of the pages done, and the chat keeps them all. A chat
    with an OCR model is not compacted (the server reads only its last turn), so no page is lost to a summary.
  - Checked on a 450-page service manual: pages 1-2, then 449-450 appended to the same file; in the web UI the bar
    counted on and "lanjut" went by itself.

## 2026-10-31 (night): faster BF16 models (the OCR model)

- **Float matrices stacked as quantized ones are** (`zero/engine.h`):
  - DeltaNet `qkv` + `z` + alpha / beta: one GEMV, since a BF16 GGUF's float matrices become the same Q8_0 rows
    first; alpha / beta join the stack when their rows come in 16s, not only 64s;
  - FFN `gate` + `up`;
  - gated attention `q` + `k` + `v`.

  Xiaomi-OCR-0 (Qwen3.5 0.8B) runs 9 GEMV launches fewer per layer:
  - generation 69.3 -> 73.2 tok/s, prompts +2 %;
  - logits bit-identical;
  - quantized models take none of these paths.

  The output head as Q4_0 was 86 tok/s, but KL 0.048 against f32 (the limit is 0.015), so it stays Q8_0. The image
  encoder (~1.44 s for a 2,000-token page) is already near the iGPU's f32 rate; its attention in f16 is left as the
  next step. Switches for A/B tests: `ARC_NO_QZ_FLT`, `ARC_NO_GU_FLT`, `ARC_NO_QKV_FLT`, `ARC_NO_QZAB16`.

## 2026-10-31 (evening): OCR mode

- **OCR models get their own task prompts** (`tools/server_ocr.h`). Xiaomi-OCR-0 (Qwen3.5 0.8B with an image
  encoder, converted to GGUF with llama.cpp's `convert_hf_to_gguf.py`, model + mmproj) knows a few English task
  prompts only:
  - from the web UI (a long system prompt, tools, thinking, sampling) it answered in Chinese;
  - asked in Indonesian it wrote a letter's title only.

  A model whose name says "OCR" is now asked the way it was trained:
  - its last turn with the image and the task its words name (text by default; `dokumen` / `document`, `tabel` /
    `table`, `rumus` / `formula`, `json` + fields);
  - no system prompt, tools or thinking, greedy;
  - fields named in Indonesian become the model's English keys (`tanggal` -> date, `nama toko` -> seller_name).

  Checked with a chat client's long system prompt, a tool, thinking and temperature 0.8:
  - an Indonesian letter: word for word (similarity 1.000), also with no text at all; as a document 0.995;
  - the README's receipt: `json: tanggal, total, nama toko, alamat toko, pajak` all right;
  - the repo's examples: 0.975 / 0.988 to its outputs.

  The web UI's message box shows the tasks.
- **OCR mode from the web UI** (the same evening): the web UI sends its images as saved files (`"type": "file"`),
  which became image parts only after the OCR rewrite looked for one, so its requests went through unchanged (and
  still came back in Chinese). The rewrite now runs after the files are expanded and skips their `[file: ...]`
  lines. A follow-up without an image ("now as a table") takes the conversation's latest image. Checked with
  files from a Downloads folder through the web UI's way (file part, system prompt, thinking, temperature 0.8), all
  in Indonesian words: no Chinese in any answer.
- **PDFs for OCR models**: a PDF in the last turn is rendered page by page by Windows (`tools/pdf_render.h`,
  Windows.Data.Pdf: every kind of page, as a viewer draws it, ~0.2 s a page at 150 dpi). Each page is read as an image
  and streamed under `## Page N / M` (at most 50 pages a request). Before, a PDF went in as its text layer, and
  the follow-up rule took the conversation's last image instead (the user's report came back as an earlier picture);
  a turn that brings a file now never looks back for an image. A 5-page report: first page after 12 s, all in 50 s.
- **OCR tables as Markdown**: the model writes tables in OTSL (`<fcel>cell<fcel>cell<nl>...`). The server turns
  them into Markdown tables in non-streamed answers, and the web UI does so while an answer streams. A report page's
  similarity to its PDF text layer went from 0.61 to 0.87.

## 2026-10-31 (afternoon): multi-GPU audit, uneven splits

- **Self audit** (docs/self-audit-2026-10-31.md):
  - a constructor error ended `runz --gpus` without a message (worker threads still running);
  - the cut tensors' host copies held the model a second time in RAM;
  - a failed shard left the others out of step;
  - the 27B over 4 GPUs could not load prompts (24 alpha / beta rows; padded to 16 now).
- **Any number of GPUs** the model's heads allow: counts that do not divide are split as evenly as their units allow
  (27B over 3 GPUs: 6 / 6 / 4 k-heads, 2 / 1 / 1 kv-heads). 27B over 2 / 3 / 4 shards: greedy text identical to one
  GPU.
- **The multi-GPU code moved apart:** `zero/mgpu/` and `zero/kernels/tp.cl`; the one-GPU code only calls hooks,
  and its logits stay bit-identical.

## 2026-10-31: one model over several GPUs (runz --gpus N)

- **Tensor parallel for dense Qwen3.5 / 3.6 / 3.8** (docs/multi-gpu.md):
  - `runz --gpus N` splits the DeltaNet and attention heads, the FFN columns and the output head's vocabulary over N
    GPUs, so each GPU reads 1/N of the weights per token.
  - The output projections' partial sums meet in pinned host memory (B60s on Windows have no peer access).
  - Checked on one GPU with `ARC_TP_SIM=1`: greedy text identical, KL against an f32 reference the same as one GPU
    (4B: 0.00113 / 0.00096 / 0.00109 for 1 / 2 / 4 shards).
  - Speed needs a machine with several GPUs; the server does not use it yet.
- **The runtime keeps its state per GPU** (`zero/clrt.h` `zrt::Dev`, `zrt::D()`):
  - one Level Zero context holds every GPU of the driver;
  - a thread works on the GPU it chose, GPU 0 by default;
  - one GPU is unchanged, with logits of the 4B and 27B bit-identical to before.

## 2026-10-31: speech to text without a chat model

- **`serverz -m none`** (`arc-server start -Model none`, the launcher's **Speech only** button): only the speech
  service loads (Qwen3-ASR alone, ~3.2 GB of GPU memory; `--whisper auto` adds Whisper), for transcripts through
  the web UI's Transcripts tab or `POST /v1/audio/transcriptions`. `/health`, `/props` and `/v1/models` answer with no
  model loaded (the model started last listed first, none `loaded`), and the first chat request naming a model loads
  it, context auto next to the speech models. A 2:43 voice message: 25 s.
- **The first transcript after a start no longer waits for the encoder**: Qwen3-ASR's audio encoder loaded its
  weights and kernels on its first clip (60 s of audio: encoder 4.1 s instead of 0.5); now when the server starts.
- **Web UI recordings: the MP3 bitrate is a choice** (Transcripts tab, "MP3 96 / 128 / 160 / 192 / 256 /
  320 kbps", or WAV) instead of a fixed 192. Windows' MP3 encoder takes at most 128 kbps for one track (a
  microphone): higher choices give 128 there; a meeting's two tracks get the bitrate chosen (checked: 96, 192, 320).
- **Recordings kept as M4A (AAC)**: "M4A 96 / 128 / 160 / 192 kbps" next to WAV and MP3 (`POST /v1/live`
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
