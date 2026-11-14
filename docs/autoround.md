# Flash Next with Intel's AutoRound INT4 experts (AR4_128)

Intel's [Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) holds
Flash Next's routed experts in INT4 calibrated by AutoRound: symmetric, one F16 scale per group of 128 weights.
inferzero takes those experts as they are and everything else from Unsloth's UD-Q4_K_XL Flash Next: the non-expert
tensors, the tokenizer, the chat metadata and the separate Q4_0 MTP head. So it is a hybrid, not Intel's whole
checkpoint, and not the UD-Q4_K_XL made faster by the engine alone:

- the UD-Q4_K_XL's down experts are Q5_1, the hybrid's calibrated INT4;
- the hybrid's expert records take 63.75 GiB against the UD-Q4_K_XL's 77.60: they fit a 64 GiB RAM tier, as on the
  author's B70 below.

The importer, the format and its decoders are chumbleyjl's (pull request #22, submitted 2026-11-13, merged
2026-11-14). His measurements on an Arc Pro B70 are in a section of their own below. The one-step download, precise
mode by itself, splits, iGPUs and the checks since are the 2026-11-14 entries of [CHANGELOG.md](../CHANGELOG.md).

## Getting it

```powershell
.\setup.cmd -Models flash-ar4
```

or, on its own, `powershell -File scripts\get-autoround.ps1`. It downloads the checkpoint's expert shards at the
importer's pinned revision, imports them, downloads the UD-Q4_K_XL and writes the package;
[install.md](install.md#flash-next-with-intels-autoround-int4-experts-flash-ar4) lists the steps, the downloads and
the disk they need. The package goes to `%USERPROFILE%\.inferzero\zero\models\Qwen3.8-Flash-Next-AutoRound-INT4\`,
where the server lists it as `qwen3.8-flash-next-autoround` (shortcut `flash-ar4`, in zero-pi too). The UD-Q4_K_XL
it was made from stays listed as a model of its own. The project ships no weights or expert files for it: each
machine imports its own.

By hand, the same steps (Linux names; on Windows the packer is `bin\import_autoround_pack.exe`, in the release
package):

```sh
python tools/import_autoround_experts.py SOURCE_DIRECTORY AutoRound.experts --experimental-format
g++ -std=c++20 -O2 -mavx2 -mfma -mf16c -Ithird_party tools/import_autoround_pack.cpp -o /tmp/import_autoround_pack
/tmp/import_autoround_pack SOURCE_Q4XL.gguf AutoRound.gguf AutoRound.experts 4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
```

- **`SOURCE_DIRECTORY`** holds, at the pinned revision: `config.json`, `model.safetensors.index.json`,
  `model-info.json` from Hugging Face's model API (with each file's SHA-256) and the expert shards the index names.
  The importer needs NumPy (Python 3.11 or newer).
- **`--experimental-format`** is required: without it the importer refuses. AR4_128 is inferzero's own type (see
  "The format"), so no build before pull request #22 and no other GGUF reader loads the package.
- **The last argument** is the checkpoint's revision, which the package keeps (see "The checkpoint check" below).

## How it runs

- **Precise mode by itself.** A model with AR4_128 experts turns Flash Next's precise mode on for its own engine
  while it is loaded, without `--precise` or `ARC_FN_PRECISE=1` (zero/engine.h `FnPreciseHold`). The log says
  `AR4_128 (AutoRound INT4) experts: Flash Next's precise mode on for this model`. Other models in the same server
  keep their own mode (per engine since the pair audit of 2026-11-14). AR4_128 experts anywhere else are refused at
  load: `AR4_128 requires precise Flash Next (--precise)` (zero/engine_stream.h).
- **In the server:** precise mode turns int8 attention off, so the system-prompt cache and saved conversations are off
  for this model (they follow the engine's mode); the `--ctx auto` key records the mode.
- **One discrete GPU:** the experts stream from the expert file through the GPU's expert cache and the RAM tier
  ([multi-gpu.md](multi-gpu.md), "Experts in RAM"), as other Flash Next files' do.
- **Split (`--gpus N`):** each GPU streams its own experts, as on one GPU. The AR4 path's KL against f32 on this
  build, on one GPU and split, is asked of the author (B70).
- **iGPUs** (Xe-LPG, Xe-LPG+, Xe3): allowed since 2026-11-14 (they have strong int4 DP4a paths). On the laptop's
  Xe-LPG, `tests/test_autoround_decode` on synthetic AR4 data (4.9 M weights): the GPU decode bit-identical to the CPU
  decoder; the expert kernels (`moe_gateup_q`, `moe_down_q`, `moe_rg`, `decode_w8_sb` / `_sl`, `gemv_v`) build for
  AR4. A whole AR4 model has not run on an iGPU yet: the import does not fit the laptop's SSD.
- **Linux:** the expert file's layer table is compared field by field (pull request #24). With `memcmp` its padding
  bytes, not set on GCC, refused the package on the author's B70.
- **The checkpoint check** (pull request #25): the package keeps the checkpoint's 40-digit SHA
  (`inferzero.autoround.revision`), the expert file its first 16 hex digits as a fingerprint. The engine
  (zero/expert_store.h) and tools/ref_forward refuse a pair that does not match, and a malformed or missing revision.
  Before it, an expert file of the same shapes from another import loaded without a word and decoded other weights.
- **The f32 reference** (tools/ref_forward) decodes AR4_128 too: a KL check compares with the package's own INT4
  weights, not with Intel's BF16 originals.

## The format

**What is imported:** the checkpoint at revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, its routed experts
only. The importer reads the 14 shards that hold them (72,296,476,000 bytes). The checkpoint's BF16 PLE table and its
MTP tensors are not imported: the UD-Q4_K_XL's are used.

**What the importer checks** (tools/import_autoround_experts.py):

- the checkpoint's identity, and every source file's SHA-256 against `model-info.json`;
- shapes and dtypes;
- zero points constant and symmetric (GPTQ stores the zero point minus one: a stored 7 is a zero point of 8);
- F16 scales finite;
- after repacking, every source code and every scale bit.

It refuses activation-order indices and an output file that exists. It streams through working arrays of bounded
size instead of loading the whole checkpoint.

**AR4_128** (`0x41523447`) is a private inferzero type, not a GGML one:

| Form | Where | Layout |
|---|---|---|
| CPU block, 128 values | `dequant_row` (src/quants.h), tools/ref_forward | one source F16 scale, then the 128 original 4-bit codes in Q4_0 nibble order: 66 bytes |
| GPU superblock, 256 values | the expert file, zero/kernels/decode.cl | the two 128-groups' F16 scales, the 256 codes (8 x 16 nibble bytes in Q4_0 order), 4 alignment bytes: 136 bytes |

- **Q4_0 nibble order:** each 32 values are 16 bytes, the low nibbles values 0-15, the high nibbles 16-31. A value
  is its group's scale x (code - 8).
- **Rows of K = 640** (the down experts) are padded to 768. Nothing else changes: no requantization, no expert
  pruning; the codes and scales are Intel's.
- **On the GPU** a 32-block's 16 bytes unpack into int8 values (code - 8), with its 128-group's scale, for the kernels
  that read the per-32 int8 form (`ar4_unpack`, zero/kernels/decode.cl).

**The files:**

- **The expert file:** 68,451,045,376 bytes (63.75 GiB plus a 4 KiB header), records of 2,785,280 bytes (one
  expert's gate, up and down in one layer: 48 layers x 512 experts). Its header holds the revision fingerprint.
- **The package** (tools/import_autoround_pack.cpp): a GGUF of 31.960 GiB with all 1,080 non-expert tensors of the
  UD-Q4_K_XL byte for byte, its metadata and tokenizer, and the 144 expert tensors as virtual AR4_128 ones, whose
  data is in the expert file.

## Measured by chumbleyjl on an Arc Pro B70 (pull request #22)

Measured before the merge, on his own build of the engine ("downstream"), not on this one. That build also has mixer
storage that keeps the source values, a two-pass expert XMX path, a growing KV cache, adaptive tiles and MTP, and
expert residency, not all of which came with the pull request: the speeds below are neither this build's nor main's.
(The precise-mode part of his build went in as pull request #21, the Q5_1 CPU reference as #20.) The comparison
includes both changes of the hybrid: other expert weights, and what stays in RAM.

**Conditions:**

- one Arc Pro B70 32 GB, i7-14700K, 96 GB of RAM, native Linux, the Level Zero / OpenCL engine;
- both models in precise mode, medium reasoning, a 262,144-token context, a 64 GiB RAM tier for the experts, adaptive
  MTP (capacity four);
- one server run per model, after the downloads, the import and the CPU reference jobs had finished: single
  measurements, no confidence intervals;
- archive requests at temperature 0; tool requests at temperature 1, top-p 0.95, top-k 20, seed 42; the generated
  histories can differ between the two models;
- prompt speed = new prompt tokens / forward time (cached tokens not counted); generation speed without the first
  token. The archive is 15,599 tokens.

| Workload | UD-Q4_K_XL: prompt / generation tok/s | AutoRound hybrid: prompt / generation tok/s |
|---|---:|---:|
| First-use archive | 484.48 / 15.43 | 300.79 / 36.02 |
| Warm archive replay | 695.63 / 25.78 | 783.59 / 42.36 |
| Weighted incremental turns | 102.41 / 24.53 | 194.15 / 38.53 |
| Plain archive decode | - / 23.31 | - / 30.79 |
| Tool call | 89.92 / 22.58 | 174.68 / 35.69 |
| Tool result | 110.71 / 24.44 | 240.65 / 37.80 |

- **Wall time** of an incremental turn, mean: 3.461 -> 2.137 s (38 % lower). Expert records 77.60 -> 63.75 GiB.
- **The first-use prompt is slower:** cold data and kernel compiles (JIT) were not measured apart. The warm gains are
  not startup or time-to-first-token gains in general.
- **Both models pass** all eight archive / replay / tool cases and the plain / MTP checks within each model.

**Accuracy:** KL against each model's own f32 reference, which decodes that model's quantized weights, not the BF16
ones. Means below what the CLI prints are written <0.000005.

| Prompt | UD-Q4_K_XL mean KL | AutoRound hybrid mean KL | AutoRound hybrid max |
|---|---:|---:|---:|
| Coding, 31 positions | <0.000005 | <0.000005 | <0.000005 |
| Workflow, 209 positions | 0.00700 | 0.01074 | 0.83681 |
| Tools, 48 positions | 0.00001 | <0.000005 | <0.000005 |
| Forced top-8 sparse attention, 48 positions | 0.00021 | 0.00018 | 0.00419 |

- All four of the hybrid's means are within 0.015; single positions are not (the workflow's maximum 0.83681; the
  UD-Q4_K_XL's 0.58798).
- Between the two models' f32 references, on the workflow: KL 0.149171, the same top token at 206 of 209 positions.
  Such a difference does not say which model is closer to the BF16 one, nor that the hybrid loses nothing against
  the UD-Q4_K_XL.

**Import checks** (his build): all 120,795,955,200 expert codes and their scale bits; a fixture of 4,915,200 real
weights (one expert's gate, up and down, as `tests/test_autoround_decode.cpp` checks) the same from the source, the
CPU decoder and the GPU, bit for bit. A small coding / tool suite: 17 of 17 functional checks (a sanity check, not a
coding benchmark).

**A long context** (the hybrid only: no paired UD-Q4_K_XL timing, no KL over the whole window):

- one request of 260,988 tokens found the keys placed early and at the midpoint: 532.52 tok/s prompt, 25.70
  generation, 497.49 s in all;
- its cached follow-up reused the 260,988 tokens, processed 214 new ones and answered correctly in 3.885 s (154.39 /
  28.81 tok/s);
- the KV cache was backed in memory for all 262,144 tokens, with at least 25.77 GiB of system RAM available.

**Pi** with the hybrid (medium reasoning, temperature 1), a synthetic read-only audit:

- 72 tool calls, 11 automatic compactions and 24 of 24 final audit results exact, in 17.3 minutes;
- no tool errors, extra inspections or guard stops; all the original instructions still in the 31 main provider
  requests;
- weighted main rates 737.51 tok/s prompt, 44.36 generation;
- this diagnostic compacts around 24K tokens (largest prompt 23,218), with the full advertised context kept. It is not
  11 compactions at 262K depth, nor proof that long coding tasks match.

## The pull request's own checks (at submission)

Against main at `c655269`, which had no private type, importer or AR4 decode yet; the pull request's description
lists the builds and the CPU and package checks.

- **Builds** on native Linux: both engine programs, the reference, the package writer and the separate tests.
- **The CPU layout check** (`tests/test_autoround_blocks.cpp`, in CI): 9,216 values over five widths, K = 640 among
  them, against code and scale expectations of its own, with the padding and an invalid width checked.
- **The package check** (`tests/test_autoround_package.cpp`) on the real package: all 1,080 non-expert tensors
  (34,305,722,880 bytes), the 144 virtual expert tensors, the metadata and the tokenizer match the UD-Q4_K_XL. No
  model file is written or overwritten for it.
- **The importer** packs as the one his build validated, with the `--experimental-format` guard added.
- **Left for before the merge:** the GPU numbers above are his build's, not runs of the pull request's code; model KL
  on the rebased code, native kernel runs, Windows / iGPU / split checks, regression dumps with it off and full-context
  integration were still to do. His running Pi service was not affected.
- **Asked in it:** how to allocate a private type id, how to version the package and keep it compatible, whether an
  import that keeps the calibration belongs in the project, and in which order the rest of his build's paths could be
  added and checked.

What was checked here at and after the merge (the other models' logits bit-identical to release 2026.11.14.10, the
scripts, the iGPU decode test) is in the 2026-11-14 entries of [CHANGELOG.md](../CHANGELOG.md). Not run here: the
full download and import (~290 GB of disk), which is for a tester.
