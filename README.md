# inferzero

*Formerly arcinferzero (renamed 2026-11-10): links to the old GitHub address still lead here, and an existing data
folder and Pi setup carry over by themselves.*

**Fast local LLMs on an Intel laptop iGPU.** A from-scratch inference engine for **Intel integrated GPUs
(Xe-LPG / Meteor Lake "Arc Graphics"; Xe2 / Xe3 such as Lunar Lake and Panther Lake's Arc B390 too)**, written in
C++ on **Level Zero**. It loads GGUF files directly, serves an **OpenAI-, Anthropic- and Ollama-compatible API with
a web UI** (web search and voice conversation included), and plugs into the [Pi](https://pi.dev) coding agent with
one command. Built and tuned on a Core Ultra 5 125H with 32 GB of RAM: no discrete GPU, no XMX matrix engines.

**[Download](https://github.com/rizagp/inferzero/releases/latest)** · [Features](docs/features.md) · [Usage guide](USAGE.md) · [Install and build](docs/install.md) · [Measured results](docs/before-after.md) · [Changelog](CHANGELOG.md)

## Why it is worth a look

- **Prompts 2.7-7.4x faster than llama.cpp** (Vulkan / SYCL) on the same GGUF files and laptop, by building every
  matrix product on the iGPU's int8 DP4a units: a 34K-token code prompt on Qwen3.8 27B takes 5.5 minutes instead
  of 40 (Vulkan).
- **Mixture-of-experts models larger than the GPU memory**, experts streamed from the SSD into a GPU cache while
  the GPU computes: Qwen3.6 35B-A3B with a 131K-262K context, Qwen3.8 Flash Next (52 GB of experts) on a 32 GB laptop.
- **Speculative decoding that never changes the answer**: the model's MTP block and a DFlash2 block drafter guess
  up to 7 tokens per step; Python code on the 35B runs at ~50 tok/s, the text identical to plain decoding.
- **Accuracy measured, not assumed**: every model is checked against an independent f32 CPU implementation, and
  perplexity is at or below llama.cpp's on the same files.
- **Deterministic**: the same prompt gives the same logits bit for bit, run after run, whatever the expert cache
  size or engine, and the drafters give exactly the text of plain decoding. Every speed change is held to that.
- **Built for agents and long documents**: prompt and session caches (a new message only processes itself), system
  prompts kept on disk per project, images and audio input, schema-constrained tool calls, model switching from
  inside Pi, two Pi sessions at once on one model (`--slots 2`): when both are writing, their next tokens run in
  one engine pass (27B prose 7.3 -> 10.5 tok/s together, 35B 20.2 -> 24.7, Gemma 4 26B 19.8 -> 26.2).
- **A web UI that searches and talks**: the model can search the web and read pages (the server fetches them,
  public addresses only), and you can talk to it: the microphone stays open (Gemma 4 E4B / 12B hear the audio
  themselves; with `--whisper auto` Qwen3-ASR writes it down for any model), it answers aloud sentence by sentence,
  can be interrupted, and looks things up while it talks. Speech to text also fills a **Transcripts** and a
  **Meetings** tab. From a phone or anywhere it works through a Cloudflare Tunnel with a password, installable as
  an app ([docs/remote-access.md](docs/remote-access.md)).
- **Long documents read properly**: a document the context takes comes whole; a longer one by its outline, page
  ranges or a search; and for all of it, `digest_file` has the model read it part by part (sections and functions
  kept whole) into notes saved with the chat. Answers say which page they come from.
- **OCR of PDFs and images** with Xiaomi-OCR-0: a PDF read page by page to its last page, saved to a Markdown
  file, tables as tables; a chat model reads that file afterwards.
- **Conversations that keep their cache**, Gemma 4 too: after a 20K-token tool result a one-line follow-up takes
  0.8 s on Gemma 4 E4B, not 38 s.

The whole tour, with how to use each feature: **[docs/features.md](docs/features.md)**.

## Try it

You need:
- Windows 11 (64-bit) and Intel's graphics driver;
- an Intel Core Ultra with Arc integrated graphics;
- 16 GB of RAM for the 4B / E4B models, 32 GB for the 12B-35B ones.

Discrete Arc B-series cards (Xe2) work too, experimentally. A user runs 1-4 Arc Pro B60s, and one model can be split
over several of them (`--gpus N`). See [Discrete GPUs](#discrete-arc-gpus-experimental).

**Linux** (Ubuntu 24.04 or newer): download `inferzero-<version>-linux-x64.tar.gz` from the same release, unpack it,
and run `./zero-launcher.sh` (a terminal menu that downloads a model and starts the headless server). It needs Intel's
GPU runtime and Python 3: [docs/install.md](docs/install.md#linux).

On Windows:

1. Download `inferzero-<version>-win64.zip` (6 MB) from the
   **[latest release](https://github.com/rizagp/inferzero/releases/latest)** and unpack it anywhere.
2. Double-click `zero-launcher.cmd` (Windows may ask once: "More info", "Run anyway").
   Packages released before the rename (up to 2026.11.10.3) are named `arcinferzero-<version>-win64.zip` and start
   with `arc-launcher.cmd`.
3. Pick a model (the one that suits the laptop's RAM is ticked) and "Download and start": the web UI opens at
   `http://127.0.0.1:8090/`.

No compiler and no Python needed. From the sources: install Visual Studio Build Tools (C++), then `.\setup.cmd`
builds everything, downloads the models you pick and connects the [Pi](https://pi.dev) coding agent
([docs/install.md](docs/install.md)). Day-to-day use, the web UI, Pi and the server's options:
**[USAGE.md](USAGE.md)**.

## Results at a glance

On the laptop (Core Ultra 5 125H, 32 GB, Windows 11; `bench/website_bench.ps1`, 2026-11-09, the 27B's 131K row and
Gemma's KL again on 2026-11-10). Prompt in tok/s with the wait for the first token; generation in tok/s for 256
tokens, plain and with the model's drafters; KL on 480 tokens of USAGE.md:

| Model | 8K prompt | 32K prompt | 131K prompt | Generation | With drafting | KL vs f32 |
|---|---|---|---|---|---|---|
| Qwen3.6 35B-A3B (MoE, experts on the SSD) | 851, 9.6 s | 637, 51 s | 324, 6.7 min | 21.6 | 38.1 (MTP + DFlash2) | 0.0040 |
| Qwen3.8 Flash Next (MoE, 52 GB of experts on the SSD) | 205, 40 s | 174, 3.1 min | 146, 15 min | 6.3 | 7.6 (DFlash2) | 0.0117 |
| Qwen3.8 27B (dense) | 135, 61 s | 125, 4.4 min | 81, 27 min | 5.1 | 12.9 (MTP + DFlash2) | 0.0005 |
| Gemma 4 26B-A4B (MoE) | 232, 35 s | 124, 4.4 min | 44, 50 min | 19.8 | – | 0.0048 |
| Qwen3.5 9B | 487, 17 s | 435, 75 s | 278, 7.8 min | 11.8 | 21.6 (MTP) | 0.0014 |
| Gemma 4 E4B | 707, 12 s | 494, 66 s | 222, 9.8 min | 19.1 | 41.8 (drafter) | 0.0027 |
| Qwen3.5 4B | 896, 9.2 s | 705, 47 s | 366, 6.0 min | 21.0 | 35.8 (MTP) | 0.0010 |

Flash Next's precise mode (`-Precise`) lowers its KL on prose to 0.0060 (3,584 tokens, 0.0105 by default) at about
half the prompt speed. The discrete cards' numbers: [Discrete GPUs](#discrete-arc-gpus-experimental).

Against llama.cpp on the same files and laptop (2026-10-25):

| vs llama.cpp (same file, same laptop) | inferzero | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B, 2K prompt | **911 tok/s** | 242 | 337 |
| Qwen3.8 27B, 2K prompt | **113 tok/s** | 23 | 25 |
| Qwen3.8 27B, 34K code prompt to first token | **327 s** | 2,418 s | 1,624 s |
| Qwen3.8 27B, generation after that prompt | **3.7 tok/s** | 2.1 | 1.1 |
| Perplexity, Qwen3.5 4B / 9B / 27B (lower is better) | **6.056 / 5.417 / 4.274** | 6.141 / 5.499 / 4.311 | 6.138 / 5.497 / 4.310 |

On a Core Ultra 5 125H with 32 GB of RAM. KL is the divergence of the next-token distributions from an
independent f32 reference (0 = identical; every model and change is held under 0.015). How each number was
measured, and every change with its before and after: **[docs/before-after.md](docs/before-after.md)**; notes on
these rows: [docs/models.md](docs/models.md).

## Supported models

| Family | Models | Input | Drafts tokens with |
|---|---|---|---|
| Qwen3.5 / 3.6 / 3.8 | 0.8B-27B dense, 35B-A3B MoE, Flash Next (512 experts; also with Intel's AutoRound INT4 experts, `setup -Models flash-ar4`) | text, images (`mmproj`) | MTP block, DFlash2 |
| DeepSeek-V4.1 | Flash (384 experts: ISTA's GSQ-RCO IQ2_XS, Unsloth's UD-Q8_K_XL), one GPU or several | text | |
| Gemma 4 | E4B, 12B, 26B-A4B MoE, 31B (Google's QAT Q4_0) | text, images, audio | Gemma 4 assistant drafters |
| OCR | Xiaomi-OCR-0 | images, PDFs | |
| Speech to text | Qwen3-ASR, Whisper large-v3 | audio, video | |

GGUF files as they are (IQ1-IQ4, Q2_K-Q8_0, Q5_1, MXFP4, F16 / BF16 small matrices); the architectures, weight types and each
model's details: [docs/models.md](docs/models.md).

## The engine

**inferzero** (`zero/`, `bin\serverz.exe`, `bin\runz.exe`) runs OpenCL-compiled kernels through **Level
Zero**. It streams mixture-of-experts weights from the SSD (only the experts in use stay in GPU memory; the next
layer's missing experts are read while the GPU computes the current one), and drafts with the MTP block, a DFlash2
drafter or both. The automatic context gives the 35B and Flash Next 131,072 tokens, Gemma 4 26B 90K-131K. On GPUs
without 8-lane sub-groups (Xe2 / Xe3) it builds its 8-lane kernels with 16 lanes, and where the GPU has XMX matrix
engines the prompt GEMMs run on them after a start-up check against the DP4a kernels.

## Discrete Arc GPUs (experimental)

Arc B-series cards (Xe2: B580, Pro B60, Pro B70) run the same engine, with switches of their own in `zero/xe2/` that
the iGPUs never take. Measured by users (Qwen3.8 27B GSQ-RCO IQ3_S unless named):
- **One Pro B60** ([issue #4](https://github.com/rizagp/inferzero/issues/4)): 4K prompt 1,522 tok/s (release
  2026.11.14.09, which prefetches the weights by default; 1,322 before), generation 23.3 tok/s (2026.11.14.09), 51.6 with MTP
  (2026.11.13.11). Qwen3.6 35B-A3B: 4K prompt 3,536 tok/s (2026.11.14.09), generation 82.4 tok/s, 111.9 with MTP
  (2026.11.13.11). The switches still off by default (`ARC_X16`, `ARC_GV_IQ3S`, `ARC_H8_NORM`, H8 on MoE models,
  `--chunk 8192`) were measured there on 2026-11-14 and none is faster: [docs/xe2-tuning.md](docs/xe2-tuning.md).
- **One Pro B70** ([issue #12](https://github.com/rizagp/inferzero/issues/12), 2026-11-13): 4K prompt 2,107 tok/s,
  generation 32.2 tok/s, 75.2 with MTP.
- **Arc 140T** (the Core Ultra 9 285H's iGPU, [issue #5](https://github.com/rizagp/inferzero/issues/5)): 4K prompt
  330 tok/s (release 2026.11.13.11, its default 32x16 XMX tiles with H8; 302 on 2026.11.11).
- **One model over several GPUs** (`runz` / `serverz --gpus N`, tensor parallel):
  - 4 Pro B60s on Windows (issue #4; no peer access there, so the sums go through host memory): the 27B's 4K prompt
    1,597 / 1,724 / 1,888 tok/s over 2 / 3 / 4 cards (2026.11.13.05 and .11), generation 35.7 / 43 / 53 tok/s (run of
    2026-10-31); over 4 a 131K-token prompt at 970 tok/s, then 32 tok/s generation (2026.11.14.05). The 35B's 4K
    prompt over 4: 4,264 tok/s (2026-11-02).
  - 2 Pro B60s on Linux (scenic6942, [issue #17](https://github.com/rizagp/inferzero/issues/17), PCIe 4.0 x8 each):
    the sums go straight from GPU to GPU (`ARC_TP_P2P`, chosen at load since 2026.11.14.06): 4K prompt 1,630 tok/s
    against 1,442 through host memory, generation 37.8 tok/s, the same logits.
  - DeepSeek-V4.1-Flash over 4 Pro B60s with 1 TB of RAM (issue #4, release 2026.11.14.09): ISTA's IQ2_XS generates
    10.3 / 18.0 / 16.6 / 17.6 tok/s on 1 / 2 / 3 / 4 cards, its 512-token prompt 171 / 227 tok/s over 3 / 4; Unsloth's
    UD-Q8_K_XL 109 / 7.6 over 3 ([docs/multi-gpu.md](docs/multi-gpu.md)).
  - A slow link sets the pace: with one card in an x4 slot at 2.4 GB/s (issue #19, the 27B UD-Q5_K_XL) the prompt over
    2 cards is half of one card's (620 against 1,223 tok/s). A layer split for such links is planned, not built.
  - Batched slots work over a split model.
  - A GPU that hangs makes the request fail and `/health` answer 503; the server itself stays up.
  - A model that cannot be split loads on one GPU.

In progress:
- an f16 XMX prompt GEMM (`ARC_X16=1`, off until measured);
- MTP / DFlash2 on a split model.

How to measure on such a card, and every switch: [docs/xe2-tuning.md](docs/xe2-tuning.md). The split model and
its failsafes: [docs/multi-gpu.md](docs/multi-gpu.md). The plan: QUEUE.md, F31.

## Documentation

| | |
|---|---|
| [docs/features.md](docs/features.md) | the best features: what each is for, how to use it, what it measured |
| [USAGE.md](USAGE.md) | step by step: the launcher, the web UI, Pi, the server's options, troubleshooting |
| [docs/install.md](docs/install.md) | the package, building from source, downloading models, checking the install |
| [docs/server.md](docs/server.md) | running the server by hand (its API, caches, options), Pi, the `runz` CLI |
| [docs/models.md](docs/models.md) | supported architectures and weight types, notes on the results |
| [docs/architecture.md](docs/architecture.md) | how it works, the source layout, how it is validated |
| [docs/before-after.md](docs/before-after.md) | every speed and accuracy change with its numbers |
| [docs/lessons-intel-arc-igpu.md](docs/lessons-intel-arc-igpu.md) | what made LLM inference fast on this iGPU, and what did not |
| [docs/review.md](docs/review.md), [docs/speed-progress.md](docs/speed-progress.md) | results, the story of the build, speed by date |
| [docs/npu-experiment.md](docs/npu-experiment.md), [docs/multi-gpu.md](docs/multi-gpu.md) | the NPU next to the GPU; one model over several GPUs |
| [docs/xe2-tuning.md](docs/xe2-tuning.md) | discrete Arc B-series cards: what was measured, the switches, the scripts to run |
| [docs/qwen3-asr.md](docs/qwen3-asr.md), [docs/whisper-native.md](docs/whisper-native.md) | speech to text, transcripts and meetings |
| [CHANGELOG.md](CHANGELOG.md), [QUEUE.md](QUEUE.md) | what changed recently; open work and decisions |

## License

inferzero is licensed under the [Apache License 2.0](LICENSE).

IQ codebook tables and Unicode category tables are generated from ggml / llama.cpp sources (MIT License).
`third_party/` contains minja (MIT) and nlohmann/json (MIT).

inferzero is an independent project, not made or endorsed by Intel. Intel, the Intel logo and Arc are trademarks of
Intel Corporation or its subsidiaries.
