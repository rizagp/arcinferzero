# arcinferzero

**Fast local LLMs on an Intel laptop iGPU.** A from-scratch inference engine for **Intel integrated GPUs
(Xe-LPG / Meteor Lake "Arc Graphics"; Xe2 / Xe3 such as Lunar Lake and Panther Lake's Arc B390 too)**, written in
C++ on **Level Zero**. It loads GGUF files directly, serves an **OpenAI-, Anthropic- and Ollama-compatible API with
a web UI** (web search and voice conversation included), and plugs into the [Pi](https://pi.dev) coding agent with
one command. Built and tuned on a Core Ultra 5 125H with 32 GB of RAM: no discrete GPU, no XMX matrix engines.

**[Download](https://github.com/rizagp/arcinferzero/releases/latest)** · [Features](docs/features.md) · [Usage guide](USAGE.md) · [Install and build](docs/install.md) · [Measured results](docs/before-after.md) · [Changelog](CHANGELOG.md)

## Why it is worth a look

- **Prompts 2.1-4.9x faster than llama.cpp** (Vulkan / SYCL) on the same GGUF files and laptop, by building every
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
  **Meetings** tab.
- **Long documents read properly**: a document the context takes comes whole; a longer one by its outline, page
  ranges or a search; and for all of it, `digest_file` has the model read it part by part (sections and functions
  kept whole) into notes saved with the chat. Answers say which page they come from.
- **OCR of PDFs and images** with Xiaomi-OCR-0: a PDF read page by page to its last page, saved to a Markdown
  file, tables as tables; a chat model reads that file afterwards.
- **Conversations that keep their cache**, Gemma 4 too: after a 20K-token tool result a one-line follow-up takes
  0.8 s on Gemma 4 E4B, not 38 s.

The whole tour, with how to use each feature: **[docs/features.md](docs/features.md)**.

## Try it

You need Windows 11 (64-bit), an Intel Core Ultra with Arc integrated graphics and Intel's graphics driver;
16 GB of RAM for the 4B / E4B models, 32 GB for the 12B-35B ones. Discrete Arc GPUs are not supported.

1. Download `arcinferzero-<version>-win64.zip` (6 MB) from the
   **[latest release](https://github.com/rizagp/arcinferzero/releases/latest)** and unpack it anywhere.
2. Double-click `arc-launcher.cmd` (Windows may ask once: "More info", "Run anyway").
3. Pick a model (the one that suits the laptop's RAM is ticked) and "Download and start": the web UI opens at
   `http://127.0.0.1:8090/`.

No compiler and no Python needed. From the sources: install Visual Studio Build Tools (C++), then `.\setup.cmd`
builds everything, downloads the models you pick and connects the [Pi](https://pi.dev) coding agent
([docs/install.md](docs/install.md)). Day-to-day use, the web UI, Pi and the server's options:
**[USAGE.md](USAGE.md)**.

## Results at a glance (2026-10-25)

| Model | Prompt | Generation | KL vs f32 reference |
|---|---|---|---|
| Qwen3.6 35B-A3B UD-IQ4_XS (MoE, experts on the SSD) | 604 tok/s (32K prompt), 324 (128K) | 20.6 tok/s; code 49.9 with DFlash2; 12.6 at 128K depth | 0.004-0.008 |
| Qwen3.8 27B IQ3_S (dense) | ~135 tok/s | 4.9 tok/s; code 18.1 with DFlash2 + MTP | 0.0003-0.0004 |
| Qwen3.8 27B-AP IQ4_XS, GPU + NPU (`--npu on`, 2026-10-25) | 153.8 tok/s (2K; off 133.5), 152.6 (11K; off 132.7), 131.3 (34K; off 116.2) | unchanged (~3.8 tok/s) | 0.0028-0.0031 (off 0.00066) |
| Qwen3.8 Flash Next IQ3_S (512 experts on the SSD) | ~164 tok/s (4K), 163 (32K), 123 (125K) | 6-7 tok/s | 0.0143 / 0.0147 (363 tokens); 0.0186 / 0.0177 at 3.5K (QSA); `-Precise` 0.0127 / 0.0132 (prompt / decode) |
| Qwen3.5 4B | 911 tok/s (2K) | 17.6 tok/s | 0.0021 |
| Gemma 4 E4B (Google QAT) | 724 tok/s (4K), 615 (14K) | 32 tok/s with its drafter | 0.00088 |
| Gemma 4 12B (Google QAT) | 137 tok/s (4K), 102 (14K); +10 % with `--npu on` | 19.0 tok/s with its drafter | 0.0002 (two-pass int8 activations) |
| Gemma 4 26B-A4B (Google QAT, MoE, experts on the SSD) | 252 tok/s (4K), 210 (14K) | ~25 tok/s with its drafter; 14 after a 14K prompt without | 0.0042 (two-pass int8 activations) |

| vs llama.cpp (same file, same laptop) | arcinferzero | llama.cpp Vulkan | llama.cpp SYCL |
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
| Qwen3.5 / 3.6 / 3.8 | 0.8B-27B dense, 35B-A3B MoE, Flash Next (512 experts) | text, images (`mmproj`) | MTP block, DFlash2 |
| Gemma 4 | E4B, 12B, 26B-A4B MoE (Google's QAT Q4_0) | text, images, audio | Gemma 4 assistant drafters |
| OCR | Xiaomi-OCR-0 | images, PDFs | |
| Speech to text | Qwen3-ASR, Whisper large-v3 | audio, video | |

GGUF files as they are (IQ1-IQ4, Q2_K-Q8_0, F16 / BF16 small matrices); the architectures, weight types and each
model's details: [docs/models.md](docs/models.md).

## The engine

**arcinferzero** (`zero/`, `bin\serverz.exe`, `bin\runz.exe`) runs OpenCL-compiled kernels through **Level
Zero**. It streams mixture-of-experts weights from the SSD (only the experts in use stay in GPU memory; the next
layer's missing experts are read while the GPU computes the current one), and drafts with the MTP block, a DFlash2
drafter or both. The automatic context gives the 35B and Flash Next 131,072 tokens, Gemma 4 26B 90K-131K. On GPUs
without 8-lane sub-groups (Xe2 / Xe3) it builds its 8-lane kernels with 16 lanes, and where the GPU has XMX matrix
engines the prompt GEMMs run on them after a start-up check against the DP4a kernels (not yet run on such a GPU).

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
| [docs/qwen3-asr.md](docs/qwen3-asr.md), [docs/whisper-native.md](docs/whisper-native.md) | speech to text, transcripts and meetings |
| [CHANGELOG.md](CHANGELOG.md), [QUEUE.md](QUEUE.md) | what changed recently; open work and decisions |

## License

arcinferzero is licensed under the [Apache License 2.0](LICENSE).

IQ codebook tables and Unicode category tables are generated from ggml / llama.cpp sources (MIT License).
`third_party/` contains minja (MIT) and nlohmann/json (MIT).
