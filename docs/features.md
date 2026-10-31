# The best of arcinferzero

A tour of what arcinferzero does well, what each feature is for, how to use it, and what it measured. The numbers
come from [before-after.md](before-after.md) (how each was measured) and the [CHANGELOG](../CHANGELOG.md); the
test laptop is a Core Ultra 5 125H (Arc iGPU, no XMX) with 32 GB of RAM. The step-by-step guide is
[USAGE.md](../USAGE.md).

## 1. Fast prompts on an iGPU

Every matrix product runs on the iGPU's int8 DP4a units, the weights decoded once per chunk. Prompts are processed
2.1-4.9x faster than llama.cpp (Vulkan / SYCL) on the same GGUF files and laptop.

| Same file, same laptop | arcinferzero | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B, 2K prompt | **911 tok/s** | 242 | 337 |
| Qwen3.8 27B, 2K prompt | **113 tok/s** | 23 | 25 |
| Qwen3.8 27B, 34K code prompt to the first token | **327 s** | 2,418 s | 1,624 s |

Why it matters: an agent's prompt (system text, tools, files) is thousands of tokens; on a laptop the prompt, not
the generation, decides how long each answer takes.

## 2. Models larger than the GPU memory

Mixture-of-experts models stream their experts from the SSD into a GPU cache while the GPU computes; only the
experts in use stay in memory. Qwen3.6 35B-A3B runs with a 131K-262K context; Qwen3.8 Flash Next (52 GB of experts)
runs on a 32 GB laptop; Gemma 4 26B-A4B likewise.

## 3. Speculative decoding that never changes the answer

The model's MTP block, the Gemma 4 assistant drafters and the DFlash2 block drafter guess several tokens; the model
checks them in one pass and keeps only those it would have chosen. Python code on the 35B runs at ~50 tok/s with
DFlash2, and the text is identical to plain decoding.

## 4. Accuracy measured, and the same output every time

- Every model is compared with an independent f32 CPU implementation: KL of the next-token distributions under
  0.015 (speech models under 0.01); perplexity at or below llama.cpp's on the same files (Qwen3.5 4B 6.056 against
  6.141).
- The same prompt gives the same logits bit for bit, run after run, whatever the expert cache size; every speed
  change is held to that (56 of 56 regression runs bit-identical).

## 5. Conversations that do not start over

A new message processes only itself:

- **Prompt cache**: the conversation's cached prefix is reused. Qwen's recurrent (DeltaNet) state is kept at the end
  of the prompt, the start of the model's turn and the turn start after the user's last message.
- **Gemma 4** (since 2026-10-31): a tool call is rendered without its reasoning every turn, so the prompt does not
  change before the tool results, and the snapshots keep a copy of the sliding-window rings, so the cache goes back
  further than the rings reach. After a 20K-token tool result, a one-line follow-up on Gemma 4 E4B took 0.8 s
  (38.4 s before).
- **Session cache**: other conversations are saved to disk and loaded when they come back; **system prompts** are
  kept on disk per project (Pi).
- **Two conversations at once** (`--slots 2`): their next tokens run in one engine pass (27B prose 7.3 -> 10.5 tok/s
  together, Gemma 4 26B 19.8 -> 26.2).

## 6. Long documents, read properly

With **Files** on in the web UI, the model reads the chat's files itself (DOCX, PDF, XLSX, PPTX, text, code,
Markdown, CSV), and every answer says which section or page it comes from:

- A document the model's context takes in one read comes **whole**, in one call (Qwen3.5 4B's 262K context: ~40K
  tokens; a 98 KB, 56-page report in one call).
- A longer one gives its **outline**; the model reads the sections or a range of pages it needs (`pages: "1-20"`),
  or finds passages with `search_file` (BM25).
- For **all** of a long document (a summary, a review, its weak points) **`digest_file`** has the server read it
  part by part with the loaded model, each section (or function, in source files) kept whole, and returns notes of
  every part with their pages. The notes are saved as `<name> (notes).md` in the chat's files, so the next
  question reads them at once. With a 16K context: README.md (57 KB) 3 parts in 3 minutes on Qwen3.5 4B; a 159 KB
  source file 4 parts on Gemma 4 E4B.
- **Scanned PDFs**: the loaded vision model transcribes the pages when they are read, and the text is kept with
  the file.
- The model also **writes** files: DOCX, PDF, RTF and XLSX from Markdown (tables become sheets, Rp amounts and
  dates become values), and any text format as written.

## 7. OCR: PDFs and images to text

An OCR model (Xiaomi-OCR-0, a Qwen3.5 0.8B with an image encoder) is asked the way it was trained, whatever the
client sends. Name the task in Indonesian or English (nothing: the text; `dokumen`: Markdown with OTSL tables shown
as tables; `tabel`, `rumus`, `json: tanggal, total`).

- A **PDF** is rendered by Windows page by page and read 50 pages a turn; the web UI goes on by itself to the last
  page (a 450-page manual in 9 turns).
- **Saved to a .md** (asked with each PDF): each page is appended as it is read, the chat shows a progress bar, and
  a chat model switched to later reads the file. A folder of your own works too (a copy stays in the chat's files).
- ~1.4 s to encode a 2,000-token page image, 69-73 tok/s for its text.

## 8. Tool calls that stay well formed

While a model writes a tool call, only tokens that keep it valid for the tool's JSON schema are drawn (Qwen and
Gemma 4 formats): no lost required argument, no argument in the wrong place, enum values only. The rest of the call
is the model's own choice. The web UI's tools: web search and page reading, files, a PowerShell terminal in the
chat's folder (each command allowed by the user), chats search.

## 9. A web UI that searches, talks and transcribes

- **Web search** and page reading through the server (public addresses only).
- **Talk**: the microphone stays open, the answer is spoken sentence by sentence, can be interrupted, and looks
  things up while it talks (Gemma 4 E4B / 12B hear the audio themselves; Qwen3-ASR writes it down for any model).
- **Transcripts and Meetings**: audio and video files, recordings from the browser (microphone or a tab, mono or
  stereo, WAV, MP3 or M4A at a chosen bitrate) and meeting recordings, with speakers and times.
- **Speech to text only**: `serverz -m none` runs Qwen3-ASR / Whisper without a chat model (a 2:43 voice message in
  25 s); a chat model loads later on request.

## 10. More hardware

- **NPU next to the GPU** for prompts (`--npu on`): Qwen3.8 27B +15 %, Gemma 4 12B +10 %
  ([npu-experiment.md](npu-experiment.md)).
- **Several GPUs** for one dense Qwen model (`runz --gpus N`, tensor parallel; each GPU reads 1/N of the weights per
  token; the KL on 2-4 shards matches one GPU's) ([multi-gpu.md](multi-gpu.md)).
- Xe2 / Xe3 iGPUs (Lunar Lake, Panther Lake) build the 8-lane kernels with 16 lanes; XMX engines are used after a
  start-up check where the GPU has them.

## 11. Easy to start

A ready-to-use package (no compiler, no Python): unpack, double-click `arc-launcher.cmd`, pick a model that suits the
laptop's RAM, and the web UI opens. The launcher shows each model's memory plan and the running server's state; one
command connects the [Pi](https://pi.dev) coding agent.
