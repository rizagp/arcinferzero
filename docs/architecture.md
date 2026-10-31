# How arcinferzero works

(Moved from the README on 2026-10-31.)

## How it works

- **GEMM on DP4a.** The Arc iGPU has no matrix engines; its fastest arithmetic is signed int8 DP4a
  (~13 TOPS measured). Activations are quantized to int8 with a scale per 32 values; weights are multiplied in
  an int8 "W8" form (int8 values, int8 scale per 16 or 32 values, float scale per 256). SIMD16 sub-groups compute
  16 columns x 16 tokens with constant-index lane broadcasts, work-groups share weight tiles across 8 token tiles
  and are swizzled so weight slices are reused from L3; long reductions (K > 8192) are split so the activation
  slab stays in L3. The fused SwiGLU (silu(gate)·up) is applied in the epilogue of the up-projection GEMM. On GPUs
  with XMX and 16-lane DPAS (Xe2 / Xe3) the same GEMM takes its int8 dot products from DPAS instructions (8 tokens
  x 16 columns x 32 values each); integer sums are exact, so the results are the DP4a kernel's bit for bit, and the
  engine checks that at start before it uses them: every weight layout on random data, launched as the engine
  launches them (split-K, accumulate, fused SwiGLU, the mixture-of-experts tiles); a kernel that fails to build or
  differs keeps DP4a (`ARC_NO_XMX=1`: off).
- **Just-in-time weight decoding.** Every GGUF quant type is decoded into W8 on the GPU: IQ/K types without
  block minimums losslessly; types with minimums (Q2_K/Q4_K/Q5_K) are requantized to signed int8 with a per-16/32
  scale (adds ~1/10 of their own quantization noise; +0.05% perplexity on Qwen3.5-4B), which lets them use the
  fast min-free kernel (Q4_K: 6.1 -> 8.8 TOPS). Small models keep the decoded form resident; large ones (e.g. 27B)
  decode just in time, and a long prompt is processed **layer by layer** (all chunks of the prompt go through a
  layer before the next one), so each matrix is decoded once per prompt and only the raw GGUF bytes plus one
  layer of decoded weights have to fit in memory.
- **Token generation** reads the raw blocks directly (padded to 8-byte multiples for coalesced sub-group block
  reads, staged through local memory, codebooks in local memory) and decodes them with SWAR bit tricks; the large
  matrices run at 60–75 GB/s of the 77 GB/s the iGPU can read from memory. The output head is allocated before all
  other weights (a buffer allocated late is read at about half speed), Gemma's logit soft-capping runs in the head's
  epilogue, and for single-token passes the residual add, the sandwich norm and the next input's norm and int8
  quantization are one kernel.
- **Gated DeltaNet** recurrence: each SIMD8 lane keeps a full 128-float state column in registers; q/k are
  block-read and broadcast, so no reductions are needed. GPUs without 8-lane sub-groups (Xe2 / Xe3) run it with 16
  lanes (their registers are twice as wide), the same arithmetic in the same order (`ARC_NO_SG8=1` forces it).
- **Mixture of experts** (`zero/kernels/moe.cl`): the router GEMM gives the top-8 experts per token (softmax,
  renormalized) and the sigmoid gate of the shared expert; the (token, expert) pairs are grouped by expert with a
  counting sort, and tiles of 16 pairs of one expert run DP4a GEMMs straight on the 2-bit codes (decoded to int8
  in registers, SwiGLU in the epilogue, routing weight on the down projection). A single token skips the sort.
- **Attention, Qwen** is int8 end to end: the KV cache holds int8 K and V (stored in 16-key blocks, one scale per
  row); scores are DP4a, the online softmax is f32 (log2 domain, rescaling only when the running max grows), and
  P·V is DP4a too, with P·vscale quantized to 8 bits per 16-key tile (SageAttention-style). In prefill, K/V tiles
  are shared through local memory by all GQA heads of several tokens; decoding splits the keys over many
  work-groups (flash-decoding) and merges the partial results. Flash Next adds QSA: an indexer scores blocks of keys
  and each query attends to its top blocks only.
- **Attention, Gemma 4** is float for accuracy (int8 attention cost it KL ~0.02 on its own): f32 queries and K
  rows, f16 V rows. Prompt chunks run `attn_fa_gh`: a work-group holds one kv head's q heads for 1-4 query rows (as
  many as fit 64 output floats per lane), so each key and value row is read once for all of them, and its 4 or 8
  sub-groups split the head dimensions (26B, 13,921-token prompt: attention 48 -> ~28 s). Generation and
  verification use flash-decoding (`attn_fa_g -DFA_SPLIT`, keys in up to 64 splits). An int8 copy of K / V stays
  for the layers the assistant drafter reads and, on E2B / E4B, for the global layers of prompts (`attn_ps`). RoPE
  angles are computed once per pass.
- **Gemma 4's two int8 passes** (26B, 12B): each matmul input is quantized to int8 and its rounding residual to a
  second int8 row; both go through the same weights and are added (single tokens and small batches read the
  weights once for both). KL 0.0985 -> 0.0042 on the 26B; E4B is accurate enough on one pass.
- **Prompt cache.** The server holds one conversation (KV cache, DeltaNet states and the tokens they stand for)
  and processes only what a request adds to the prefix it shares. Attention K/V are stored by position, so moving
  back is free, but the DeltaNet states can only be restored where a snapshot was taken: at the end of the system
  prompt (host memory, and on disk per project folder), at the start of the last `<|im_start|>assistant` header
  (where the next turn diverges, also after an aborted reply the client drops) and at the end of the last prompt
  (GPU). A request resumes from the latest one inside its shared prefix. Gemma 4 has no recurrent state and moves
  back to any position its sliding-window rings still cover. A cancelled prompt rolls the block in progress back
  (DeltaNet states from a host copy taken at the block start) and keeps the blocks before it.
- Compiled kernels are cached in `%USERPROFILE%\.arcinferzero\kernels` (keyed by source, options and driver), so
  only the first use of a kernel variant pays the driver's compile time.
- Prompts longer than 16,384 tokens go through the layer-major prefill in segments of 16,384, which keeps the
  residual-stream buffer small and lets the long contexts above fit.
- At start the server also runs a 512-token warm-up prompt, so the first request does not pay the first use of
  the large-chunk kernels (35B with its drafters: first 1,749-token request 8.6 -> 4.2 s).
- Tokenizer (byte-level BPE with the Qwen pre-tokenizer), chat templates (ChatML + `<think>` + XML tool calls for
  Qwen, the GGUF's Jinja template through minja for Gemma 4) and sampling are implemented natively; the tokenizer output is identical to llama.cpp's.

## Layout

```
src/        shared code: quants.h (GGUF decoding/repacking), tokenizer.h, chat.h (Qwen), chat_gemma4.h +
            chat_jinja.h (Gemma 4, minja templates), image.h + vision_gemma4.h (image input), audio.h +
            audio_gemma4.h (audio input), json.h, http.h, gguf.h, clrt.h / cl_min.h (OpenCL loader for clinfo, the
            benchmarks and tests)
zero/       the engine: engine.h (setup, memory, slots, sessions, dispatch) with its parts engine_gemma.h,
            engine_qwen.h, engine_flashnext.h, engine_moe.h, engine_drafters.h (forward passes by model family,
            included inside class Engine), clrt.h (OpenCL-style calls on Level Zero), xcache.h +
            expert_store.h (MoE experts on the SSD, GPU cache), serverz.cpp / runz.cpp, vision_gemma4v.h /
            vision_qwen3vl.h / audio_gemma4a.h (encoders), kernels/ (gemm.cl DP4a GEMM, decode.cl JIT decode + GEMV,
            ops.cl norms / DeltaNet / attention / quantization, moe.cl expert routing and GEMMs), npu/ (the NPU
            next to the GPU: engine_npu.h, npu_rt.h graph runtime, npu_ir.h OpenVINO IR writer, probes), grf_probe.cpp
tools/      server.cpp (the HTTP API, built by zero/serverz.cpp) with server_cache.h / server_chat.h /
            server_catalog.h / server_load.h, api_compat.h (Anthropic / Ollama / Responses formats), web_tools.h (web
            search / page fetch), server_files.h (the web UI's chat folders), doc_writer.h (DOCX / PDF / RTF
            from Markdown), doc_reader.h + inflate.h (reading DOCX / PDF: outline, BM25 search),
            server_terminal.h (the terminal tool), ref_forward.cpp (f32 CPU reference: Qwen3.5, Qwen3.5-MoE, Gemma 4, Flash
            Next), gemma4v_ref.cpp (f32 Gemma 4 image encoder), convert_gsq.py (ISTA GSQ checkpoint -> GGUF),
            tok_test.cpp, chat_render.cpp, gguf_dump.cpp, clinfo.cpp
bench/      micro-benchmarks (gemv_bench, attn_dec_bench, attn_bench, bw_bench, peak_bench, ...), run_bench.ps1
            (comparison with llama.cpp), prompt_bench.py, review_data.py (review data)
scripts/    arc-server.ps1 (server in the background), arc-launcher.ps1 (launcher window, arc-launcher.cmd),
            pi-arc.ps1 (Pi launcher), install-pi.ps1 (commands and Pi's model list), pi-models.ps1 (writes that list)
tests/      run_tests.ps1 and its reference data
webui/      index.html (chat, files and documents, terminal, web search tools, Talk voice mode)
docs/       before-after.md (every change, before / after, accuracy), review.md (results), speed-progress.md
            (speed by date), lessons-intel-arc-igpu.md, flash-next-spec.md, npu-experiment.md (the NPU next to
            the GPU), qwen3-asr.md + whisper-native.md + whisper-native-status.md (speech to text), kernel-audit and
            self-audit reports, archive/ (arcinferzero-log.md, review_notes.md, f21-q2-speed.md: older logs, notes
            and finished plans)
QUEUE.md    open work, measurements and the decisions waiting for the user
setup.cmd   setup.ps1: checks, builds, downloads models, connects Pi;  build.ps1: builds one or more tools
```

## Validation

- `bench/gemv_bench` and `bench/attn_dec_bench` check the generation kernels against a CPU reference on real
  model tensors. `tests/test_quant_rows` (manual) prints one dequantized row of a tensor, to compare with another
  decoder; with `PAD=1` it checks the padding of IQ4_NL / Q2_0 rows.
- `tools/ref_forward` is an independent f32 CPU implementation of the model; `runz --kl-ref` reports the KL
  divergence of the engine's next-token distributions against it (averaged over the prompt positions), and
  `--kl-decode` does the same through the token-by-token generation path.
- `tests/run_tests.ps1` runs all of the above plus chat-template rendering against the GGUF Jinja templates and an
  OpenAI API test (streaming tool calls, prefix reuse); the Gemma 4 models found (or given with `-ModelGemma`)
  get a KL check against the f32 reference and drafter output identical to plain decoding; the 35B-A3B its greedy,
  MTP and DFlash2 checks, and a KL check only with `-Ref35B`. Every model found also
  runs `runz --slots-check` (two conversations alone, interleaved and batched: bit-identical; misuse refused; a
  slot's memory counted exactly), and `tests/fail_inject.ps1` makes a generation pass fail on purpose
  (`ARC_FAIL_PASS=N`): the requests of that pass get an error (a streamed one an error event), the server stays up,
  and the next answers are the same as before the failure.
- Mean KL of the next-token distributions vs the f32 reference: Qwen3.5-4B 0.0021; Qwen3.6-35B-A3B 0.0008 with the
  GSQ file, 0.004-0.008 with UD-IQ4_XS (generation path); Gemma 4 (480-token code prompt) 26B 0.0042, 12B 0.0002,
  E4B 0.00088; Flash Next 0.0143 (363 tokens), 0.0186 at 3.5K. Gemma 4 is about 7x more sensitive to int8 rounding than Qwen, hence
  its float attention and two passes. `tools/ref_forward` can also simulate each engine rounding on its own
  (`REF_Q`, `REF_Q2`, `REF_ATTQ`, `REF_WQ8`) to find which one costs the KL.
- Consistency: logits are compared bit for bit before and after every speed change (`runz --dump-logits`); the
  SSD expert streaming was run with small caches, simulated GPU out-of-memory (`ARC_FAKE_OOM`) and 40 prompts in a
  row, 56 of 56 runs identical; speculative decoding must give the text of plain decoding. The record:
  [docs/before-after.md](before-after.md).
- `tools/tok_test` compares tokenization with reference token ids.
- `tools/tool_stream_test` checks the streamed tool calls: for every prefix of a response the arguments sent so far
  are a prefix of the final ones, and the final ones equal the whole-response parse.
- Speeds measured while another GPU process runs (a serverz, a stray `runz.exe`, llama.cpp) are not
  representative: they share the iGPU and its memory. `runz.exe` prints a note when it finds one.
