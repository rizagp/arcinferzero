# Multi-GPU: one model split over several GPUs

`runz --gpus N` splits a dense Qwen3.5 / 3.6 / 3.8 model (the 27B, the 9B, the 4B) or the Qwen3.6 35B-A3B MoE
(and fine-tunes of it such as Ornith 1.5 35B) over N GPUs with tensor parallelism; see "Mixture of experts" below. Each GPU reads 1/N of the weights per token, so token generation can get up to N times faster.
Generation is bound by memory bandwidth, so this is the only kind of split that speeds it up; a layer split only adds
capacity. The server does not use it yet (see "Not yet").

```
.\bin\runz C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 512 --n 64 --gpus 2
.\bin\runz C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 512 --n 64 --gpus 3
.\bin\runz C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 512 --n 64 --gpus 4
```

The output adds:
- **per prompt run:** how many sums the GPUs made and how long GPU 0 waited in them;
- **after generation:** the sums per token (2 per layer: 128 for the 27B) and the milliseconds of each token spent
  waiting for them.

## How it is split

Code, apart from the one-GPU engine: `zero/mgpu/engine_tp_shard.h` (what one shard holds), `zero/mgpu/engine_tp.h`
(`TpEngine`: the shards on their threads and the sums), kernels in `zero/kernels/tp.cl`. The one-GPU code only calls
hooks (`T`, `tp_head`, `tp_out`, `gdn_ab`, `tp_conv_lws`, `tp_loaded`) that do what it did before when the model is
not split.

Every GPU keeps the whole residual stream and computes the norms itself. The heads and FFN columns are divided:

| Part | What each of N GPUs holds |
|---|---|
| DeltaNet layers | k-heads `[k0, k1)` and the v-heads that read them (`h = k + r * nk`, the kernels pair v-head h with k-head h % nk), stored r-major so the kernels are unchanged. Rows of `attn_qkv`, `attn_gate`, `ssm_alpha` / `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`; columns of `ssm_out` |
| Attention layers | kv-head groups with their q-heads: rows of `attn_q` / `attn_k` / `attn_v`, columns of `attn_output`, that part of the KV cache |
| FFN | columns `[f0, f1)` of `ffn_gate` / `ffn_up`, the same columns of `ffn_down` |
| Output head | vocabulary rows `[v0, v1)`; the logits are joined on the host |

The three output projections (`ssm_out`, `attn_output`, `ffn_down`) give partial sums. These are added over the GPUs
before they reach the residual stream:
1. Each GPU copies its part to pinned host memory and waits for its own GPU.
2. The GPUs meet at a barrier.
3. Each GPU adds every part to its residual stream, in rank order.

All GPUs compute the same float sum, so their residual streams stay identical. Host memory is used because Arc B60s on
Windows have no peer access to each other: `mgpu_probe` on 4× B60 showed every direct copy failing, and copies through
host memory taking ~16 µs for 20 KB.

A count that does not divide by N is split as evenly as its unit allows, the first GPUs taking one unit more:
- k-heads in pairs (a pair's v-values are one 256-value block of `ssm_out`'s columns);
- kv-heads one by one;
- FFN columns 256 at a time;
- vocabulary rows 16 at a time.

Over 3 GPUs the 27B holds 6 / 6 / 4 k-heads and 2 / 1 / 1 kv-heads. The GPU with the largest part sets the pace: ~37 %
of the weights instead of 33 %.

A shard's DeltaNet alpha / beta projection can have a number of rows the GEMM does not take (24 for the 27B over 4
GPUs). It gets rows of zeros up to 16, into a scratch buffer packed back by `tp_pack`.

The limits:
- N at most half the k-heads, at most the kv-heads, at most the FFN width / 256;
- 2 to 4 GPUs for the 27B and 4B.

MTP / DFlash2 drafting, slots (batch decoding) and the NPU are off in a split model.

## Mixture of experts (Qwen3.6 35B-A3B, 2026-11-01)

The MoE layers are divided by expert instead of by FFN column: GPU r of N holds the routed experts [e0, e1) (256
experts over 2: 0-127 and 128-255). Every GPU routes every token itself (the router and its input are the same on
all of them), then only its own (token, expert) pairs get slots and tiles (`tp_moe_mask` / `tp_moe_scatter` after
the prompt routing, `tp_moe_tiles1` after a single token's). The expert kernels are the one-GPU ones. Each GPU's
combine (`tp_moe_combine`) writes the sum of its experts, GPU 0 adding the gated shared expert (which every GPU
computes), and the parts are summed as for the dense projections. Expert ids stay global everywhere (router, expert
file, cache tables).

- **Each GPU streams and caches only its experts.** The expert cache (`zero/xcache.h`) has a range: warm-up,
  prefetch, the sizes and the saved cache state (`<model>.gpu<r>of<N>.lru`) cover the GPU's own experts. Over 2 GPUs
  each needs half the cache for the same hit rate; 4 B60s (96 GB) hold all of the 35B's 14.7 GiB of experts many
  times over.
- **Attention whole when the kv-heads are few.** The 35B has 2 kv-heads. Over 3 or 4 GPUs its 10 attention layers
  stay whole on every GPU (the same output everywhere, no sum: 70 sums per token instead of 80), while DeltaNet and
  the experts are divided. Over 2 GPUs everything is divided.
- **Off in a split model:** decode + GEMM for prompt experts (it decodes every expert of a layer; moe_rg runs the
  prompt chunks instead, the same results bit for bit on one GPU), and the cached / missing split of a token's
  experts (ARC_XSPLIT): with experts missing it gave run-to-run differences in a split model, see QUEUE F30.

Checked with `ARC_TP_SIM=1` on the laptop (Qwen3.6 35B-A3B UD-IQ4_XS, 1,687 tokens of `docs/architecture.md`, then
32 greedy tokens):

| GPUs | KL vs 1 GPU, first 4 positions | Greedy text vs 1 GPU |
|---|---|---|
| 2 | 0.0093, 0.0000, 0.0158, 0.0038 | the same 3 tokens, then a near tie (318 vs 513) |
| 3 | 0.0085, 0.0000, 0.0234, 0.0063 | the same 27 tokens |
| 4 | 0.0074, 0.0000, 0.0115, 0.0065 | the same 4 tokens |
| 1, other chunk size (512 vs 8192) | 0.0026, 0.0000, 0.0174, 0.0030 | the same 32 tokens |

The last row is the yardstick: the same model on one GPU moves as much when only its prompt chunks change, so the
split adds no error beyond this MoE's own rounding sensitivity (near ties of the router). One GPU is unchanged: its
logits are bit-identical to the build before (full cache, and a 1.2 GiB cache at 50 % hits three times), and its
speed the same (best of two: 608 vs 612 tok/s prompt, 21.6 vs 21.8 tok/s generation). Two GPUs give the same logits
run after run, with full and with small (2 GiB) caches.

## Experts in RAM (GPUs with memory of their own)

Experts are read from the SSD unbuffered (no Windows file cache), so a machine with a large RAM never used it for
them. On a GPU with memory of its own (`zrt::integrated()` false: Arc B580 / B60 / A770, not the iGPUs) each record
read from the SSD now goes into pinned host memory and stays there, up to the free RAM less 16 GiB (shared by a
split model's GPUs; `serverz --expert-ram-gb N`, `ARC_XRAM_GB`, 0: off). A later miss of that expert is no SSD
read: the GPU copies it from RAM (PCIe DMA), no CPU copy in between (a memcpy into the staging memory ran at only
3-4 GB/s on the laptop). The tiers are then GPU cache -> RAM -> SSD.

On the laptop (forced on, iGPU, 4 GiB cache: not a case it is meant for) the logits are bit-identical with and
without it; generation the same (13.1 vs 13.3 tok/s), the prompt slower (275 vs 372 tok/s: the RAM is the iGPU's
memory too). The numbers that count are a B60 machine's: `serverz --expert-ram-gb 0` against the default.

## Accuracy (checked on one GPU, 2026-10-31)

`ARC_TP_SIM=1` runs the N shards on GPU 0, each with its own command list. It checks the split; its speed means
nothing.

- **Greedy text:** identical to one GPU on both models.
  - 4B: 32 tokens after a 2,280-token prompt, 2, 3 and 4 shards.
  - 27B: 16 tokens after a 552-token prompt, 2, 3 and 4 shards.
- **Logits:** not bit-identical, because the sums are added in another order. KL against one GPU is 0.0007-0.001
  (top-1 the same at every position; 27B over 2 / 3 / 4: 0.00087 / 0.00078 / 0.00068).
- **Against an f32 reference** (`tools/ref_forward`, 4B, 200 positions through the decode path), the split is as
  accurate as one GPU:

| | KL mean | KL max | top-1 |
|---|---|---|---|
| 1 GPU | 0.00113 | 0.0143 | 196/200 |
| 2 shards | 0.00096 | 0.0091 | 196/200 |
| 3 shards | 0.00110 | 0.0208 | 198/200 |
| 4 shards | 0.00109 | 0.0251 | 197/200 |

- **One GPU is unchanged:** the 4B and 27B logits are bit-identical to the build before multi-GPU.

## Measured on 4 Arc Pro B60 (issue #4, 2026-10-31)

Qwen3.8 27B GSQ-RCO IQ3_S, `runz --bench-pp 512 --n 64 --gpus N`, two runs each (the owner's machine: no peer access
between the GPUs, every sum through host memory). Generation by the median step (the tg64 total includes a slow
first step):

| GPUs | Generation step | tok/s | vs 1 GPU | pp512 | vs 1 GPU | Waiting for the sums |
|---|---|---|---|---|---|---|
| 1 | 45.7 ms | ~21.9 | 1.00x | 383-395 tok/s | 1.00x | - |
| 2 | 28.0 ms | ~35.7 | 1.63x | 489-503 | 1.28x | 21.5-21.9 ms of a token |
| 3 | 23.2 ms | ~43 | 1.97x | 531-532 | 1.36x | 33.6 ms (a slow step in the run) |
| 4 | 18.7 ms | ~53 | 2.44x | 531-541 | 1.38x | 12.7 ms |

- "Waiting" counts a shard's time in its sums, the host round trip and the wait for the slowest shard's kernels
  alike, so it is not the transfer alone. Against the ideal (4 GPUs: 45.7 / 4 = 11.4 ms), the 4-GPU step has ~7 ms
  more: ~57 µs for each of its 128 sums, not the 16-20 µs the probe measured for a single copy.
- Prompts gain little: each sum of a 512-row chunk moves 10 MB through host memory, 128 times a chunk, about 0.4 s
  of the ~1 s per chunk.
- One run of 3 GPUs and one of 4 ended while loading in the driver's "Abort was called at 252 line in file:",
  right after the shards' XMX checks. The shards after the first were made at the same time; since then they are
  made one after another (`ARC_TP_PARALLEL_LOAD=1`: together, for tests). The owner's check with the fix (commit
  79edbcd): 20 runs each of 1, 2, 3 and 4 GPUs, 80 loads, no abort; loading 8.7 / 14.9 / 11.7 / 9.3 s (1-4 GPUs,
  averages); every run's median step within 0.4 ms (2: 27.7-28.1 ms, 3: 22.8-23.1, 4: 18.5-18.7) and pp512 within
  2 tok/s (512.8 / 542.1 / 546.5).

Next for speed: f16 parts for prompts (half the bytes), the next GPU waiting on an event instead of a host wait and
a barrier, and batch decoding (a step's sums carry several rows for the cost of one).

## Expected speed (the estimate before the measurement)

One B60 runs the 27B IQ3_S at 21.5 tok/s, reading 11.3 GB per token in 45.3 ms. With N GPUs each reads 1/N, plus
128 sums of ~16-20 µs each (2-2.6 ms per token). Estimates:

| GPUs | Generation | Prompt |
|---|---|---|
| 2 | ~38-40 tok/s | ~2x |
| 3 | ~50 tok/s (the largest part ~37 %) | ~2.5x |
| 4 | ~65-70 tok/s | ~3x (a 4096-token chunk moves 80 MB per sum through host memory at ~6.6 GB/s) |

## Not yet

- The server and launcher (`serverz --gpus`, a choice of which GPUs): slots and batch decoding (a step's sums carry
  B rows instead of 1, so their cost per token falls), DeltaNet snapshots, the prompt cache and MTP on a split model.
- MoE models: Flash Next (hyper-connections, its two-pass precise mode) and Gemma 4 26B; the 35B is split (above).
  Measured numbers of the 35B over B60s.
- Fewer host round trips per sum (an event the next GPU waits on instead of a host wait and a barrier), and f16 parts
  for prompts (half the bytes).
