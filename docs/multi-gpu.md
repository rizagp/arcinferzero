# Multi-GPU: one model split over several GPUs

`runz --gpus N` splits a dense Qwen3.5 / 3.6 / 3.8 model (the 27B, the 9B, the 4B) over N GPUs with tensor
parallelism. Each GPU reads 1/N of the weights per token, so token generation can get up to N times faster.
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

## Expected speed (not measured yet: no machine here has two GPUs)

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
- MoE models (Qwen3.6 35B, Flash Next): experts spread over the GPUs, those that do not fit in pinned host RAM
  instead of the SSD (the B60 machine has ~990 GB free); Gemma 4.
- Fewer host round trips per sum (an event the next GPU waits on instead of a host wait and a barrier), and f16 parts
  for prompts (half the bytes).
