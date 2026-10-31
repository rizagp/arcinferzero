# Multi-GPU: one model split over several GPUs

`runz --gpus N` splits a dense Qwen3.5 / 3.6 / 3.8 model (the 27B, the 9B, the 4B) over N GPUs with tensor
parallelism. Each GPU reads 1/N of the weights per token, so token generation can get up to N times faster.
Generation is bound by memory bandwidth, so this is the only kind of split that speeds it up; a layer split only adds
capacity. The server does not use it yet (see "Not yet").

```
.\bin\runz C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 512 --n 64 --gpus 2
.\bin\runz C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --bench-pp 512 --n 64 --gpus 4
```

The output adds:
- **per prompt run:** how many sums the GPUs made and how long GPU 0 waited in them;
- **after generation:** the sums per token (2 per layer: 128 for the 27B) and the milliseconds of each token spent
  waiting for them.

## How it is split

Code: `zero/engine_tp_shard.h` (what one shard holds), `zero/engine_tp.h` (`TpEngine`: the shards on their threads
and the sums).

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

What the split needs (checked at load; an error names what does not divide):
- DeltaNet k-heads divisible by N, with an even count per GPU;
- kv-heads divisible by N;
- FFN width divisible by 256·N;
- vocabulary divisible by 16·N.

The 27B and 4B split over 2 or 4 GPUs. MTP / DFlash2 drafting, slots and the NPU are off in a split model.

## Accuracy (checked on one GPU, 2026-10-31)

`ARC_TP_SIM=1` runs the N shards on GPU 0, each with its own command list. It checks the split; its speed means
nothing.

- **Greedy text:** identical to one GPU on both models.
  - 4B: 32 tokens after a 2,280-token prompt, 2 and 4 shards.
  - 27B: 16 tokens after a 552-token prompt, 2 shards.
- **Logits:** not bit-identical, because the sums are added in another order. KL against one GPU is 0.0007-0.001
  (top-1 the same at every position).
- **Against an f32 reference** (`tools/ref_forward`, 4B, 200 positions through the decode path), the split is as
  accurate as one GPU:

| | KL mean | KL max | top-1 |
|---|---|---|---|
| 1 GPU | 0.00113 | 0.0143 | 196/200 |
| 2 shards | 0.00096 | 0.0091 | 196/200 |
| 4 shards | 0.00109 | 0.0251 | 197/200 |

- **One GPU is unchanged:** the 4B and 27B logits are bit-identical to the build before multi-GPU.

## Expected speed (not measured yet: no machine here has two GPUs)

One B60 runs the 27B IQ3_S at 21.5 tok/s, reading 11.3 GB per token in 45.3 ms. With N GPUs each reads 1/N, plus
128 sums of ~16-20 µs each (2-2.6 ms per token). Estimates:

| GPUs | Generation | Prompt |
|---|---|---|
| 2 | ~38-40 tok/s | ~2x |
| 4 | ~65-70 tok/s | ~3x (a 4096-token chunk moves 80 MB per sum through host memory at ~6.6 GB/s) |

## Not yet

- The server (`serverz --gpus`): slots, DeltaNet snapshots, the prompt cache and MTP on a split model.
- MoE models (Qwen3.6 35B: experts spread over the GPUs) and Gemma 4.
- Fewer host round trips per sum (an event the next GPU waits on instead of a host wait and a barrier), and f16 parts
  for prompts (half the bytes).
