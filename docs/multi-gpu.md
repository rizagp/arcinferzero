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

That is for generation (1-63 rows: the cost of a sum is its latency). Prompt chunks (64 rows or more, since
2026-11-02) sum through a **reduce-scatter in f16** instead, as the bytes are what costs there (4 B60s, 35B: 65-80 %
of a split prompt went to the sums):
1. Each GPU converts its part to f16 and copies every row but its own block (rows `[T·r/N, T·(r+1)/N)`) to host memory.
2. After a barrier, each GPU sums its own block over all parts (its own part in f32, in rank order), rounds the sum
   to f16 once, adds that value to its residual stream and copies it to host memory.
3. After a second barrier, each GPU adds the other blocks' sums.

Per GPU and sum about 3 (N - 1) / N part sizes of f16 cross PCIe, against 1 + N part sizes of f32 before: 4.4x fewer
bytes on 4 GPUs, 4x on 2. `ARC_TP_SUM=f32` keeps the one-step f32 sum for prompts too. `ARC_TP_RS_MIN=N` moves the
64-row threshold, to measure on B60s. A GPU's own block is only range-checked by the f16 conversion (`tp_half`), not
written: `tp_rs_sum` reads it in f32 (bit-identical, 1/N fewer writes). Accuracy (ARC_TP_SIM on the
laptop, 1,687-token prompt + 8 tokens, KL against one GPU): 27B mean 0.0016-0.0018, max 0.0035 (f32 sums 0.0009-0.0012),
greedy text the same; 35B within the split's own noise (f16 vs f32 sums at the same split 0.0025-0.019 at the prompt's
last position, 2 vs 4 GPUs with f32 sums 0.0168).

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

MTP / DFlash2 drafting and the NPU are off in a split model. Slots (batched decoding) work since 2026-11-02 (see
below).

### The sums in pieces (`ARC_TP_PIPE`, 2026-11-12, on by default since the same evening; `=0` off)

The three steps above run one after another, and each uses one direction of a GPU's link at a time: step 1 sends
(GPU -> host), steps 2 and 3 mostly receive (the kernels read host memory). The pieces (`ARC_TP_PIPE`) cut every GPU's block
of rows into P pieces (2-8, ~2 MB of f16 each; `ARC_TP_PIPE_PIECES=N`), the same on every GPU, and overlaps them,
after manoreken2's HostCopy (PR #10, two B60s: a 40 MB block 1.85x faster with a piece in each direction at once):
1. Each GPU converts its part to f16 (`tp_half`), then a copy list of its own (`zrt::tp_copy_list`, a copy engine)
   puts the pieces of the other GPUs' blocks to host memory, piece by piece, each signalling an event.
2. When every GPU has put its first piece, every f16 flag is final: all GPUs decide alike whether to fall back to
   the one-step f32 sum (then every put finishes first, and the f32 sum runs as before).
3. Piece by piece, once every GPU has put it (host counters `TpComm::put_n`): the GPU's own rows of that piece are
   summed (`tp_rs_sum`) while its copy list still puts the later pieces; the piece's sums are put next.
4. Piece by piece, once every GPU has put its sums of it (`TpComm::res_n`): the other blocks' sums are added
   (`tp_rs_add_r`).

The arithmetic is the same as without it, so the logits are the same bit for bit (checked with `ARC_TP_SIM` on the
27B over 2, 3 and 4 GPUs, with 1, 3 and the default pieces, and with the f16 failsafe forced). A buffer half is
written again two sums later only after every GPU has put its first piece of the sum in between, which its copy
list does only after its kernels that read that half. Every wait (an event, the counters) keeps the barrier's time
limit and leaves when another GPU failed; the hang watchdog counts the pieces as progress.

Per GPU and sum about one part size goes out and 2 (N - 1) / N parts come in. One after another that is about 2
part-times on 2 GPUs and 2.5 on 4; overlapped, 1 and 1.5: the sums up to ~2x faster on 2 GPUs and ~1.7x on 4, if
the links and the host memory carry both directions at once (manoreken2's 1.85x says they can on B60s). Not
measured yet: off by default.

### Sums beside the next chunk's work (`ARC_TP_OVERLAP=1`, 2026-11-12, opt-in, stage 1)

A tester's two Arcs (the second on PCIe x4, the 27B IQ4_XS, 32K prompt over 2 GPUs): 904 tok/s, 1,034 with
`ARC_TP_PIPE=1`, and ~31 s of a 31.7-s pass inside the sums (1,024 of them). That figure includes the GPU's own work
before each sum, but most of it is the GPU idle while its shard's thread waits for a sum. Stage 1 lets the GPU work on
the next chunk during the FFN sums.

**Where it applies:** prompts that run layer by layer (`forward_layer_major`: just-in-time weights and a prompt longer
than a chunk, the tester's case), dense split models (not the 35B MoE). Generation, chunk-by-chunk prompts (resident
weights) and MoE sum as before.

**Why the FFN sum can wait:** at layer L, chunk c's FFN sum is read first when layer L+1 reaches chunk c. Before that,
layer L computes chunk c+1, whose attention reads chunk c's KV cache and DeltaNet state; both were written before that
sum. The attention / DeltaNet sums stay on the critical path: the FFN of the same chunk reads them next.

**How:**
- Each shard has a **sum thread** with a Level Zero device handle of its own (`zrt::Dev`, "guarded" by the shard's:
  the same time limit, abort and fault flags), an immediate command list on the compute group (its second queue when
  the GPU has one, else the same queue), its own copy list (`ARC_TP_PIPE`) and its own kernel objects of `tp.cl`.
- The main thread matmuls the output projection into one of **two partial buffers** (instead of the one `tp_part_`);
  its list signals the job's `ready` event; the job (rows, the chunk's view of `x_all_`, the buffer, `ready`, `done`)
  goes to the sum thread's queue.
- The sum thread runs the sums one after another, in the same order on every shard, through the same code as before
  (`tp_reduce_at` / `tp_reduce_rs` / `tp_reduce_rs_pipe`, which now take their buffers as arguments): the same
  barriers, counters, time limits and f16 failsafe. Its list waits for `ready` and signals `done`.
- The main thread waits for `done` right away after an attention / DeltaNet sum; after an FFN sum only before the
  chunk's next layer, before a partial buffer is written again, and at the end of the pass.
- **Every wait between the two lists is a host wait first** (`zrt::event_wait`, with the time limit), then a GPU wait
  on the already signalled event (device scope, so the waiting list sees the other's writes). A GPU wait for an event
  not yet signalled never ends on a GPU with one compute queue: both lists run in that queue. Seen on the laptop: both
  shards stopped until the 120-s limit.
- Events: two per job, never reused within a pass, reset at the next pass's start (after a host wait for the main
  list, so both lists are idle).
- The same kernels on the same values in the same order: **the logits are the same bit for bit** as without the
  switch.

**Failsafes:**
- A sum that throws (a barrier's time limit, another GPU's failure, a GPU wait that gave up) is kept. The rest of the
  queue is skipped, each skipped job's `done` signalled from the host so the main list never waits for good. The
  error is thrown on the main thread at its next submit or wait, as if the main thread had thrown it.
- A pass that ends early with sums still queued (any exception on the main thread) marks the split model failed, so
  the other shards leave their barriers, and waits for its sum thread.
- A served model's cancel check meets the other shards at a barrier: every queued sum finishes first (the cost: the
  layer's last FFN sum is not overlapped when the server has a cancel callback).
- A sum thread stuck for good (`ARC_TP_HANG=r:k` makes it sleep) leaves its shard's thread waiting for it; the
  watchdog (`TpEngine::wait_done`) gives up on that shard as on any stuck shard. Its progress already counts the sum
  threads' work: the barriers and the pipelined pieces are theirs now.
- Profiles (`--prof`) do not show the sum thread's kernels.

**Memory:** one more partial buffer per GPU, `chunk x n_embd x 4` bytes (80 MB for the 27B at chunk 4096).

**Checked** on the laptop with `ARC_TP_SIM=1` (one GPU, one compute queue: the sums do not run beside the GEMMs
there, so no speed is measured), the 27B IQ3_S, `--budget 0` (layer by layer), every prompt position's logits and 3
generated tokens' (`runz --dump-logits`), compared byte for byte:
- the switch off against the build before (6640be9) and the switch on against off: 2, 3 and 4 shards, chunk 512
  (8,678 and 10,171 tokens: 17-20 chunks, 2,176-2,560 sums); `ARC_TP_PIPE=1` over 2 and 4; the f16 failsafe forced
  (`ARC_TP_F16_LIMIT=0`) over 2, and over 3 with `ARC_TP_PIPE=1`; chunk 4096 (3 chunks); a 4,136-token prompt
  whose last chunk of 40 rows takes the one-step f32 sum, over 2, and over 3 with `ARC_TP_PIPE=1`: all identical;
- `serverz --gpus 2` (the cancel check per layer): the same greedy reply to a 2,021-token prompt off and on;
- `ARC_TP_TIMEOUT=5` with `ARC_TP_HANG=1:20` (an FFN sum), `1:21` (an attention sum), `1:20:wait`, `0:20:wait`, and
  `1:20` / `1:20:wait` with `ARC_TP_PIPE=1`: every run ended with the expected error in 46-64 s (20 s of it the load),
  no process left.

**Speed:** unmeasured; it needs two or more real GPUs. At best stage 1 hides the FFN sums (half the sums) behind the
next chunk's attention work.

**Self audit (threads, lifetimes, events, deadlocks, exceptions), fixed before the commit:**
- GPU-side waits between the two lists deadlocked on a one-queue GPU (the 120-s stop above): host waits first.
- A host join left the sum list's last kernels possibly still running while the main list went on (a cancel's state
  restore, the next generation step's sums reuse `tp_part_`): every join also waits for the last two sums on the GPU.
- A sum on rows that are not a chunk view (the sum thread cannot name them as a kernel argument) threw; it now runs
  on the main thread after the queue is empty and its kernels are done.
- Checked and left as they are: the sum thread allocates nothing and its Dev's buffer list is fixed before it
  starts (no concurrent writes); each list and each kernel object is used by one thread only; events are reset only
  with both lists idle; skipped jobs' events are signalled from the host, so neither list waits for good after a
  failure; `release_all` drains the main list, then the sum thread's lists, before any buffer is freed; a faulted
  model is never freed (its sum thread may still run). Known limit: `ARC_ZE_WATCH`'s launch ring (a debug switch) is
  not made for two threads of one GPU.

**Next stages (not built):**
- **Stage 2: hide the attention sums too.** Reorder a layer's chunks so that chunk c+1's attention / DeltaNet half
  runs while chunk c's attention sum is on its way, and chunk c's FFN after it: attn(c+1) ‖ sum_attn(c), then
  FFN(c) ‖ sum_attn(c+1) ... The DeltaNet state and KV cache of chunk c+1 depend only on chunk c's attention input,
  so the order is free; the sums must keep one order on every shard (they do: the same schedule everywhere). It
  needs a partial buffer per sum in flight (3-4) and the FFN input of chunk c kept until its FFN runs.
- **Stage 3: chunk-major prompts (resident weights).** With resident weights a prompt runs chunk by chunk (each
  chunk through all layers): there is no next chunk at the same layer to overlap with. Split each chunk into two
  sub-chunks of rows and run them as stage 1 / 2 do (sub-chunk b's attention while sub-chunk a's sum is on its way);
  the GEMMs get half as many rows, so only where the sums cost more than the smaller GEMMs lose.

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
- **Or by q-head (`ARC_TP_ATT_QSPLIT=1`, 2026-11-02, opt-in).** The rule: N a multiple of the kv-heads, and each
  kv-head's q-heads divisible by N / kv-heads. Over 4 GPUs the 35B then gives each GPU 4 q-heads of one kv-head:
  - that kv-head's K / V on 2 GPUs;
  - `wo`'s columns of its q-heads;
  - the output summed like the other layers'.

  The kernels run at 4 q-heads per kv-head, as for the 9B. Each GPU reads half the K / V and does a quarter of the
  attention, for 10 more sums per token: a gain at long contexts and prompts, a small loss at short ones (estimated
  for B60s: ~0.57 ms more sums against ~0.3 ms less reading per token). Laptop, ARC_TP_SIM, 1,687-token prompt, KL to
  one GPU over the positions before the greedy texts part: 0.0109, against 0.0076 with the attention whole. Both are
  within the 4-GPU split's own range (0.0074-0.0115), and 2 GPUs are bit-identical to before. 4 B60s, 2026-11-06,
  `--bench-pp 16384 --n 64`: prompt 3,435-3,442 -> **4,287-4,292 tok/s** (+25 %), median step 8.8 -> 8.1 ms, each GPU
  5.54 instead of 6.11 GiB. It stays off for now: no KL on B60s, and its one generation run spent about 4.4 s on the
  first token (cause not known, perhaps kernels compiling). An earlier comment in issue #4 gives 4,260 tok/s over 4
  without saying this was on; whether the default went down since has not been checked.
- **Off in a split model:** decode + GEMM for prompt experts (it decodes every expert of a layer; moe_rg runs the
  prompt chunks instead, the same results bit for bit on one GPU), and the cached / missing split of a token's
  experts (ARC_XSPLIT): with experts missing it gave run-to-run differences in a split model, see QUEUE F30.
- **Gate / up through moe_rg too (2026-11-02).** Until then only the down projection of a split prompt chunk took
  moe_rg; gate / up ran the per-tile kernel. Laptop, ARC_TP_SIM, 35B over 2, 1,024-token prompt: 498.7 -> 511.9
  tok/s. KL against one GPU over the 5 positions before the greedy texts part: mean 0.0113 -> 0.0066 (one GPU's
  long chunks are decode + GEMM, which moe_rg equals). `ARC_TP_RG_GU=0`: per tile as before.

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

**27B with `ARC_H8=1`, 2026-11-06** (commit 9c18778, `--bench-pp 4096 --n 16`; one GPU with H8 570 tok/s, 44.9 ms):

| GPUs | Prompt | Against one GPU | Median step | About | Waiting per token |
|---|---|---|---|---|---|
| 2 | 863 tok/s | 1.51x | 27.3 ms | 36.6 tok/s | 22.63 ms |
| 4 | 1,232 tok/s | 2.16x | 18.2 ms | 55 tok/s | 13.17 ms |

- `tg16` says 11.63 and 42.36 tok/s because it counts the slow first step. Use the median.
- That was the per-256 H8.

**27B, 2026-11-07** (commit 515f416, 2D block loads with a prefetch; one GPU 764 tok/s, 44.8 ms; H8 918 tok/s, 44.9 ms):

| GPUs | Prompt | Against one GPU | Median step | H8 now: prompt |
|---|---|---|---|---|
| 2 | 1,097 tok/s | 1.44x | 27.3 ms | not run |
| 4 | 1,501 tok/s | 1.96x | 18.4 ms | 1,570 tok/s (1.71x its one GPU) |

Over 4 GPUs a chunk takes 2.6-2.7 s where the GPU time per card would allow 1.1-1.3 s: the sums are the limit now.

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

## Batched decoding over a split model

Since 2026-11-02 a split model holds conversation slots like one GPU does (each GPU its part of every slot's KV
cache and DeltaNet state), and `TpEngine::forward_batch` runs one generation step of several slots in one pass: the
weights are read once for all of them, and each of the step's sums carries B rows for the latency of one, so the
cost of the sums per token falls with B. In `runz`:

```powershell
.\bin\runz.exe <model.gguf> --bench-pp 512 --gpus 4 --bench-batch 8
```

It first checks that 8 greedy steps of every slot batched give the same logits as each slot alone, bit for bit, then
times a batched step for 1..N slots against one slot alone. Laptop (ARC_TP_SIM, one iGPU: only the ratios mean
something): both checks bit-identical; 4 slots together 3.26x one slot's tokens per second (27B over 2) and 2.41x
(35B over 4; one GPU 2.82x).

## The server: `serverz --gpus N`

```powershell
.\bin\serverz.exe -m <model.gguf> --gpus 4 --slots 4 --ctx 32768
```

The server drives shard 0 as its one engine. Every engine call that changes a conversation's state (forward,
batched steps, binding a slot, reset, the DeltaNet snapshots that let a conversation's next turn reuse its prefix)
begins with a `TpMirror` (`zero/mgpu/tp_mirror.h`): on a served shard 0 the same call runs on the other shards' threads
at once, and their logits are joined into shard 0's, so the server sees the whole vocabulary. On one GPU the guard
is one pointer test (logits bit-identical to the build before). A request's cancel (the web UI's stop) is asked on
shard 0 and its answer shared, so every shard stops at the same point.

Not with `--gpus` yet: MTP / DFlash2, images, the prompt and session caches on disk, the warm-up.

**The sums in pieces by default (2026-11-12):** measured faster on every machine that ran it (27B 4K prompt: 2 Pro
B60s 1,298.9 -> 1,368.8 tok/s, 4 Pro B60s 1,629.6 -> 1,734.0; a 27B IQ4_XS 32K prompt over two Arcs, the second
on PCIe x4, 903.9 -> 1,033.7), so `ARC_TP_PIPE` is on unless `=0`. The gain is far below the ~2x the link's two
directions suggested: most of a sum's time is not the copy itself (the waiting counts each GPU's work before it
too). The next step is to overlap the sums with the next chunk's work (the `ARC_TP_OVERLAP` branch, stage 1).

2026-11-13, with the wide-tile H8 GEMM on by default on each shard (release 2026.11.13.05, issue #4): the 27B's 4K
prompt over 2 Pro B60s 1,596.6 tok/s, over 4 1,887.6 (from 1,368.8 / 1,734.0).

**`--ctx auto` on a split model (2026-11-12):** the dry probe is shard 0 of N (`tp_size` N, `tp_rank` 0: the
largest shard, which takes the extra unit of a count that does not divide), so the context is what one GPU's share
of the weights and the KV leaves on GPU 0. Before, it sized the whole model on one GPU, and a model bigger than one
card got the 4,096-token floor. Laptop, the 27B: weights 10.94 -> 5.47 GiB and 32.9 -> 16.4 MiB of context per 1K
tokens on 2 GPUs (auto context 208,896 -> 262,144, the trained maximum); the 4B split over 2 with `ARC_TP_SIM` loaded
at 262,144 tokens, 5.41 GiB on each GPU (the dry run's estimate: 5.5). A model that cannot be split is sized as on one
GPU (the load then falls back to one GPU, as before). Only GPU 0's memory is probed: a smaller second card is caught by
the split's own load, which then loads on one GPU. `serverz --list-models --gpus N` lists the contexts the same way;
`/props` says `gpus`, and zero-server's Pi sync passes it on, so Pi's model list matches a split server.
`zero-server -Gpus N`, `zero-pi --gpus N` and the launcher's "GPUs (one model split)" start one (kept on a restart and
a model switch like `-Slots`; `-Gpus 1` goes back to one GPU).

Checked over HTTP on the laptop (35B, ARC_TP_SIM 2): the same greedy answers as the one-GPU server (short questions
and a 1,710-token prompt), two chats at once the same as each alone, a request cancelled mid-prompt then another one
the same as before.

### When a GPU hangs

A GPU that stops answering would otherwise leave the others at a barrier for good, and the server with them:
- **Barrier timeout:** a shard waiting longer than `ARC_TP_TIMEOUT` seconds (default 120, far beyond one layer of a
  prompt chunk; 5 s to an hour) stops every shard; the error names the GPUs that did not come.
- **GPU waits** (`zrt::sync` on a split model's GPUs) wait in pieces of 100 ms: they give up after the same timeout,
  or as soon as another shard failed. One GPU waits as before, with no limit.
- **A thread stuck in the driver** 10 s after a failure is given up on (left running, its shard's memory left).
- **No progress at all:** a pass, slot change or mirrored server call during which no barrier completes for
  `ARC_TP_TIMEOUT` + 30 s is given up on the same way, with no shard having failed: a shard stuck *inside* a driver
  call (not in a wait) throws nothing, and the request waited forever (self audit 2026-11-03, MG1). Loads and frees
  are not watched so (they meet at no barrier).
- **After a fault that left a split model unfreed**, no other split model loads in that server process (its threads
  may still run and would set the new model's flags): the server loads on one GPU and says to restart.
- The request gets the error (HTTP 500), the server stays up, `/health` answers 503 ("restart the server") and later
  requests fail at once. The way back is a restart: a GPU that hung keeps its memory and usually its fault until the
  driver resets it, so the server does not load the model again by itself.

After such a fault (a barrier or GPU wait that gave up) the split model is never taken as "in step" again: a call
counts as cleanly failed only when every GPU stopped on its own at the same point (a cancel, a context overflow), none
left a barrier or gave up a wait, all made the same number of sums and none is inside a barrier. Unloading it frees
nothing (a GPU may still run queued work; a thread given up on may still touch it): the memory waits for a restart.

Other failsafes:
- **f16 range:** a prompt chunk's partial sum beyond 65504 / N (the sum of N parts must fit f16 too), or not a
  number, is flagged by the conversion kernel on any GPU; then every GPU sums that projection in f32 (logged up to
  three times as "summed in f32 (failsafe)"). `ARC_TP_F16_LIMIT=0` (tests) sends every non-zero part there: the
  logits then equal `ARC_TP_SUM=f32` bit for bit.
- **The same layout on every GPU:** shards whose GPUs chose another weight layout, chunk or context than GPU 0 would
  run different passes and miss each other's barriers; the load fails with the values to give instead.
- **Host memory of the sums:** `2 x N x chunk x n_embd x 4` bytes of pinned host memory, outside `--budget` (the 27B
  over 4 GPUs: 671 MB at chunk 4096, 1.3 GB at 8192): a larger `--chunk` doubles it.
- **`serverz`:** a model that cannot be split (Gemma 4, e.g. for transcripts; Flash Next) or a split that fails to
  load is loaded on one GPU instead, with a line in the log. A shard that runs out of GPU or host memory while it
  is made counts as such a failure (`zrt::oom_throws`: its allocation throws instead of ending the process).
  `ARC_TP_OOM=r` (tests) fails shard r's first allocation of 256 MB or more.

Only the split model's own work takes the wait limit (`zrt::tp_guarded`: the shard threads, and the server's thread
inside a mirrored call). GPU 0's other users (Whisper, Qwen3-ASR, images) wait as always, so a long wait of theirs
never marks the split model faulty.

Two consequences:
- The server's own GPU 0 work outside a mirrored call (reading logits, sampling) waits without a limit, as on one GPU.
- Whisper runs on GPU 0's command list too, so a guarded wait there also waits for any Whisper work queued ahead of it.

A faulted split model that is left unfreed (`TpEngine::disarm(true)`) does two things:
- it gives the GPUs their limits back, so the next model loaded does not inherit them;
- it sets every GPU's abort, so a thread given up on that is still polling its wait leaves it instead of polling
  forever.

The limit, abort and fault fields of a GPU are atomic, because they are written while such a thread may read them.

During loading, the staging buffer for the weight upload also comes from the allocation that throws, so a shard
short of host memory there falls back to one GPU as well. Freeing a half-built shard swallows a second error
(a GPU wait that gives up while freeing) and keeps the first.

`ARC_TP_HANG=r:k` (tests) makes shard r stop for good at its k-th sum; `ARC_TP_HANG=r:k:wait` makes it throw there as
a GPU wait that gave up. Laptop, `ARC_TP_TIMEOUT=5`: runz ended with
"GPU 1 did not reach the sum within 5 s" after ~15 s; the server answered that request with HTTP 500 after 17 s,
`/health` 503, the next requests at once with the error.

## Sums without the CPU in the middle? `bin\event_probe.exe` (2026-11-02)

Today each sum of a generated token goes through the CPU:
1. each GPU copies its part to host memory;
2. its thread waits for it;
3. the threads meet at a spin barrier;
4. each GPU reads every part.

On 4 B60s that is ~57 us per sum, ~7 of the 27B's 18.7 ms per token.

Level Zero lets a GPU wait on an event that another GPU signals. The event comes from a pool made for several
devices, which the driver keeps in host memory (compute-runtime `event.cpp`), so no peer access is needed. Whether
that holds on Windows B60s is what the probe measures:
- whether the pool can be made;
- whether a 20 KB part arrives intact GPU a -> host -> GPU b, waited on by the GPU, for every pair;
- microseconds per sum, today's path against events, for 2..N GPUs;
- whether a GPU waiting on an event that no GPU will signal is released by the host (the failsafe);
- whether the driver has the experimental GPU memory waits (`zexCommandListAppendWaitOnMemory64`).

On the laptop, `--sim 2` runs two command lists on the one iGPU. That is a test of the tool, not of a link:
- pairs intact;
- 22.4 us per sum today vs 16.7 with events;
- the failsafe released the waiting list in 0.05 ms;
- the zex functions are present.

If the B60s show a clear gain, the decode sums get an `ARC_TP_SYNC=event` path (QUEUE F31.3).

```powershell
.\bin\event_probe.exe
```

## Not yet

- The launcher (a choice of which GPUs), and with `serverz --gpus`: MTP / DFlash2, images, the prompt and session
  caches on disk.
- MoE models: Flash Next (hyper-connections, its two-pass precise mode) and Gemma 4 26B; the 35B is split (above).
  Measured numbers of the 35B over B60s.
- Fewer host round trips per sum (an event the next GPU waits on instead of a host wait and a barrier). The f16
  reduce-scatter for prompts is in (2026-11-02), not yet measured on B60s.
