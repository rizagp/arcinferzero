# DeepSeek-V4.1-Flash (`deepseek41`): implementation spec

Status 2026-11-11: research notes, written before any code (as docs/flash-next-spec.md was), so the
implementation can be checked against them. The real model is ~231 GB as GGUF (ISTA-DASLab IQ1_M 120.7 GB +
Engram 110.6 GB): it can only be run on a tester's machine.

## Status in the engine (2026-11-12)

Implemented (`zero/engine_ds41.h`, `zero/kernels/ops_ds41.cl`, the MoE path of `zero/engine_moe.h` with `moe.cl
-DROUTE_DS` and `-DSWI_CLAMP`), and checked against the f32 reference `tools/ref_forward` (`deepseek41_layers`,
written from section 3) on synthetic models from `tools/make_tiny_deepseek41.py`. It has not run the real model yet;
the accuracy check of the real model is against llama.cpp on the same prompts (the user's decision, 2026-11-11).

The real file's metadata (`bin\gguf_dump.exe` on manoreken2's `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.gguf`, issue
#4, 2026-11-12): one file of 247.56 GiB, 1,043 tensors; every key the engine reads present with the values of
section 1 (compress ratios 43 entries, KV sources 2 / 8 / 14 / 20, index sources 2 / 8 / 14 / 20 / 24 / 28 / 32 /
36, candidate source 20 with 2048 blocks of 8, Engram layers 1 / 14 with 4-grams x 8 heads, rows of 256, 48 primes
just above 16,000,000; 40 SwiGLU clamps of 10; `hash_layer_count` 0; `expert_gating_func` 4; no
`hyper_connection.single_pass` key). Types: Q5_K (token embedding, head, q_b, output LoRA, shared experts), Q8_0
(q_a, kv), BF16 router, F32 norms / mixes / sinks / biases, routed experts IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS
(142 GB), the Engram table IQ4_NL (105 GB, 768,022,850 rows). The tokenizer's pre-tokenizer is `joyai-llm`, which
inferzero's BPE does not have yet (it splits as Qwen's): first runs take token ids from llama.cpp.

| Test model | What it covers | KL against f32 (mean over the positions) |
|---|---|---|
| tiny, `--nu 16` (10 layers, 8 heads, window 16, top-k 8, 16 experts all used) | every layer role, window wrap, top-k, the candidate pool, Engram | 100 tokens 0.0030 prompt / 0.0032 token by token; 300 tokens 0.0037 / 0.0037 |
| tiny, `--nu 16 --heads 16` | the 16-head prompt attention (`ds_attn_gh`) | 300 tokens 0.0031 |
| tiny (4 of 16 experts) | the routing | 0.003-0.05: expert near-ties flip under int8 rounding (traced layer by layer) |
| `--real` (4 layers at the real widths: 5120, 64 heads, top-k 512, a 2048-block pool, 32 indexer heads, 384 experts) | kernel and buffer limits at the real sizes | median 0.001 at every depth over 1,200 positions; mean 0.05 from a random router's expert ties |

What the server does with it: prompts and generation, the prompt-cache snapshots and saved conversations (format 3;
`runz --snap-test` bit-identical). Not yet: slots, drafters (the DSpark MTP GGUF, section 3.9), images, multi-GPU.

Switches (all default on; each one's off value restores the earlier kernel):

| Switch | What it turns off |
|---|---|
| `ARC_DS_KV32=1` | f16 key caches (window ring, compressed rows, index keys): f32 |
| `ARC_DS_MIX1=1` | the hyper-connection mixes over 16 work-groups a token: one work-group |
| `ARC_DS_ATTN1=1` | the split attention of 1-8 rows and the 16-head prompt attention: one sub-group per (row, head) |
| `ARC_DS_ATTN_GH=0` | the 16-head prompt attention only |
| `ARC_DS_IX1=1` | the lane-per-row indexer scores: one work-item per score |

Speed on the laptop (Core Ultra 5 125H), the `--real` test model's 4 layers, a 1,200-token prompt: prompt 407 ->
~990 tok/s and generation 32 -> 41 tok/s over the night's changes (docs/before-after.md, 11-11 / 11-12). What is left
there is the experts and the large matrix products at memory speed; the real model's speed depends on where its
experts live (RAM or SSD) on the tester's machine.

Sources are numbered [S1]..[S20] (list at the end). Statements without a source tag are derived arithmetic. "GUESS" marks
inferences that were not seen in code. Most code was read through a summarising web fetcher, not byte for byte, so short
code quotes are faithful to what the fetcher returned but should be checked against the source before relying on a
single character (for example the sign or an eps).

**Important correction to the "already known" list:** V4.1-Flash has **no compress ratio 4**. The ratios are 0, 2 and 1
only [S1][S2]; llama.cpp throws on a ratio above 2 for this architecture [S7]. (Ratios 4 / 128, the compressor `ape`
tensor, the hash-routed first layers and the per-head q RMS norm belong to DeepSeek-V4, not V4.1 [S8][S10].)

---

## 0. Model at a glance

| item | value | src |
|---|---|---|
| layers | 40 (20 "encoder" + 20 "decoder" in the card's Causal Encoder-Decoder wording) | S1 S15 |
| hidden `n_embd` | 5120 | S1 |
| vocab | 129280; bos 0, eos 1, pad 2; image token 129264 | S1 |
| attention | 64 query heads, 1 KV head, head_dim 512 (rope part 64, nope part 448), K = V (same tensor) | S1 S3 S7 |
| q LoRA rank | 1280 | S1 |
| output LoRA | o_groups 8, o_lora_rank 1024 | S1 |
| sliding window | 128 tokens, current token included | S1 S3 |
| compressed attention | per-layer ratio (0 = window only, 2, 1); top-512 compressed rows per query from a lightning indexer (32 heads x 128) | S1 S2 |
| MoE | 384 routed (6 active) + 1 shared, expert FFN 2304, sqrt(softplus) router, bias for selection, renorm, x1.5 | S1 S2 S12 |
| SwiGLU clamp | limit 10.0 (up clamped to [-10,10], gate clamped above at 10) | S2 S3 S12 |
| residual | 4 hyper-connection streams ("single-pass mHC"), Sinkhorn 20 iters, eps 1e-6 | S1 S3 S5 |
| RMSNorm eps | **1e-20** (all norms, mHC rms, Engram rms) | S1 S2 S3 |
| RoPE | theta 10000, no YaRN on window-only layers; theta 160000 + YaRN (factor 16, orig 65536, beta_fast 32, beta_slow 1) on compressed layers | S2 S3 |
| Engram | layers 1 and 14; n-grams 2..4, 8 heads per n-gram => 24 hash columns per layer; row dim 256; tables 384,006,168 + 384,016,682 rows | S1 S2 S4 |
| MTP | none in the main GGUF; DSpark drafter (3 layers) is a separate GGUF (`dflash` arch) | S7 S11 |
| params | 552B backbone + 196B Engram (ISTA card: 551.9B + 196.6B, 763.2B) | S15 S16 |

Layer roles (card names: Full / Reindex / Reuse [S15]):

| layers | ratio | KV source (compressor + index keys) | index source (indexer q, top-k) |
|---|---|---|---|
| 0, 1 | 0 | none (window only) | none |
| 2..7 | 2 | 2 | 2 |
| 8..13 | 2 | 8 | 8 |
| 14..19 | 2 | 14 | 14 |
| 20..23 | 1 | 20 | 20 |
| 24..27 | 1 | 20 | 24 |
| 28..31 | 1 | 20 | 28 |
| 32..35 | 1 | 20 | 32 |
| 36..39 | 1 | 20 | 36 |

A consumer uses "the most recent KV source and index source at or before it" [S7][S3]. Layer 20 is also the candidate
(block pre-selection) source for index layers 24, 28, 32, 36 [S1][S3][S7].
`compress_ratios` in config has **43** entries: 40 layers + 3 trailing zeros for the 3 DSpark/MTP layers [S1][S2].

---

## 1. GGUF hparams (`deepseek41.*`)

Key strings from `src/llama-arch.cpp` [S9]; `%s` = `deepseek41`. Values: from config [S1][S2] via the converter
[S11][S12][S18]; the actual GGUF value was **not** read from a file (the HF viewer shows no metadata) - verify with
`gguf-dump` on shard 1 (77 KV entries, 1043 tensors [S19]).

### Standard keys (written by the base `TextModel` / `DeepseekV4Model` converter)
| key | meaning | expected value |
|---|---|---|
| `general.architecture` | | `deepseek41` |
| `deepseek41.block_count` | layers | 40 |
| `deepseek41.context_length` | | 1048576 |
| `deepseek41.embedding_length` | n_embd | 5120 |
| `deepseek41.attention.head_count` | | 64 |
| `deepseek41.attention.head_count_kv` | | 1 |
| `deepseek41.attention.key_length` / `value_length` | | 512 / 512 |
| `deepseek41.rope.dimension_count` | rotated dims per head | 64 (`qk_rope_head_dim`) [S12] |
| `deepseek41.rope.freq_base` | window-only layers | 10000 |
| `deepseek41.rope.scaling.type` | | `yarn` |
| `deepseek41.rope.scaling.factor` | | 16 |
| `deepseek41.rope.scaling.original_context_length` | | 65536 |
| `deepseek41.rope.scaling.yarn_beta_fast` / `yarn_beta_slow` | | 32 / 1 |
| `deepseek41.attention.layer_norm_rms_epsilon` | | 1e-20 |
| `deepseek41.attention.q_lora_rank` | | 1280 [S12] |
| `deepseek41.attention.sliding_window` | n_swa | 128 [S12] |
| `deepseek41.expert_count` / `expert_used_count` | | 384 / 6 |
| `deepseek41.expert_feed_forward_length` | n_ff_exp | 2304 [S12] |
| `deepseek41.expert_shared_count` | | 1 [S12] |
| `deepseek41.expert_weights_scale` | routed_scaling_factor | 1.5 [S12] |
| `deepseek41.expert_weights_norm` | norm_topk_prob | true [S12] |
| `deepseek41.expert_gating_func` | must be SQRT_SOFTPLUS or the loader throws [S8] | enum value of `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` (integer not seen) |
| `deepseek41.swiglu_clamp_exp` | per-layer array | [10.0] x block_count [S12] |
| `deepseek41.swiglu_clamp_shexp` | per-layer array (falls back to exp) | [10.0] x block_count [S12][S8] |
| `deepseek41.attention.output_group_count` | o_groups | 8 |
| `deepseek41.attention.output_lora_rank` | | 1024 |
| `deepseek41.attention.compress_ratios` | per-layer array | config list passed through unchanged (43 entries) [S12]; how the loader treats the 3 extra is not seen |
| `deepseek41.attention.compress_rope_freq_base` | theta for compressed layers | 160000 |
| `deepseek41.attention.indexer.head_count` | | 32 |
| `deepseek41.attention.indexer.key_length` | | 128 |
| `deepseek41.attention.indexer.top_k` | | 512 |
| `deepseek41.hyper_connection.count` | | 4 |
| `deepseek41.hyper_connection.sinkhorn_iterations` | | 20 |
| `deepseek41.hyper_connection.epsilon` | | 1e-6 |
| `deepseek41.hash_layer_count` | V4 hash-routed layers | from `num_hash_layers`, absent in V4.1 config => presumably 0 / absent (GUESS) |
| `deepseek41.vocab_size`, tokenizer keys | | 129280 |

`feed_forward_length`, `leading_dense_block_count`, `expert_group_count` were not seen for this arch; V4.1 has no dense
layers and no expert groups [S3].

### V4.1-specific keys (written by `conversion/deepseek41.py` [S11]; renamed list in [S18])
| key | type | meaning | value |
|---|---|---|---|
| `deepseek41.attention.kv_source_layers` | u32[] | layers that own a compressor + index keys | [2, 8, 14, 20] |
| `deepseek41.attention.indexer.source_layers` | u32[] | layers that run an indexer query + top-k | [2, 8, 14, 20, 24, 28, 32, 36] |
| `deepseek41.attention.indexer.candidate_source_layer` | i32 | layer that builds the block candidate pool | 20 |
| `deepseek41.attention.indexer.candidate_top_k_blocks` | u32 | blocks kept | 2048 |
| `deepseek41.attention.indexer.candidate_block_size` | u32 | compressed rows per block | 8 |
| `deepseek41.ple.layers` | u32[] | Engram layers | [1, 14] |
| `deepseek41.ple.ngram_size` | u32 | max n-gram | 4 |
| `deepseek41.ple.heads_per_ngram` | u32 | | 8 |
| `deepseek41.embedding_length_per_layer_input` | u32 | Engram row dim | 256 |
| `deepseek41.ple.layer_multipliers` | u64[] | odd hash multipliers, layer-major, `ngram_size` per layer | 2 x 4 = 8 values |
| `deepseek41.ple.head_offsets` | u64[] (llama.cpp stores u32) | absolute first row of each hash column in the concatenated table | 2 x 24 = 48 values |
| `deepseek41.ple.head_vocab_sizes` | u64[] (stored u32) | prime bucket count per column | 48 primes, each just above 16,000,000 |
| `deepseek41.engram.pad_token_id` | u32 | pad id (original vocab) | 2 |
| `deepseek41.engram.token_map` | i32[129280] | vocab id -> compressed id (0..99091) | computed by the converter |
| `deepseek41.hyper_connection.single_pass` | bool | single-pass mHC | written true for the DSpark GGUF [S11]; reportedly **not** written for the main model by the saver [S6 bot review] |

Derived in the loader [S7]: `ple_n_heads = (ngram_size-1)*heads_per_ngram = 24`; `n_embd_out_impl = n_embd` (no HC head,
unlike V4 where it is `hc*n_embd` [S8]); `dsv41_kv_src[il]`, `dsv41_topk_src[il]` per layer; checks `offset + prime <=
INT32_MAX`, `prime >= 2`, ratios in {1,2}, source layers have nonzero ratio, candidate source is an index source, and
candidate consumers share its ratio.

---

## 2. Tensors

Names from [S9]; shapes in ggml order `{ne0, ne1, ...}` (ne0 = input dim for matmul weights) from `load_arch_tensors` [S7];
types in the ISTA IQ2_XXS build from [S19].

Notation: E=5120, H=64, D=512 (head), R=64 (rope dims), QL=1280, OL=1024, G=8, HC=4, HCD=HC*E=20480,
MIX=(2+HC)*HC=24, IH=32, ID=128, NE=384, FF=2304, PH=24 (Engram columns), PD=256.

### Global
| tensor | shape | role | ISTA type |
|---|---|---|---|
| `token_embd.weight` | {E, 129280} | embedding (no scaling [S3]) | Q5_K |
| `output_norm.weight` | {E} | final RMSNorm | F32 |
| `output.weight` | {E, 129280} | LM head (reference computes it in f32 [S3]) | Q5_K |
| `per_layer_token_embd.weight` | {PD=256, 768,022,850} | both Engram tables concatenated (layer 1 rows first, then layer 14), read lazily by rows [S7][S18] | IQ4_NL (110.6 GB, separate shard 6) |

No `output_hc_*` tensors in V4.1 (those are V4's HC head) [S7][S8].

### Every layer `blk.N.`
| tensor | shape | role | ISTA type |
|---|---|---|---|
| `attn_norm.weight` | {E} | RMSNorm before attention | F32 |
| `attn_q_a.weight` | {E, QL} | q down-projection | Q8_0 |
| `attn_q_a_norm.weight` | {QL} | RMSNorm on q latent `qr` | F32 |
| `attn_q_b.weight` | {QL, H*D=32768} | q up-projection | Q5_K |
| `attn_kv.weight` | {E, D} | single KV head (K = V) | Q8_0 |
| `attn_kv_a_norm.weight` | {D} | RMSNorm on kv | F32 |
| `attn_sinks.weight` | {H} | per-head attention sink logit | F32 |
| `attn_output_a.weight` | {H*D/G=4096, OL, G} (stored 2D [4096, 8192] in GGUF, reshaped at load, `TENSOR_ALLOW_RESHAPE`) [S7][S18] | grouped output LoRA A | Q5_K |
| `attn_output_b.weight` | {G*OL=8192, E} | output LoRA B | Q5_K |
| `hc_attn_fn.weight` | {HCD, MIX} = {20480, 24} | mHC mix projection, attention site | F32 |
| `hc_attn_base.weight` | {24} | mHC bias | F32 |
| `hc_attn_scale.weight` | {3} | mHC scales (pre, post, comb) | F32 |
| `hc_ffn_fn/base/scale.weight` | same | mHC, FFN site | F32 |
| `ffn_norm.weight` | {E} | RMSNorm before MoE | F32 |
| `ffn_gate_inp.weight` | {E, NE} | router (forced F32/BF16; logits accumulated in F32 [S7][S12]) | BF16 |
| `exp_probs_b.bias` | {NE} | selection bias (text tokens) | F32 |
| `exp_probs_b_vl.bias` | {NE} | selection bias for image tokens (optional) | F32 |
| `ffn_gate_exps.weight` | {E, FF, NE} | routed w1 | IQ1_M/IQ2_XXS/IQ2_XS per layer |
| `ffn_up_exps.weight` | {E, FF, NE} | routed w3 | same |
| `ffn_down_exps.weight` | {FF, E, NE} | routed w2 | same |
| `ffn_gate_shexp.weight` | {E, FF} | shared w1 | Q5_K |
| `ffn_up_shexp.weight` | {E, FF} | shared w3 | Q5_K |
| `ffn_down_shexp.weight` | {FF, E} | shared w2 | Q5_K |

### KV-source layers only (2, 8, 14, 20) [S7][S19]
| tensor | shape | role | ISTA type |
|---|---|---|---|
| `attn_compressor_kv.weight` | {E, D} | compressor value projection | BF16 |
| `attn_compressor_gate.weight` | {E, D} | compressor score projection, **only if ratio > 1** (layers 2, 8, 14; absent on 20) | BF16 |
| `attn_compressor_norm.weight` | {D} | RMSNorm on compressed latent | F32 |
| `indexer.attn_k.weight` | {D, ID} | index key from latent (512 -> 128) | BF16 |
| `indexer.k_norm.weight` | {ID} | RMSNorm on index key | F32 |

### Index-source layers only (2, 8, 14, 20, 24, 28, 32, 36)
| tensor | shape | role | ISTA type |
|---|---|---|---|
| `indexer.attn_q_b.weight` | {QL, IH*ID=4096} | indexer query from `qr` | BF16 |
| `indexer.proj.weight` | {E, IH} | per-head indexer weights from the normed attention input | BF16 |

### Engram layers only (1, 14)
| tensor | shape | role | ISTA type |
|---|---|---|---|
| `engram_kv.weight` | {PH*PD=6144, E*(HC+1)=25600} | projects the 24 concatenated rows to 4 keys + 1 value | Q5_K |
| `engram_gate.weight` | {E, HC} | `q_weight * k_weight` (product precomputed by the converter, exact) [S11][S18] | F32 |

The ISTA allocation file confirms exactly this tensor set (1043 tensors; quant counts BF16 67, F32 531, IQ1_M 60,
IQ2_XS 18, IQ2_XXS 42, IQ4_NL 1, Q5_K 244, Q8_0 80) [S19].

---

## 3. Forward pass

Reference = `inference/model.py` + `engram.py` + `kernel.py` [S3][S4][S5]. llama.cpp = `src/models/deepseek41.cpp`,
base `deepseek4.cpp`, `llama-graph.cpp` [S7][S8][S10]. They agree unless section 4 says otherwise.

RMSNorm everywhere: `y = w * x * rsqrt(mean(x^2) + 1e-20)`, computed in f32 [S3].

### 3.1 Embedding and stream init
1. `h = token_embd[id]` (no scale) [S3].
2. Expand to 4 identical streams: `X[s] = h` for s = 0..3, shape [T, 4, E] [S3][S7].
3. `pre_mix = one_hot(0)` (stream 0) - the collapse weights for layer 0's attention input [S3]. llama.cpp just copies
   stream 0 for layer 0 [S7].

### 3.2 Engram hashing (host side; done before the graph in both implementations)
Per token at absolute position `p`, for each Engram layer `e` (0 for layer 1, 1 for layer 14) [S4][S7]:
1. Compress the token: `c = token_map[id]` (dense ids 0..99091). Image / embedding tokens and out-of-range ids become
   `DEAD = -1`. `pad = token_map[engram_pad_token_id]`.
2. Lookback tokens: `t[0] = c(p)`, `t[k] = c(p-k)` for k = 1..3. If `p-k < 0` or `c(p-k)` is DEAD, the shift is
   blocked; blocking is cumulative (once shift k is blocked, all longer shifts are blocked too); a blocked shift uses
   `pad`. The token itself being DEAD blocks shift 0 (becomes pad) [S4][S7]. The history is per sequence and spans
   prefill and decode (the reference keeps a [batch, max_seq] cache of compressed ids) [S4].
3. `mult = ple.layer_multipliers[e*4 + k]` (u64, odd), one per lookback position k, shared by all heads and n-gram sizes.
4. Rolling hash, u64 with wraparound: `r = t[0]*mult[0]`; for order o = 1..3 (n-gram size o+1):
   `r ^= t[o]*mult[o]`; for head h = 0..7: `col = (o-1)*8 + h`;
   `row[col] = (r % prime[e*24+col]) + offset[e*24+col]` (int32) [S4][S7].
   Primes and offsets come from GGUF; offsets are absolute rows into `per_layer_token_embd` (prefix sums over all
   layers and columns) [S18].
5. How the converter derives the constants (only needed to re-create them) [S4][S11]: multipliers:
   `bound = max(1, (INT64_MAX // 99092) // 2)`; `rng = numpy.random.default_rng(10007 * layer_id)`;
   `v = rng.integers(0, bound, size=4)`; `mult = 2*v + 1`. Primes: walk "next prime not yet used" upward from
   `engram_vocab_size - 1 = 15,999,999`, per layer, per n-gram size, per head; no prime reused; each layer's primes
   sum to `engram_num_embeddings[layer]`. Token map: decode each id alone; key = raw token string if the text has U+FFFD,
   else NFKC -> NFD -> strip accents -> lowercase -> collapse whitespace -> strip (empty -> raw text); dense ids in
   first-seen order. Not needed at runtime if the GGUF arrays are used.

### 3.3 Engram block (applied to the streams **before** layers 1 and 14) [S3][S7]
For one token, X = [4, E] streams:
1. `rows = concat_{col=0..23} per_layer_token_embd[row[col]]` -> 6144 values (the 24 rows concatenated, not summed).
2. `kv = engram_kv @ rows` -> 25600. `key[s] = kv[s*E:(s+1)*E]` for s = 0..3; `value = kv[4E:5E]`.
3. Per stream s: `dot_s = sum_d( X[s,d] * key[s,d] * gate[s,d] ) * rsqrt(mean(X[s]^2)+eps) * rsqrt(mean(key[s]^2)+eps) / sqrt(E)`,
   `gate = engram_gate` (= q_weight*k_weight), eps = 1e-20. (llama.cpp: `sum_rows(rms(x) * rms(key) * egate) / sqrt(n_embd)`
   with weightless rms norms - identical.)
4. `g_s = sigmoid( sign(dot_s) * sqrt(max(|dot_s|, 1e-6)) )` (sign(0) = +1 in llama.cpp).
5. `g_s *= keep` (0 for image tokens, 1 otherwise).
6. `X[s] += g_s * value` (one value shared by all four streams).

### 3.4 mHC mixes ("single pass") [S3][S5][S8]
For a site (attention or FFN) with weights `fn` {20480, 24}, `base` {24}, `scale` {3}, from stream state X:
1. `flat = X` flattened to 20480 values (stream-major: stream 0's 5120 values first), f32.
2. `m = (fn @ flat) * rsqrt(mean(flat^2) + 1e-20)` -> 24 values (one rms over all 20480 values; equal to a weightless
   RMSNorm then matmul).
3. Split: `pre = m[0:4]`, `post = m[4:8]`, `comb = m[8:24]` (flat index `j*4+k`).
4. `pre[j]  = sigmoid(pre[j]*scale[0] + base[j]) + 1e-6`.
5. `post[j] = 2*sigmoid(post[j]*scale[1] + base[4+j])` (no eps).
6. `c[j][k] = comb[j*4+k]*scale[2] + base[8+j*4+k]`, then Sinkhorn with eps = 1e-6, iters = 20:
   - `c = softmax over k (each row j)`, then `c += eps`;
   - column normalise: `c[j][k] /= (sum_j c[j][k] + eps)`;
   - repeat 19 times: row normalise `c[j][k] /= (sum_k c[j][k] + eps)`, then column normalise as above.
   (Total: softmax+eps, 1 column step, 19 row/column pairs; ends on a column step.) [S5][S8]
7. Collapse ("hc_pre") with a weight vector w[4]: `y = sum_j w[j] * X[j]` (f32) [S3].
8. Expand ("hc_post") with sublayer output f [E], the residual streams Xr = the X the mixes were computed from:
   `Xnew[k] = post[k] * f + sum_j c[j][k] * Xr[j]` - i.e. `c[j][k]` = weight from source stream j to destination k
   [S3 hc_post einsum]; llama.cpp indexes the same element as `comb[dst*nb0 + src*nb1]` [S8].

Single-pass wiring per block (the pre-mix a site computes is used by the **next** site) [S3][S7]:
```
# input: X (streams), pre_in (= previous block's ffn_pre, or one_hot(0) for layer 0)
if layer is Engram layer: X = engram(X)            # 3.3, before the block
Xr = X
(a_pre, a_post, a_comb) = mixes(X, hc_attn_*)       # from the block input
h  = attn_norm( hc_pre(X, pre_in) )
f  = attention(h)
X  = hc_post(f, Xr, a_post, a_comb)
Xr = X
(f_pre, f_post, f_comb) = mixes(X, hc_ffn_*)        # from the post-attention stream
h  = ffn_norm( hc_pre(X, a_pre) )                   # uses the ATTENTION site's pre
f  = moe(h)
X  = hc_post(f, Xr, f_post, f_comb)
return X, f_pre                                      # f_pre feeds the next block's attention input
```
So each layer does 2 mix projections; no third projection for the output.

### 3.5 Attention (per layer) [S3][S7]
Input `x` = `attn_norm(...)`, [E] per token. Let `pos` = absolute position.

RoPE parameters per layer:
- ratio 0 (layers 0, 1): theta 10000, **no YaRN** (`freq_scale 1, ext_factor 0`) [S3][S7].
- ratio > 0 (layers 2..39): theta 160000 with YaRN (factor 16, original 65536, beta_fast 32, beta_slow 1). Used for q,
  window kv, compressed rows, indexer q and indexer k of that layer [S3][S7].
- Reference YaRN (frequency interpolation only, **no magnitude/mscale factor**) [S3]:
  `f_i = theta^(-2i/64)`, i = 0..31; `cd(r) = 64*ln(65536/(2*pi*r)) / (2*ln theta)`;
  `low = max(floor(cd(32)),0)`, `high = min(ceil(cd(1)), 63)`;
  `ramp_i = clamp((i-low)/max(high-low,1e-3), 0, 1)`; `f_i = f_i/16*ramp_i + f_i*(1-ramp_i)`; angle = pos*f_i.
  llama.cpp passes `attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l)` [S7]; body not seen - GUESS: it
  cancels ggml's built-in YaRN mscale so the result matches the reference (no scaling).
- Rotated dims: the **last 64** of the 512 (dims 448..511), interleaved adjacent pairs (x[2i], x[2i+1]) as complex
  numbers [S3] (GPT-J / ggml "NORM" style; llama.cpp's rope_type for the arch was not seen). For the indexer (128-wide),
  the last 64 dims are rotated [S3][S7].

Steps:
1. `qr = q_a_norm(attn_q_a @ x)` [1280]; `q = attn_q_b @ qr` -> [64 heads, 512]; RoPE on q[..., 448:512] at `pos`.
   **No per-head q RMS norm in V4.1** (V4 has a weightless one) [S3][S7][S8].
2. `kv = kv_a_norm(attn_kv @ x)` [512]; RoPE on kv[448:512] at `pos`. This one vector is both K and V [S3][S7].
   Write it into this layer's own window cache (ring of 128; slot `pos % 128`). Every layer, including consumers, has
   its own window KV [S3].
3. Window keys for query at `pos`: positions max(0,pos-127)..pos (128 incl. current) [S3].
4. Ratio 0 layers: attend over window only (go to step 9).
5. Compressor (KV-source layers only: 2, 8, 14, 20), input = the same normed `x`:
   - ratio 1 (layer 20): `latent = compressor_norm(compressor_kv @ x)`; one row per token, no gate [S3].
   - ratio 2 (layers 2, 8, 14): `v_t = compressor_kv @ x_t`, `s_t = compressor_gate @ x_t` (f32) for the two tokens of a
     group; per channel d: `latent_d = sum_t softmax_t(s_t,d) * v_t,d`; then `compressor_norm`. Groups are
     non-overlapping: tokens (2j, 2j+1). In decode the row is emitted when `(pos+1) % 2 == 0`; the first token's
     kv/score wait in a 2-slot state [S3][S7].
   - Index key (from the latent **before** RoPE) [S7]: `ik = indexer_k_norm(indexer_attn_k @ latent)` [128], RoPE on
     ik[64:128] at the group position.
   - RoPE the latent's last 64 dims at the group position `j*ratio` (the group's first token) [S3][S7].
   - Append latent (and ik) to the shared compressed cache. Row j becomes visible to query `s` iff `j < floor((s+1)/ratio)`
     (the group is complete) [S3]. For ratio 1 this includes the current token's own row.
6. Indexer (index-source layers only: 2, 8, 14, 20, 24, 28, 32, 36) [S3][S7]:
   - `iq = indexer_attn_q_b @ qr` -> [32, 128]; RoPE on iq[..., 64:128] at `pos`.
   - `w = indexer_proj @ x` [32] `* (1/sqrt(128)) * (1/sqrt(32))` (llama.cpp: `1/sqrt(n_idx_embd*n_idx_head)`).
   - Keys: `ik` rows of this layer's KV source (e.g. layer 24 scores against layer 20's index keys).
   - `score[t] = sum_h w[h] * relu(iq[h] . ik[t])` over visible rows t; invisible rows -inf.
   - Candidate pool (two levels, ratio-1 decoder only) [S3][S7]:
     - At the candidate source (layer 20): block score = max of `score` over blocks of 8 rows (padded with -inf);
       force the newest block (the one containing the query's newest visible row) to +inf (llama.cpp: +1e30); keep the
       top 2048 blocks (drop -inf blocks). Layer 20 then takes its own top-k from its **unmasked** scores.
     - At later index layers (24, 28, 32, 36): `score` outside the kept blocks set to -inf, then top-k.
   - Top-k: `k = min(512, n_visible)`; reference sorts the selected indices; invalid slots = -1 [S3]; llama.cpp
     `ggml_top_k`, rows sorted before the gather [S6][S7].
7. Consumers (not index sources) reuse the top-k indices of their index source from the same forward step; layers
   that are not KV sources read their KV source's compressed cache and run no compressor [S3][S7].
8. Keys/values = `[window rows (<=128) ; selected compressed rows (<=512)]` (window first) [S3][S7].
9. Attention (one KV head shared by 64 q heads), scale `1/sqrt(512)` [S3][S7]:
   `p_t = exp(q.k_t*scale - m)`, `out = sum_t p_t v_t / (sum_t p_t + exp(sink_h - m))` - the sink adds only to the
   denominator [S5]; masked / -1 slots contribute nothing; an all-masked row gives 0 [S5]. V = K (the 512-vector incl.
   its roped dims).
10. Output: inverse RoPE on out[..., 448:512] at the query position (`ggml_rope_ext_back`, same params) [S3][S7];
    reshape [64*512] into 8 groups of 4096 (heads 8g..8g+7); `oa_g = wo_a[g] @ out_g` (4096 -> 1024); concat -> 8192;
    `y = attn_output_b @ oa` -> 5120 [S3][S7].

### 3.6 MoE (every layer) [S3][S7][S10]
Input `h = ffn_norm(hc_pre(X, a_pre))`.
1. `z = ffn_gate_inp @ h` (f32 accumulate) [384]; `s = sqrt(softplus(z))` (llama.cpp softplus via log1p to avoid
   underflow [S6]); reference `gate_temp = 1.0`.
2. Selection: `idx = top6(s + b)`, `b = exp_probs_b` (or `exp_probs_b_vl` for image tokens). Flat top-k, **no expert
   groups**, no hash routing [S3].
3. Weights: `w = s[idx]` (unbiased); `w /= (sum(w) + 1e-20)`; `w *= 1.5` [S3][S10].
4. Expert e (routed and shared): `g = W1 h`, `u = W3 h` (f32); `u = clamp(u, -10, 10)`; `g = min(g, 10)`;
   `a = silu(g) * u`; `out = W2 a` [S3]. llama.cpp: fused `ggml_swiglu_clamp` for this arch (internals not seen); its
   unfused fallback clamps `silu(gate)` from above instead of `gate` (see 4) [S10].
5. `y = sum_k w_k * expert_{idx_k}(h) + shared(h)` (shared expert unweighted, also clamped) [S3][S7].

### 3.7 Final norm and head
`h = hc_pre(X, f_pre_of_layer_39)`; `logits = output @ output_norm(h)` (reference: f32 head, last position only) [S3][S7].

### 3.8 Prompt (prefill) vs one token
Same math. Differences are only in bookkeeping [S3][S4]:
- Engram hashes are computed for all prompt tokens from the token history (lookback across the batch and the cache).
- Window: each query sees its own last 128 positions (a causal band mask in prefill; a 128-slot ring in decode).
- Compressor: prefill emits `floor(S/ratio)` rows and keeps the remainder in the ratio state; decode emits on group
  completion. Each query's visible compressed rows obey `j < floor((s+1)/ratio)`.
- Indexer / candidate / top-k per query row (llama.cpp chunks queries to bound the score tensor [S7]).

### 3.9 MTP / DSpark
Not part of the main graph: llama.cpp rejects the MTP graph type for `deepseek41` and loads the DSpark drafter as a
separate GGUF (`dflash` arch, `--dspark` at conversion) [S6][S7][S11]. Facts for a later port [S3][S11]: 3 draft blocks
(layer ids 40..42, window-only attention, MoE with 128 experts / 3 active), block size 5, noise/mask token 128799,
input features = mean over the 4 streams of the hidden state entering layers 37, 38, 39 (after Engram), projected by
`main_proj` (3E -> E) + `main_norm`; final stage has a Markov head (rank 256) and an f32 confidence head; embedding and
LM head shared with the main model.

---

## 4. Where llama.cpp differs from the reference, and open questions

Differences found:
1. **Simulated quantisation.** The reference quant-dequantises in the forward pass: window KV fp8 e4m3 (block 32, ue8m0
   scales), compressed latent fp4 (block 16, e4m3 scale), indexer q/k fp4 (block 32), attention output fp8 before
   `wo_a`, linear inputs inside `linear()` [S3][S5]. llama.cpp does not simulate this (optional Hadamard rotation only
   when the K cache is quantised) [S7]. The PR author measured mean KL 4.8e-3 (wiki) / 2.8e-3 (17k ctx) for a
   near-lossless GGUF against the f32 reference, while the reference's own native kernels gave 2.6e-2 / 1.5e-2 [S14].
   So an f32 reference without the fp8/fp4 simulation is the right KL target.
2. **SwiGLU clamp placement.** Reference clamps the gate pre-activation: `silu(min(g,10))`. llama.cpp's unfused fallback
   does `min(silu(g),10)`; for this arch it uses fused `ggml_swiglu_clamp`, whose exact semantics were not read [S3][S10].
   The difference is tiny (silu(10) = 9.99955 vs 10) but not bit-identical.
3. **Candidate pin** +1e30 vs +inf [S3][S7] (same effect).
4. Router weight normalisation: llama.cpp adds 1e-20 to the sum for this arch only (matches the reference); other archs
   clamp the sum at 6.1e-5 [S10].
5. Softplus in llama.cpp changed to `log1pf(expf(x))` [S6].
6. Known bug: issue #30944 "single-token output corruption when a cached prompt prefix is reused (Engram n-gram
   history)" [S14] - the Engram history must be rebuilt for reused prefixes.
7. `LLAMA_DSV41_SPARSE=0` selects a dense reference path (attend to all compressed rows with a top-k mask) [S7].

Open questions:
- Exact `dsv4_rope_attn_factor` body (YaRN mscale cancellation assumed) and the arch's ggml rope type. The reference is
  unambiguous (no mscale, interleaved pairs, last 64 dims); implement that and compare logits.
- How the llama.cpp loader handles the 43-entry `compress_ratios` (presumably first `n_layer` entries).
- Exact GGUF values in the ISTA files (read with `gguf-dump`): especially `expert_gating_func` integer,
  `hash_layer_count`, `hyper_connection.single_pass`, rope scaling keys, and whether `swiglu_clamp_*` are arrays of 40.
- Whether the reference indexer key uses the pre-RoPE latent (llama.cpp does [S7]; the reference text read did not say).
- The tech report PDF [S1 repo] was not read; the card describes FP4 main-KV caching (E2M1, one E4M3 scale per 16
  channels, "890 bytes per token") and "SWA bounded replay" as deployment features [S15].
- Per-layer routed-expert quant types in the ISTA build came through a summariser table; check against the allocation
  file [S19] before relying on them.

---

## Sources
- [S1] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json (and repo tree https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main)
- [S2] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/config.json
- [S3] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py
- [S4] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/engram.py
- [S5] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/kernel.py
- [S6] https://github.com/ggml-org/llama.cpp/pull/31086 (merged 2026-10-25, commit 74bca39; the upstream support)
- [S7] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/deepseek41.cpp
- [S8] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/deepseek4.cpp
- [S9] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp
- [S10] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-graph.cpp
- [S11] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/conversion/deepseek41.py
- [S12] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/conversion/deepseek.py (class DeepseekV4Model)
- [S13] https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/deepseek_v41/config and .../deepseek_v41/model (config defaults, block structure)
- [S14] https://github.com/ggml-org/llama.cpp/pull/30817 (earlier closed PR by danielhanchen; KL numbers, issue #30944)
- [S15] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/README.md
- [S16] https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (model card: b11870+, file layout, Engram IQ4_NL shard)
- [S17] https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/main
- [S18] https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/raw/main/convert_ds41_upstream.py
- [S19] https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/raw/main/tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.rco-allocation.txt
- [S20] https://github.com/ggml-org/llama.cpp/tree/master/conversion and https://github.com/ggml-org/llama.cpp/tree/master/src/models (file listings)
