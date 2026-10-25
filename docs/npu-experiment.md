# The NPU next to the GPU for prompts (`--npu on`)

The Meteor Lake NPU ("Intel AI Boost", NPU 3720) computes part of some matrix products of a prompt chunk while the
GPU computes the rest. Everything goes through Level Zero (the NPU's driver is a second Level Zero driver with the
graph extension); there is no OpenVINO. Off by default; with it off the engine gives the same bits as before it
existed (built on the branch npu-experiment, merged into `main` on 2026-10-25).

## Turning it on

| Where | On | Off |
|---|---|---|
| `bin\serverz.exe`, `bin\runz.exe` | `--npu on` (`--npu force`: also on models where it does not pay off) | `--npu off` or nothing |
| `arc-server start` | `-Npu on` | `-Npu off`, `arc-server restart`, or a start after `stop` |
| `pi-arc` | `--npu on` | `--npu off` |
| environment (tests) | `ARC_NPU=1` (`2`: force), used when no `--npu` is given | |

The setting belongs to the server, not to a model: a request for another model (Pi's `/model`, an OpenAI client's
`model` field) loads it with the same setting, and so does `pi-arc` / `arc-server start` on another model while the
server runs. Models where the NPU gains nothing (below) say so in the log and run without it; switching back to the
27B uses it again. Only a restart (or a stop) of the server turns it off.

## Result

Qwen3.8 27B (agentionai Qwen3.8-27B-AP IQ4_XS), code prompts, this laptop (thermals not ideal, runs alternated):

| Prompt | off | `--npu on` | KL vs f32 (128 tokens) |
|---|---|---|---|
| 4,096 tokens | 132-140 tok/s | **153-162 tok/s (+15-20 %)** | 0.00066 -> **0.0028-0.0031** |
| 11,240 tokens | 133.8-134.0 tok/s | **154.7 tok/s (+15.6 %)** | |

Benchmark (`bench/run_bench.ps1 -Engines arcinferzero,arcinferzero-npu,arcinferzero`, 2026-10-25, off measured
before and after the NPU round; results in `bench/results/npu-2026-10-25.jsonl`):

| 27B-AP IQ4_XS | off | `--npu on` |
|---|---|---|
| pp512 (one 512-row chunk: no NPU) | 94.7-100.2 tok/s | 97.7 |
| pp2048 | 125.9-133.5 tok/s | **153.8 (+15 %)** |
| 11,240-token code prompt, time to first token | 84.7-86.2 s (130-133 tok/s) | **73.7 s (152.6 tok/s, +15 %)** |
| 33,937-token code prompt, time to first token | 292.1-295.3 s (115-116 tok/s) | **258.4 s (131.3 tok/s, +13 %)** |
| generation (tg64; 32 tokens after the 11K / 34K prompt) | 3.67-4.04; 3.76-3.79; 3.65-3.68 tok/s | 3.91; 3.83; 3.46 (unchanged: noise and heat) |

- Off: logits bit-identical to the engine without the NPU code (27B and Qwen3.6 35B, prompt and generation).
- On: bit-identical run to run; MTP and DFlash2 drafting give the same logits as plain decoding (with the NPU on
  both); the MTP block's prompt pass uses the NPU too. Generation is unchanged (it never uses the NPU).
- Not worth it on models whose W8 weights stay resident on the GPU, measured: Qwen3.5 4B +2 % (noise), 9B +4-6 % with
  KL 0.008-0.013 (over the 0.01 target), Qwen3.6 35B-A3B +0.5 %. `--npu on` leaves them off (the log says why);
  `--npu force` forces it. Gemma 4 (E4B) was not built: small and resident like the 4B. Other architectures: off.
- Cost: 0.37 GiB of system RAM outside the GPU budget for the 27B (the shared block) and 33 MiB of GPU staging. The
  NPU's weight rows are not kept: per layer the GPU decodes them from its own W8 form into the shared block
  (`npu_w8rows`, once per layer and prompt segment; until 2026-10-25 a resident copy took 5.8 GiB more, same speed).
- The first prompt with a new chunk height (multiples of 1024 rows, 2048-4096) compiles its graphs (10-30 s each
  kind), later ones load them in milliseconds from `%LOCALAPPDATA%\arcinferzero\npu`.

## What runs where

Per prompt chunk of >= 2048 rows (`ARC_NPU_MIN`) of the 27B, per layer:

| Matrix | NPU share | NPU form | Why |
|---|---|---|---|
| `ffn_gate`, `ffn_up` | 1/3 of the columns (`ARC_NPU_SHARE`) | int8 per row ("fqi") | ~40 % of the prompt time; their input comes out of an RMS norm, so one scale per row is accurate enough |
| `ffn_down` | 27.5 % (`ARC_NPU_DOWN`) | f16 on int8 weights with one scale per 32 values ("w8g") | its input (silu(gate) * up) has large outliers: int8 per row would cost KL |
| `attn_qkv` (DeltaNet layers) | 29 % (`ARC_NPU_QKV`) | f16 ("w8g") | |

The shares are where the GPU and the NPU finish together (swept: 1/3 of gate/up 1.44x on those GEMMs, 0.30 1.39x,
0.37 1.27x; down / qkv 0.275 / 0.29 best of 0.20-0.38).

The flow of one FFN: the FFN norm also writes its f32 rows; a GPU kernel quantizes them per row into a memory block
that both Level Zero contexts map (`ZE_extension_external_memmap_sysmem`); the NPU starts its graph on its weight
rows (decoded just before by the GPU into the same block, once per layer and prompt segment) while the GPU runs the engine's GEMMs on its columns (persistent copies of the weights with a smaller N, the
same JIT decode); the GPU waits for the NPU and writes silu(gate) * up of the NPU's columns. ffn_down and attn_qkv
go the same way with f16 inputs and outputs.

## How the NPU is programmed (zero/npu/)

- `ze_npu.h`: the graph extension's declarations (written from the published layout; the header is not on this
  machine), `npu_probe` / `npu_graph_props`: the NPU is driver 1, extension 1.8, compiler 8.3, opset 11.
- `npu_ir.h`: an OpenVINO IR v11 writer. The driver compiles a serialized IR (compiler version, 2 parts, xml size +
  xml, weights size + weights; `ZE_GRAPH_FORMAT_NGRAPH_LITE`).
- `npu_rt.h`: the runtime (graph compile with a blob cache: the native binary loads only from page-aligned memory,
  `ZE_RESULT_ERROR_INVALID_NATIVE_BINARY` otherwise; external memory import with stype `0x00020037`; arguments by
  name; start / wait; teardown).
- `engine_npu.h`: the engine's part (inside `class Engine`), hooks in `engine.h` (init, setup, teardown) and
  `engine_qwen.h` (`layer_ffn`, `layer_gdn`).

What the NPU compiler does with a graph, measured (27B FFN up shape, 4096 rows):

| Graph form | Throughput |
|---|---|
| f16 MatMul | 2.0-3.1 TFLOPS (by shape) |
| int8 inputs + Convert + Multiply | 1.7 (it goes back to f16) |
| FakeQuantize with bounds given at run time | 1.2 (no int8 units) |
| **int8 inputs -> Convert -> FakeQuantize with constant bounds [-127, 127] -> [-1, 1], scales after the product** | **3.8 TOPS**, exact per-row int8 |
| weights as graph constants | +40 % over weights as inputs (not used: a second copy) |

The GPU's own int8 GEMM runs at 8.4-8.8 TOPS; both at once: GPU -3 %, NPU unchanged.

## Accuracy

`tools/ref_forward` simulates the NPU's share (`REF_NPU=f`, `REF_NPU_SEL`, `REF_NPU_F16=1`). 128-token code prompt,
KL vs f32:

| Model | engine | 1/3 of every matrix f16 | 1/3 of every matrix int8 per row | 1/2 of gate / up int8 |
|---|---|---|---|---|
| 27B | 0.00046 | 0.00030 | 0.0294 | 0.0084 |
| 4B | 0.00105 | 0.00082 | 0.0090 | 0.0022 |
| 9B | 0.00308 | 0.00138 | 0.0350 | 0.0094 |
| E4B | 0.00145 | 0.00112 | 0.0108 | 0.0020 |

Per-row int8 is only safe on inputs that come out of a norm (the activations' outliers decide it, not the weights:
per-channel weights alone cost little). The engine with `--npu on` measures 0.0028-0.0031 on the 27B (the shares above).

## Open

- The NPU's memory is not in the GPU budget (`--ctx auto` and the expert cache do not see it); the log prints it.
- A graph per chunk height (multiples of 1024): a token's result can depend on the height of its chunk (results stay
  deterministic for the same prompt).
- Faster: weights as graph constants (+40 % on the NPU, a second copy of its share), the row quantization fused into
  the FFN norm, z / ssm_out shares (~0.3 s each on a 4K prompt).
