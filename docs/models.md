# Supported models

(Moved from the README on 2026-10-31.)


| GGUF architecture | Models | Notes |
|---|---|---|
| `qwen35` | Qwen3.5 (0.8B–27B dense), Qwen3.6, Qwen3.8 27B | hybrid Gated DeltaNet + gated attention; MTP speculative decoding; image input with a Qwen3-VL `mmproj` next to the model (encoded on the GPU; Qwen3.5 4B and 9B, Qwen3.8 27B) |
| `qwen35moe` | Qwen3.6-35B-A3B: standard GGUFs (e.g. Unsloth UD-IQ4_XS / UD-Q3_K_M) or ISTA-DASLab GSQ 2-bit | mixture of experts (256 experts, 8 active); standard GGUF experts (IQ / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0) load as they are; GSQ converted from the safetensors checkpoint with `tools/convert_gsq.py`; Unsloth's MTP repository adds speculative decoding and its `mmproj` image input (arcinferzero: both together; the automatic context reserves 128K tokens of context, then sizes the expert cache: 131K tokens with ~80 % of the UD-IQ4_XS experts cached; up to 262K with `--ctx`, with a smaller cache; context sweep in docs/speed-progress.md) |
| `gemma4` | Gemma 4 12B, Gemma 4 E4B, Gemma 4 26B-A4B (Google's QAT Q4_0 GGUFs) | sliding-window + global attention, per-layer embeddings (E4B); image and audio input (`mmproj`: 12B gemma4uv / gemma4ua, E4B gemma4v ViT and gemma4a conformer, 26B gemma4v with 72-dim heads, on the GPU); speculative decoding with the Gemma 4 assistant drafters (`mtp-*.gguf`; 12B, E4B, 26B). 26B-A4B: a dense FFN beside 128 routed experts (8 active, GELU, f32 router) streamed from the SSD like the 35B's. Accuracy: float attention (f32 q / K, f16 V; E4B's global layers in prompts read an int8 copy of K / V, `ARC_G4_ATT8=0`: float); 26B and 12B run their matmuls on two int8 passes (value and rounding residual: KL 0.0042 / 0.0002), E4B on one (KL 0.00088) |
| `qwen4exp` | Qwen3.8 Flash Next (ISTA-DASLab GSQ-RCO IQ3_S, GGUF in 2 parts) | **arcinferzero only**; 4 hyper-connection streams, Gated DeltaNet + gated attention with QSA sparse attention, 512 experts (10 active) streamed from the SSD, PLE n-gram table read from part 2 per token; no MTP block; images with an `mmproj-*` next to it (bartowski/Qwen3.8-Flash-Next-GGUF `mmproj-Qwen3.8-Flash-Next-f16.gguf`, 0.9 GB) (incoai/Qwen3.8-Flash-Next-DFlash2 drafts with `--dflash DIR`: same text, ~79 % accepted, no faster on this laptop's SSD: QUEUE F21.16); runs up to 262,144 tokens, accuracy measured up to a 3.5K-token prompt (KL 0.0143 / 0.0147 (363 tokens); 0.0186 / 0.0177 at 3.5K (QSA); `ARC_FN_PRECISE=1` 0.0127 / 0.0132) |

Weight types: `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_XS` (mixed per tensor, e.g. Unsloth dynamic or ISTA-DASLab GSQ-RCO quants), plus `F32/F16/BF16` small matrices;
GSQ 2-bit experts (`tools/convert_gsq.py`); `Q4_0` drafter matrices (requantized to Q8_0 at load).


## About the results table

The README's results at a glance, measured on a Core Ultra 5 125H:

Gemma 26B / 12B rows measured on 2026-10-23 with the laptop on a cooling pad, E4B without it (two runs each
within 1-2 %; a hot laptop measured up to 25 % slower on long prompts). Images: the Gemma 4 26B encoder takes 7.4 s
for a 1,064-token image (12.7 s before 2026-10-24), E4B 3.1 s, Qwen3.5 9B / Qwen3.8 27B ~2.8 s; since 2026-10-27
the Qwen encoders of the dense models run their matrix products on two int8 passes, 22-30 % faster (a 768-token
image 27B 1.26-1.35 s, 9B 1.28-1.30, 4B 0.96-0.97; the 35B and Flash Next keep the f16 encoder).
KL is the divergence of the next-token distributions from the f32 reference (0 = identical); run-to-run int8
rounding noise is ~0.0003. The limit kept for every model and change is KL < 0.015. Flash Next's default meets it on short prompts and
is just above it at 3.5K; its precise mode (`arc-server -Precise`, `--precise`; BF16 hyper-connection rows and router, two int8 passes everywhere)
stays under it at about half the prompt speed, opt-in (QUEUE.md item "Flash Next KL"). How each number was measured, every change with its before and after, and the
consistency checks (56 of 56 regression runs bit-identical): **[docs/before-after.md](before-after.md)**.
