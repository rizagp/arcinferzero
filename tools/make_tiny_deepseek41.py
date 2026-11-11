# A small DeepSeek-V4.1-Flash (`deepseek41`) GGUF with the real layout and random weights, for building and checking
# the engine's path against tools/ref_forward without the 231 GB model (docs/deepseek41-spec.md).
#
# The same keys and tensor names as the ISTA-DASLab GGUFs, smaller sizes: 10 layers covering every layer role of the
# real 40 (window only; ratio 2 with its own compressor and indexer; a ratio 2 consumer; ratio 1 with the candidate
# pool; a later index layer that uses the pool; consumers of each), two Engram layers (one a KV source, as 14 is), a
# sliding window of 16 so that a ~100-token prompt passes every case (window wrap, more compressed rows than top_k,
# more blocks than the pool keeps). Head size 512 with 64 rotated dimensions and the 128-wide indexer stay as in the
# real model, since kernels may be written for them.
#
# --real: the real model's widths instead (embedding 5120, 64 heads, q LoRA 1280, 8 output groups of 1024, window 128,
# top-k 512, a 2048-block candidate pool of 8 rows, 32 indexer heads, 384 experts with 6 active, Engram rows of 256),
# so that kernels and buffers meet their real sizes; but 4 layers (window only; ratio 2 with Engram; ratio 1 with the
# candidate pool; an index layer using it), an expert FFN of 256 instead of 2304 and a vocabulary of 2048 to keep the
# file at ~3.5 GB (experts in Q4_0). Its data are written tensor by tensor.
#
# usage: python tools/make_tiny_deepseek41.py OUT.gguf [--seed N] [--nu N] [--real]
import struct, sys
import numpy as np

ARCH = "deepseek41"
E, NL, H, D, R, QL, G, OL = 256, 10, 8, 512, 64, 256, 2, 128
SWA, TOPK, IH, ID = 16, 8, 16, 128
NE, NU, FF, NSH = 16, 4, 256, 1
HC, PD, NGRAM, HPN = 4, 64, 4, 8
VOCAB = 512
RATIOS = [0, 0, 2, 2, 2, 2, 1, 1, 1, 1]
KV_SRC = [2, 4, 6]
IDX_SRC = [2, 4, 6, 8]
CAND_SRC, CAND_BLOCKS, CAND_BS = 6, 4, 4
PLE_LAYERS = [1, 4]

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_BF16 = 0, 1, 2, 8, 30


def real_config():
    global E, NL, H, D, R, QL, G, OL, SWA, TOPK, IH, ID, NE, NU, FF, NSH, HC, PD, NGRAM, HPN, VOCAB
    global RATIOS, KV_SRC, IDX_SRC, CAND_SRC, CAND_BLOCKS, CAND_BS, PLE_LAYERS
    E, NL, H, D, R, QL, G, OL = 5120, 4, 64, 512, 64, 1280, 8, 1024
    SWA, TOPK, IH, ID = 128, 512, 32, 128
    NE, NU, FF, NSH = 384, 6, 256, 1
    HC, PD, NGRAM, HPN = 4, 256, 4, 8
    VOCAB = 2048
    RATIOS = [0, 2, 1, 1]
    KV_SRC = [1, 2]
    IDX_SRC = [1, 2, 3]
    CAND_SRC, CAND_BLOCKS, CAND_BS = 2, 2048, 8
    PLE_LAYERS = [0, 1]


def encode(data, ggml_type):
    if ggml_type == GGML_F32:
        return data.astype("<f4").tobytes()
    if ggml_type == GGML_F16:
        return data.astype("<f2").tobytes()
    if ggml_type == GGML_BF16:
        u = data.astype("<f4").view("<u4")
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()  # round to nearest even
    if ggml_type == GGML_Q8_0:
        x = data.astype("<f4").reshape(-1, 32)
        amax = np.abs(x).max(axis=1)
        d = (amax / 127.0).astype("<f2")
        df = d.astype("<f4")
        inv = np.where(df > 0, 1.0 / np.where(df > 0, df, 1), 0)
        q = np.clip(np.rint(x * inv[:, None]), -127, 127).astype("<i1")
        blk = np.zeros(x.shape[0], dtype=[("d", "<f2"), ("q", "<i1", 32)])
        blk["d"], blk["q"] = d, q
        return blk.tobytes()
    if ggml_type == GGML_Q4_0:  # (as ggml: d = the value of largest magnitude / -8, q = x / d + 8 in 0..15)
        x = data.astype("<f4").reshape(-1, 32)
        mx = x[np.arange(x.shape[0]), np.abs(x).argmax(axis=1)]
        d = (mx / -8.0).astype("<f2")
        df = d.astype("<f4")
        inv = np.where(df != 0, 1.0 / np.where(df != 0, df, 1), 0)
        q = np.clip(np.trunc(x * inv[:, None] + 8.5), 0, 15).astype("<u1")
        blk = np.zeros(x.shape[0], dtype=[("d", "<f2"), ("q", "<u1", 16)])
        blk["d"], blk["q"] = d, q[:, :16] | (q[:, 16:] << 4)
        return blk.tobytes()
    raise ValueError(ggml_type)


def type_bytes(n, ggml_type):
    return {GGML_F32: 4 * n, GGML_F16: 2 * n, GGML_BF16: 2 * n, GGML_Q8_0: n // 32 * 34, GGML_Q4_0: n // 32 * 18}[ggml_type]


class Writer:
    def __init__(self):
        self.kv = []
        self.tensors = []

    def add(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def u32(self, k, v): self.add(k, 4, int(v))
    def i32(self, k, v): self.add(k, 5, int(v))
    def f32(self, k, v): self.add(k, 6, float(v))
    def bool(self, k, v): self.add(k, 7, bool(v))
    def str(self, k, v): self.add(k, 8, v)
    def arr(self, k, etype, vals): self.add(k, 9, (etype, list(vals)))

    def tensor(self, name, data, ggml_type, shape=None):
        # data: numpy array in row-major [.., ne1, ne0] (ne0 = the contiguous dimension, as ggml writes it), or with
        # `shape` a function giving the encoded bytes piece by piece at write time (big tensors)
        if shape is None:
            raw = encode(data, ggml_type)
            self.tensors.append((name, list(data.shape[::-1]), ggml_type, len(raw), lambda raw=raw: [raw]))
        else:
            n = int(np.prod(shape))
            self.tensors.append((name, list(shape[::-1]), ggml_type, type_bytes(n, ggml_type), data))

    @staticmethod
    def s(b):
        b = b.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def value(self, vtype, v):
        if vtype == 4: return struct.pack("<I", v)
        if vtype == 5: return struct.pack("<i", v)
        if vtype == 6: return struct.pack("<f", v)
        if vtype == 7: return struct.pack("<B", 1 if v else 0)
        if vtype == 8: return self.s(v)
        if vtype == 10: return struct.pack("<Q", v)
        if vtype == 9:
            et, vals = v
            return struct.pack("<IQ", et, len(vals)) + b"".join(self.value(et, x) for x in vals)
        raise ValueError(vtype)

    def write(self, path):
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
        for k, t, v in self.kv:
            out += self.s(k) + struct.pack("<I", t) + self.value(t, v)
        off = 0
        for name, dims, ty, nb, _ in self.tensors:
            out += self.s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            out += struct.pack("<IQ", ty, off)
            off = (off + nb + 31) // 32 * 32
        out += b"\0" * ((32 - len(out) % 32) % 32)
        with open(path, "wb") as f:
            f.write(out)
            for name, dims, ty, nb, gen in self.tensors:
                got = 0
                for piece in gen():
                    f.write(piece)
                    got += len(piece)
                if got != nb:
                    raise RuntimeError(f"{name}: {got} bytes written, {nb} expected")
                f.write(b"\0" * ((32 - nb % 32) % 32))


def primes_from(start, n, used):
    def isprime(x):
        if x < 2: return False
        i = 2
        while i * i <= x:
            if x % i == 0: return False
            i += 1
        return True
    out, x = [], start
    while len(out) < n:
        if isprime(x) and x not in used:
            out.append(x)
            used.add(x)
        x += 1
    return out


def main():
    out = sys.argv[1]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 41
    if "--real" in sys.argv:
        real_config()
    global NU  # (--nu N: experts per token; --nu 16 = every expert, no selection to flip on near-ties)
    if "--nu" in sys.argv:
        NU = int(sys.argv[sys.argv.index("--nu") + 1])
    rng = np.random.default_rng(seed)
    w = Writer()
    p = ARCH + "."

    def mat(n_out, n_in, scale=1.0):  # a matmul weight [n_out][n_in] (ne0 = n_in), outputs of about unit size
        return (rng.standard_normal((n_out, n_in)) * scale / np.sqrt(n_in)).astype(np.float32)

    def normw(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    w.str("general.architecture", ARCH)
    w.str("general.name", "DeepSeek-V4.1-Flash tiny (random weights, inferzero test)")
    w.u32("general.alignment", 32)
    w.u32(p + "block_count", NL)
    w.u32(p + "context_length", 4096)
    w.u32(p + "embedding_length", E)
    w.u32(p + "attention.head_count", H)
    w.u32(p + "attention.head_count_kv", 1)
    w.u32(p + "attention.key_length", D)
    w.u32(p + "attention.value_length", D)
    w.u32(p + "rope.dimension_count", R)
    w.f32(p + "rope.freq_base", 10000.0)
    w.str(p + "rope.scaling.type", "yarn")
    w.f32(p + "rope.scaling.factor", 16.0)
    w.u32(p + "rope.scaling.original_context_length", 65536)
    w.f32(p + "rope.scaling.yarn_beta_fast", 32.0)
    w.f32(p + "rope.scaling.yarn_beta_slow", 1.0)
    w.f32(p + "attention.layer_norm_rms_epsilon", 1e-20)
    w.u32(p + "attention.q_lora_rank", QL)
    w.u32(p + "attention.sliding_window", SWA)
    w.u32(p + "expert_count", NE)
    w.u32(p + "expert_used_count", NU)
    w.u32(p + "expert_feed_forward_length", FF)
    w.u32(p + "expert_shared_count", NSH)
    w.f32(p + "expert_weights_scale", 1.5)
    w.bool(p + "expert_weights_norm", True)
    w.arr(p + "swiglu_clamp_exp", 6, [10.0] * NL)
    w.arr(p + "swiglu_clamp_shexp", 6, [10.0] * NL)
    w.u32(p + "attention.output_group_count", G)
    w.u32(p + "attention.output_lora_rank", OL)
    w.arr(p + "attention.compress_ratios", 4, RATIOS + [0, 0, 0])  # (as the real file: the 3 drafter layers' zeros)
    w.f32(p + "attention.compress_rope_freq_base", 160000.0)
    w.u32(p + "attention.indexer.head_count", IH)
    w.u32(p + "attention.indexer.key_length", ID)
    w.u32(p + "attention.indexer.top_k", TOPK)
    w.arr(p + "attention.kv_source_layers", 4, KV_SRC)
    w.arr(p + "attention.indexer.source_layers", 4, IDX_SRC)
    w.i32(p + "attention.indexer.candidate_source_layer", CAND_SRC)
    w.u32(p + "attention.indexer.candidate_top_k_blocks", CAND_BLOCKS)
    w.u32(p + "attention.indexer.candidate_block_size", CAND_BS)
    w.u32(p + "hyper_connection.count", HC)
    w.u32(p + "hyper_connection.sinkhorn_iterations", 20)
    w.f32(p + "hyper_connection.epsilon", 1e-6)
    w.u32(p + "vocab_size", VOCAB)

    # Engram: hash constants as the converter makes them, scaled down (primes from 101 upward, no prime reused)
    ncol = (NGRAM - 1) * HPN
    mults, primes, offsets = [], [], []
    used, base = set(), 0
    for li, layer in enumerate(PLE_LAYERS):
        r2 = np.random.default_rng(10007 * layer)
        bound = max(1, ((2**63 - 1) // 99092) // 2)
        mults += [int(2 * v + 1) for v in r2.integers(0, bound, size=NGRAM)]
        pr = primes_from(101 + 40 * li, ncol, used)
        for q in pr:
            offsets.append(base)
            base += q
        primes += pr
    n_rows = base
    token_map = [(i * 7) % (VOCAB - 100) for i in range(VOCAB)]  # (several ids share a compressed id, as the real map)
    w.arr(p + "ple.layers", 4, PLE_LAYERS)
    w.u32(p + "ple.ngram_size", NGRAM)
    w.u32(p + "ple.heads_per_ngram", HPN)
    w.u32(p + "embedding_length_per_layer_input", PD)
    w.arr(p + "ple.layer_multipliers", 10, mults)
    w.arr(p + "ple.head_offsets", 4, offsets)
    w.arr(p + "ple.head_vocab_sizes", 4, primes)
    w.u32(p + "engram.pad_token_id", 2)
    w.arr(p + "engram.token_map", 5, token_map)

    # a token list so that tokenizers load (ids are fed directly)
    w.str("tokenizer.ggml.model", "gpt2")
    w.arr("tokenizer.ggml.tokens", 8, ["<bos>", "<eos>", "<pad>"] + [f"t{i}" for i in range(3, VOCAB)])
    w.arr("tokenizer.ggml.token_type", 5, [3, 3, 3] + [1] * (VOCAB - 3))
    w.arr("tokenizer.ggml.merges", 8, [])
    w.u32("tokenizer.ggml.bos_token_id", 0)
    w.u32("tokenizer.ggml.eos_token_id", 1)
    w.u32("tokenizer.ggml.padding_token_id", 2)

    Q8, F32, BF16, F16 = GGML_Q8_0, GGML_F32, GGML_BF16, GGML_F16
    w.tensor("token_embd.weight", rng.standard_normal((VOCAB, E)).astype(np.float32), Q8)
    w.tensor("output_norm.weight", normw(E), F32)
    w.tensor("output.weight", mat(VOCAB, E, 1.5), Q8)
    w.tensor("per_layer_token_embd.weight", rng.standard_normal((n_rows, PD)).astype(np.float32), F16)
    hcd, mix = HC * E, (2 + HC) * HC
    for il in range(NL):
        b = f"blk.{il}."
        w.tensor(b + "attn_norm.weight", normw(E), F32)
        w.tensor(b + "attn_q_a.weight", mat(QL, E), Q8)
        w.tensor(b + "attn_q_a_norm.weight", normw(QL), F32)
        w.tensor(b + "attn_q_b.weight", mat(H * D, QL), Q8)
        w.tensor(b + "attn_kv.weight", mat(D, E), Q8)
        w.tensor(b + "attn_kv_a_norm.weight", normw(D), F32)
        w.tensor(b + "attn_sinks.weight", rng.standard_normal(H).astype(np.float32), F32)
        w.tensor(b + "attn_output_a.weight", mat(G * OL, H * D // G), Q8)
        w.tensor(b + "attn_output_b.weight", mat(E, G * OL), Q8)
        for site in ("attn", "ffn"):
            w.tensor(b + f"hc_{site}_fn.weight", mat(mix, hcd, 2.0), F32)
            w.tensor(b + f"hc_{site}_base.weight", (0.5 * rng.standard_normal(mix)).astype(np.float32), F32)
            w.tensor(b + f"hc_{site}_scale.weight", (1.0 + 0.2 * rng.standard_normal(3)).astype(np.float32), F32)
        w.tensor(b + "ffn_norm.weight", normw(E), F32)
        w.tensor(b + "ffn_gate_inp.weight", mat(NE, E, 8.0), BF16)  # (decisive: fewer near-ties)
        w.tensor(b + "exp_probs_b.bias", (0.1 * rng.standard_normal(NE)).astype(np.float32), F32)
        if "--real" in sys.argv:  # (per expert at write time, Q4_0: 384 experts of 5120 x 256)
            def gen_exps(n_out, n_in, scale, seed_e):
                def g():
                    r2 = np.random.default_rng(seed_e)
                    for e in range(NE):
                        yield encode((r2.standard_normal((n_out, n_in)) * scale / np.sqrt(n_in)).astype(np.float32), GGML_Q4_0)
                return g
            sd = int(rng.integers(1 << 30))
            w.tensor(b + "ffn_gate_exps.weight", gen_exps(FF, E, 3.0, sd), GGML_Q4_0, shape=(NE, FF, E))
            w.tensor(b + "ffn_up_exps.weight", gen_exps(FF, E, 3.0, sd + 1), GGML_Q4_0, shape=(NE, FF, E))
            w.tensor(b + "ffn_down_exps.weight", gen_exps(E, FF, 1.0, sd + 2), GGML_Q4_0, shape=(NE, E, FF))
        else:
            w.tensor(b + "ffn_gate_exps.weight", (rng.standard_normal((NE, FF, E)) * 3 / np.sqrt(E)).astype(np.float32), Q8)
            w.tensor(b + "ffn_up_exps.weight", (rng.standard_normal((NE, FF, E)) * 3 / np.sqrt(E)).astype(np.float32), Q8)
            w.tensor(b + "ffn_down_exps.weight", (rng.standard_normal((NE, E, FF)) / np.sqrt(FF)).astype(np.float32), Q8)
        w.tensor(b + "ffn_gate_shexp.weight", mat(FF * NSH, E, 3.0), Q8)
        w.tensor(b + "ffn_up_shexp.weight", mat(FF * NSH, E, 3.0), Q8)
        w.tensor(b + "ffn_down_shexp.weight", mat(E, FF * NSH), Q8)
        if il in KV_SRC:
            w.tensor(b + "attn_compressor_kv.weight", mat(D, E), BF16)
            if RATIOS[il] > 1:
                w.tensor(b + "attn_compressor_gate.weight", mat(D, E, 2.0), BF16)
            w.tensor(b + "attn_compressor_norm.weight", normw(D), F32)
            w.tensor(b + "indexer.attn_k.weight", mat(ID, D), BF16)
            w.tensor(b + "indexer.k_norm.weight", normw(ID), F32)
        if il in IDX_SRC:
            w.tensor(b + "indexer.attn_q_b.weight", mat(IH * ID, QL), BF16)
            w.tensor(b + "indexer.proj.weight", mat(IH, E, 4.0), BF16)
        if il in PLE_LAYERS:
            w.tensor(b + "engram_kv.weight", mat(E * (HC + 1), ncol * PD), Q8)
            w.tensor(b + "engram_gate.weight", normw(HC * E).reshape(HC, E), F32)
    w.write(out)
    print(f"{out}: {len(w.tensors)} tensors, {len(w.kv)} keys, Engram rows {n_rows}")


if __name__ == "__main__":
    main()
