"""f32 reference of the Gemma 4 vision encoder (mmproj gemma4v: E2B / E4B), written from the mmproj and llama.cpp's
tools/mtmd/models/gemma4v.cpp (FFN: gelu_tanh, as Google's config). Compares with the GPU encoder's output that
bin/vision_testz.exe writes (<image>.g4v.bin, and the resized pixels <image>.g4v.rgb).
usage: python gemma4v_ref.py mmproj.gguf image.png   (after: vision_testz model.gguf mmproj.gguf image.png 70)
Lower-precision simulations (environment): REF_VW=q8 weights as Q8_0 (int8 per 32 values); REF_VA=1 / 2 activations
int8 per 32 values on one / two passes (the value, then its rounding residual); REF_VATT=f16 / i8 attention q / K / V
in f16, or int8 per row with P in 8 bits; REF_VOUT=file writes the embeddings as vision_testz does (header n_tok,
n_out, 0; f32), for vision_testz ARC_VIS_EMB (the model's KL on them)."""
import os
import struct
import sys

import numpy as np

GT = {0: (np.float32, 4), 1: (np.float16, 2), 30: (np.uint16, 2)}  # F32, F16, BF16


def read_gguf(path):
    f = open(path, 'rb')
    u32 = lambda: struct.unpack('<I', f.read(4))[0]
    u64 = lambda: struct.unpack('<Q', f.read(8))[0]
    s = lambda: f.read(u64()).decode('utf-8', 'replace')
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    FMT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = u32(), u64()
            return [val(et) for _ in range(n)]
        return struct.unpack(FMT[t], f.read(SZ[t]))[0]

    assert f.read(4) == b'GGUF'
    u32()
    nt, nk = u64(), u64()
    kv = {}
    for _ in range(nk):
        k = s()
        kv[k] = val(u32())
    infos = []
    for _ in range(nt):
        name = s()
        nd = u32()
        dims = [u64() for _ in range(nd)]
        t = u32()
        off = u64()
        infos.append((name, dims, t, off))
    align = kv.get('general.alignment', 32)
    base = (f.tell() + align - 1) // align * align
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    T = {}
    for name, dims, t, off in infos:
        dt, bs = GT[t]
        n = int(np.prod(dims))
        a = np.frombuffer(mm, dtype=dt, count=n, offset=base + off)
        if t == 30:
            a = (a.astype(np.uint32) << 16).view(np.float32)
        T[name] = a.astype(np.float32).reshape(list(reversed(dims)))
    return kv, T


def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * w if w is not None else y


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def rope_neox(x, pos, theta=100.0):  # x [..., n, d]: pairs (i, i + d/2), pos [n]
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d // 2, dtype=np.float64) * 2.0 / d)
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def main():
    kv, T = read_gguf(sys.argv[1])
    img = sys.argv[2]
    raw = open(img + '.g4v.rgb', 'rb').read()
    W, H = struct.unpack('<ii', raw[:8])
    px = np.frombuffer(raw[8:], dtype=np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
    E = kv['clip.vision.embedding_length']
    nh = kv['clip.vision.attention.head_count']
    P = kv['clip.vision.patch_size']
    L = kv['clip.vision.block_count']
    eps = kv.get('clip.vision.attention.layer_norm_epsilon', 1e-6)
    pool = 3
    nx, ny = W // P, H // P
    N = nx * ny
    # patches (row-major), each (channel, row, column), pixels mapped to [-1, 1]
    p = (2.0 * px - 1.0).reshape(ny, P, nx, P, 3).transpose(0, 2, 4, 1, 3).reshape(N, 3 * P * P)
    wpe = T['v.patch_embd.weight'].reshape(E, 3 * P * P)
    x = p @ wpe.T
    tab = T['v.position_embd.weight']  # [2][pos_size][E]
    cols, rows = np.arange(N) % nx, np.arange(N) // nx
    x = x + tab[0][cols] + tab[1][rows]

    vw, va, vatt = os.environ.get('REF_VW', ''), int(os.environ.get('REF_VA', '0')), os.environ.get('REF_VATT', '')

    def q8(m, passes=1):  # int8 per 32 values along the last axis (absmax / 127), `passes` times on the residual
        sh = m.shape
        r = m.reshape(-1, 32).astype(np.float32)
        out = np.zeros_like(r)
        for _ in range(passes):
            d = np.abs(r).max(axis=1, keepdims=True) / 127.0
            qv = np.round(r / np.where(d > 0, d, 1)) * d
            out += qv
            r = r - qv
        return out.reshape(sh)

    wcache = {}

    def lin(name, a):
        w = T[name + '.weight']
        if vw == 'q8' and w.shape[1] % 32 == 0:
            if name not in wcache:
                wcache[name] = q8(w)
            w = wcache[name]
        lo, hi = (T.get(name + '.input_min'), T.get(name + '.input_max'))
        if lo is not None:
            a = np.clip(a, lo.item(), hi.item())
        if va and a.shape[-1] % 32 == 0:
            a = q8(a, va)
        y = a @ w.T
        lo, hi = (T.get(name + '.output_min'), T.get(name + '.output_max'))
        if lo is not None:
            y = np.clip(y, lo.item(), hi.item())
        return y

    for il in range(L):
        b = f'v.blk.{il}.'
        h = rms(x, T[b + 'ln1.weight'], eps)
        hd = E // nh  # (64; Gemma 4 26B-A4B 72)
        q = lin(b + 'attn_q', h).reshape(N, nh, hd).transpose(1, 0, 2)
        k = lin(b + 'attn_k', h).reshape(N, nh, hd).transpose(1, 0, 2)
        v = lin(b + 'attn_v', h).reshape(N, nh, hd).transpose(1, 0, 2)
        q = rms(q, T[b + 'attn_q_norm.weight'], eps)
        k = rms(k, T[b + 'attn_k_norm.weight'], eps)
        v = rms(v, None, eps)
        q = np.concatenate([rope_neox(q[..., :hd // 2], cols), rope_neox(q[..., hd // 2:], rows)], axis=-1)
        k = np.concatenate([rope_neox(k[..., :hd // 2], cols), rope_neox(k[..., hd // 2:], rows)], axis=-1)
        if vatt == 'f16':
            q, k, v = (t.astype(np.float16).astype(np.float32) for t in (q, k, v))
        elif vatt == 'i8':
            def i8row(t):
                d = np.abs(t).max(axis=-1, keepdims=True) / 127.0
                return np.round(t / np.where(d > 0, d, 1)) * d
            q, k, v = i8row(q), i8row(k), i8row(v)
        s = q @ k.transpose(0, 2, 1)  # unscaled
        s = s - s.max(axis=-1, keepdims=True)
        a = np.exp(s)
        if vatt == 'i8':
            a = np.round(a * 255.0) / 255.0  # (P in 8 bits before the sum, as an int8 P.V would)
        a /= a.sum(axis=-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(N, E)
        o = lin(b + 'attn_out', o)
        x = x + rms(o, T[b + 'attn_post_norm.weight'], eps)
        h = rms(x, T[b + 'ln2.weight'], eps)
        g = gelu_tanh(lin(b + 'ffn_gate', h)) * lin(b + 'ffn_up', h)
        d = lin(b + 'ffn_down', g)
        x = x + rms(d, T[b + 'ffn_post_norm.weight'], eps)
    # 3x3 average pooling, times sqrt(E), RMS norm, projection
    g = x.reshape(ny // pool, pool, nx // pool, pool, E).mean(axis=(1, 3)).reshape(-1, E) * np.sqrt(E)
    if 'v.std_bias' in T:  # (26B-A4B: llama.cpp gemma4v's std_bias / std_scale)
        g = (g - T['v.std_bias']) * T['v.std_scale']
    ref = rms(g, None, eps) @ T['mm.input_projection.weight'].T
    if os.environ.get('REF_VOUT'):
        with open(os.environ['REF_VOUT'], 'wb') as f:
            f.write(np.array([ref.shape[0], ref.shape[1], 0], dtype=np.int32).tobytes())
            f.write(ref.astype(np.float32).tobytes())
    hdr = np.fromfile(img + '.g4v.bin', dtype=np.int32, count=3)
    gpu = np.fromfile(img + '.g4v.bin', dtype=np.float32, offset=12).reshape(hdr[0], hdr[1])
    cos = np.sum(ref * gpu, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(gpu, axis=1))
    rel = np.linalg.norm(ref - gpu, axis=1) / np.linalg.norm(ref, axis=1)
    print(f'{ref.shape[0]} tokens: cosine min {cos.min():.6f} mean {cos.mean():.6f}; '
          f'relative error mean {rel.mean() * 100:.3f} % max {rel.max() * 100:.3f} %')


main()
