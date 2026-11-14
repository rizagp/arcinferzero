#!/usr/bin/env python3
# Token generation settings on this GPU (made for GPUs with memory of their own, e.g. Arc Pro B60): the same greedy
# answer to one coding question plain, with the model's MTP block, with a DFlash2 drafter, and with the GEMV's
# tuning switches, each the best of 2 runs. Paste the whole output into the GitHub issue.
# The Python port of bench/decode_try.ps1, for Windows and Linux: the same runs in the same order and the same output.
# The two must stay the same; change both together.
#   Windows: python bench\decode_try.py --models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--dflash <DFlash2 drafter folder>] [--n 256]
#   Linux:   python3 bench/decode_try.py --models ~/hf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--dflash <DFlash2 drafter folder>] [--n 256]
#   (several models: --models a.gguf b.gguf)
# Runs (each: bin/runz <model> --text-file <question> --n N; bin\runz.exe on Windows):
#   plain          no drafts
#   mtp            --mtp (the model's own MTP block, when it has one)
#   mtp7           --mtp --mtp-drafts 7
#   dflash         --dflash DIR --draft auto (with --dflash)
#   GEMV_GRF=256   plain, the single-token GEMV with 256 registers per thread (ARC_GEMV_GRF=256)
#   GVROWS=1/2/4   plain, rows per sub-group of the codebook GEMV (ARC_GVROWS; default: by matrix size)
#   mtp MT_GRF=256 mtp, the 2-8 token GEMV of the verification with 256 registers, as before 2026-11-03 (ARC_GEMV_MT_GRF=256)
#   mtp MT_GRF=128 the same with 128 registers (the discrete cards' default; iGPUs default to 256)
#   GEMV_PF=0      plain, without the single-token GEMV issuing all its rows' reads of a step first (ARC_GEMV_PF=0;
#                  on by default on discrete cards: dense models since 2026-11-08, MoE models since 2026-11-12, a
#                  B60's 35B 79.9 without it, 88.9 with)
#   GV_IQ3S=1      plain, IQ3_S weights through gemv_v (straight into registers) instead of gemv's local window
#                  (ARC_GV_IQ3S=1); also with GVROWS=2 / 4 (ARC_GVROWS: its rows per sub-group)
#   BATCH=16/32/128  plain, kernels recorded into command lists of up to n kernels, each appended to the immediate
#                  list when full (ARC_BATCH=n; off by default); also "mtp BATCH=32"
#   XALL=0         MoE models: without skipping the host's per-layer reads of expert ids when every expert is
#                  cached (ARC_XALL: on by default on discrete cards since 2026-11-09, a B60's 35B 78.8 -> 88.1);
#                  also with GEMV_PF=1 (both as before 2026-11-09) and "mtp XALL=0". Their tokens are checked against plain.
#   WIDE_ROWS=0    plain, the row kernels that quantize for the next GEMV (swiglu_quant, rmsnorm_quant,
#                  gated_norm_quant, attn_gate_quant) in one work-group per row, as before 2026-11-10
#                  (ARC_WIDE_ROWS=0); tokens checked against plain
#   SMALL_WG=1     plain, the GEMV of matrices with few rows (DeltaNet's alpha / beta, MoE routers) in work-groups
#                  of 64 (ARC_GEMV_SMALL_WG=1; the default 2026-11-10 to 11-12, off since: the B60's 35B 30 % slower
#                  with it); tokens checked against plain
#   GEMV_GU=1      dense models: plain, the FFN gate and up GEMVs in one launch instead of two (ARC_GEMV_GU=1,
#                  QUEUE 33.5; bit-identical; MoE models take their own FFN, so it equals plain there); tokens checked
#   (at the end, per model: a --prof-gen table, the GPU time of a generation step by kernel)
#   MOE_GR_GU=2/4, MOE_GR_DN=2/8   MoE models: rows per sub-group of the experts' gate / up and down GEMVs for one
#                  token (ARC_MOE_GR_GU, default 1; ARC_MOE_GR_DN, default 4)
# Added 2026-11-13 (a GEMV sweep for the B60 / B70, issue #12):
#   GVROWS=8       plain, 8 rows per sub-group (the most ARC_GVROWS takes)
#   GV4=0          plain, the codebook types (IQ1 / IQ2 / IQ3_XXS) at 2 rows per sub-group instead of 4 (ARC_GV4=0)
#   GV_IQ3S=1 GVROWS=1/8   the IQ3_S path at the ends of the range
#   mtp MOE_GR_DN_S=1/2/8, MOE_GRF=128   MoE models: rows per sub-group of the experts' down GEMV for 2-8 tokens (the
#                  verification; default 4), the experts' GEMVs with the default register file (ARC_MOE_GRF=128)
#   long context (--ctx, default 16384; 0 skips it): the prompt of that length, then 64 tokens, best of 2
#   (runz --bench-pp N --n 64): plain, ATTN_DEC_GRF=128 (the generation attention with the default register file),
#   ATTN_DEC_SG=256/1024 (the sub-groups its key split aims for, default 512), ATTN_DEC_SMAX=128/512 (keys per split
#   at most, default 256), and a --prof-gen table there
# Python 3.8 or newer, standard library only.
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""
RUNZ = os.path.join(ROOT, "bin", "runz" + EXE)
PROBE = os.path.join(ROOT, "bin", "mgpu_probe" + EXE)
VARS = ["ARC_GEMV_GRF", "ARC_GVROWS", "ARC_GEMV_MT_GRF", "ARC_GEMV_PF", "ARC_GV_IQ3S", "ARC_MOE_GR_GU", "ARC_MOE_GR_DN", "ARC_BATCH", "ARC_XALL",
        "ARC_WIDE_ROWS", "ARC_GEMV_SMALL_WG", "ARC_GEMV_GU", "ARC_GV4", "ARC_MOE_GR_DN_S", "ARC_MOE_GRF", "ARC_ATTN_DEC_GRF",
        "ARC_ATTN_DEC_SG", "ARC_ATTN_DEC_SMAX"]

ap = argparse.ArgumentParser(
    description="Token generation settings on this GPU (the Python port of bench/decode_try.ps1). "
                "Paste the whole output into the GitHub issue.")
ap.add_argument("--models", nargs="+", required=True, help="one or more models (.gguf)")
ap.add_argument("--dflash", default="", help="a DFlash2 drafter folder: adds the dflash row")
ap.add_argument("--n", type=int, default=256, help="tokens generated per run (default 256)")
ap.add_argument("--ctx", type=int, default=16384, help="the long-context rows' prompt length (default 16384; 0 skips them)")
args = ap.parse_args()
N = args.n
CTX = args.ctx
try:
    sys.stdout.reconfigure(errors="replace")
except AttributeError:
    pass

if not os.path.exists(RUNZ):
    sys.exit("bin/runz%s not found (run from the inferzero folder)" % EXE)

# (Qwen chat format, thinking off: the answer starts right away)
q = os.path.join(tempfile.gettempdir(), "inferzero-decode-try.txt")
text = ("<|im_start|>user\nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using "
        "a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n")
with open(q, "w", encoding="utf-8", newline="") as f:
    f.write(text)


def out(s=""):
    print(s, flush=True)


def leaf(path):
    return os.path.basename(path.rstrip("/\\"))


def lines_of(o):
    # (the .ps1 splits runz's output at "\n"; the "\r" left of Windows line ends is dropped here)
    return [l.rstrip("\r") for l in o.split("\n")]


def first_error(o):
    for l in lines_of(o):
        if re.search(r"error|cannot|failed|exception", l, re.I):
            return l
    return ""


def env_with(envs):
    # a copy of this process's environment without the script's variables, plus envs (the launcher's stays as it is)
    e = dict(os.environ)
    for v in VARS:
        e.pop(v, None)
    e.update(envs)
    return e


def runz(a, envs):
    # one runz call; returns its output (stdout and stderr)
    p = subprocess.run([RUNZ] + [str(x) for x in a], env=env_with(envs), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode("utf-8", "replace")


class Result:
    def __init__(self, tps, note, moe=False, ids=None):
        self.tps = tps
        self.note = note
        self.moe = moe
        self.ids = ids


def invoke_run(model, extra, envs):
    best = None
    for _ in range(2):
        o = runz([model, "--text-file", q, "--n", N] + list(extra), envs)
        m = re.search(r"generation: \d+ tokens in [\d.]+ ms \(([\d.]+) t/s\)", o)
        if not m:
            return Result(0, "failed: " + first_error(o))
        a = re.search(r"MTP: \d+ drafts, \d+ accepted \(([\d.]+) %\), ([\d.]+) tokens per main-model pass", o)
        note = "%s %% accepted, %s tokens per pass" % (a.group(1), a.group(2)) if a else ""
        g = re.search(r"greedy:([ \d]*)", o)
        r = Result(float(m.group(1)), note, re.search(r"expert cache", o, re.I) is not None, g.group(1).strip() if g else "")
        if best is None or r.tps > best.tps:
            best = r
    return best


def line(what, r):
    return "{0:<22} {1:8.2f} tok/s  {2}".format(what, r.tps, r.note)


def same(what, r, base):
    # (switches that must not change the answer: their tokens against plain's)
    if r.tps == 0:
        n = r.note
    elif r.ids == base.ids:
        n = "same tokens as plain"
    else:
        n = "TOKENS DIFFER from plain"
    return "{0:<22} {1:8.2f} tok/s  {2}".format(what, r.tps, n)


PROF_RE = r"^tg\d+:|tg steps|weights read|GPU time by kernel|^\s+\w+\s+[\d.]+ ms"


def prof_lines(o):
    return [l.rstrip() for l in lines_of(o) if re.search(PROF_RE, l, re.I)]


def tg(model, envs, prof=False):
    best = 0.0
    txt = ""
    for _ in range(1 if prof else 2):
        a = ["--bench-pp", str(CTX), "--n", "64"] + (["--prof-gen"] if prof else [])
        o = runz([model] + a, envs)
        m = re.search(r"tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s", o)
        if m and float(m.group(1)) > best:
            best = float(m.group(1))
        txt = o
    if prof:
        return txt
    if best == 0:
        return "failed: " + first_error(txt)
    return "{0:8.2f} tok/s".format(best)


if not os.path.exists(PROBE):
    sys.exit("bin/mgpu_probe%s not found" % EXE)
p = subprocess.run([PROBE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for l in p.stdout.decode("utf-8", "replace").splitlines():
    if re.match(r"^\s+GPU \d+:", l, re.I):
        out(l.strip())
out("generation of %d tokens (greedy), best of 2 runs" % N)
for model in args.models:
    if not os.path.exists(model):
        out("skipped (not found): " + model)
        continue
    out()
    out("=== " + leaf(model))
    invoke_run(model, ["--n", "8"], {})  # (warm-up: the first run after a start is slower)
    plain = invoke_run(model, [], {})
    out(line("plain", plain))
    # (2026-11-10: the wide row kernels, on by default: without them; the small GEMV work-groups, off by default
    #  since 2026-11-12: with them)
    out(same("WIDE_ROWS=0", invoke_run(model, [], {"ARC_WIDE_ROWS": "0"}), plain))
    out(same("SMALL_WG=1", invoke_run(model, [], {"ARC_GEMV_SMALL_WG": "1"}), plain))
    out(same("GEMV_GU=1", invoke_run(model, [], {"ARC_GEMV_GU": "1"}), plain))
    # (meant bit for bit, so checked bit for bit: every one of 32 steps' logits against plain's)
    lg = []
    for gu in ("0", "1"):
        d = os.path.join(tempfile.gettempdir(), "inferzero-decode-try-gu%s.bin" % gu)
        runz([model, "--text-file", q, "--n", "32", "--dump-logits", d], {"ARC_GEMV_GU": "1"} if gu == "1" else {})
        if os.path.exists(d):
            with open(d, "rb") as f:
                lg.append(hashlib.sha256(f.read()).hexdigest())
            try:
                os.remove(d)
            except OSError:
                pass
        else:
            lg.append("")
    out("GEMV_GU=1 logits       " + ("not written" if not lg[0] or not lg[1]
                                     else "identical to plain (32 steps)" if lg[0] == lg[1] else "DIFFER from plain"))
    out(line("mtp", invoke_run(model, ["--mtp"], {})))
    out(line("mtp7", invoke_run(model, ["--mtp", "--mtp-drafts", "7"], {})))
    if args.dflash:
        out(line("dflash", invoke_run(model, ["--dflash", args.dflash, "--draft", "auto"], {})))
    out(line("GEMV_GRF=256", invoke_run(model, [], {"ARC_GEMV_GRF": "256"})))
    for g in (1, 2, 4, 8):
        out(line("GVROWS=%d" % g, invoke_run(model, [], {"ARC_GVROWS": str(g)})))
    out(same("GV4=0", invoke_run(model, [], {"ARC_GV4": "0"}), plain))
    out(line("mtp MT_GRF=256", invoke_run(model, ["--mtp"], {"ARC_GEMV_MT_GRF": "256"})))
    out(line("mtp MT_GRF=128", invoke_run(model, ["--mtp"], {"ARC_GEMV_MT_GRF": "128"})))
    # (the prefetch is on by default on discrete cards, dense and MoE models: the row without it)
    out(line("GEMV_PF=0", invoke_run(model, [], {"ARC_GEMV_PF": "0"})))
    out(line("GV_IQ3S=1", invoke_run(model, [], {"ARC_GV_IQ3S": "1"})))
    for g in (1, 2, 4, 8):
        out(line("GV_IQ3S=1 GVROWS=%d" % g, invoke_run(model, [], {"ARC_GV_IQ3S": "1", "ARC_GVROWS": str(g)})))
    for b in (16, 32, 128):
        out(same("BATCH=%d" % b, invoke_run(model, [], {"ARC_BATCH": str(b)}), plain))
    out(line("mtp BATCH=32", invoke_run(model, ["--mtp"], {"ARC_BATCH": "32"})))
    if plain.moe:  # (MoE models: the experts' per-tile GEMVs of one token, rows per sub-group)
        # (XALL is on by default on discrete cards since 2026-11-09: the rows without it)
        out(same("XALL=0", invoke_run(model, [], {"ARC_XALL": "0"}), plain))
        out(same("XALL=0 GEMV_PF=0", invoke_run(model, [], {"ARC_XALL": "0", "ARC_GEMV_PF": "0"}), plain))
        out(line("mtp XALL=0", invoke_run(model, ["--mtp"], {"ARC_XALL": "0"})))
        for g in (2, 4):
            out(line("MOE_GR_GU=%d" % g, invoke_run(model, [], {"ARC_MOE_GR_GU": str(g)})))
        for g in (2, 8):
            out(line("MOE_GR_DN=%d" % g, invoke_run(model, [], {"ARC_MOE_GR_DN": str(g)})))
        for g in (1, 2, 8):
            out(line("mtp MOE_GR_DN_S=%d" % g, invoke_run(model, ["--mtp"], {"ARC_MOE_GR_DN_S": str(g)})))
        out(same("MOE_GRF=128", invoke_run(model, [], {"ARC_MOE_GRF": "128"}), plain))
    # (generation after a long prompt: the attention's share grows with the context)
    if CTX > 0:
        out("--- after a %d-token prompt: 64 tokens, best of 2" % CTX)
        ctx_runs = {
            "plain": {},
            "ATTN_DEC_GRF=128": {"ARC_ATTN_DEC_GRF": "128"},
            "ATTN_DEC_SG=256": {"ARC_ATTN_DEC_SG": "256"},
            "ATTN_DEC_SG=1024": {"ARC_ATTN_DEC_SG": "1024"},
            "ATTN_DEC_SMAX=128": {"ARC_ATTN_DEC_SMAX": "128"},
            "ATTN_DEC_SMAX=512": {"ARC_ATTN_DEC_SMAX": "512"},
        }
        for k in ctx_runs:
            out("{0:<22} {1}".format("ctx " + k, tg(model, ctx_runs[k])))
        out("--- generation profile after the %d-token prompt (plain)" % CTX)
        for l in prof_lines(tg(model, {}, prof=True)):
            out(l)
    # (where a generation step's GPU time goes, per kernel, plain: --prof-gen over 32 steps after a 128-token prompt)
    out("--- generation profile (plain, 32 steps)")
    for l in prof_lines(runz([model, "--bench-pp", "128", "--n", "32", "--prof-gen"], {})):
        out(l)
    # (kernel launches and host waits per token: plain, and for MoE models with ARC_XALL=0)
    out("--- launches per token")
    runs = [("plain", {})]
    if plain.moe:
        runs.append(("XALL=0", {"ARC_XALL": "0"}))
    for name, e in runs:
        envs = {"ARC_LAUNCH_STATS": "1"}
        envs.update(e)
        o = runz([model, "--bench-pp", "128", "--n", "32"], envs)
        for l in lines_of(o):
            if re.search(r"^launches:|expert cache", l, re.I):
                out(name + ": " + l.strip())
try:
    os.remove(q)
except OSError:
    pass
