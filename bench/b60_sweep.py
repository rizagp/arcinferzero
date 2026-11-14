#!/usr/bin/env python3
# Sweeps the prompt GEMM's settings on a discrete Arc card (made for the Pro B60, issue #4) around the two best paths
# so far: the default (XMX with 2D loads and their prefetch: 764 tok/s on the 27B, 4K prompt) and H8 (ARC_H8=1,
# Hadamard-rotated int8 with one scale per column and token: 918 tok/s). Each setting is changed alone from each base,
# the winners are put together, then measured at prompt sizes of real use, and their accuracy is checked against the
# default by KL. With --moe-model also the 35B-A3B's expert GEMM shapes (ARC_MOE_CG / ARC_MOE_WY) and H8 there.
# Paste the whole output into the GitHub issue.
# The Python port of bench/b60_sweep.ps1, for Windows and Linux: the same runs in the same order and the same output.
# The two must stay the same; change both together.
#   Windows: python bench\b60_sweep.py --model C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--moe-model C:\hf\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf]
#   Linux:   python3 bench/b60_sweep.py --model ~/hf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--moe-model ~/hf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf]
#   [--tokens 4096]  phase A's prompt length   [--quick]  fewer values, no 16K row   [--gpus 2 4] (or 2,4)  phase E's GPU counts
# A setting wins only 3 % ahead of its base; "best" is the faster of all winners together and the single best change.
# Settings (defaults on discrete cards in brackets):
#   TM      ARC_GEMM_TM, tokens per sub-group [16]      WGY    ARC_GEMM_WGY, sub-groups per work-group [8]
#   RG      ARC_GEMM_RG, rows per weight slice [4]       CG     ARC_GEMM_CG, column groups per work-group [1]
#   GRF     ARC_GEMM_GRF, registers per thread [128]     SPLIT  ARC_GEMM_SPLIT, superblocks per split-K piece [17]
#   X2D     2D loads: pf (with the prefetch) / on / off [pf]
#   WPF     ARC_GEMM_WPF, the weights prefetched n 64-value chunks ahead, 0 = off [0]
#   FUSE    ARC_H8_FUSE, H8's weight rotation in the decode pass (1) or after it (0) [1]; H8 only, bit-identical
#   MOECG / MOEWY  ARC_MOE_CG / ARC_MOE_WY, sub-groups per work-group of the expert GEMM [1 / 1] (35B only)
# Python 3.8 or newer, standard library only.
import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""
RUNZ = os.path.join(ROOT, "bin", "runz" + EXE)
PROBE = os.path.join(ROOT, "bin", "mgpu_probe" + EXE)
VARS = ["ARC_H8", "ARC_GEMM_TM", "ARC_GEMM_WGY", "ARC_GEMM_RG", "ARC_GEMM_CG", "ARC_GEMM_GRF", "ARC_GEMM_SPLIT", "ARC_GEMM_2D",
        "ARC_GEMM_2D_PF", "ARC_MOE_CG", "ARC_MOE_WY", "ARC_GEMM_WPF", "ARC_H8_FUSE"]


def int_list(values):
    # (--gpus 2 4 and --gpus 2,4 both work, as -Gpus 2,4 does in the .ps1)
    out = []
    for v in values:
        out += [int(x) for x in v.split(",") if x.strip() != ""]
    return out


ap = argparse.ArgumentParser(
    description="Sweeps the prompt GEMM's settings on a discrete Arc card (the Python port of bench/b60_sweep.ps1). "
                "Paste the whole output into the GitHub issue.")
ap.add_argument("--model", required=True, help="the dense model (.gguf), e.g. the 27B")
ap.add_argument("--moe-model", default="", help="the 35B-A3B (.gguf): phase D, and its rows in phase E")
ap.add_argument("--tokens", type=int, default=4096, help="phase A's prompt length (default 4096)")
ap.add_argument("--gpus", nargs="+", default=["2", "4"], help="phase E's GPU counts (default 2 4; 2,4 works too)")
ap.add_argument("--quick", action="store_true", help="fewer values, no 16K row")
args = ap.parse_args()
args.gpus = int_list(args.gpus)
try:
    sys.stdout.reconfigure(errors="replace")
except AttributeError:
    pass

if not os.path.exists(RUNZ):
    sys.exit("bin/runz%s not found (run from the inferzero folder)" % EXE)
if not os.path.exists(args.model):
    sys.exit("model not found: " + args.model)


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


def env_of(sset):
    # the environment of a setting set (name -> value as in the table above)
    e = {}
    for k, val in sset.items():
        v = str(val)
        if k == "H8":
            e["ARC_H8"] = "1" if v == "on" else "0"  # (off: ARC_H8=0, since H8 is on by default for Qwen dense models)
        elif k == "TM":
            e["ARC_GEMM_TM"] = v
        elif k == "WGY":
            e["ARC_GEMM_WGY"] = v
        elif k == "RG":
            e["ARC_GEMM_RG"] = v
        elif k == "CG":
            e["ARC_GEMM_CG"] = v
        elif k == "GRF":
            e["ARC_GEMM_GRF"] = v
        elif k == "SPLIT":
            e["ARC_GEMM_SPLIT"] = v
        elif k == "X2D":
            e["ARC_GEMM_2D"] = "0" if v == "off" else "1"
            e["ARC_GEMM_2D_PF"] = "1" if v == "pf" else "0"
        elif k == "WPF":
            e["ARC_GEMM_WPF"] = v
        elif k == "FUSE":
            e["ARC_H8_FUSE"] = v
        elif k == "MOECG":
            e["ARC_MOE_CG"] = v
        elif k == "MOEWY":
            e["ARC_MOE_WY"] = v
    return e


def clean_env():
    # a copy of this process's environment without the variables the script sets (the launcher's stays as it is)
    e = dict(os.environ)
    for v in VARS:
        e.pop(v, None)
    return e


def run_runz(sset, a):
    # one runz call with the settings sset and the arguments a; returns its output (stdout and stderr)
    e = clean_env()
    e.update(env_of(sset))
    p = subprocess.run([RUNZ] + [str(x) for x in a], env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode("utf-8", "replace")


class Result:
    def __init__(self, tps, path):
        self.tps = tps
        self.path = path


def bench(model, sset, pp, chunk=0, gpus=1):
    # a prompt benchmark: best of 3 passes, and the path the engine took
    a = [model, "--bench-pp", pp, "--n", 1]
    if chunk > 0:
        a += ["--chunk", chunk]
    if gpus > 1:
        a += ["--gpus", gpus]
    o = run_runz(sset, a)
    tps = [float(m.group(1)) for m in re.finditer(r"pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s", o)]
    if re.search(r"XMX on for the prompt GEMMs", o, re.I):
        path = "XMX"
    elif re.search(r"XMX results differ", o, re.I):
        path = "DP4a (XMX differed!)"
    else:
        path = "DP4a"
    if re.search(r"with 2D block loads", o, re.I):
        path += " + 2D"
    elif re.search(r"2D block loads \(ARC_GEMM_2D\) do not build or differ", o, re.I):
        path += " (2D differed!)"
    if re.search(r"H8 on for the prompt GEMMs", o, re.I):
        path += " + H8"
    # (as in the .ps1: no pass, or one pass at 0, counts as failed)
    if not tps or tps == [0.0]:
        return Result(0, "failed: " + first_error(o))
    return Result(max(tps), path)


def line(what, r):
    return "{0:<40} {1:9.1f} tok/s  {2}".format(what, r.tps, r.path)


def desc(sset):
    if len(sset) == 0:
        return "defaults"
    return " ".join("%s=%s" % (k, sset[k]) for k in sorted(sset))


def probe_gpus(env):
    # the "  GPU n: ..." lines of mgpu_probe, trimmed
    if not os.path.exists(PROBE):
        sys.exit("bin/mgpu_probe%s not found" % EXE)
    p = subprocess.run([PROBE], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o = p.stdout.decode("utf-8", "replace")
    return [l.strip() for l in o.splitlines() if re.match(r"^\s+GPU \d+:", l, re.I)]


model = args.model
moe_model = args.moe_model
tokens = args.tokens
quick = args.quick

grid = {  # (in this order)
    "TM": [16, 32],
    "WGY": [4, 8, 16] if quick else [2, 4, 8, 16],
    "RG": [2, 4, 8] if quick else [1, 2, 4, 8],
    "CG": [1, 2, 4],
    "GRF": [128, 256],
    "SPLIT": [0, 17] if quick else [0, 9, 17, 34],
    "X2D": ["pf", "on", "off"],
    "WPF": [0, 4] if quick else [0, 2, 4, 8],
    "FUSE": [1, 0],
}
defaults = {"TM": 16, "WGY": 8, "RG": 4, "CG": 1, "GRF": 128, "SPLIT": 17, "X2D": "pf", "WPF": 0, "FUSE": 1}

for l in probe_gpus(dict(os.environ)):
    out(l)
out("model: " + leaf(model) + "; phase A prompt: %d tokens; tok/s = best of 3 passes" % tokens)
bench(model, {}, 512)  # (warm-up: the first run after a start is slower)
together = {}
for base_name, base_set in (("default (no H8)", {"H8": "off"}), ("H8", {"H8": "on"})):
    out()
    out("=== A. %s: one setting at a time" % base_name)
    b = bench(model, base_set, tokens)
    out(line("base", b))
    # (H8 runs at 4 column groups by default since 2026-11-08: its base is CG 4, so CG 1 and 2 are the changes)
    bdef = dict(defaults)
    if base_set.get("H8") == "on":
        bdef["CG"] = 4
    best_set = dict(base_set)
    single = dict(base_set)
    single_t = b.tps
    for p in grid:
        if p == "FUSE" and base_set.get("H8") != "on":
            continue  # (H8's switch: nothing to measure without it)
        bv = bdef[p]
        bt = b.tps
        for v in grid[p]:
            if str(v) == str(bdef[p]):
                continue  # (the base: measured above)
            s = dict(base_set)
            s[p] = v
            r = bench(model, s, tokens)
            out(line("%s=%s" % (p, v), r))
            # (a winner must be 3 % ahead of the base, past the run-to-run drift)
            if r.tps > bt * 1.03 and r.tps > b.tps * 1.03:
                bt = r.tps
                bv = v
            if r.tps > single_t and r.tps > b.tps * 1.03:
                single_t = r.tps
                single = dict(s)
        if str(bv) != str(bdef[p]):
            best_set[p] = bv
    # (winners alone do not always add up: settings of the same work-group can work against each other, so the single
    #  best change is kept too, and whichever of the two is faster is "best")
    r = bench(model, best_set, tokens)
    out(line("best of each together", r))
    out("  (" + desc(best_set) + ")")
    out(line("base again (drift)", bench(model, base_set, tokens)))
    out("  single best change: " + desc(single))
    together[base_name] = best_set if r.tps >= single_t else single
    out("  taken as best: " + desc(together[base_name]))

out()
out("=== B. prompt sizes of real use (default chunks unless noted)")
sets = {"default": {}, "no H8": {"H8": "off"}, "H8": {"H8": "on"}, "best without H8": together["default (no H8)"],
        "best with H8": together["H8"]}
sizes = [(512, 0, "512 (a chat turn)"), (4096, 0, "4096 (a document)"), (16384, 0, "16384 (a long prompt)"),
         (16384, 4096, "16384 in 4096-token chunks")]
if quick:
    sizes = sizes[0:2]
for z in sizes:
    for n in sets:
        out(line("{0:<27} {1:<16}".format(z[2], n), bench(model, sets[n], z[0], z[1])))

out()
out("=== C. accuracy against the plain path (no H8): KL over 4 perplexity chunks of USAGE.md")
kl = os.path.join(tempfile.gettempdir(), "b60_sweep_base.kl")
text = os.path.join(ROOT, "USAGE.md")
o = run_runz({"H8": "off"}, [model, "--perplexity", text, "--ppl-chunks", 4, "--ppl-kl-save", kl])  # (the plain path, as before H8 was a default)
m = [l for l in lines_of(o) if re.search(r"^Final|perplexity|PPL", l, re.I)]
if m:
    out("no H8: " + m[-1].strip())
for n in ("default", "H8", "best without H8", "best with H8"):
    o = run_runz(sets[n], [model, "--perplexity", text, "--ppl-chunks", 4, "--ppl-kl", kl])
    kls = [l for l in lines_of(o) if re.search(r"KL vs base", l, re.I)]
    ppl = re.search(r"perplexity: ([\d.]+)", o)
    out("{0:<18} {1}  perplexity {2}  ({3})".format(n, kls[-1].strip() if kls else "no KL line",
                                                    ppl.group(1) if ppl else "?", desc(sets[n])))
try:
    os.remove(kl)
except OSError:
    pass

if moe_model != "":
    out()
    out("=== D. " + leaf(moe_model) + ": the expert GEMM's work-group shapes, 16384-token prompt")
    if not os.path.exists(moe_model):
        out("skipped (not found): " + moe_model)
    else:
        bench(moe_model, {}, 512)
        moe = {
            "defaults (1 x 1)": {}, "MOECG=2": {"MOECG": 2}, "MOECG=4": {"MOECG": 4}, "MOEWY=2": {"MOEWY": 2},
            "MOEWY=4": {"MOEWY": 4}, "MOEWY=8": {"MOEWY": 8}, "MOECG=2 MOEWY=4": {"MOECG": 2, "MOEWY": 4},
            "MOECG=4 MOEWY=4": {"MOECG": 4, "MOEWY": 4}, "H8": {"H8": "on"},
        }
        for n in moe:
            out(line(n, bench(moe_model, moe[n], 16384)))
        out(line("defaults again (drift)", bench(moe_model, {}, 16384)))

out()
out("=== E. several GPUs (--gpus), the 27B at 4096 and 16384 tokens" + (", the 35B at 16384" if moe_model != "" else ""))
ngpu = len(probe_gpus(clean_env()))
for g in args.gpus:
    if g > ngpu:
        out("skipped --gpus %d (%d GPUs found)" % (g, ngpu))
        continue
    for n in ("default", "best with H8"):
        for pp in (4096, 16384):
            out(line("--gpus {0} 27B {1:<14} {2:>5}".format(g, n, pp), bench(model, sets[n], pp, 0, g)))
    if moe_model != "" and os.path.exists(moe_model):
        out(line("--gpus {0} 35B default        16384".format(g), bench(moe_model, {}, 16384, 0, g)))
        out(line("--gpus {0} 35B MOEWY=4        16384".format(g), bench(moe_model, {"MOEWY": 4}, 16384, 0, g)))
