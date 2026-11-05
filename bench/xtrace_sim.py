# Replays an expert-route trace (ARC_XTRACE=file, zero/xcache.h) through cache policies, with the cache's own pools
# and slots, and counts the records each would read from the SSD: generation misses and prompt reads apart.
# usage: python bench/xtrace_sim.py trace [--slru 0.5,0.7] [--lfu 0.25]
#   lru     the engine's policy (generation hits and misses to the front, a prompt's misses to the back)
#   slru:p  segmented LRU: experts used again in generation move to a protected part (p of the slots); victims come
#           from the other part first. A prompt's hits do not promote (a chunk touches nearly every expert)
#   lfu:w   victims among the w least recent slots: the one with the fewest uses (counts halved every 64 tokens)
#   opt     Belady (the expert used farthest in the future goes): the most any policy could save, not buildable
import struct, sys, heapq
from collections import OrderedDict

MAGIC = 0x58545231


def read_trace(path):
    data = open(path, "rb").read()
    v = memoryview(data).cast("i")
    i, hdr, calls = 0, None, []
    while i < len(v):
        if v[i] == MAGIC:
            NL, NE, G = v[i + 1], v[i + 2], v[i + 3]
            i += 4
            pools = []
            for _ in range(G):
                n, k = v[i], v[i + 1]
                keys = list(v[i + 2:i + 2 + k])
                i += 2 + k
                pools.append((n, keys))
            pool_of = list(v[i:i + NL])
            i += NL
            if hdr is None:
                hdr = (NL, NE, pools, pool_of)
            calls.append(None)  # (a new process: the run boundary)
            continue
        l, prompt, n = v[i], v[i + 1], v[i + 2]
        ids = v[i + 3:i + 3 + n]
        i += 3 + n
        seen, u = set(), []
        for e in ids:
            if e not in seen and 0 <= e < hdr[1]:
                seen.add(e)
                u.append(e)
        calls.append((l, prompt, u))
    return hdr, calls


class Lru:
    """One pool: OrderedDict, last = most recent."""

    def __init__(self, n, keys):
        self.n = n
        self.od = OrderedDict()
        for k in reversed(keys[:n]):  # (keys: most recent first)
            if k >= 0:
                self.od[k] = None

    def victim(self, cur):
        for k in self.od:
            if k not in cur:
                return k
        raise RuntimeError("no victim")

    def access(self, keys, prompt, cur):
        miss = 0
        for k in keys:
            if k in self.od:
                self.od.move_to_end(k)
        for k in keys:
            if k in self.od:
                continue
            miss += 1
            if len(self.od) >= self.n:
                del self.od[self.victim(cur)]
            self.od[k] = None
            if prompt:
                self.od.move_to_end(k, last=False)
        return miss


class Slru(Lru):
    def __init__(self, n, keys, p):
        super().__init__(n, keys)
        self.prot = OrderedDict()
        self.cap = int(n * p)

    def size(self):
        return len(self.od) + len(self.prot)

    def victim2(self, cur):
        for d in (self.od, self.prot):
            for k in d:
                if k not in cur:
                    return d, k
        raise RuntimeError("no victim")

    def access(self, keys, prompt, cur):
        miss = 0
        for k in keys:
            if k in self.prot:
                self.prot.move_to_end(k)
            elif k in self.od:
                if prompt:
                    self.od.move_to_end(k)
                else:
                    del self.od[k]
                    self.prot[k] = None
                    while len(self.prot) > self.cap:
                        d, _ = self.prot.popitem(last=False)
                        self.od[d] = None
        for k in keys:
            if k in self.od or k in self.prot:
                continue
            miss += 1
            if self.size() >= self.n:
                d, v = self.victim2(cur)
                del d[v]
            self.od[k] = None
            if prompt:
                self.od.move_to_end(k, last=False)
        return miss


class Lfu(Lru):
    def __init__(self, n, keys, w, cnt):
        super().__init__(n, keys)
        self.w = max(1, int(n * w))
        self.cnt = cnt

    def victim(self, cur):
        best, bc, j = None, None, 0
        for k in self.od:
            if k in cur:
                continue
            c = self.cnt.get(k, 0)
            if best is None or c < bc:
                best, bc = k, c
            j += 1
            if j >= self.w or bc == 0:
                break
        if best is None:
            raise RuntimeError("no victim")
        return best


def run(hdr, calls, make, counts=None):
    NL, NE, pools, pool_of = hdr
    P = [make(n, keys) for n, keys in pools]
    gen_miss = prompt_read = gen_calls = 0
    for c in calls:
        if c is None:
            continue
        l, prompt, ids = c
        keys = [l * NE + e for e in ids]
        cur = set(keys)
        if counts is not None and not prompt:
            gen_calls += 1
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
            if gen_calls % (64 * NL) == 0:
                for k in list(counts):
                    counts[k] >>= 1
        m = P[pool_of[l]].access(keys, prompt, cur)
        if prompt:
            prompt_read += m
        else:
            gen_miss += m
    return gen_miss, prompt_read


def opt(hdr, calls):
    NL, NE, pools, pool_of = hdr
    seq = []  # (call index, key)
    for ci, c in enumerate(calls):
        if c is None:
            continue
        l, prompt, ids = c
        for e in ids:
            seq.append((ci, l * NE + e))
    INF = 1 << 60
    nxt = [INF] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i][1]
        nxt[i] = last.get(k, INF)
        last[k] = i
    cache = [set(k for k in keys[:n] if k >= 0) for n, keys in pools]
    heaps = [[] for _ in pools]
    nu = {}  # key -> its next use (current)
    first = {}
    for i, (ci, k) in enumerate(seq):
        first.setdefault(k, i)
    for g, s in enumerate(cache):
        for k in s:
            nu[k] = first.get(k, INF)
            heapq.heappush(heaps[g], (-nu[k], k))
    gen_miss = prompt_read = 0
    i = 0
    for ci, c in enumerate(calls):
        if c is None:
            continue
        l, prompt, ids = c
        g = pool_of[l]
        S, H, n = cache[g], heaps[g], pools[g][0]
        cur = set(l * NE + e for e in ids)
        for e in ids:
            k = l * NE + e
            if k not in S:
                if prompt:
                    prompt_read += 1
                else:
                    gen_miss += 1
                if len(S) >= n:
                    keep = []
                    while True:
                        nk, v = heapq.heappop(H)
                        if v not in S or -nk != nu.get(v):
                            continue  # (stale)
                        if v in cur:
                            keep.append((nk, v))
                            continue
                        S.discard(v)
                        break
                    for x in keep:
                        heapq.heappush(H, x)
                S.add(k)
            nu[k] = nxt[i]
            heapq.heappush(H, (-nu[k], k))
            i += 1
    return gen_miss, prompt_read


def main():
    path = sys.argv[1]
    hdr, calls = read_trace(path)
    NL, NE, pools, pool_of = hdr
    ng = sum(1 for c in calls if c and not c[1])
    gen_routes = sum(len(c[2]) for c in calls if c and not c[1])
    print("%s: %d layers x %d experts, %d pools (%s slots), %d runs, %d generation calls (%d tokens), %d routes" %
          (path, NL, NE, len(pools), "/".join(str(n) for n, _ in pools), calls.count(None), ng,
           ng // max(1, sum(1 for x in pool_of if x >= 0)), gen_routes))
    rows = [("lru (engine)", lambda n, k: Lru(n, k), None)]
    for p in (0.3, 0.5, 0.7, 0.8):
        rows.append(("slru:%.1f" % p, lambda n, k, p=p: Slru(n, k, p), None))
    for w in (0.05, 0.25, 0.5, 1.0):
        cnt = {}
        rows.append(("lfu:%.2f" % w, lambda n, k, w=w, cnt=cnt: Lfu(n, k, w, cnt), cnt))
    base = None
    for name, mk, cnt in rows:
        gm, pr = run(hdr, calls, mk, cnt)
        if base is None:
            base = gm
        print("  %-14s generation misses %7d (%5.1f %% of routes, %+6.1f %% vs lru) | prompt reads %7d" %
              (name, gm, 100.0 * gm / max(1, gen_routes), 100.0 * (gm - base) / max(1, base), pr))
    gm, pr = opt(hdr, calls)
    print("  %-14s generation misses %7d (%5.1f %% of routes, %+6.1f %% vs lru) | prompt reads %7d" %
          ("opt (bound)", gm, 100.0 * gm / max(1, gen_routes), 100.0 * (gm - base) / max(1, base), pr))


if __name__ == "__main__":
    main()
