"""Interleaved A/B on ONE resident server (flag flips only, no reload). Prompts: the 4 x 128-token greedy runs of tools/served_ds41.py
(prose + code, reps 0 and 1). Per arm and depth: tok/s (decode, from the stream) and md5 of the 4 replies.
usage: train_ab.py PORT cache [--limits 256,1024,4096,16384]   cache limit sweep (levers as the server started)
       train_ab.py PORT lever NAME[,NAME]                      ALL_OFF vs ALL_OFF + lever(s) (A B A B), md5 of B against A
       train_ab.py PORT final                                  ALL_OFF vs the winners (--on JSON)
       train_ab.py PORT kl NAME[,NAME]                         teacher-forced KL of the lever(s) against ALL_OFF (plain decode)
Options: --depths 0,3  --rounds 2  --n 128  --cache-mb N (set before the run)  --on JSON  --out FILE"""
import argparse
import hashlib
import http.client
import json
import statistics
import sys
import os
import time

ALL_OFF = {"glue_fuse": 0, "glue_fuse_norm": 0, "idx_fast": 0, "idx_fuse": 0, "moe_fastdec": 0, "moeroof": 0, "dense_fast": 0, "moefuse": 0, "pf_glu": 0,
           "pf_split_gu": 0, "port_sumoe": 0, "port_suglue": 0, "dspol": 0, "bound_layers": 0, "idx_qblock": 0}
LEVERS = {"glue": {"glue_fuse": 1}, "gluenorm": {"glue_fuse": 1, "glue_fuse_norm": 1}, "idxfast": {"idx_fast": 1}, "idxfuse": {"idx_fuse": 1},
          "densefast": {"dense_fast": 1}, "pfglu": {"pf_glu": 1}, "pfsplit": {"pf_split_gu": 1}, "dspol": {"dspol": 1},
          "bound": {"bound_layers": 1}, "qblock": {"idx_qblock": 512}, "moefastdec": {"moe_fastdec": 1}, "moeroof": {"moeroof": 1}, "moeroof1": {"moeroof": 1, "moeroof_dec": 1},
          "moeroofnopf": {"moeroof": 1, "moeroof_pf": 0}, "moeroof+fastdec": {"moeroof": 1, "moe_fastdec": 1}, "moefuse": {"moefuse": 1},
          "suglue": {"port_suglue": 1}, "sumoe": {"port_sumoe": 1}, "sumoe+fastdec": {"port_sumoe": 1, "moe_fastdec": 1}}
P = ["Write a long, detailed essay about the history of the Roman Empire, its institutions, its economy and its decline. ",
     "Write a complete Python implementation of a red-black tree with insert, delete and search, with comments and tests. "]
PROMPTS = [p + str(rep) for rep in range(2) for p in P]


class Srv:
    def __init__(self, port):
        self.port = port

    def call(self, method, path, body=None, retry=True):
        for _ in range(40):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
            c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            if r.status == 409 and retry:
                time.sleep(0.5)
                continue
            if r.status != 200:
                raise SystemExit(f"{path}: HTTP {r.status} {data[:300]!r}")
            return json.loads(data)
        raise SystemExit(f"{path}: busy")

    def arm(self, flags, depth=None, cache_mb=None):
        body = dict(flags)
        if depth is not None:
            body["mtp_depth"] = depth
            body.setdefault("mtp_fixed", depth if os.environ.get("TRAIN_FIXED") else -1)       # TRAIN_FIXED=1: draft at exactly `depth` (no cost policy)
        if cache_mb is not None:
            body["cache_limit_mb"] = cache_mb
        out = self.call("POST", "/admin/flags", body)
        self.chat("Say hello in one sentence.", 16)                      # warm: compiled regions retrace once after a switch
        return out

    def chat(self, q, n):
        body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        text, t_first, t_last, ntok = [], None, None, n
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            for ch in d.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    t_last = time.perf_counter()
                    t_first = t_first or t_last
                    text.append(piece)
        c.close()
        return "".join(text), ntok, (t_first, t_last)

    def run4(self, n):
        sums, rates = [], []
        for q in PROMPTS:
            txt, ntok, (t0, t1) = self.chat(q, n)
            sums.append(hashlib.md5(txt.encode()).hexdigest()[:12])
            if t0 and t1 and t1 > t0 and ntok > 1:
                rates.append((ntok - 1) / (t1 - t0))
        self.last_rates = rates
        self.last_rates = rates
        return sums, statistics.median(rates) if rates else 0.0


def ab(srv, arms, depths, rounds, n, log):
    """arms: [(name, flags, cache_mb)]; interleaved A B A B per depth. Returns {(depth, name): (md5s, [rates])}."""
    res = {}
    for d in depths:
        for rd in range(rounds):
            for name, flags, cmb in arms:
                srv.arm({**ALL_OFF, **flags}, depth=d, cache_mb=cmb)
                sums, rate = srv.run4(n)
                m, rs = res.setdefault((d, name), (sums, []))
                rs.append(rate)
                if sums != m:
                    res[(d, name)] = (m, rs)
                    log(f"  NOTE md5 changed between rounds for {name} d{d}: {sums} vs {m}")
                pr = self_rates = srv.last_rates
                kinds = (f"prose {statistics.mean(pr[0::2]):.1f} code {statistics.mean(pr[1::2]):.1f}" if len(pr) == 4 else "")
                log(f"  d{d} round{rd} {name}: {rate:.1f} tok/s ({kinds}) md5 {' '.join(sums)}")
    return res


def prefill_ttft(srv, arms, rounds, log, words=4500):
    """TTFT of a long fresh prompt (prefix cache is dropped by every flag switch) per arm, A B A B."""
    for rd in range(rounds):
        for name, flags, cmb in arms:
            srv.arm({**ALL_OFF, **flags}, depth=None, cache_mb=cmb)
            q = ("Summarize this text. " + " ".join(f"word{rd}x{i}" for i in range(words)))
            t0 = time.perf_counter()
            _, ntok, (t1, _) = srv.chat(q, 2)
            st = srv.call("GET", "/v1/stats")
            log(f"  prefill round{rd} {name}: ttft {t1 - t0:.2f} s")


def summarize(res, base, log):
    for (d, name), (sums, rs) in sorted(res.items()):
        ref = res.get((d, base))
        eq = "REF" if name == base else ("EQUAL" if ref and sums == ref[0] else "DIFF")
        log(f"RESULT d{d} {name} {eq} median {statistics.median(rs):.1f} tok/s (n={len(rs)}) md5 {' '.join(sums)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int); ap.add_argument("exp"); ap.add_argument("what", nargs="?", default="")
    ap.add_argument("--depths", default="0,3"); ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--limits", default="256,1024,4096,16384"); ap.add_argument("--cache-mb", type=int, default=None)
    ap.add_argument("--on", default="{}"); ap.add_argument("--out")
    a = ap.parse_args()
    out = open(a.out, "a") if a.out else None

    def log(s):
        print(s, flush=True)
        if out:
            out.write(s + "\n"); out.flush()
    srv = Srv(a.port)
    depths = [int(x) for x in a.depths.split(",")]
    if a.cache_mb:
        srv.call("POST", "/admin/flags", {"cache_limit_mb": a.cache_mb})
    base = [("off", {}, a.cache_mb)]
    if a.exp == "cache":
        arms = [(f"cache{m}", {}, int(m)) for m in a.limits.split(",")]
        summarize(ab(srv, arms, depths, a.rounds, a.n, log), "cache256", log)
    elif a.exp == "prefill":
        prefill_ttft(srv, [(f"cache{m}", {}, int(m)) for m in a.limits.split(",")], a.rounds, log)
    elif a.exp == "prefill":
        prefill_ttft(srv, [(f"cache{m}", {}, int(m)) for m in a.limits.split(",")], a.rounds, log)
    elif a.exp == "lever":
        names = a.what.split(",")
        arms = base + [(nm, LEVERS[nm], a.cache_mb) for nm in names]
        summarize(ab(srv, arms, depths, a.rounds, a.n, log), "off", log)
    elif a.exp == "mtp":                                          # depth policy A/B at mtp depth 3; --on = the lever set under test
        on = json.loads(a.on)
        arms = [("constant", {**on, "mtp_policy": "constant", "mtp_fixed": -1}, a.cache_mb), ("robust", {**on, "mtp_policy": "robust", "mtp_fixed": -1}, a.cache_mb)]
        arms += [(f"fixed{k}", {**on, "mtp_policy": "robust", "mtp_fixed": k}, a.cache_mb) for k in (1, 2, 3)]
        summarize(ab(srv, arms, [3], a.rounds, a.n, log), "constant", log)
    elif a.exp == "final":
        arms = base + [("winners", json.loads(a.on), a.cache_mb)]
        summarize(ab(srv, arms, depths, a.rounds, a.n, log), "off", log)
    elif a.exp == "kl":
        for nm in a.what.split(","):
            srv.arm(ALL_OFF, depth=0, cache_mb=a.cache_mb)
            srv.call("POST", "/admin/tap", {"mode": "ref"})
            txt, ntok, _ = srv.chat(PROMPTS[0], a.n)
            srv.call("POST", "/admin/tap", {"mode": "off"})
            srv.arm({**ALL_OFF, **LEVERS[nm]}, depth=0, cache_mb=a.cache_mb)
            srv.call("POST", "/admin/tap", {"mode": "cmp"})
            res = []
            for q in PROMPTS[:1]:
                srv.chat(q, a.n)
            srv.call("POST", "/admin/tap", {"mode": "off"})
            log(f"KL {nm} {json.dumps(srv.call('GET', '/admin/tap'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
