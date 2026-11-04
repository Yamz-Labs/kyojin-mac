"""day2 served lever gate on ONE resident server: the levers switch through POST /admin/flags (kyojin_mlx.flags), no reload between arms.
Per arm (every runtime lever OFF + that lever ON): 4 greedy replies (temperature 0, prose + code, 2 reps, --n tokens) -> md5 of each reply vs the
all-off arm; then one tok/s A/B all-off vs all-on with the golive prompts (4 prose + 4 code, --bench-n tokens, no temperature as Hermes).
The Engram gate is fixed when the server starts (KYOJIN_ENGRAM_GATE in the caller's env): tools/day2_gate.sh gates it with fresh processes.
usage: day2_served.py TREE PACK PORT [--depth 3] [--arms a,b] [--n 128] [--bench-n 256] [--no-md5] [--no-served] [--on-extra JSON] [--log FILE]
       day2_served.py --list   (arm names)
Output lines: "MD5 <arm> <EQUAL|DIFF|REF> <md5 x4>" and "SERVED <off|on> prose <median> code <median> tok/s"."""
import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ALL_OFF = {"glue_fuse": 0, "glue_fuse_norm": 0, "idx_fast": 0, "moe_fastdec": 0, "moeroof": 0, "dense_fast": 0, "moefuse": 0, "pf_glu": 0, "pf_split_gu": 0,
           "dspol": 0, "bound_layers": 0, "idx_qblock": 0, "pfroof2": 0}
ARMS = {"glue": {"glue_fuse": 1}, "gluenorm": {"glue_fuse": 1, "glue_fuse_norm": 1}, "idxfast": {"idx_fast": 1}, "moefastdec": {"moe_fastdec": 1},
        "moeroof": {"moeroof": 1}, "moeroof1": {"moeroof": 1, "moeroof_dec": 1}, "moeroofnopf": {"moeroof": 1, "moeroof_pf": 0},
        "moeroof+fastdec": {"moeroof": 1, "moe_fastdec": 1},
        "densefast": {"dense_fast": 1}, "moefuse": {"moefuse": 1}, "pfglu": {"pf_glu": 1}, "pfsplit": {"pf_split_gu": 1}, "dspol": {"dspol": 1},
        "bound": {"bound_layers": 1}, "qblock": {"idx_qblock": 512}, "pfroof2": {"pfroof2": 1, "idx_qblock": 512}}
ALL_ON = {"glue_fuse": 1, "idx_fast": 1, "moe_fastdec": 1, "dense_fast": 1, "moefuse": 1, "pf_glu": 1, "dspol": 1, "bound_layers": 1, "idx_qblock": 512}
MD5_P = ["Write a long, detailed essay about the history of the Roman Empire, its institutions, its economy and its decline. ",
         "Write a complete Python implementation of a red-black tree with insert, delete and search, with comments and tests. "]
PROSE = ["Write a long, detailed essay about the history of the Roman Empire, its institutions, its economy and its decline.",
         "Explain in detail how the immune system fights a viral infection, from first contact to long-term memory.",
         "Describe the causes, the course and the consequences of the French Revolution as a narrative for a curious reader.",
         "Write a thoughtful essay on how cities should plan for climate change over the next fifty years."]
CODE = ["Write a complete Python implementation of a red-black tree with insert, delete and search, with comments and tests.",
        "Write a Python module that implements an LRU cache, a thread-safe variant, and unit tests for both.",
        "Write a Python implementation of Dijkstra and A* on a weighted grid graph, with a small command line and tests.",
        "Write a Python class for a B-tree with insert, search and delete, with docstrings and a test suite."]


def main():
    if "--list" in sys.argv:
        print(" ".join(ARMS))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("tree"); ap.add_argument("pack"); ap.add_argument("port", type=int)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--bench-n", type=int, default=256)
    ap.add_argument("--no-md5", action="store_true"); ap.add_argument("--no-served", action="store_true")
    ap.add_argument("--on-extra", default="{}", help='extra flags of the all-on arm, e.g. {"pf_split_gu": 1}')
    ap.add_argument("--vlm", default=os.environ.get("KYOJIN_VLM_PKG", "")); ap.add_argument("--log", default=None)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        print(f"unknown arm(s) {bad}; known: {' '.join(ARMS)}", file=sys.stderr)
        return 2
    base = f"http://127.0.0.1:{a.port}"
    logp = a.log or f"/tmp/day2_served_{a.port}.log"
    env = {**os.environ, "PYTHONPATH": a.tree + (":" + a.vlm if a.vlm else ""), "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "kyojin_mlx.serve.server", a.pack, "--port", str(a.port), "--sessions", "1", "--ctx", "8192",
           "--mtp-depth", str(a.depth), "--prefill-chunk", "2048"]
    pr = subprocess.Popen(cmd, cwd=a.tree, env=env, stdout=open(logp, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    print("server pid", pr.pid, "log", logp, flush=True)

    def call(path, body=None, timeout=900):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(base + path, data, {"Content-Type": "application/json"} if data else {})
        try:
            return json.load(urllib.request.urlopen(r, timeout=timeout))
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{path}: HTTP {e.code} {e.read()[:400]!r}")

    def chat(q, n, temperature):
        body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": n, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        if temperature is not None:
            body["temperature"] = temperature
        t0 = time.perf_counter()
        d = call("/v1/chat/completions", body)
        return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"], time.perf_counter() - t0

    def arm(flags):
        want = {**ALL_OFF, **flags}
        for _ in range(20):                                       # 409 while the engine still counts the last request: retry
            try:
                got = call("/admin/flags", want)["flags"]
                break
            except SystemExit as e:
                if "HTTP 409" not in str(e):
                    raise
                time.sleep(0.5)
        else:
            raise SystemExit("/admin/flags: still busy (409) after 10 s")
        wrong = {k: got[k]["value"] for k, v in want.items() if got[k]["value"] != str(int(v))}
        if wrong:
            raise SystemExit(f"flags not applied: {wrong}")
        chat("Say hello in one sentence.", 16, 0.0)                   # warm: the compiled regions retrace once after a switch

    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2)
                break
            except Exception:                                         # noqa: BLE001
                if pr.poll() is not None:
                    print(open(logp).read()[-3000:])
                    raise SystemExit("server died")
                time.sleep(2)
        print(f"server up after {time.time() - t0:.0f} s; engram gate env {os.environ.get('KYOJIN_ENGRAM_GATE', 'auto')}", flush=True)
        if not a.no_md5:
            ref = None
            for name in ["off"] + arms + ["off"]:                       # the closing off arm checks that switching back restores the bits
                arm(ARMS.get(name, {}))
                sums = []
                for rep in range(2):
                    for p in MD5_P:
                        txt, n, dt = chat(p + str(rep), a.n, 0.0)
                        sums.append(hashlib.md5(txt.encode()).hexdigest()[:12])
                res = "REF" if ref is None else ("EQUAL" if sums == ref else "DIFF")
                ref = ref or sums
                print(f"MD5 {name} {res} {' '.join(sums)}", flush=True)
        if not a.no_served:
            on = {**ALL_ON, **json.loads(a.on_extra)}
            for name, flags in (("off", {}), ("on", on)):
                arm(flags)
                med = {}
                for kind, qs in (("prose", PROSE), ("code", CODE)):
                    v = []
                    for q in qs:
                        _, n, dt = chat(q, a.bench_n, None)
                        v.append(n / dt)
                        print(f"  {name} {kind}: {n} tok in {dt:.2f} s = {n / dt:.1f} tok/s", flush=True)
                    med[kind] = statistics.median(v)
                print(f"SERVED {name} prose {med['prose']:.1f} code {med['code']:.1f} tok/s", flush=True)
        return 0
    finally:
        for f in (lambda: os.killpg(pr.pid, signal.SIGTERM), lambda: os.kill(pr.pid, signal.SIGTERM)):
            try:
                f()
            except Exception:                                         # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(main())
