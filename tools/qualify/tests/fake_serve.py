"""Stand-in for kyojin_mlx.serve.server in the qualify tests: same CLI subset, /health 503 then 200, SSE chat completions with usage + timings.
--ctx above LIMIT exits like a Metal OOM at start-up."""
import argparse, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--port", type=int); ap.add_argument("--ctx", type=int); ap.add_argument("--sessions", type=int, default=1)
ap.add_argument("--model-id"); ap.add_argument("--prefill-chunk"); ap.add_argument("--prefix-block"); ap.add_argument("--mtp-depth"); ap.add_argument("--tokenizer")
a = ap.parse_args()
a.c = a.ctx
LIMIT = int(os.environ.get("LIMIT", "50000"))
if a.c > LIMIT:
    print("torch.OutOfMemoryError: Metal out of memory", flush=True); sys.exit(1)
ready = time.time() + 0.5


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def do_GET(self):
        ok = time.time() >= ready
        self.send_response(200 if ok else 503); self.end_headers(); self.wfile.write(b"{}")

    def do_POST(self):
        n = int(self.headers["Content-Length"]); b = json.loads(self.rfile.read(n))
        if self.path == "/v1/tokenize":
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"count": max(1, len(b["messages"][-1]["content"]) // 4)}).encode()); return
        text = b["messages"][-1]["content"]
        pt = max(1, len(text) // 4); mt = b["max_tokens"]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if pt + mt > a.c:
            self.wfile.write(b'data: {"error": "context overflow"}\n\n'); return
        time.sleep(pt / 2e5)
        for i in range(min(mt, 20)):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": f"w{(hash(text[:50]) + i) % 7 if False else i} "}}]}).encode() + b"\n\n"); self.wfile.flush(); time.sleep(0.002)
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": min(mt, 20)}, "timings": {"ttft_s": round(pt / 2e5, 4), "predicted_per_second": 123.0}}).encode() + b"\n\ndata: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
