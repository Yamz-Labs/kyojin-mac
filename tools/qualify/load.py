"""Served-side measurements, stdlib only (the conformance check uses the pack's tokenizer through transformers, already in the venv).
Adapted from the CUDA/ROCm load.py (nanb bench.py/agent.py, rel174c3 drv.py/bench_n.py, gdrv.py sha set, lane-kv-conformance.py rule 1). Prompts are cut
from a plain-text corpus (QUALIFY_CORPUS, default ~/qualify_data; falls back to the Python stdlib docs + sources when absent); the chars/token
ratio is calibrated on the server's own usage.prompt_tokens, so no tokenizer import is needed for sizing. The speeds are the server's own timings
(`timings.ttft_s`, `predicted_per_second`) when it sends them, else the client's clock."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CORPORA = ("c4.utf8", "wiki.utf8", "code.utf8", "technical.utf8", "multilingual.utf8")
FR = ["Explique en quatre ou cinq phrases pourquoi le ciel est bleu, puis donne une analogie simple.",
      "Écris un court poème sur la mer en six vers, puis explique le choix des rimes.",
      "Décris les étapes pour préparer une bonne pâte à crêpes, avec les quantités pour six personnes."]
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False}


def corpus_dir() -> Path:
    return Path(os.environ.get("QUALIFY_CORPUS", Path.home() / "qualify_data")).expanduser()


def stdlib_corpus() -> str:
    """Fallback corpus with no data files: pydoc topics (English technical prose) + the sources of a few stdlib modules."""
    import pydoc_data.topics as T
    import sysconfig
    parts = ["\n\n".join(T.topics[k] for k in sorted(T.topics))]
    lib = Path(sysconfig.get_paths()["stdlib"])
    for m in ("argparse.py", "http/server.py", "json/decoder.py", "subprocess.py", "threading.py", "typing.py", "dataclasses.py", "pathlib.py"):
        try:
            parts.append((lib / m).read_text(errors="replace"))
        except OSError:
            pass
    return "\n\n".join(parts)


def load_corpus(d: Path | None = None) -> str:
    d = d or corpus_dir()
    big = "\n\n".join((d / n).read_text(encoding="utf8", errors="replace") for n in CORPORA if (d / n).exists())
    return big if len(big) >= 1_000_000 else (big + "\n\n" + stdlib_corpus())


class Client:
    def __init__(self, port: int, model: str = "qual", corpus: str | None = None):
        self.url = f"http://127.0.0.1:{port}"
        self.model = model
        self.big = corpus if corpus is not None else load_corpus()
        if len(self.big) < 500_000:
            raise RuntimeError(f"no corpus under {corpus_dir()} and the stdlib fallback is too small")
        self.cpt = 3.2          # chars per token, recalibrated by calibrate()
        self.salt = 0

    def post(self, path: str, body: dict, timeout: int = 3000):
        return urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer x"})

    def get(self, path: str) -> dict:
        return json.load(urllib.request.urlopen(self.url + path, timeout=30))

    def stream(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "stream_options": {"include_usage": True}, **NOTHINK}
        t0 = time.time(); first = None; usage = None; txt = []; err = None; tm = None
        try:
            for line in urllib.request.urlopen(self.post("/v1/chat/completions", body), timeout=3000):
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                j = json.loads(line[5:])
                if j.get("error"):
                    err = str(j["error"])[:300]
                usage = j.get("usage") or usage
                tm = j.get("timings") or tm
                for c in j.get("choices", []):
                    d = c.get("delta", {})
                    s = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if s:
                        first = first or time.time(); txt.append(s)
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:240]}"       # the server's own message (context overflow, bad field)
        except Exception as e:
            err = repr(e)[:300]
        te = time.time(); ct = (usage or {}).get("completion_tokens", 0)
        ttft = (tm or {}).get("ttft_s") if tm else None
        ttft = ttft if ttft is not None else (first or te) - t0
        dec = (tm or {}).get("predicted_per_second") or (round((ct - 1) / (te - first), 2) if first and te > first and ct > 1 else None)
        return dict(ptok=(usage or {}).get("prompt_tokens"), ntok=ct, ttft=round(ttft, 3), total=round(te - t0, 2), dec_tps=dec,
                    cached=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                    sha=hashlib.sha1("".join(txt).encode()).hexdigest()[:10], err=err, t0=t0, first=first, end=te)

    def prompt(self, tokens: int, slot: int) -> str:
        """~tokens of corpus text, distinct for every (slot, call): fresh offsets so no prefix cache can answer."""
        self.salt += 1
        T = self.big
        n = int(tokens * self.cpt)
        if n + 1000 > len(T):                           # longer than the corpus: tile it with a changing header per tile
            reps = n // max(1, len(T) - 2000) + 1
            T = "\n\n".join(f"[tile {k}.{self.salt}] " + self.big[k * 1013:] for k in range(reps + 1))
        off = (30_000 + slot * 337_000 + self.salt * 91_003) % max(1, len(T) - n - 10)
        return f"[{slot}.{self.salt}] Voici un document.\n\n{T[off:off + n]}\n\nEn deux phrases, dis quels sujets ce document mélange."

    def count(self, prompt: str) -> int | None:
        """Prompt tokens as the server counts them (its chat template included): POST /v1/tokenize on the same body a chat request would carry."""
        try:
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], **NOTHINK}
            return json.load(urllib.request.urlopen(self.post("/v1/tokenize", body), timeout=300))["count"]
        except Exception:
            return None

    def fit(self, tokens: int, slot: int, tol: float = 0.04) -> str:
        """A prompt of `tokens` (within -tol..0 of it, never above): the chars per token of a corpus slice depends on the slice (code vs prose), so the
        estimate is corrected against the server's own count (<= 5 rounds)."""
        best = None
        for _ in range(5):
            p = self.prompt(tokens, slot)
            n = self.count(p)
            if n is None:
                return p
            if n <= tokens:
                best = p
                if n >= tokens * (1 - tol):
                    return p
            self.cpt *= max(0.5, min(1.5, tokens / n)) if n else 1.0
            self.cpt = max(0.3, self.cpt)
        return best or self.prompt(int(tokens * 0.8), slot)

    def calibrate(self, tokens: int = 1500) -> float:
        r = self.stream(self.prompt(tokens, 0), 1)
        if r["ptok"]:
            self.cpt = self.cpt * tokens / r["ptok"]
        return self.cpt

    def near_full(self, ctx: int, slot: int, fill: float = 0.97, reply: int = 32) -> dict:
        """One request whose prompt is `fill` of --ctx (rounded down so prompt + reply < ctx)."""
        want = int(ctx * fill) - reply
        return self.stream(self.fit(want, slot), reply)

    def concurrent(self, n: int, tokens: int, max_tokens: int = 128) -> dict:
        res = [None] * n
        prompts = [self.fit(tokens, i) for i in range(n)]
        bar = threading.Barrier(n)

        def run(i):
            bar.wait(); res[i] = self.stream(prompts[i], max_tokens)
        th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in th]; [t.join() for t in th]
        ok = [r for r in res if not r["err"] and r["dec_tps"] and r["first"]]
        span = (max(r["end"] for r in ok) - min(r["first"] for r in ok)) if ok else 0
        return dict(n=n, ptok=[r["ptok"] for r in res], ttft=[r["ttft"] for r in res], dec_tps=[r["dec_tps"] for r in res],
                    agg_dec_tps=round(sum(r["ntok"] - 1 for r in ok) / span, 1) if span > 0 else None, errors=[r["err"] for r in res if r["err"]])

    def sha_set(self, max_tokens: int = 256) -> list:
        return [{k: r[k] for k in ("sha", "ntok", "dec_tps", "err")} for r in (self.stream(p, max_tokens) for p in FR)]


def conformance(pack: str) -> dict:
    """lane-kv-conformance rule 1: the template renders TOOLS BEFORE the system text (else every new session re-pays the tool schemas).
    Rendered with the pack's own tokenizer / template (the Mac server has no /apply-template). DS4.1 uses its built-in encoder: checked there."""
    msgs = [{"role": "system", "content": "SYSMARK"}, {"role": "user", "content": "USERMARK"}]
    tools = [{"type": "function", "function": {"name": "TOOLMARK", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    try:
        cfg = json.loads((Path(pack) / "config.json").read_text())
        from kyojin_mlx import modeldesc
        row = modeldesc.by_model_type(cfg.get("model_type", ""))
        if row is not None and row.builtin_template == "ds41_chat":
            from kyojin_mlx.serve import ds41_chat
            p = ds41_chat.render({"messages": msgs, "tools": tools}).prompt if hasattr(ds41_chat, "render") else None
            if p is None:
                return dict(ok=True, detail="ds41 built-in encoder: no jinja to check")
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(pack)
            tpl = (Path(pack) / "chat_template.jinja")
            p = tok.apply_chat_template(msgs, tools=tools, tokenize=False, chat_template=tpl.read_text() if tpl.exists() else None)
        a, b = p.find("TOOLMARK"), p.find("SYSMARK")
        return dict(ok=0 <= a < b, detail=f"tools@{a} sys@{b}")
    except Exception as e:
        return dict(ok=False, detail=repr(e)[:200])
