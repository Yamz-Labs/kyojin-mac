# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server (stdlib only) on top of the continuous-batching engine.

    POST /v1/chat/completions   streaming and not; tools, tool_choice, reasoning_content, usage (+ prompt_tokens_details.cached_tokens)
    POST /v1/completions        raw prompt (text or token ids)
    GET  /v1/models  /health  /v1/stats

python -m kyojin_mlx.serve.server MODEL_DIR --sessions 4 --ctx 32768 --port 8080     (tools/mac_serve.sh sets the environment)
The HTTP threads never touch MLX: they tokenize, hand the ids to the engine thread and detokenize / parse the tokens it queues."""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .chat import ChatError, FAMILIES, family_of, render_resume, token_cuts
from .sampling import Sampling
from .tokinc import IncTok
from . import autoconfig
from . import hostloop as HL
from . import loopguard as LG
from .. import engram_rows as ER
from .. import fired as FD
from ..ds41_vision import VID_BASE
from .hostloop import Detok, Reply           # noqa: F401  (the stream text classes live next to the other host-loop levers; no mlx import there)
from .. import gpuspan as _GS

# Probe plumbing (flag gpu_span / gpu_kern, both default 0): the command-buffer counters name every Metal pipeline at creation, so they must
# be installed before the model compiles its first kernel. Dispatch recording stays off unless gpu_kern is on.
_GS.install()


class RawParser:
    """/v1/completions: the text as it comes."""

    def feed(self, text):
        return [("content", text)] if text else []

    def finish(self):
        return []


class App:
    def __init__(self, engine, tokenizer, family, model_id: str, defaults: dict, stop_ids: set, log=print):
        self.engine, self.tokenizer, self.family, self.model_id = engine, tokenizer, family, model_id
        self.defaults, self.stop_ids, self.log = defaults, stop_ids, log
        self.tok_lock = threading.Lock()
        self.inc = IncTok(tokenizer)
        self.count = 0
        self.t0 = time.time()
        self.tap_ref = self.tap_last = None                      # kltap results (POST / GET /admin/tap)

    def encode(self, text: str) -> list:
        with self.tok_lock:
            return self.inc.encode(text)                         # flag tok_inc: a long prompt from where it differs from a recent one

    def encode_chat(self, r) -> tuple:
        """A rendered chat -> (ids, cuts, spans, real ids). Without an image: spans and real ids are None, ids are what the model reads."""
        ids = self.encode(r.prompt)
        if not r.images:
            return ids, token_cuts(r.cut_texts, ids, self.encode), None, None
        v = self.family.vision
        plans = [v.plan(rec) for rec in r.images]
        ids, spans = v.expand(ids, plans)
        cuts = token_cuts(r.cut_texts, ids, lambda t: v.expand(self.encode(t), plans, whole=False)[0])
        return ids, cuts, spans, v.real_ids(ids)

    def decode(self, ids: list) -> str:
        with self.tok_lock:
            return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ---------------------------------------------------------------------------- request fields
    def sampling(self, body: dict) -> Sampling:
        def num(key, default, typ=float, lo=None, hi=None):
            v = body.get(key)
            if v is None:
                v = self.defaults.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ChatError(f"{key} must be a number")
            v = typ(v)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                raise ChatError(f"{key} out of range")
            return v
        seed = body.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ChatError("seed must be an integer")
        return Sampling(temperature=num("temperature", 1.0, lo=0.0, hi=5.0), top_p=num("top_p", 1.0, lo=0.0, hi=1.0),
                        top_k=num("top_k", 0, int, lo=0), min_p=num("min_p", 0.0, lo=0.0, hi=1.0), seed=seed)

    @staticmethod
    def max_tokens(body: dict):
        for key in ("max_completion_tokens", "max_tokens"):
            v = body.get(key)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ChatError(f"{key} must be a positive integer")
            return v
        return None

    @staticmethod
    def stops(body: dict) -> list:
        s = body.get("stop")
        if s is None:
            return []
        s = [s] if isinstance(s, str) else s
        if not isinstance(s, list) or len(s) > 16 or not all(isinstance(x, str) for x in s):
            raise ChatError("stop must be a string or an array of up to 16 strings")
        return s

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:24]}"


def usage_of(stats: dict, n_completion: int) -> dict:
    p = stats["prompt_tokens"]
    return {"prompt_tokens": p, "completion_tokens": n_completion, "total_tokens": p + n_completion,
            "prompt_tokens_details": {"cached_tokens": stats["cached_tokens"]}}


TIMINGS_PERIOD_S = 1.0        # a streamed reply carries a partial `timings` at most this often


def timings_from(queue_s: float, ttft_s: float, total_s: float, cached: int, prompt: int, n_completion: int) -> dict:
    """One `timings` shape on every Kyojin server: the llama.cpp keys (prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms,
    predicted_per_second, cache_n) plus this server's queue_s, ttft_s and total_s."""
    gen = max(total_s - ttft_s, 1e-9)
    pre = max(ttft_s - queue_s, 1e-9)
    return {"queue_s": round(queue_s, 4), "ttft_s": round(ttft_s, 4), "total_s": round(total_s, 4),
            "cache_n": cached, "prompt_n": prompt - cached, "prompt_ms": round(pre * 1000, 1),
            "prompt_per_second": round((prompt - cached) / pre, 2),
            "predicted_n": n_completion, "predicted_ms": round(gen * 1000, 1),
            "predicted_per_second": round(max(n_completion - 1, 0) / gen, 2)}


def timings_of(stats: dict, n_completion: int) -> dict:
    out = timings_from(stats["queue_s"], stats["ttft_s"], stats["total_s"], stats["cached_tokens"], stats["prompt_tokens"], n_completion)
    if stats.get("stages") is not None:                  # flag ttft_stages: the same ttft split into its stages (ms)
        out["ttft_stages"] = stats["stages"]
    return out


def live_timings(s, now: float):
    """The partial `timings` of a running session (None before its first token), same shape as timings_of, read from the engine's Session."""
    if s is None or not s.t_first:
        return None
    return timings_from((s.t_start or s.t_first) - s.t_submit, s.t_first - s.t_submit, now - s.t_submit, s.cached, s.n_prompt, s.n_gen)


def engage(backend=None) -> dict:
    """Path counters of every loaded kyojin_mlx module (module-level STATS / HITS dicts) and the CED stats of the model: a fast path that never
    engages (its fallback counter grows) shows here instead of as a silent slowdown."""
    out = {}
    for name, mod in list(sys.modules.items()):
        if name.startswith("kyojin_mlx.") and mod is not None:
            for attr in ("STATS", "HITS"):
                d = getattr(mod, attr, None)
                if isinstance(d, dict) and any(isinstance(v, (int, float)) and v for v in d.values()):
                    out[name[11:] + ("" if attr == "STATS" else ".hits")] = {k: v for k, v in d.items() if isinstance(v, (int, float))}
    from .. import ced
    st = getattr(getattr(backend, "lm", None), ced.STATE_ATTR, None)
    if st is not None:
        out["ced"] = {"on": bool(st.on), "tail": st.tail, **st.stats}
    return out


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.") or host == "::ffff:127.0.0.1"


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "kyojin-mac"
        _pend = None                                            # SSE frames of the batch being handled (flag serve_coalesce); None = write each frame

        @property
        def disable_nagle_algorithm(self):                      # read by StreamRequestHandler.setup for every connection (flag serve_nodelay)
            FD.hit("serve_nodelay" if HL.NODELAY else "serve_nodelay_off")     # one count per accepted connection
            return HL.NODELAY

        def log_message(self, fmt, *args):                      # request lines are logged by the endpoints
            pass

        # ------------------------------------------------------------------------ plumbing
        def _send(self, code: int, obj, headers=None):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str):
            self._send(code, {"error": {"message": msg, "type": "invalid_request_error" if code < 500 else "server_error"}})

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 64 * 2 ** 20:
                raise ChatError("a JSON body of up to 64 MB is required")
            try:
                body = json.loads(self.rfile.read(n))
            except json.JSONDecodeError as exc:
                raise ChatError(f"invalid JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise ChatError("the body must be a JSON object")
            return body

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj):
            data = b"data: " + (json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() if obj != "[DONE]" else b"[DONE]") + b"\n\n"
            if self._pend is not None:
                self._pend.append(data)
                return
            self.wfile.write(data)
            self.wfile.flush()

        def _sse_flush(self):
            frames, self._pend = self._pend, None
            if frames:
                self.wfile.write(b"".join(frames))
                self.wfile.flush()

        # ------------------------------------------------------------------------ routes
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/healthz"):
                self._send(200, {"status": "ok", "model": app.model_id, "family": app.family.name, **app.engine.status()})
            elif path == "/v1/stats":
                self._send(200, {**app.engine.status(), "uptime_s": round(time.time() - app.t0, 1), "requests": app.count,
                                 "max_decode_tick_ms": round(app.engine.max_step_gap * 1e3, 1), "eng_read": dict(ER.STATS),
                                 "timing": {k: round(v, 4) for k, v in app.engine.stats.items() if k[:2] in ("t_", "n_") or k.startswith(("gpu_", "ps_"))}, "engage": engage(getattr(app.engine, "backend", None))})
            elif path == "/admin/flags":
                if self._admin_ok():
                    from .. import flags as FL
                    self._send(200, {"flags": FL.get_all(), "gen": FL.GEN})
            elif path == "/admin/tap":
                if self._admin_ok():
                    t = app.engine.tap or app.tap_last
                    self._send(200, t.result() if t else {})
            elif path in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": app.model_id, "object": "model", "created": int(app.t0),
                                                             "owned_by": "kyojin-mac", "max_model_len": app.engine.ctx}]})
            else:
                self._error(404, f"no such endpoint: {path}")

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/v1/chat/completions", "/chat/completions"):
                    self._completion(self._body(), chat=True)
                elif path in ("/v1/completions", "/completions"):
                    self._completion(self._body(), chat=False)
                elif path == "/admin/flags":
                    self._admin_flags()
                elif path == "/admin/tap":
                    self._admin_tap()
                elif path == "/v1/tokenize" or path == "/tokenize":
                    body = self._body()
                    if "prompt" in body:
                        ids = app.encode(body.get("prompt"))
                    else:
                        ids, _, _, real = app.encode_chat(app.family.render(body))
                        ids = real if real is not None else ids
                    self._send(200, {"tokens": ids, "count": len(ids)})
                else:
                    self._error(404, f"no such endpoint: {path}")
            except (ChatError, ValueError) as exc:
                self._error(400, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:                                # noqa: BLE001
                app.log("request error:\n" + traceback.format_exc())
                try:
                    self._error(500, f"{type(exc).__name__}: {exc}")
                except Exception:                                    # noqa: BLE001
                    pass

        # ------------------------------------------------------------------------ owner admin (lever A/B on one loaded model)
        def _admin_ok(self) -> bool:
            """Owner-only: the loopback peer only (a server bound to 0.0.0.0 still refuses remote callers)."""
            if is_loopback(self.client_address[0]):
                return True
            self._error(403, "admin endpoints answer on localhost only")
            return False

        def _admin_tap(self):
            """POST /admin/tap {"mode": "ref"|"cmp"|"off"}: arm the KL probe for the next request (see kltap)."""
            if not self._admin_ok():
                return
            from .kltap import Tap
            body = self._body()
            mode = body.get("mode", "off")
            self._body_save = body.get("save")
            if body.get("load"):                                # a saved reference (other server run, other spec)
                app.tap_ref = Tap.load(body["load"])
            if mode == "off":
                t = app.engine.tap
                if t is not None:
                    app.tap_last = t
                    if t.mode == "ref":
                        app.tap_ref = t
                        if self._body_save:
                            t.save(self._body_save)
                app.engine.tap = None
            elif mode == "ref":
                app.engine.tap = Tap("ref")
            elif mode == "cmp" and app.tap_ref is not None:
                app.engine.tap = Tap("cmp", app.tap_ref)
            else:
                self._error(400, "mode: ref | cmp (needs a finished ref) | off")
                return
            self._send(200, {"mode": mode})

        def _admin_flags(self):
            """POST /admin/flags {name: value, ...}: switch lever flags (kyojin_mlx.flags) between requests. 409 while a request runs; the memory
            prefix cache is dropped after a switch (its snapshots were computed with the old levers; a --kv-disk store is not touched)."""
            if not self._admin_ok():
                return
            from .. import flags as FL
            body = self._body()
            st = app.engine.status()
            if st["waiting"] or st["prefilling"] or st["decoding"]:
                self._error(409, "requests are running: switch flags when the server is idle")
                return
            body = dict(body)
            cache_mb = body.pop("cache_limit_mb", None)
            depth = body.pop("mtp_depth", None)
            policy = body.pop("mtp_policy", None)                   # "robust" | "constant"
            fixed = body.pop("mtp_fixed", None)                      # forced draft depth for every greedy session; -1 = back to the cost policy
            try:
                if cache_mb is not None:
                    import mlx.core as mx
                    cache_mb = int(cache_mb)
                    if cache_mb < 0:
                        raise ValueError("cache_limit_mb must be >= 0")
                flags = FL.set_many(body)
                if depth is not None:
                    app.engine.set_depth(int(depth))
                if policy is not None:
                    if policy not in ("robust", "constant"):
                        raise ValueError("mtp_policy must be robust or constant")
                    app.engine.robust = policy == "robust"
                if fixed is not None:
                    app.engine.depth_override = None if int(fixed) < 0 else int(fixed)
                if cache_mb is not None:
                    mx.set_cache_limit(cache_mb * 2 ** 20)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            except KeyError as exc:
                self._error(400, f"unknown flag {exc.args[0]!r} (known: {', '.join(FL.FLAGS)})")
                return
            except PermissionError as exc:
                self._error(400, str(exc))
                return
            dropped = app.engine.prefix.shrink(0)
            app.log(f"admin: flags {body} (gen {FL.GEN}, prefix cache {'dropped' if dropped else 'empty'})")
            self._send(200, {"flags": flags, "gen": FL.GEN, "prefix_dropped": dropped, "cache_limit_mb": cache_mb})

        # ------------------------------------------------------------------------ completions
        def _completion(self, body: dict, chat: bool):
            from .. import flags as FL
            with FL.request_override(FL.check_request_flags(body.get("kyojin_flags"))):         # held until the reply is fully sent
                self._completion_run(body, chat)

        def _completion_run(self, body: dict, chat: bool):
            if (body.get("n") or 1) != 1:
                raise ChatError("n must be 1")
            stream = body.get("stream") is True
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            from .engine import ttft_on
            stages = {} if ttft_on() else None               # flag ttft_stages: the stages this thread owns (render, tokenize)
            t_r = time.perf_counter()
            if chat:
                rendered = render_resume(app.family, body)
                parser = app.family.parser(rendered)
            else:
                cuts, spans, real_ids = [], None, None
                prompt = body.get("prompt")
                if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
                    prompt = prompt[0]
                if not isinstance(prompt, str) and not (isinstance(prompt, list) and prompt and all(isinstance(t, int) and t < VID_BASE for t in prompt)):
                    raise ChatError("prompt must be a string or an array of token ids")
                parser = RawParser()
            t_k = time.perf_counter()
            if chat:
                ids, cuts, spans, real_ids = app.encode_chat(rendered)
            elif isinstance(prompt, str):
                ids = app.encode(prompt)
            else:
                ids = list(prompt)
            if stages is not None:
                stages["render_ms"] = round((t_k - t_r) * 1e3, 3)
                stages["tokenize_ms"] = round((time.perf_counter() - t_k) * 1e3, 3)
            sampling = app.sampling(body)
            stops = app.stops(body)
            handle = app.engine.submit(ids, max_new=app.max_tokens(body), sampling=sampling,
                                       stop_ids=() if body.get("ignore_eos") is True else app.stop_ids, cuts=cuts, vl=spans, real_ids=real_ids,
                                       stages=stages)
            reply = Reply(app, parser, stops)
            app.count += 1
            rid = app.new_id("chatcmpl" if chat else "cmpl")
            created = int(time.time())
            state = {"started": False, "tm_at": time.perf_counter()}

            def tick():
                """At most once per TIMINGS_PERIOD_S, queue the partial timings for the next chunk of a streamed reply."""
                now = time.perf_counter()
                if stream and now - state["tm_at"] >= TIMINGS_PERIOD_S and (tm := live_timings(getattr(handle, "session", None), now)) is not None:
                    state["tm"], state["tm_at"] = tm, now

            def carry():
                """No chunk took the queued timings (held text, tool-call XML): an empty delta carries them."""
                if "tm" in state:
                    self._sse(chunk(text="") if not chat else chunk())
            msg = {"content": [], "reasoning": [], "calls": []}
            obj_name = "chat.completion.chunk" if chat else "text_completion"

            def chunk(delta=None, finish=None, text=None, extra=None):
                if "tm" in state:                                  # a progressive timings is due: it rides on the next chunk
                    extra = {**(extra or {}), "timings": state.pop("tm")}
                if chat:
                    c = {"index": 0, "delta": delta if delta is not None else {}, "finish_reason": finish}
                else:
                    c = {"index": 0, "text": text or "", "finish_reason": finish}
                return {"id": rid, "object": obj_name, "created": created, "model": app.model_id, "choices": [c], **(extra or {})}

            def emit(events):
                for ev in events:
                    kind = ev[0]
                    if kind == "content":
                        msg["content"].append(ev[1])
                        if stream:
                            self._sse(chunk({"content": ev[1]}) if chat else chunk(text=ev[1]))
                    elif kind == "reasoning":
                        msg["reasoning"].append(ev[1])
                        if stream:
                            self._sse(chunk({"reasoning_content": ev[1]}))
                    elif kind == "tool_start":
                        call = {"id": ev[2], "type": "function", "function": {"name": ev[3], "arguments": ""}}
                        if ev[4] is not None:
                            call["namespace"] = ev[4]
                        msg["calls"].append(call)
                        if stream:
                            self._sse(chunk({"tool_calls": [{"index": ev[1], **call}]}))
                    elif kind == "tool_args":
                        msg["calls"][ev[1]]["function"]["arguments"] += ev[2]
                        if stream:
                            self._sse(chunk({"tool_calls": [{"index": ev[1], "function": {"arguments": ev[2]}}]}))

            stats, reason = None, "stop"
            lg = LG.ReplyGuard(parser) if chat else None             # flag loop_guard: thinking loops, EOS inside an open think block
            if lg is not None and not lg.on:
                lg = None                                            # (a prompt with images is guarded and resumed too: see resume())
            n_carry = 0                                              # tokens of earlier engine requests of this reply

            def resume(cause: str) -> bool:
                """Close the think block (the template's close tag), then let the model write its answer in a new engine request that continues
                the text so far. Once per reply; False = the caller ends the reply."""
                nonlocal handle, n_carry
                limit = app.max_tokens(body)
                close = app.encode(LG.THINK_CLOSE)
                if limit is not None and len(lg.gen) + len(close) >= limit:
                    return False
                if not lg.closes():
                    return False
                if not handle.cancelled.is_set():
                    handle.cancel()
                app.log(f"{rid[-6:]} loop guard: {cause} after {len(lg.gen)} tokens ({lg.hit or 'eos in think'}): think block closed")
                emit([ev for t in close for ev in reply.push(t)])
                n_carry = len(lg.gen) + len(close)
                # A prompt with images keeps its virtual ids and its image spans (they sit before the reply, so they still hold); the real ids
                # of the reply are the sampled ids themselves.
                handle = app.engine.submit(ids + lg.gen + close, max_new=None if limit is None else limit - n_carry, sampling=sampling,
                                           stop_ids=() if body.get("ignore_eos") is True else app.stop_ids, cuts=cuts, stages=stages,
                                           vl=spans, real_ids=None if real_ids is None else real_ids + lg.gen + close)
                return True
            try:
                if stream:
                    self._sse_start()
                    state["started"] = True
                    if chat:
                        self._sse(chunk({"role": "assistant", "content": ""}))
                while True:
                    try:
                        ev = handle.events.get(timeout=10.0)
                    except queue.Empty:                          # a long prefill or a wait for a slot: tell a stream client we are alive
                        if stream:                               # (and find out if it left)
                            self.wfile.write(b": waiting\n\n")
                            self.wfile.flush()
                        continue
                    was_think = lg is not None and LG.in_think(parser)
                    if ev[0] == "tok" and HL.COALESCE:           # everything the engine queued meanwhile: one write, one frame per token
                        toks, tail = HL.drain(handle.events, ev, queue.Empty)
                        FD.hit("serve_coalesce")
                        FD.hit("serve_coalesce_toks", len(toks))
                        self._pend = [] if stream else None
                        try:
                            tick()
                            evs = HL.drain_events(reply, toks)
                            emit(evs)
                            carry()
                        finally:
                            self._sse_flush()
                        if lg is not None and not reply.stopped and lg.observe(toks, evs, was_think):
                            if resume("loop"):
                                continue
                            reply.stopped = True                 # a second loop: the reply ends here, with the text so far
                        if reply.stopped and not handle.cancelled.is_set():
                            handle.cancel()
                        if tail is not None:
                            reason, stats = tail[1], tail[2]
                            if lg is not None and not reply.stopped and lg.eos_open(reason) and resume("eos"):
                                continue
                            break
                    elif ev[0] == "tok":
                        tick()
                        evs = reply.push(ev[1])
                        emit(evs)
                        carry()
                        if lg is not None and not reply.stopped and lg.observe([ev[1]], evs, was_think):
                            if resume("loop"):
                                continue
                            reply.stopped = True
                        if reply.stopped and not handle.cancelled.is_set():
                            handle.cancel()
                    else:
                        reason, stats = ev[1], ev[2]
                        if lg is not None and not reply.stopped and lg.eos_open(reason) and resume("eos"):
                            continue
                        break
                emit(reply.finish())
            except (BrokenPipeError, ConnectionResetError):
                handle.cancel()
                return
            except Exception:
                handle.cancel()
                raise
            if reason in ("error", "shutdown"):
                if stream:
                    self._sse({"error": {"message": "engine error", "type": "server_error"}})
                    self._sse("[DONE]")
                    return
                raise RuntimeError("engine error")
            finish = "stop" if reply.stopped or reason == "cancelled" else reason
            if finish == "stop" and msg["calls"]:
                finish = "tool_calls"
            if n_carry:                                          # a resumed reply: the usage covers the whole text, not the last engine request
                stats = {**stats, "prompt_tokens": len(ids), "completion_tokens": stats["completion_tokens"] + n_carry}
            n_comp = reply.n_tokens if reply.stopped else stats["completion_tokens"]
            usage, timings = usage_of(stats, n_comp), timings_of(stats, n_comp)
            app.log(f"{rid[-6:]} {'chat' if chat else 'cmpl'} prompt={stats['prompt_tokens']} cached={stats['cached_tokens']} gen={n_comp} "
                    f"queue={stats['queue_s']:.2f}s ttft={stats['ttft_s']:.2f}s total={stats['total_s']:.2f}s "
                    f"prefill={timings['prompt_per_second']:.1f} tok/s decode={timings['predicted_per_second']:.1f} tok/s {finish}")
            if stream:
                self._sse(chunk(finish=finish, extra={"timings": timings}))
                if include_usage:
                    self._sse({"id": rid, "object": obj_name, "created": created, "model": app.model_id, "choices": [], "usage": usage})
                self._sse("[DONE]")
                return
            if chat:
                message = {"role": "assistant", "content": "".join(msg["content"]) if msg["content"] or not msg["calls"] else None}
                if msg["reasoning"]:
                    message["reasoning_content"] = "".join(msg["reasoning"])
                if msg["calls"]:
                    message["tool_calls"] = msg["calls"]
                choice = {"index": 0, "message": message, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": "".join(msg["content"]), "finish_reason": finish}
            self._send(200, {"id": rid, "object": "chat.completion" if chat else "text_completion", "created": created,
                             "model": app.model_id, "choices": [choice], "usage": usage, "timings": timings})

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    allow_reuse_address = True


# ------------------------------------------------------------------------------------------------ main
def stop_token_ids(model_dir: str, cfg: dict, tokenizer, family: str) -> set:
    ids: set = set()

    def add(v):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, list):
            ids.update(x for x in v if isinstance(x, int))
    add(cfg.get("eos_token_id"))
    add(cfg.get("text_config", {}).get("eos_token_id"))
    gc = Path(model_dir) / "generation_config.json"
    if gc.is_file():
        add(json.loads(gc.read_text()).get("eos_token_id"))
    add(tokenizer.eos_token_id)
    from .. import modeldesc
    for t in modeldesc.stop_tokens_of(family):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id:
            ids.add(i)
    return ids


def sampling_defaults(model_dir: str) -> dict:
    d = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}       # the Qwen card values (the box servers' fallback)
    gc = Path(model_dir) / "generation_config.json"
    if gc.is_file():
        cfg = json.loads(gc.read_text())
        d.update({k: cfg[k] for k in ("temperature", "top_p", "top_k", "min_p") if cfg.get(k) is not None})
    return d


def open_kv_disk(args, backend, log=print):
    """The disk KV tier (--kv-disk DIR), or None. The namespace holds everything that changes the cached values: architecture, pack files, CED
    setting, prefill chunk (the CED tail and the MoE paths depend on the chunk grid), the drafter ring layout."""
    if not args.kv_disk:
        return None
    ns = backend.disk_namespace()
    if ns is None:
        log(f"kv-disk: not available for the {backend.kind} backend, running without it")
        return None
    from ..kvdisk import KvDisk
    disk = KvDisk(args.kv_disk, {**ns, "chunk": args.prefill_chunk}, backend.make_cache, block=args.kv_disk_block,
                  cap_bytes=int(args.kv_disk_gib * 2 ** 30), log=log)
    log(f"kv-disk: {disk.describe()} (block {disk.block} tokens)")
    return disk



def auto_prefill_chunk(free: int | None) -> int:
    """Default prefill chunk from the free unified memory after the load (engine.free_unified_bytes): 2048 with 4 GiB or more (the chunk's
    transient buffers are ~1.3 GiB at 2048 and 1024 alike, measured on the 4B pack, and the 64K-prompt peak is the KV, not the chunk), 1024 with
    1.5 GiB or more, else 256. 256 costs 15-20 % of the prefill speed against 1024-2048 (dense decode + matmul wins above 512 rows)."""
    from .engine import GIB
    if free is None:
        return 1024
    return 2048 if free >= 4 * GIB else 1024 if free >= 1.5 * GIB else 256


def has_stacked_experts(backend) -> bool:
    """True when the model holds foreign stacked-expert packs (ForeignSwitchGLU): their prefill decode cost is per chunk, not per token."""
    from ..foreign_linear import ForeignSwitchGLU
    m = getattr(backend, "model", None)
    return m is not None and any(isinstance(x, ForeignSwitchGLU) for _, x in m.named_modules())


def resolve_chunk(args, backend, log=print):
    """--prefill-chunk / --prefix-block left unset: the chunk from the free memory, the block as the smallest multiple of it >= 512."""
    if args.prefill_chunk is not None and args.prefill_chunk > 8192:
        raise SystemExit(f"--prefill-chunk {args.prefill_chunk}: chunks above 8192 rows are refused (16384-row chunks panicked the Ultra on 29/10 00:26)")
    if args.prefill_chunk is None:
        from .engine import free_unified_bytes
        args.prefill_chunk = auto_prefill_chunk(free_unified_bytes())
        log(f"prefill chunk: {args.prefill_chunk} (automatic, from the free unified memory)")
        big = int(os.environ.get("KYOJIN_STACK_CHUNK", "0"))               # flag: a pack with stacked experts decodes every expert to fp16 once per chunk (prefill gathered matmul), so a larger chunk amortises it
        if big > args.prefill_chunk and args.prefill_chunk == 2048 and has_stacked_experts(backend):
            args.prefill_chunk = min(big, 8192)
            log(f"prefill chunk: {args.prefill_chunk} (KYOJIN_STACK_CHUNK, stacked experts decoded once per chunk)")
    if args.prefill_chunk != 2048 and hasattr(backend, "warm"):          # the load warm-up compiled the 2048-row chunk; the first long prompt would pay the compile of this one
        backend.warm(log, chunk=args.prefill_chunk, steps=0)
    if args.prefix_block is None:
        args.prefix_block = -(-512 // args.prefill_chunk) * args.prefill_chunk
    if args.prefix_block % args.prefill_chunk:
        raise SystemExit(f"--prefix-block {args.prefix_block} must be a multiple of --prefill-chunk {args.prefill_chunk}")
    if args.kv_disk and args.kv_disk_block % args.prefill_chunk:
        raise SystemExit(f"--kv-disk-block {args.kv_disk_block} must be a multiple of --prefill-chunk {args.prefill_chunk}")


CACHE_TIGHT_GIB = 4.0                                              # room after the weights under which the MLX buffer cache stays at its floor


CACHE_FLOOR_MAX_MB = int(os.environ.get("KYOJIN_CACHE_FLOOR_MAX_MB", "2048"))


def cache_limit_mb_for(weight_bytes: int, recommended_bytes: int = 0, room_bytes: float | None = None) -> int:
    """MLX buffer cache limit: weights / 10, between a floor and 16 GiB. A 256 MB cap made the 172 GiB DS4.1 pack prefill 40 % slower (21.5K tokens: 88 s vs
    52 s at 16 GiB, 29/10 train A/B). The floor follows the machine, not the weights: stale KV buffers fill a small limit, every recycle then evicts the hot
    per-step buffers and each step allocates them again (real 0.6B GGUF, 02/11: 18-21 ms a token at 256-512 MB, 5.4 ms at 1 GiB, same GPU work).
    recommended_bytes / 32, capped at CACHE_FLOOR_MAX_MB: 2 GiB on a 256 GiB Mac (was 7.6 GiB: gpt-oss-20b served 22 GB vs 16 GB at 2 GiB, same speed at N1 /
    N4 / N8 / pf8K, 12/11), 256 MB on a 16 GiB one.
    room_bytes (the GPU ceiling less the weights; None = unknown): under CACHE_TIGHT_GIB the KV, the prefill temporaries and the cache all live in that room, and the
    cache takes the floor. 16 GB Air, 27B GGUF (8.8 GiB, room 1.2 GiB): 899 MB of cache cost 0.66 GB of footprint for 2.5 % of the prefill speed (6K tokens, 163 vs 159 tok/s)."""
    floor = max(256, min(CACHE_FLOOR_MAX_MB, recommended_bytes / 32 / 2 ** 20))
    if room_bytes is not None and room_bytes < CACHE_TIGHT_GIB * 2 ** 30:
        return int(floor)
    return int(min(16384, max(floor, weight_bytes / 10 / 2 ** 20)))


def build_app(args, log=print):
    import mlx.core as mx
    from transformers import AutoTokenizer
    from .backends import load_backend
    from .engine import Engine
    autoconfig.resolve_serve_args(args)                            # the pack-derived defaults (idempotent; --print-config already did it)
    if args.cache_limit_mb:
        mx.set_cache_limit(args.cache_limit_mb * 2 ** 20)
    elif args.cache_limit_mb is None:
        mx.set_cache_limit(256 * 2 ** 20)                              # until the weights are loaded (sized below)
    t0 = time.time()
    if args.mtp_depth is None and os.environ.get("KYOJIN_MTP_DEPTH") is not None:
        args.mtp_depth = int(os.environ["KYOJIN_MTP_DEPTH"])        # 0 = off, N = depth, unset = automatic
    mtp = None if args.mtp_depth is None else args.mtp_depth > 0     # None: on wherever the pack carries the vendor's draft head
    ced_tail = args.ced_tail if args.ced_tail in (None, "exact") else int(args.ced_tail)
    if args.no_ced:
        log("ced: off (--no-ced)")
    backend, cfg = load_backend(args.model, mtp=mtp, mtp_head=args.mtp_head, ced=not args.no_ced, ced_tail=ced_tail, log=log)
    if args.mtp_depth is None or getattr(backend, "foreign", False):     # read as stored: no draft head path yet (load_q35 logged it)
        args.mtp_depth = backend.default_mtp_depth if backend.supports_spec else 0
    elif args.mtp_depth > 0 and not backend.supports_spec:
        raise SystemExit("--mtp-depth needs the pack's draft head (mtp.* tensors, a Qwen3.5 mlx pack: its mtp/ sidecar, tools/mtp_sidecar.py, or --mtp-head DIR)")
    for k in ("row_cost", "draft_cost"):
        if getattr(args, k) is None and hasattr(backend, k):
            setattr(args, k, getattr(backend, k))
    from .. import tuning                                              # measured priors of this pack on this device (tools/ds41_bench.py --tuning), else the constants
    args.row_cost = tuning.serve("row_cost", 0.4) if args.row_cost is None else args.row_cost
    args.draft_cost = tuning.serve("draft_cost", 0.15) if args.draft_cost is None else args.draft_cost
    log(f"depth policy: row cost {args.row_cost}, draft cost {args.draft_cost} (tuning {tuning.INFO['status']})")
    if args.cache_limit_mb is None:
        info = mx.device_info()
        room = None
        if os.environ.get("KYOJIN_CACHE_TIGHT", "1") not in ("0", "") and info.get("memory_size") and info.get("max_recommended_working_set_size"):
            from .engine import kv_ceiling
            room = kv_ceiling(int(info["memory_size"]), int(info["max_recommended_working_set_size"])) - mx.get_active_memory()
        mb = cache_limit_mb_for(mx.get_active_memory(), info.get("max_recommended_working_set_size", 0), room)
        mx.set_cache_limit(mb * 2 ** 20)
        log(f"MLX buffer cache limit {mb} MB (weights / 10, working set / 32..16 GiB, the floor when the weights leave under 4 GiB; --cache-limit-mb overrides)")
    from .. import memfit
    memfit.apply(log)                                                  # KYOJIN_WIRED_FIT / KYOJIN_CACHE_FIT (both off by default)
    log(f"model loaded in {time.time() - t0:.1f}s ({backend.kind}), peak {mx.get_peak_memory() / 2 ** 30:.2f} GB")
    from .. import kvwin
    from .. import tq
    if tq.config() is not None:
        log(f"kv: TurboQuant pool + fp16 overlay {tq.config()} (--kv-mode tq)")
    if kvwin.config() is not None:
        log(f"kvwin: quantized KV pool + fp16 overlay {kvwin.config()} (KYOJIN_KVWIN, opt-in)")
    tdir = Path(args.tokenizer or args.model)
    if (tdir / "tokenizer_config.json").is_file():
        tok = AutoTokenizer.from_pretrained(str(tdir))
    else:                                                          # a bare tokenizer.json (DS4.1 reference dir)
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=str(tdir / "tokenizer.json"))
    fam_name = args.family or _family_name(args.model, cfg)
    template = Path(args.chat_template).read_text() if args.chat_template else None
    defaults = sampling_defaults(args.model)
    fdef = {}
    if args.thinking is not None:
        fdef[_thinking_key(fam_name)] = args.thinking
    if args.reasoning_effort:
        fdef["reasoning_effort"] = args.reasoning_effort
    family = FAMILIES[fam_name](tok, args.model, template, fdef)
    if getattr(backend, "vision", None) is not None and hasattr(family, "vision") and family.vision is None:   # flag visiongen (generic backend)
        family.vision = backend.vision
    from .engine import GIB, default_kv_budget
    kv_budget = int(args.kv_budget * GIB) if args.kv_budget is not None else default_kv_budget()
    log("kv budget: " + ("none (free memory unknown)" if kv_budget is None else f"{kv_budget / GIB:.2f} GiB"
                         + ("" if args.kv_budget is not None else " (RAM less headroom, weights and buffer cache)")))
    resolve_chunk(args, backend, log)
    disk = open_kv_disk(args, backend, log)
    spec_costs = None
    if args.mtp_depth > 0:
        from .. import tuning
        spec_costs = tuning.spec_costs(args.model, "tq" if tq.config() is not None else "fp16")
        log("mtp depth policy: " + (f"measured round costs at {sorted(spec_costs)} tokens (tuning spec)" if spec_costs else
                                    f"constant costs row {args.row_cost:g} draft {args.draft_cost:g} (measured table: KYOJIN_SPEC_COSTS=1)"))
    if os.environ.get("KYOJIN_ABLIT_MU_JOB"):              # "<token ids json>:<out json>", once, before serving
        from .. import ablit as AB
        src, out = os.environ["KYOJIN_ABLIT_MU_JOB"].split(":")
        AB.mu_job(backend, src, out, log=log)
    if args.sessions is None:
        args.sessions = _auto_sessions(backend, kv_budget, log)
    engine = Engine(backend, sessions=args.sessions, ctx=args.ctx, chunk=args.prefill_chunk, block=args.prefix_block,
                    prefix_bytes=int(args.prefix_cache_mb * 2 ** 20), decode_per_chunk=args.decode_per_chunk, log=log,
                    mtp_depth=args.mtp_depth, kv_budget=kv_budget, row_cost=args.row_cost, draft_cost=args.draft_cost, disk=disk,
                    spec_costs=spec_costs, depth_override=int(os.environ["KYOJIN_MTP_FIXED"]) if os.environ.get("KYOJIN_MTP_FIXED") else None)
    model_id = args.model_id or Path(args.model.rstrip("/")).name
    stops = stop_token_ids(args.model, cfg, tok, fam_name)
    bos = cfg.get("bos_token_id", cfg.get("text_config", {}).get("bos_token_id"))
    engine.bad_ids = frozenset({bos} - stops) if isinstance(bos, int) else frozenset()
    tg = [tok.convert_tokens_to_ids(x) for x in ("<think>", "</think>")]   # flag spec_typical_phase: the think tags, when the tokenizer has them as single tokens
    if all(isinstance(i, int) and i >= 0 and i != tok.unk_token_id for i in tg):
        engine.think_ids = (tg[0], tg[1])
    engine.detok = lambda ids: tok.decode(ids, skip_special_tokens=False)   # flag spec_typical_cooldown: the line signal reads the session's text
    app = App(engine, tok, family, model_id, defaults, stops, log)
    return app


def _family_name(model: str, cfg: dict) -> str:
    """Chat family of the pack. A bare GGUF of a mapped architecture without a modeldesc row gets the generic family (the file's own chat template and stop ids)."""
    from .. import modeldesc
    mt = cfg.get("model_type", "")
    if modeldesc.backend_desc(mt) is None:
        from .. import foreign_linear as FL
        if FL.is_foreign(model) == "gguf":
            from .chat import family_of_gguf
            return family_of_gguf(mt)
    return family_of(mt)


def _thinking_key(fam_name: str) -> str:
    """Chat-template kwarg that switches thinking (the modeldesc row of the family)."""
    from .. import modeldesc
    return next((d.thinking_key for d in modeldesc.DESCS if d.server == fam_name), "enable_thinking" if fam_name == "generic" else "thinking")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?", help="pack directory, fixed defaults (--model PATH derives them from the pack)")
    ap.add_argument("--model", dest="model_opt", metavar="PATH", help="pack directory; options not given are derived from the pack")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sessions", type=int, default=None, help="requests decoded together (also the most that prefill / wait for a slot at once); default: derived "
                    "from the KV budget and one session's KV at the TQ_FROM depth, 4..8")
    ap.add_argument("--ctx", type=int, default=None, help="context tokens one request may use (prompt + reply); default: the pack's max_position_embeddings, else 32768")
    ap.add_argument("--prefill-chunk", type=int, default=None, help="prefill chunk (tokens); a decode step runs between two chunks. Default: 8192 for a long-context pack, else automatic 2048 / 1024 / 256 by the free unified memory after the load")
    ap.add_argument("--prefix-block", type=int, default=None, help="prefix snapshots every N tokens (a multiple of the prefill chunk; default 512, or the chunk when larger)")
    ap.add_argument("--prefix-cache-mb", type=float, default=None, help="memory for prefix snapshots (0 = no prefix sharing); default: ctx / 64 between 1024 and 16384 MB")
    ap.add_argument("--decode-per-chunk", type=int, default=1)
    ap.add_argument("--no-ced", action="store_true",
                    help="turn off CED prefill. CED (DeepSeek's own encoder / decoder prefill, arXiv 2609.19969) is on by default for packs whose layer "
                         "layout supports it (DS4.1): the second half of the layers runs on the last --ced-tail rows of each prefill chunk only; the "
                         "global KV and the encoder half stay exact. It pays when --prefill-chunk is several times the tail (use 2048 or more). "
                         "Tools: tools/ced_gate.py (quality), tools/ced_bench.py (speed)")
    ap.add_argument("--ced-tail", default=None, metavar="ROWS|exact",
                    help="rows of the decoder half per CED chunk (default: the sliding window = 128, DeepSeek's setting; with --mtp-depth at least the "
                         "drafter tap window). 'exact' = the receptive field (D * (w - 1) + w rows): last-row logits equal the exact prefill, less saved")
    ap.add_argument("--kv-disk", metavar="DIR",
                    help="checkpoint prompt KV to DIR every --kv-disk-block tokens and reload the longest match instead of prefilling again (also after "
                         "a restart). The cache of a prefix is bit for bit what the prefill produced; directories are namespaced by pack, CED setting "
                         "and --prefill-chunk, so a changed setup never reads old files. Default: <user cache>/kyojin_mlx/kvdisk/<pack>-<hash>")
    ap.add_argument("--no-kv-disk", action="store_true", help="turn off the default disk KV tier (does not cancel an explicit --kv-disk DIR)")
    ap.add_argument("--kv-disk-gib", type=float, default=None, metavar="GiB", help="disk cap of --kv-disk, least recently used chains go first (0 = no cap); default 100")
    ap.add_argument("--kv-disk-block", type=int, default=autoconfig.KV_DISK_BLOCK, help="tokens per checkpoint file (a multiple of --prefill-chunk)")
    ap.add_argument("--mtp-head", metavar="DIR", default=None, help="Qwen3.5 mlx pack: directory with the mtp.* tensors (default: <pack>/mtp)")
    ap.add_argument("--mtp-depth", type=int, default=None,
                    help="speculative decode: most drafts per session and round (0 = off; default: the pack's draft block size (DS4.1: 5) when draft-head "
                         "tensors exist, else off; env KYOJIN_MTP_DEPTH). Needs a pack with draft-head tensors (DS4.1: mtp.*). "
                         "Every active session adds its pending token and its drafts to one verify pass of the batched core (the code of a plain decode step) "
                         "and one draft pass serves all sessions; greedy output equals plain decode (bit-exact on EXL3 packs; on mlx packs the verify rows differ from the 1 row step at rounding level, so exact logit ties can flip a token). The depth of a "
                         "session adapts to its acceptance (0 allowed).")
    ap.add_argument("--row-cost", type=float, default=None, help="depth policy: cost of one more verify row, in solo decode steps (measure with tools/spec_gen.py)")
    ap.add_argument("--draft-cost", type=float, default=None, help="depth policy: cost of one draft call, in solo decode steps (measure with tools/spec_gen.py)")
    ap.add_argument("--kv-budget", type=float, default=None, metavar="GiB",
                    help="memory for session caches (active and parked) plus prefix snapshots. Over it, the oldest parked session's KV is dropped and "
                         "rebuilt on resume; active sessions are never dropped. Default: free unified memory after the load minus 1 GiB")
    ap.add_argument("--cache-limit-mb", type=int, default=None,
                    help="MLX buffer cache limit in MB (default: weights / 10 between 256 MB and 16 GiB)")
    ap.add_argument("--model-id"); ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--tokenizer", help="directory with tokenizer.json when the pack has none (slice packs)")
    ap.add_argument("--chat-template", help="Jinja file instead of the model's own")
    ap.add_argument("--thinking", type=lambda v: v.lower() in ("1", "true", "yes", "on"), default=None,
                    help="default thinking switch when a request names none (qwen: enable_thinking; ds41: thinking); default: on when the pack's template has a thinking switch")
    ap.add_argument("--reasoning-effort", default=None, help="default reasoning effort when a request names none (default: medium)")
    ap.add_argument("--print-config", action="store_true", help="print the effective pack-derived options as JSON and exit without loading the model")
    ap.add_argument("--kv-mode", choices=["default", "tq", "fp16"], default="default",
                    help="tq: TurboQuant KV pool on the Qwen3.5 full-attention layers (rotation + Lloyd-Max codebook, one fp16 norm per head vector) with the fp16 "
                         "sink + window overlay (= KYOJIN_TQ). default: tq (6-bit, 4 sink, 256 window) unless KYOJIN_TQ / KYOJIN_KVWIN is set; "
                         "fp16: opt out, plain fp16 cache (= KYOJIN_TQ=0)")
    ap.add_argument("--k-bits", type=int, default=6, choices=[4, 5, 6], help="--kv-mode tq: bits per K value (6 = tq6w, 5 = tq5w)")
    ap.add_argument("--v-bits", type=int, default=None, choices=[4, 5, 6], help="--kv-mode tq: bits per V value (default: --k-bits)")
    ap.add_argument("--tq-sink", type=int, default=4, help="--kv-mode tq: first tokens kept fp16")
    ap.add_argument("--tq-window", type=int, default=256, help="--kv-mode tq: last tokens kept fp16")
    return ap



def _row_kv(model: str) -> str | None:
    """kv field of the modeldesc row of a pack directory (None: no config.json / no row)."""
    try:
        from .. import modeldesc
        mt = json.load(open(os.path.join(model, "config.json"))).get("model_type", "")
        row = modeldesc.backend_desc(mt)
        return row.kv if row is not None else None
    except (OSError, ValueError):
        return None


def _auto_sessions(backend, kv_budget, log) -> int:
    """Default --sessions: one fresh session cache at the TQ_FROM depth is measured (256 prefilled tokens, scaled), then autoconfig.auto_sessions."""
    import mlx.core as mx
    from .. import tq
    depth = tq.FROM if tq.FROM > 0 else 4096
    try:
        c = backend.make_cache_for(depth)
        mx.eval(backend.prefill_chunk(c, [0] * 256))
        per = backend.nbytes(c) / 256 * depth
        del c
    except Exception as e:                                    # a backend that cannot prefill a bare chunk here: the fixed default
        log(f"sessions: {autoconfig.DEFAULT_SESSIONS} (KV per session not measured: {type(e).__name__})")
        return autoconfig.DEFAULT_SESSIONS
    n = autoconfig.auto_sessions(per, kv_budget, getattr(backend, "max_rows", 16))
    log(f"sessions: {n} (one session ~{per / 2 ** 30:.2f} GiB of KV at {depth} tokens, budget "
        + ("unknown" if not kv_budget else f"{kv_budget / 2 ** 30:.1f} GiB") + f"; {autoconfig.DEFAULT_SESSIONS}..{autoconfig.SESSIONS_CAP})")
    return n

def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.model is None:
        args.model = args.model_opt
    if args.model is None:
        ap.error("a pack directory is required (positional or --model PATH)")
    from ..foreign_names import gguf_serve_dir
    args.model = gguf_serve_dir(args.model)                        # a GGUF without its config / tokenizer files: built from the metadata
    if args.model_opt:
        args.model_opt = args.model
    if args.kv_mode == "default" and _row_kv(args.model) == "fp16":
        args.kv_mode = "fp16"                                      # the row asks for a plain cache (attention sinks: the TurboQuant pool cannot serve them)
    if args.kv_mode == "default" and "KYOJIN_TQ" not in os.environ and not os.environ.get("KYOJIN_KVWIN", "").strip():
        args.kv_mode = "tq"                                        # TurboQuant is the default KV cache (28/10 owner; 9B 64K d4 +18 %, KL below k6v6)
    if args.kv_mode == "fp16":
        os.environ["KYOJIN_TQ"] = "0"
    if args.kv_mode == "tq":
        os.environ["KYOJIN_TQ"] = f"{args.k_bits}:{args.v_bits or args.k_bits}:{args.tq_sink}:{args.tq_window}"
    args = autoconfig.resolve_serve_args(args)
    if args.kv_disk and args.prefill_chunk and args.kv_disk_block % args.prefill_chunk:
        ap.error(f"--kv-disk-block {args.kv_disk_block} must be a multiple of --prefill-chunk {args.prefill_chunk}")
    if args.ced_tail not in (None, "exact") and not args.ced_tail.isdigit():
        ap.error("--ced-tail takes a number of rows or 'exact'")
    if args.print_config:
        json.dump(autoconfig.effective_config(args), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)   # noqa: E731
    app = build_app(args, log)                                     # the model is loaded on this thread, so the engine loop runs on it too
    srv = Server((args.host, args.port), make_handler(app))        # (MLX streams are per thread: a lazy array cannot cross threads)
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()

    def bye(signum, frame):
        log(f"signal {signum}: stopping")
        app.engine.stop()
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    log(f"serving {app.model_id} ({app.family.name}) on http://{args.host}:{args.port}  sessions={args.sessions} ctx={args.ctx} "
        f"chunk={args.prefill_chunk} prefix-block={args.prefix_block} prefix-cache={args.prefix_cache_mb:g} MB mtp-depth={args.mtp_depth}")
    try:
        app.engine.run()
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
