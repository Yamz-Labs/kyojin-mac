"""Hermes Agent conformance of the GLM-5.3 chat server: the real HTTP server and Engine on a scripted backend (needs mlx, no model).

Same rig as test_hermes_conformance.py (DS4.1), with GLM's template, tokens and reply format: `<think>` is in the prompt, the reply is
`reasoning</think>prose<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>...` and ends on <|observation|> (after calls) or <|user|>.
The tags are single tokens, as in the real tokenizer.
"""
import http.client
import json
import threading

import mlx.core as mx
import pytest

from kyojin_mlx import modeldesc
from kyojin_mlx.serve.backends import Backend
from kyojin_mlx.serve.chat import GlmFamily
from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.server import App, Server, make_handler

V = 4096
MODEL = "glm-5.3"
THINK, THINK_END, USER, ASSIST, OBS, SYSTEM, EOT = "<think>", "</think>", "<|user|>", "<|assistant|>", "<|observation|>", "<|system|>", "<|endoftext|>"
CALL, CALL_END, ARGK, ARGK_END, ARGV, ARGV_END = "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
SPECIALS = ["[gMASK]", "<sop>", THINK, THINK_END, USER, ASSIST, OBS, SYSTEM, EOT, CALL, CALL_END, ARGK, ARGK_END, ARGV, ARGV_END,
            "<tool_response>", "</tool_response>"]


class Tok:
    """One id per special token, one id per character otherwise (ids handed out on first use)."""

    def __init__(self):
        self.ids, self.strs = {}, []
        for s in SPECIALS:
            self._id(s)

    def _id(self, s):
        if s not in self.ids:
            assert len(self.strs) < V, "char vocab is full"
            self.ids[s] = len(self.strs)
            self.strs.append(s)
        return self.ids[s]

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            sp = next((s for s in SPECIALS if text.startswith(s, i)), None)
            out.append(self._id(sp or text[i]))
            i += len(sp) if sp else 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.strs[i] for i in ids)


class Script(Backend):
    """Plays reply n (n = number of <|assistant|> tokens of the history) after the `<think>` (or `<think></think>`, thinking off) of the prompt."""
    kind = "script"
    max_rows = 4

    def __init__(self, tok, replies, off=False):
        self.tok, self.replies, self.off = tok, replies, off
        self.assistant, self.stop = tok.ids[ASSIST], tok.ids[USER]

    def _next(self, h):
        n = h.count(self.assistant)
        if n == 0 or n > len(self.replies):
            return self.stop
        last = len(h) - 1 - h[::-1].index(self.assistant)
        offset = len(h) - last - (3 if self.off else 2)
        text = self.replies[n - 1]
        reply = self.tok.encode(text) + [self.tok.ids[OBS if CALL_END in text else USER]]
        return reply[offset] if 0 <= offset < len(reply) else self.stop

    def _logits(self, rows):
        out = [[0.0] * V for _ in rows]
        for r, h in zip(out, rows):
            r[self._next(h)] = 100.0
        return mx.array(out, dtype=mx.float32)

    def make_cache(self):
        return {"h": []}

    def snapshot(self, cache):
        return self.fork(cache)

    def fork(self, cache):
        return {"h": list(cache["h"])}

    def prefill_chunk(self, cache, ids):
        cache["h"].extend(ids)
        return self._logits([cache["h"]])

    def batch(self, caches):
        return list(caches)

    def unbatch(self, bc, n):
        return list(bc)

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return self._logits([c["h"] for c in bc])

    def state(self, cache):
        return []

    def nbytes(self, cache):
        return sum(8 * len(c["h"]) for c in cache) if isinstance(cache, list) else 8 * len(cache["h"])


def call_text(name, **kv):
    return f"{CALL}{name}" + "".join(f"{ARGK}{k}{ARGK_END}{ARGV}{v}{ARGV_END}" for k, v in kv.items()) + CALL_END


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
SYSTEM_TEXT = "You are Hermes, a careful assistant. " * 12
CODE = "def f(x):\n    return x < 3 and '<b>' or \"\"\n"
R1 = ("The user wants the weather and a file; call both tools." + THINK_END + "Checking both." +
      call_text("get_weather", city="Paris", days="3") + call_text("write_file", path="007.py", content=CODE))
R2 = "Both tools answered." + THINK_END + "Paris is sunny, and the file is written."
R3 = "Short follow-up." + THINK_END + "Tomorrow looks the same."
R_OFF = "Plain answer without thinking."


class Rig:
    def __init__(self, replies, off=False, block=16, chunk=8, ctx=16384, defaults=None):
        self.tok = Tok()
        for r in replies:
            self.tok.encode(r)
        self.engine = Engine(Script(self.tok, replies, off), sessions=2, ctx=ctx, chunk=chunk, block=block, prefix_bytes=1 << 22)
        self.family = GlmFamily(self.tok, "unused", None, defaults or {})
        stops = {self.tok.ids[t] for t in (USER, OBS, EOT)}
        self.app = App(self.engine, self.tok, self.family, MODEL, {}, stops, log=lambda *a: None)
        self.srv = Server(("127.0.0.1", 0), make_handler(self.app))
        self.port = self.srv.server_address[1]
        self.engine.start()
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.engine.stop()

    def post(self, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode()
        conn.close()
        return resp.status, raw


def sse(raw):
    out = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith(":") or not line.strip():
                continue
            assert line.startswith("data: "), line
            out.append(line[6:] if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


def assemble(events):
    """What an OpenAI streaming client builds from the chunks."""
    msg = {"role": None, "content": "", "reasoning_content": "", "calls": {}, "finish": None, "usage": None}
    for ev in events[:-1]:
        if not ev["choices"]:
            msg["usage"] = ev["usage"]
            continue
        ch = ev["choices"][0]
        d = ch["delta"]
        msg["role"] = d.get("role", msg["role"])
        msg["content"] += d.get("content") or ""
        msg["reasoning_content"] += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            acc = msg["calls"].setdefault(tc["index"], {"id": None, "type": None, "function": {"name": "", "arguments": ""}})
            acc["id"] = tc.get("id") or acc["id"]
            acc["type"] = tc.get("type") or acc["type"]
            acc["function"]["name"] += (tc.get("function") or {}).get("name") or ""
            acc["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
        if ch["finish_reason"]:
            msg["finish"] = ch["finish_reason"]
    return msg


def hermes_body(messages, **kw):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "stream": True,
            "stream_options": {"include_usage": True}, "max_completion_tokens": 512, "parallel_tool_calls": True,
            "chat_template_kwargs": {"reasoning_effort": "high"}}
    body.update(kw)
    return body


def echo(msg):
    calls = [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
             for _, c in sorted(msg["calls"].items())]
    out = {"role": "assistant", "content": msg["content"], "reasoning_content": msg["reasoning_content"]}
    if calls:
        out["tool_calls"] = calls
    return out


@pytest.fixture
def rig():
    r = Rig([R1, R2, R3])
    yield r
    r.close()


def test_glm_row_selects_the_family():
    assert modeldesc.chat_desc("glm5_next").server == "glm"


def test_three_turn_session_with_parallel_tool_calls(rig):
    m1 = [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": "Weather in Paris, and write 007.py?"}]
    status, raw = rig.post(hermes_body(m1))
    assert status == 200
    ev = sse(raw)
    assert ev[-1] == "[DONE]"
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    t1 = assemble(ev)
    assert t1["reasoning_content"] == "The user wants the weather and a file; call both tools."
    assert t1["content"] == "Checking both."
    assert t1["finish"] == "tool_calls"
    calls = [t1["calls"][i] for i in sorted(t1["calls"])]
    assert [c["function"]["name"] for c in calls] == ["get_weather", "write_file"]
    assert all(c["id"].startswith("call_") and c["type"] == "function" for c in calls) and calls[0]["id"] != calls[1]["id"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert json.loads(calls[1]["function"]["arguments"]) == {"path": "007.py", "content": CODE}      # string parameters keep their text
    firsts = {}
    for e in ev[:-1]:
        for tc in (e["choices"][0]["delta"].get("tool_calls") or []) if e["choices"] else []:
            assert "index" in tc
            firsts.setdefault(tc["index"], tc)
    assert all(f.get("id") and f.get("type") == "function" and f["function"].get("name") for f in firsts.values())
    u1 = t1["usage"]
    assert u1["prompt_tokens"] > 64 and u1["completion_tokens"] > 0
    assert "cached_tokens" in u1["prompt_tokens_details"]

    m2 = m1 + [echo(t1), {"role": "tool", "tool_call_id": calls[0]["id"], "content": "sunny, 21C"},
               {"role": "tool", "tool_call_id": calls[1]["id"], "content": "written"}]
    p1 = rig.family.render(hermes_body(m1)).prompt
    p2 = rig.family.render(hermes_body(m2)).prompt
    assert p2.startswith(p1 + R1)                 # the history renders back to the exact turn-1 prompt and reply: the prefix cache can reuse it
    assert p2.endswith("<tool_response>sunny, 21C</tool_response><tool_response>written</tool_response><|assistant|><think>")
    t2 = assemble(sse(rig.post(hermes_body(m2))[1]))
    assert t2["finish"] == "stop" and t2["content"] == "Paris is sunny, and the file is written." and not t2["calls"]
    assert t2["reasoning_content"] == "Both tools answered."
    c2 = t2["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert c2 >= u1["prompt_tokens"] - 16 and c2 > 0

    m3 = m2 + [echo(t2), {"role": "user", "content": "And tomorrow?"}]
    t3 = assemble(sse(rig.post(hermes_body(m3))[1]))
    assert t3["content"] == "Tomorrow looks the same." and t3["finish"] == "stop"
    c3 = t3["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert c3 > c2 and c3 >= t2["usage"]["prompt_tokens"] - 16


def test_non_stream_reply_shape(rig):
    m1 = [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": "Weather in Paris, and write 007.py?"}]
    status, raw = rig.post(hermes_body(m1, stream=False))
    assert status == 200
    out = json.loads(raw)
    ch = out["choices"][0]
    assert out["object"] == "chat.completion" and ch["finish_reason"] == "tool_calls"
    msg = ch["message"]
    assert msg["role"] == "assistant" and msg["content"] == "Checking both."
    assert msg["reasoning_content"] == "The user wants the weather and a file; call both tools."
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["get_weather", "write_file"]
    assert all(isinstance(c["function"]["arguments"], str) and c["id"].startswith("call_") for c in msg["tool_calls"])
    assert json.loads(msg["tool_calls"][1]["function"]["arguments"])["content"] == CODE


def test_tool_call_only_reply_has_null_content():
    r = Rig(["Call it." + THINK_END + call_text("get_weather", city="Rome")])
    try:
        out = json.loads(r.post(hermes_body([{"role": "user", "content": "Rome?"}], stream=False))[1])
        m = out["choices"][0]["message"]
        assert m["content"] is None and m["reasoning_content"] == "Call it." and m["tool_calls"][0]["function"]["arguments"] == '{"city": "Rome"}'
        assert out["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        r.close()


@pytest.mark.parametrize("extra", [{"chat_template_kwargs": {"reasoning_effort": "none"}}, {"chat_template_kwargs": {"enable_thinking": False}},
                                   {"thinking": {"type": "disabled"}}, {"reasoning": {"enabled": False}}, {"reasoning": {"effort": "none"}}])
def test_thinking_off(extra):
    r = Rig([R_OFF], off=True)
    try:
        body = hermes_body([{"role": "user", "content": "Hi"}], **extra)
        if "chat_template_kwargs" not in extra:
            body.pop("chat_template_kwargs")
        assert r.family.render(body).prompt.endswith("<|assistant|><think></think>")
        t = assemble(sse(r.post(body)[1]))
        assert t["reasoning_content"] == "" and t["content"] == R_OFF and t["finish"] == "stop"
    finally:
        r.close()


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
def test_every_effort_level_renders(effort):
    r = GlmFamily(None, "unused", None, {}).render({"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"reasoning_effort": effort}})
    assert r.thinking_first and r.prompt.endswith("<|assistant|><think>")


def test_hermes_extra_fields_never_400(rig):
    body = hermes_body([{"role": "user", "content": "Hi"}], stream=False, temperature=0.6, top_p=0.95, user="hermes", metadata={"s": "x"},
                       reasoning={"enabled": True, "effort": "medium"}, tool_choice="none")
    status, raw = rig.post(body)
    assert status == 200, raw


def test_tool_choice_none_does_not_advertise_or_parse():
    r = Rig(["x" + THINK_END + "Use " + call_text("get_weather", city="Rome") + " later."])
    try:
        body = hermes_body([{"role": "user", "content": "Hi"}], stream=False, tool_choice="none")
        assert "get_weather" not in r.family.render(body).prompt
        out = json.loads(r.post(body)[1])["choices"][0]
        assert "tool_calls" not in out["message"] and out["finish_reason"] == "stop" and "<tool_call>get_weather" in out["message"]["content"]
    finally:
        r.close()


def test_reply_cut_inside_a_call_is_not_a_call():
    r = Rig(["Plan." + THINK_END + "Go." + call_text("get_weather", city="Rome")[:-12]])
    try:
        out = json.loads(r.post(hermes_body([{"role": "user", "content": "Hi"}], stream=False))[1])["choices"][0]
        assert "tool_calls" not in out["message"] and out["message"]["content"] == "Go." and out["finish_reason"] == "stop"
    finally:
        r.close()
