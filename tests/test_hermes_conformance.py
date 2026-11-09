"""Hermes Agent conformance of the DS4.1 chat server: the real HTTP server and Engine on a scripted backend (needs mlx, no model).

Hermes is an OpenAI chat-completions client: stream=true, stream_options.include_usage, tools with tool_choice auto,
parallel calls, role tool + tool_call_id, assistant tool_calls with reasoning_content echoed back, and the thinking
switch in chat_template_kwargs.reasoning_effort. A char-level tokenizer and a backend that plays scripted replies drive
the real `Engine` (prefix cache included), so cached_tokens on turns 2 and 3 comes from the real cache.
"""
import http.client
import json
import os
import threading

import mlx.core as mx
import pytest

from kyojin_mlx.serve import ds41_chat as C
from kyojin_mlx.serve import ds41_parse as P
from kyojin_mlx.serve.backends import Backend
from kyojin_mlx.serve.chat import Ds41Family
from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.server import App, Server, make_handler

V = 4096
D = C.DSML
MODEL = "deepseek-v4.1-flash"
SPECIALS = [C.BOS, C.EOS, C.THINK_OPEN, C.THINK_CLOSE, C.USER_SP, C.ASSISTANT_SP, C.SYSTEM_SP]


class CharTok:
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
    """Plays reply n (n = number of <｜Assistant｜> tokens in the history) after the <think> / </think> of the prompt."""
    kind = "script"
    max_rows = 4

    def __init__(self, tok, replies):
        self.tok, self.replies = tok, replies
        self.assistant = tok.ids[C.ASSISTANT_SP]
        self.eos = tok.ids[C.EOS]

    def _next(self, h):
        n = h.count(self.assistant)
        if n == 0 or n > len(self.replies):
            return self.eos
        last = len(h) - 1 - h[::-1].index(self.assistant)
        offset = len(h) - last - 2                      # past <｜Assistant｜> and the <think> / </think> after it
        reply = self.tok.encode(self.replies[n - 1]) + [self.eos]
        return reply[offset] if 0 <= offset < len(reply) else self.eos

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


def call_text(name, params):
    """DSML block of one call; params = [(name, text, string_flag)]."""
    body = "".join(f'<{D} parameter name="{k}" string="{"true" if s else "false"}">{v}</{D} parameter>\n' for k, v, s in params)
    return f'<{D} invoke name="{name}">\n{body}</{D} invoke>\n'


def calls_text(*calls):
    return f"\n\n<{D} calls>\n" + "".join(calls) + f"</{D} calls>"


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "Local time of a time zone.", "parameters": {
        "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}},
]
SYSTEM = "You are Hermes, a careful assistant. " * 12
R1 = ("The user wants weather and time; call both tools.</think>Checking both." +
      calls_text(call_text("get_weather", [("city", "Paris", True), ("days", "3", False)]),
                 call_text("get_time", [("tz", "Europe/Paris", True)])))
R2 = "Both tools answered.</think>Paris is sunny, 21C, and it is 14:05."
R3 = "Short follow-up.</think>Tomorrow looks the same."
R_OFF = "Plain answer without thinking."


class Rig:
    def __init__(self, replies, block=16, chunk=8, ctx=16384):
        self.tok = CharTok()
        for r in replies:
            self.tok.encode(r)
        self.engine = Engine(Script(self.tok, replies), sessions=2, ctx=ctx, chunk=chunk, block=block, prefix_bytes=1 << 22)
        self.family = Ds41Family(self.tok, "unused")
        self.app = App(self.engine, self.tok, self.family, MODEL, {}, {self.tok.ids[C.EOS]}, log=lambda *a: None)
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
    """SSE body -> list of data payloads (dicts, then the "[DONE]" string). Comment lines are skipped."""
    out = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith(":") or not line.strip():
                continue
            assert line.startswith("data: "), line
            data = line[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
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
    """The request shape Hermes sends: no temperature, stream with usage, tools auto, reasoning_effort in chat_template_kwargs."""
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "stream": True,
            "stream_options": {"include_usage": True}, "max_completion_tokens": 512, "parallel_tool_calls": True,
            "chat_template_kwargs": {"reasoning_effort": "high"}}
    body.update(kw)
    return body


def echo(msg):
    """The assistant message Hermes stores and sends back (reasoning echoed for deepseek model names)."""
    calls = [{"id": c["id"], "type": "function",
              "function": {"name": c["function"]["name"], "arguments": json.dumps(json.loads(c["function"]["arguments"]))}}
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


def test_three_turn_session_with_parallel_tool_calls(rig):
    m1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather and time in Paris?"}]
    status, raw = rig.post(hermes_body(m1))
    assert status == 200
    ev = sse(raw)
    assert ev[-1] == "[DONE]"
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(e["object"] == "chat.completion.chunk" and e["model"] == MODEL for e in ev[:-1])
    t1 = assemble(ev)
    assert t1["reasoning_content"] == "The user wants weather and time; call both tools."
    assert t1["content"] == "Checking both."
    assert t1["finish"] == "tool_calls"
    calls = [t1["calls"][i] for i in sorted(t1["calls"])]
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
    assert all(c["id"].startswith("call_") and c["type"] == "function" for c in calls) and calls[0]["id"] != calls[1]["id"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert json.loads(calls[1]["function"]["arguments"]) == {"tz": "Europe/Paris"}
    # every tool_calls delta carries its index; the first one per call carries id, type and name
    firsts = {}
    for e in ev[:-1]:
        for tc in (e["choices"][0]["delta"].get("tool_calls") or []) if e["choices"] else []:
            assert "index" in tc
            firsts.setdefault(tc["index"], tc)
    assert all(f.get("id") and f.get("type") == "function" and f["function"].get("name") for f in firsts.values())
    u1 = t1["usage"]
    assert u1["prompt_tokens"] > 64 and u1["completion_tokens"] > 0
    assert u1["total_tokens"] == u1["prompt_tokens"] + u1["completion_tokens"]
    assert "cached_tokens" in u1["prompt_tokens_details"]

    a1 = echo(t1)
    m2 = m1 + [a1, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "sunny, 21C"},
               {"role": "tool", "tool_call_id": calls[1]["id"], "content": "14:05"}]
    # the history renders back to the exact turn-1 prompt and reply text: the prefix cache can reuse it
    p1 = rig.family.render(hermes_body(m1)).prompt
    p2 = rig.family.render(hermes_body(m2)).prompt
    assert p2.startswith(p1 + R1 + C.EOS)
    t2 = assemble(sse(rig.post(hermes_body(m2))[1]))
    assert t2["finish"] == "stop" and t2["content"] == "Paris is sunny, 21C, and it is 14:05." and not t2["calls"]
    assert t2["reasoning_content"] == "Both tools answered."
    c2 = t2["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert c2 >= u1["prompt_tokens"] - 16 and c2 > 0          # the whole turn-1 prompt, cut to the 16-token block grid

    m3 = m2 + [echo(t2), {"role": "user", "content": "And tomorrow?"}]
    t3 = assemble(sse(rig.post(hermes_body(m3))[1]))
    assert t3["content"] == "Tomorrow looks the same." and t3["finish"] == "stop"
    c3 = t3["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert c3 > c2 and c3 >= t2["usage"]["prompt_tokens"] - 16


def test_reasoning_effort_none_turns_thinking_off():
    rig = Rig([R_OFF])
    try:
        body = hermes_body([{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": "none"})
        assert rig.family.render(body).prompt.endswith(C.ASSISTANT_SP + C.THINK_CLOSE)
        t = assemble(sse(rig.post(body)[1]))
        assert t["reasoning_content"] == "" and t["content"] == R_OFF and t["finish"] == "stop"
    finally:
        rig.close()


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
def test_every_effort_level_renders(effort):
    fam = Ds41Family(None, "unused")
    r = fam.render({"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"reasoning_effort": effort}})
    assert r.prompt.endswith(C.ASSISTANT_SP + C.THINK_OPEN) and r.thinking_first


@pytest.mark.parametrize("reasoning,thinking", [({"enabled": True, "effort": "medium"}, True),
                                                ({"enabled": False, "effort": "none"}, False),
                                                ({"enabled": True, "effort": "ultra"}, True),
                                                ({"enabled": True, "effort": "unknown-level"}, True)])
def test_hermes_reasoning_object(rig, reasoning, thinking):
    """Hermes sends `reasoning: {enabled, effort}` in extra_body for reasoning-capable routes: accepted, never a 400."""
    body = hermes_body([{"role": "user", "content": "Hi"}], stream=False, reasoning=reasoning)
    body.pop("chat_template_kwargs")
    r = rig.family.render(body)
    assert r.prompt.endswith(C.ASSISTANT_SP + (C.THINK_OPEN if thinking else C.THINK_CLOSE))
    status, raw = rig.post(body)
    assert status == 200, raw


def test_non_stream_tool_call_reply(rig):
    m1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather and time in Paris?"}]
    status, raw = rig.post(hermes_body(m1, stream=False))
    assert status == 200
    out = json.loads(raw)
    ch = out["choices"][0]
    assert out["object"] == "chat.completion" and ch["finish_reason"] == "tool_calls"
    assert ch["message"]["content"] == "Checking both."
    assert ch["message"]["reasoning_content"].startswith("The user wants")
    assert [c["function"]["name"] for c in ch["message"]["tool_calls"]] == ["get_weather", "get_time"]
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] >= 0


def test_request_fields_hermes_may_send(rig):
    body = hermes_body([{"role": "user", "content": "Hi"}], stream=False, temperature=0.6, top_p=0.95, max_tokens=64,
                       tool_choice="none", reasoning_effort="low", user="hermes", metadata={"session": "x"})
    body.pop("max_completion_tokens")
    status, raw = rig.post(body)
    assert status == 200, raw


def test_history_variants_render_the_same():
    """content "" or None beside tool_calls, reasoning under `reasoning`, ASCII-escaped arguments: one prompt."""
    call = {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Zürich"}'}}
    tool = {"role": "tool", "tool_call_id": "call_a", "content": "cold"}
    user = {"role": "user", "content": "Weather?"}
    base = {"role": "assistant", "reasoning_content": "think", "tool_calls": [call], "content": "Checking."}
    variants = [
        dict(base),
        {**{k: v for k, v in base.items() if k != "reasoning_content"}, "reasoning": "think"},
        {**base, "tool_calls": [{**call, "function": {**call["function"], "arguments": json.dumps({"city": "Zürich"})}}]},
    ]
    prompts = {C.render_chat([user, v, tool], TOOLS, thinking=True) for v in variants}
    assert len(prompts) == 1
    empty = {**base, "content": ""}
    none = {**base, "content": None}
    assert C.render_chat([user, empty, tool], TOOLS, thinking=True) == C.render_chat([user, none, tool], TOOLS, thinking=True)


# ---- parser: coercion and repair ----------------------------------------------------------------------------


def args_of(text, tools=TOOLS, cut=None):
    p = P.StreamParser(thinking=False, parse_tools=True, tools=tools)
    events = []
    step = cut or len(text) or 1
    for i in range(0, len(text), step):
        events += p.feed(text[i:i + step])
    events += p.finish()
    msg = P.split_events(events)
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg["tool_calls"]], p


@pytest.mark.parametrize("cut", [1, 3, 7, None])
def test_schema_coercion(cut):
    text = "ok" + calls_text(call_text("get_weather", [("city", "42", False), ("days", "3", True)]))
    calls, _ = args_of(text, cut=cut)
    assert calls == [("get_weather", {"city": "42", "days": 3})]
    calls, _ = args_of(text, tools=None, cut=cut)                   # no tools: the official decoding, untouched
    assert calls == [("get_weather", {"city": 42, "days": "3"})]


def test_coerced_value_re_renders_with_the_schema_flag():
    """A coerced value is rendered back as string="false": the history no longer matches the reply tokens there.
    No cache is lost (reply tokens are never snapshotted), the prompt just carries the corrected value."""
    reply = "ok" + calls_text(call_text("get_weather", [("city", "Paris", True), ("days", "3", True)]))
    msg = P.parse_completion(reply, thinking=False, tools=TOOLS)
    hist = [{"role": "user", "content": "w"}, {"role": "assistant", "content": msg["content"], "tool_calls": msg["tool_calls"]},
            {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "sunny"}]
    out = C.render_chat(hist, TOOLS, thinking=False)
    assert f'<{D} parameter name="days" string="false">3</{D} parameter>' in out


@pytest.mark.parametrize("raw,want", [
    ("{'unit': 'c', 'ok': True}", {"unit": "c", "ok": True}),
    ("[1, 2, None]", [1, 2, None]),
    ('{"a": [1, 2', {"a": [1, 2]}),
    ("not json", "not json"),
])
def test_invalid_json_is_repaired_or_kept_as_a_string(raw, want):
    calls, _ = args_of("x" + calls_text(call_text("get_weather", [("opts", raw, False)])))
    assert calls == [("get_weather", {"opts": want})]


def test_cut_off_call_closes_into_valid_json():
    text = "x" + f'\n\n<{D} calls>\n<{D} invoke name="get_weather">\n<{D} parameter name="city" string="true">Par'
    calls, p = args_of(text)
    assert calls == [("get_weather", {"city": "Par"})] and p.truncated


def test_declared_namespaced_tool_keeps_its_full_name():
    tools = [{"type": "function", "function": {"name": "mcp::search", "parameters": {"type": "object", "properties": {}}}}]
    msg = P.parse_completion("x" + calls_text(call_text("mcp::search", [("q", "a", True)])), thinking=False, tools=tools)
    assert msg["tool_calls"][0]["function"]["name"] == "mcp::search" and "namespace" not in msg["tool_calls"][0]
    msg = P.parse_completion("x" + calls_text(call_text("mcp::search", [("q", "a", True)])), thinking=False)
    assert msg["tool_calls"][0]["function"]["name"] == "search" and msg["tool_calls"][0]["namespace"] == "mcp"


# ---- real tokenizer (skipped without the model) ------------------------------------------------------------

TOKENIZER = os.environ.get("DS41_TOKENIZER", os.path.expanduser("~/ds41/model/tokenizer.json"))


@pytest.mark.skipif(not os.path.exists(TOKENIZER), reason="no DS4.1 tokenizer.json")
@pytest.mark.parametrize("thinking", [True, False])
def test_real_tokens_of_the_history_extend_the_turn_one_tokens(thinking):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER)
    enc = lambda s: tok.encode(s, add_special_tokens=False)          # noqa: E731
    reply = R1 if thinking else R1.split(C.THINK_CLOSE)[1]
    m1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather and time in Paris?"}]
    p1 = C.render_chat(m1, TOOLS, thinking=thinking)
    msg = P.parse_completion(reply, thinking=thinking, tools=TOOLS)
    calls = [{**c, "function": {**c["function"], "arguments": json.dumps(json.loads(c["function"]["arguments"]))}}
             for c in msg["tool_calls"]]
    a = {"role": "assistant", "content": msg["content"], "tool_calls": calls}
    if thinking:
        a["reasoning"] = msg["reasoning_content"]
    m2 = m1 + [a] + [{"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in calls]
    t1, t2 = enc(p1) + enc(reply) + enc(C.EOS), enc(C.render_chat(m2, TOOLS, thinking=thinking))
    assert t2[:len(t1)] == t1


def test_ds41_medium_is_numeric_line_only_and_guidance_is_opt_in(monkeypatch):
    from kyojin_mlx.serve.ds41_chat import render_chat
    monkeypatch.delenv("KYOJIN_EFFORT_GUIDANCE", raising=False)
    plain = render_chat([{"role": "user", "content": "hi"}], reasoning_effort="medium")
    assert "Reasoning Effort: 62 (range" in plain and "Match the depth" not in plain
    monkeypatch.setenv("KYOJIN_EFFORT_GUIDANCE", "1")
    assert "Between tool calls, think only about the next step." in render_chat([{"role": "user", "content": "hi"}], reasoning_effort="medium")


def test_ds41_medium_tools_lead_the_system_text():
    from kyojin_mlx.serve.ds41_chat import render_chat
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    medium = render_chat([{"role": "system", "content": "SYSTEXT"}, {"role": "user", "content": "hi"}], tools, reasoning_effort="medium")
    assert "Reasoning Effort: 62 " in medium and medium.index("## Tools") < medium.index("SYSTEXT")
    assert "Reasoning Effort: 75 " in render_chat([{"role": "user", "content": "hi"}], reasoning_effort="high")
