"""A Hermes Agent session against the live server, stdlib only: 3 turns, one tool call, the requests Hermes sends.

usage: hermes_session_check.py [--host 127.0.0.1] [--port 8080] [--model ID] [--effort high] [--json FILE]

Turn 1  big system prompt + tools, user asks for the weather: the model must call get_weather (valid JSON arguments).
Turn 2  the tool result comes back (reasoning_content echoed, as Hermes does for deepseek model names): the model answers.
Turn 3  a follow-up user message.
Asserts: tool_calls parsed, finish_reason "tool_calls" on turn 1, reasoning_content streamed on turn 1 (thinking on),
usage in the last chunk, prompt_tokens_details.cached_tokens > 0 on turns 2 and 3.
Prints per turn: prompt / cached / completion tokens, TTFT (to the first reasoning, content or tool delta) and decode tok/s.
Exit code 1 when an assertion fails.
"""
import argparse
import http.client
import json
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--model", help="model id (default: the first one of /v1/models)")
ap.add_argument("--effort", default="high", help="chat_template_kwargs.reasoning_effort (none = thinking off)")
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--json", help="write the per-turn numbers here")
a = ap.parse_args()

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file of the workspace.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        "required": ["path"]}}},
    {"type": "function", "function": {"name": "terminal", "description": "Run a shell command and return its output.", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
]

RULES = [
    "Answer in the language of the user.", "Use a tool when the answer depends on live data.",
    "Never invent tool results; wait for them.", "Keep answers short and concrete.",
    "Prefer reading files over guessing their content.", "Quote paths exactly as the user wrote them.",
    "When a command fails, read the error before you retry.", "Do not repeat a tool call that already succeeded.",
]
SYSTEM = ("You are Hermes, an agent that runs on the user's own machine. You have tools for the weather, files and the "
          "terminal.\n\n" + "\n".join(f"## Section {i}\n" + " ".join(f"Rule {i}.{j}: {r}" for j, r in enumerate(RULES))
                                     for i in range(40)))


def conn(timeout=1800):
    return http.client.HTTPConnection(a.host, a.port, timeout=timeout)


def model_id():
    if a.model:
        return a.model
    c = conn(30)
    c.request("GET", "/v1/models")
    return json.loads(c.getresponse().read())["data"][0]["id"]


def turn(messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "stream": True,
            "stream_options": {"include_usage": True}, "max_completion_tokens": a.max_tokens, "parallel_tool_calls": True,
            "chat_template_kwargs": {"reasoning_effort": a.effort}}
    c = conn()
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read().decode()[:500]}")
    res = {"content": "", "reasoning": "", "calls": {}, "finish": None, "usage": None, "t_first": None, "t_last": None, "done": False}
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):               # blank separators and ": waiting" keepalives
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            res["done"] = True
            break
        ev = json.loads(data)
        if "error" in ev:
            raise SystemExit(f"stream error: {ev['error']}")
        if not ev["choices"]:
            res["usage"] = ev.get("usage")
            continue
        ch = ev["choices"][0]
        d = ch.get("delta") or {}
        got = False
        if d.get("reasoning_content"):
            res["reasoning"] += d["reasoning_content"]
            got = True
        if d.get("content"):
            res["content"] += d["content"]
            got = True
        for tc in d.get("tool_calls") or []:
            acc = res["calls"].setdefault(tc["index"], {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            acc["id"] = tc.get("id") or acc["id"]
            fn = tc.get("function") or {}
            acc["function"]["name"] += fn.get("name") or ""
            acc["function"]["arguments"] += fn.get("arguments") or ""
            got = True
        if got:
            now = time.perf_counter() - t0
            res["t_first"] = res["t_first"] if res["t_first"] is not None else now
            res["t_last"] = now
        if ch.get("finish_reason"):
            res["finish"] = ch["finish_reason"]
    c.close()
    res["total"] = time.perf_counter() - t0
    return res


FAILS = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        FAILS.append(what)


def report(n, res):
    u = res["usage"] or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    comp = u.get("completion_tokens", 0)
    ttft = res["t_first"] if res["t_first"] is not None else float("nan")
    span = (res["t_last"] or 0) - (res["t_first"] or 0)
    tps = (comp - 1) / span if comp > 1 and span > 0 else float("nan")
    print(f"turn {n}: prompt={u.get('prompt_tokens')} cached={cached} completion={comp} ttft={ttft:.2f}s "
          f"decode={tps:.1f} tok/s total={res['total']:.2f}s finish={res['finish']}")
    return {"turn": n, "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached, "completion_tokens": comp,
            "ttft_s": round(ttft, 3), "decode_tok_s": round(tps, 2), "total_s": round(res["total"], 3), "finish": res["finish"]}


def echo(res):
    msg = {"role": "assistant", "content": res["content"]}
    if res["reasoning"]:
        msg["reasoning_content"] = res["reasoning"]
    calls = [res["calls"][i] for i in sorted(res["calls"])]
    if calls:
        msg["tool_calls"] = calls
    return msg


MODEL = model_id()
thinking = a.effort != "none"
print(f"model {MODEL} on {a.host}:{a.port}, reasoning_effort={a.effort}")
rows = []

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
r1 = turn(messages)
rows.append(report(1, r1))
calls = [r1["calls"][i] for i in sorted(r1["calls"])]
check(r1["done"] and r1["usage"] is not None, "turn 1: [DONE] and a usage chunk")
check(bool(calls), "turn 1: the model called a tool")
if not calls:
    print(f"  turn 1 content: {r1['content'][:300]!r}")
    sys.exit(1)
check(r1["finish"] == "tool_calls", "turn 1: finish_reason tool_calls")
parsed = True
for c in calls:
    try:
        json.loads(c["function"]["arguments"])
    except ValueError:
        parsed = False
check(parsed and all(c["id"] and c["function"]["name"] for c in calls), "turn 1: every call has an id, a name and JSON arguments")
check(any(c["function"]["name"] == "get_weather" for c in calls), "turn 1: get_weather was called")
if thinking:
    check(bool(r1["reasoning"]), "turn 1: reasoning_content streamed")
print(f"  calls: {[(c['function']['name'], c['function']['arguments']) for c in calls]}")

messages += [echo(r1)] + [{"role": "tool", "tool_call_id": c["id"],
                           "content": json.dumps({"city": "Paris", "temperature_c": 21, "sky": "sunny"})
                           if c["function"]["name"] == "get_weather" else "unavailable"} for c in calls]
r2 = turn(messages)
rows.append(report(2, r2))
check(bool(r2["content"]) or bool(r2["calls"]), "turn 2: an answer after the tool result")
check(rows[-1]["cached_tokens"] > 0, "turn 2: cached_tokens > 0")
if thinking:                                            # the model may skip thinking after a tool result: report only
    print(f"  turn 2 reasoning: {len(r2['reasoning'])} chars")

messages.append(echo(r2))
calls2 = [r2["calls"][i] for i in sorted(r2["calls"])]  # a second call: answer it so the history stays valid
messages += [{"role": "tool", "tool_call_id": c["id"], "content": "unavailable"} for c in calls2]
messages.append({"role": "user", "content": "Thanks. Should I take a jacket tonight? One sentence."})
r3 = turn(messages)
rows.append(report(3, r3))
check(bool(r3["content"]) or bool(r3["calls"]), "turn 3: an answer")
check(rows[-1]["cached_tokens"] > 0, "turn 3: cached_tokens > 0")
if thinking:
    print(f"  turn 3 reasoning: {len(r3['reasoning'])} chars")

if a.json:
    with open(a.json, "w") as f:
        json.dump({"model": MODEL, "effort": a.effort, "turns": rows, "fails": FAILS}, f, indent=1)
print("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS))
sys.exit(1 if FAILS else 0)
