"""Hy4-flash chat-template oracle: the GGUF's own tokenizer.chat_template rendered by transformers (render_jinja_template, the apply_chat_template
path) on a fixed set of conversations, written to tests/data/hy4_chat_oracle.json.

    python tools/hy4_chat_oracle.py --gguf <Hy4-flash ...-00001-of-0000N.gguf> [--out tests/data/hy4_chat_oracle.json]

Only the GGUF header is read (gguf_header.read_header, the first 64 MiB at most). Every case holds the OpenAI request a client sends (`body`:
tool-call arguments as JSON strings, reasoning switches as the client spells them) and what transformers is given for it (`hf`: messages with
arguments as objects, tools, template kwargs), and the reference text (`expected`). tests/test_hy4_chat.py checks serve.chat.Hy4Family renders
each `body` to exactly `expected`."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                                                                                      "filters": {"type": "object"}, "tags": {"type": "array", "items": {"type": "string"}},
                                                                                      "safe": {"type": "boolean"}},
                                                     "required": ["query"]}}},
]
SYS = {"role": "system", "content": "You are a careful assistant."}
U1 = {"role": "user", "content": "What is the weather in Paris?"}


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _asst_calls(calls, content="", reasoning=None):
    m = {"role": "assistant", "content": content, "tool_calls": calls}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def _tool(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


W1 = _call("call_1", "get_weather", {"city": "Paris", "unit": "c"})
S1 = _call("call_2", "search", {"query": "météo Paris", "limit": 3, "filters": {"site": "meteo.fr", "days": [1, 2]}, "tags": ["fr", "météo"], "safe": True})

# (name, hf messages, tools, template kwargs, request extras: how a client asks for those kwargs)
CASES = [
    ("user_only", [U1], None, {}, {}),
    ("system_user", [SYS, U1], None, {}, {}),
    ("multi_turn", [SYS, U1, {"role": "assistant", "content": "It is 18 C and sunny."}, {"role": "user", "content": "And tomorrow?"}], None, {}, {}),
    ("history_reasoning_dropped", [U1, {"role": "assistant", "content": "18 C.", "reasoning_content": "The user wants Paris weather."},
                                   {"role": "user", "content": "Thanks!"}], None, {}, {}),
    ("history_reasoning_preserved", [U1, {"role": "assistant", "content": "18 C.", "reasoning_content": "The user wants Paris weather."},
                                     {"role": "user", "content": "Thanks!"}], None, {"preserved_thinking": True}, {"chat_template_kwargs": {"preserved_thinking": True}}),
    ("clear_thinking_false", [U1, {"role": "assistant", "content": "18 C.", "reasoning_content": "Paris."}, {"role": "user", "content": "ok"}], None,
     {"preserved_thinking": True}, {"clear_thinking": False}),
    ("reasoning_field_openrouter", [U1, {"role": "assistant", "content": "18 C.", "reasoning": "OpenRouter spelling."}, {"role": "user", "content": "ok"}], None,
     {"preserved_thinking": True}, {"chat_template_kwargs": {"preserved_thinking": True}}),
    ("effort_low", [U1], None, {"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
    ("effort_high", [U1], None, {"reasoning_effort": "high"}, {"chat_template_kwargs": {"reasoning_effort": "high"}}),
    ("effort_max", [SYS, U1], None, {"reasoning_effort": "max"}, {"reasoning_effort": "max"}),
    ("effort_no_think", [U1], None, {"reasoning_effort": "no_think"}, {"reasoning_effort": "no_think"}),
    ("enable_thinking_false", [SYS, U1], None, {"reasoning_effort": "no_think"}, {"chat_template_kwargs": {"enable_thinking": False}}),
    ("effort_none", [U1], None, {"reasoning_effort": "no_think"}, {"reasoning_effort": "none"}),
    ("reasoning_object_off", [U1], None, {"reasoning_effort": "no_think"}, {"reasoning": {"enabled": False}}),
    ("effort_medium_alias", [U1], None, {"reasoning_effort": "high"}, {"reasoning_effort": "medium"}),
    ("no_think_history", [U1, {"role": "assistant", "content": "18 C.", "reasoning_content": "Hidden."}, {"role": "user", "content": "ok"}], TOOLS,
     {"reasoning_effort": "no_think"}, {"thinking": {"type": "disabled"}}),
    ("tools_user", [U1], TOOLS, {}, {}),
    ("tools_system_user", [SYS, U1], TOOLS, {}, {}),
    ("tools_empty_system", [{"role": "system", "content": ""}, U1], TOOLS, {}, {}),
    ("tool_call_and_result", [SYS, U1, _asst_calls([W1], reasoning="I should call get_weather."), _tool("call_1", "{\"temp\": 18, \"sky\": \"sunny\"}")],
     TOOLS, {}, {}),
    ("tool_call_content_and_result", [U1, _asst_calls([W1], content="Let me check."), _tool("call_1", "18 C")], TOOLS, {}, {}),
    ("parallel_calls_results_reordered", [U1, _asst_calls([W1, S1], reasoning="Two calls."), _tool("call_2", "3 results"), _tool("call_1", "18 C")],
     TOOLS, {}, {}),
    ("tool_loop_then_answer_then_user", [SYS, U1, _asst_calls([W1], reasoning="Call it."), _tool("call_1", "18 C"),
                                          {"role": "assistant", "content": "It is 18 C in Paris.", "reasoning_content": "Got 18."},
                                          {"role": "user", "content": "Search météo too, 3 results."}], TOOLS, {"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
    ("tools_preserved_off", [U1, _asst_calls([W1], reasoning="Hidden thought."), _tool("call_1", "18 C"), {"role": "user", "content": "ok"}], TOOLS,
     {"preserved_thinking": False}, {"chat_template_kwargs": {"preserved_thinking": False}}),
    ("second_system_mid", [SYS, U1, {"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Encore ?"}], None, {}, {}),
    ("unicode", [{"role": "user", "content": "Écris « bonjour » en chinois 你好 et un emoji 🙂, puis du code: def f(x): return {x: 1}"}], None, {}, {}),
    ("text_parts", [{"role": "system", "content": [{"type": "text", "text": "Be brief. "}, {"type": "text", "text": "Use metric."}]},
                    {"role": "user", "content": [{"type": "text", "text": "Weather in "}, {"type": "text", "text": "Lyon?"}]}], None, {}, {}),
    ("assistant_last_prefill", [U1, {"role": "assistant", "content": "Sure,"}], None, {}, {}),
]


def _openai(messages):
    """The client's spelling of an hf message list: tool-call arguments as JSON strings."""
    out = copy.deepcopy(messages)
    for m in out:
        for c in m.get("tool_calls") or []:
            c["function"]["arguments"] = json.dumps(c["function"]["arguments"], ensure_ascii=False)
    return out


def build(template: str) -> list:
    from transformers.utils.chat_template_utils import render_jinja_template
    cases = []
    for name, msgs, tools, kw, req in CASES:
        out, _ = render_jinja_template(conversations=[copy.deepcopy(msgs)], tools=copy.deepcopy(tools), chat_template=template, add_generation_prompt=True, **kw)
        body = {"messages": _openai(msgs), **({"tools": tools} if tools else {}), **req}
        cases.append({"name": name, "body": body, "hf": {"messages": msgs, "tools": tools, "kwargs": kw}, "expected": out[0]})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "tests", "data", "hy4_chat_oracle.json"))
    a = ap.parse_args()
    from kyojin_mlx.gguf_header import read_header
    import transformers
    kv, _ = read_header(a.gguf)
    t = kv["tokenizer.chat_template"]
    cases = build(t)
    with open(a.out, "w") as fh:
        json.dump({"reference": f"transformers {transformers.__version__} render_jinja_template", "gguf": os.path.basename(a.gguf),
                   "template_sha256": hashlib.sha256(t.encode()).hexdigest(), "template": t, "cases": cases}, fh, ensure_ascii=False, indent=1)
    print(f"{len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
