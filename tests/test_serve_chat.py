"""Chat templates and reply parsers of the Mac server: a reply cut anywhere must parse like the whole reply (no model, no MLX)."""
import json
import random

import pytest

from kyojin_mlx.serve import ds41_parse as P
from kyojin_mlx.serve.chat import (ChatError, Ds41Family, GlmFamily, GlmParser, QwenParser, glm_parse_completion, FAMILIES, family_of, parse_tool_calls)

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}, "required": ["city"]}}}


def fold(events):
    reasoning, content, calls = [], [], []
    for ev in events:
        if ev[0] == "reasoning":
            reasoning.append(ev[1])
        elif ev[0] == "content":
            content.append(ev[1])
        elif ev[0] == "tool_start":
            assert ev[1] == len(calls), "tool indexes must be 0, 1, 2 ..."
            calls.append([ev[3], ""])
        elif ev[0] == "tool_args":
            calls[ev[1]][1] += ev[2]
        else:
            assert ev[0] == "tool_end"
    return "".join(reasoning), "".join(content), [(n, json.loads(a)) for n, a in calls]


def run(make, text, cuts):
    parser = make()
    events, last = [], 0
    for c in list(cuts) + [len(text)]:
        events += parser.feed(text[last:c])
        last = c
    return fold(events + parser.finish())


def every_cut(make, text, expect):
    assert run(make, text, []) == expect
    for k in range(len(text) + 1):
        assert run(make, text, [k]) == expect, f"cut at {k}: {text[max(0, k - 12):k]!r} | {text[k:k + 12]!r}"
    rng = random.Random(5)
    for _ in range(150):
        cuts = sorted(rng.sample(range(len(text) + 1), 4))
        assert run(make, text, cuts) == expect, cuts
    assert run(make, text, range(len(text))) == expect


QWEN = ("The user wants the weather.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n</function>\n</tool_call>")


def test_qwen_reply_every_cut():
    expect = ("The user wants the weather.", "Let me check.", [("get_weather", {"city": "Paris", "days": 3, "metric": True})])
    every_cut(lambda: QwenParser(True, [WEATHER]), QWEN, expect)


def test_qwen_two_calls_and_prose_after():
    text = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>")
    expect = ("", "", [("get_weather", {"city": "Rome"}), ("get_weather", {"city": "Oslo"})])
    every_cut(lambda: QwenParser(False, [WEATHER]), text, expect)


def test_qwen_plain_answer_and_tool_choice_none():
    text = "Hello <tool_call> is just a word when tools are off."
    r = run(lambda: QwenParser(False, None, parse_tools=False), text, [7])
    assert r[2] == [] and r[1].startswith("Hello")


def test_qwen_unfinished_call_is_not_a_call():
    assert parse_tool_calls("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", [WEATHER]) == []
    r = run(lambda: QwenParser(False, [WEATHER]), "ok\n<tool_call>\n<function=get_wea", [5])
    assert r[1] == "ok"


GLM_CALL = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>"


def test_glm_reply_every_cut():
    text = "I should call it.</think>Sure, checking.\n" + GLM_CALL
    expect = ("I should call it.", "Sure, checking.\n", [("get_weather", {"city": "Paris", "days": 3})])
    got = run(lambda: GlmParser(True), text, [])
    assert got[0] == expect[0] and got[1].strip() == "Sure, checking." and got[2] == expect[2]
    for k in range(len(text) + 1):
        assert run(lambda: GlmParser(True), text, [k]) == got, k
    assert run(lambda: GlmParser(True), text, range(len(text))) == got


def test_glm_no_reasoning():
    got = run(lambda: GlmParser(False), "plain answer", [3])
    assert got == ("", "plain answer", [])


DS_CALL = (f"<{P.DSML} calls>\n<{P.DSML} invoke name=\"get_weather\">\n<{P.DSML} parameter name=\"city\" string=\"true\">Paris</{P.DSML} parameter>\n"
           f"<{P.DSML} parameter name=\"days\" string=\"false\">3</{P.DSML} parameter>\n</{P.DSML} invoke>\n</{P.DSML} calls>")


def test_ds41_reply_every_cut():
    text = "Think about it.</think>Checking now.\n\n" + DS_CALL
    got = run(lambda: P.StreamParser(True, True), text, [])
    assert got[0] == "Think about it." and got[1].strip() == "Checking now." and got[2] == [("get_weather", {"city": "Paris", "days": 3})]
    every_cut(lambda: P.StreamParser(True, True), text, got)


def test_family_names(monkeypatch):
    assert family_of("qwen3_5") == "qwen" and family_of("glm5_next") == "glm" and family_of("deepseek_v41") == "ds41"
    assert family_of("llama") == "jinja"                                      # no row: the pack's own template + the <tool_call> parser (KYOJIN_CHAT_FALLBACK is on by default)
    monkeypatch.setenv("KYOJIN_CHAT_FALLBACK", "0")
    with pytest.raises(ValueError):
        family_of("llama")
    assert set(FAMILIES) == {"qwen", "glm", "ds41", "generic", "jinja", "harmony", "hy4"}


# ------------------------------------------------------------------------------------------------ rendering
class FakeTok:
    chat_template = None


def test_ds41_render_text_only_and_errors():
    fam = Ds41Family(FakeTok(), "", None, {})
    body = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "tools": [WEATHER]}
    r = fam.render(body)
    assert r.prompt.startswith("<｜begin▁of▁sentence｜>") and "get_weather" in r.prompt and r.prompt.rstrip().endswith("<think>")
    assert r.thinking_first is True
    r2 = fam.render(dict(body, chat_template_kwargs={"thinking": False}))
    assert r2.thinking_first is False and r2.prompt.rstrip().endswith("</think>")
    with pytest.raises(ChatError):
        fam.render({"messages": []})
    with pytest.raises(ChatError):
        fam.render({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})


def test_glm_render_has_tools_and_think():
    fam = GlmFamily(FakeTok(), "", None, {})
    r = fam.render({"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER]})
    assert "get_weather" in r.prompt and r.prompt.endswith("<|user|>Weather in Paris?<|assistant|><think>") and r.thinking_first
    none = fam.render({"messages": [{"role": "user", "content": "x"}], "tools": [WEATHER], "tool_choice": "none"})
    assert "get_weather" not in none.prompt and none.parse_tools is False


def test_glm_medium_guidance_is_opt_in():
    fam = GlmFamily(FakeTok(), "", None, {})
    msgs = [{"role": "user", "content": "x"}]
    assert "Match the depth" not in fam.render({"messages": msgs}).prompt
    on = fam.render({"messages": msgs, "chat_template_kwargs": {"effort_guidance": True}}).prompt
    assert "Match the depth" in on and "think only about the next step" in on


def test_stop_ids_without_eos_token_in_tokenizer(tmp_path):
    """A bare tokenizer.json has no eos_token: the family's end token must still stop the reply (a DS4.1 tool call ends with it)."""
    from kyojin_mlx.serve.server import stop_token_ids

    class Tok:
        eos_token_id = None
        unk_token_id = None
        ids = {"<｜end▁of▁sentence｜>": 1, "<|im_end|>": 7, "<|endoftext|>": 8}

        def convert_tokens_to_ids(self, t):
            return self.ids.get(t, -1)
    assert stop_token_ids(str(tmp_path), {"model_type": "deepseek_v41"}, Tok(), "ds41") == {1}
    assert stop_token_ids(str(tmp_path), {"text_config": {"eos_token_id": 8}}, Tok(), "qwen") == {7, 8}


# ------------------------------------------------------------------------------------------------ glm: reasoning split + tool calls
SEARCH = {"type": "function", "function": {"name": "search", "description": "s", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "filters": {"type": "object"}, "exact": {"type": "boolean"}}}}}


def glm_call(name, **kv):
    return f"<tool_call>{name}" + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in kv.items()) + "</tool_call>"


def test_glm_schema_types_and_exact_strings():
    """A string parameter keeps its text (no JSON guess, no strip); integer / boolean / object parameters are typed; no schema = JSON when it parses."""
    text = ("Look it up.</think>Searching.\n" + glm_call("search", query="007", limit="5", exact="true", filters='{"lang": "fr"}')
            + "\n" + glm_call("search", query="  padded\nline  \n"))
    expect = ("Look it up.", "Searching.", [("search", {"query": "007", "limit": 5, "exact": True, "filters": {"lang": "fr"}}), ("search", {"query": "  padded\nline  \n"})])
    every_cut(lambda: GlmParser(True, True, [SEARCH]), text, expect)
    got = run(lambda: GlmParser(True), "x</think>" + glm_call("search", query="7", limit="5"), [])
    assert got[2] == [("search", {"query": 7, "limit": 5})]                    # unknown tool: JSON guess


def test_glm_parallel_calls_and_event_shapes():
    text = "think</think>" + glm_call("get_weather", city="Rome") + "\n" + glm_call("get_weather", city="Oslo", days=2)
    p = GlmParser(True, True, [WEATHER])
    ev = p.feed(text) + p.finish()
    assert [e[0] for e in ev] == ["reasoning", "tool_start", "tool_args", "tool_end", "tool_start", "tool_args", "tool_end"]
    starts = [e for e in ev if e[0] == "tool_start"]
    assert [e[1] for e in starts] == [0, 1] and starts[0][2] != starts[1][2] and starts[0][2].startswith("call_")
    assert json.loads(ev[2][2]) == {"city": "Rome"} and json.loads(ev[5][2]) == {"city": "Oslo", "days": 2}


def test_glm_tool_text_inside_reasoning_is_not_a_call():
    text = "The format is <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>, so.</think>Done."
    got = run(lambda: GlmParser(True, True, [WEATHER]), text, [])
    assert got[2] == [] and "<tool_call>name" in got[0] and got[1] == "Done."
    every_cut(lambda: GlmParser(True, True, [WEATHER]), text, got)


def test_glm_tool_choice_none_keeps_the_text():
    text = "ok</think>Use " + glm_call("get_weather", city="Rome") + " later."
    got = run(lambda: GlmParser(True, False), text, [4])
    assert got[2] == [] and "<tool_call>get_weather" in got[1] and got[1].endswith("later.")


def test_glm_unfinished_and_reasoning_only_replies():
    cut = "plan</think>" + glm_call("get_weather", city="Rome")[:-6]               # max_tokens inside the call: not a call
    assert run(lambda: GlmParser(True, True, [WEATHER]), cut, [3]) == ("plan", "", [])
    assert run(lambda: GlmParser(True, True, [WEATHER]), "still thinking", [3]) == ("still thinking", "", [])
    assert run(lambda: GlmParser(True, True, [WEATHER]), "</think>" + glm_call("get_weather"), []) == ("", "", [("get_weather", {})])


def test_glm_model_opened_think_block():
    got = run(lambda: GlmParser(False, True, [WEATHER]), "<think>hm</think>answer", [2])
    assert got == ("hm", "answer", [])
    assert run(lambda: GlmParser(False), "plain answer", [3]) == ("", "plain answer", [])


def test_glm_parse_completion():
    m = glm_parse_completion("<think>why</think>Sure.\n" + glm_call("get_weather", city="Rome"), [WEATHER])
    assert m["reasoning_content"] == "why" and m["content"] == "Sure."
    assert m["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Rome"}'}


def test_glm_thinking_switch():
    fam = GlmFamily(FakeTok(), "", None, {})
    msgs = [{"role": "user", "content": "hi"}]
    assert fam.render({"messages": msgs}).thinking_first
    for body in ({"chat_template_kwargs": {"enable_thinking": False}}, {"thinking": {"type": "disabled"}}, {"chat_template_kwargs": {"thinking": False}}):
        r = fam.render({"messages": msgs, **body})
        assert r.prompt.endswith("<|assistant|><think></think>") and not r.thinking_first, body
        assert fam.parser(r).sp.mode == "content"
    assert GlmFamily(FakeTok(), "", None, {"thinking": False}).render({"messages": msgs}).prompt.endswith("<think></think>")
    assert GlmFamily(FakeTok(), "", None, {"thinking": False}).render({"messages": msgs, "thinking": {"type": "enabled"}}).thinking_first


def test_glm_render_tool_round_trip():
    """What a client sends back after a tool call: JSON-string arguments, a tool message, reasoning_content (or `reasoning`)."""
    fam = GlmFamily(FakeTok(), "", None, {})
    call = {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": "paris", "limit": 3, "filters": {"lang": "fr"}})}}
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Find Paris"},
            {"role": "assistant", "content": None, "reasoning": "need a search", "tool_calls": [call, dict(call, id="call_2", function={"name": "search", "arguments": ""})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result one"}, {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "result two"}]}]
    r = fam.render({"messages": msgs, "tools": [SEARCH]})
    tail = r.prompt[r.prompt.index("<|user|>Find Paris"):]
    assert tail == ("<|user|>Find Paris<|assistant|><think>need a search</think>"
                    '<tool_call>search<arg_key>query</arg_key><arg_value>paris</arg_value><arg_key>limit</arg_key><arg_value>3</arg_value>'
                    '<arg_key>filters</arg_key><arg_value>{"lang": "fr"}</arg_value></tool_call><tool_call>search</tool_call>'
                    "<|observation|><tool_response>result one</tool_response><tool_response>result two</tool_response><|assistant|><think>")
    assert r.thinking_first and r.cut_texts and all(r.prompt.startswith(c) for c in r.cut_texts)
    with pytest.raises(ChatError):
        fam.render({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [dict(call, function={"name": "s", "arguments": "[1]"})]}]})
