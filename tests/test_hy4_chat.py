"""Hy4-flash chat family (serve.chat.Hy4Family): the GGUF's own chat template rendered byte for byte as transformers' render_jinja_template renders
it (tests/data/hy4_chat_oracle.json, tools/hy4_chat_oracle.py), reasoning control (reasoning_effort no_think / low / high / max, thinking off =
no_think), GLM-shaped tool calls whose tags carry the suffix ':6124c78e' rendered then parsed back (stream and whole-completion paths), and the
suffixed think tag wherever the server reads one (resume, loop-guard close, spec_typical_phase think ids). CPU only, no weights."""
import json
import os
import random

import pytest

from kyojin_mlx.serve import chat as C
from kyojin_mlx.serve import loopguard as LG

HERE = os.path.dirname(__file__)
ORACLE = os.path.join(HERE, "data", "hy4_chat_oracle.json")
S = ":6124c78e"
START, MIDDLE, END = f"<｜hy_start{S}｜>", f"<｜hy_middle{S}｜>", f"<｜hy_end{S}｜>"
THINK, UNTHINK = f"<think{S}>", f"</think{S}>"


def _oracle():
    return json.load(open(ORACLE))


def _family(defaults=None):
    return C.Hy4Family(None, HERE, _oracle()["template"], defaults)


def _cases():
    return [pytest.param(c, id=c["name"]) for c in _oracle()["cases"]]


class _Tok:
    """convert_tokens_to_ids of the real Hy4 vocab for the tags read here (120029 / 120030); anything else is unknown (None, as transformers gives)."""
    unk_token_id = None
    ids = {THINK: 120029, UNTHINK: 120030, f"<tool_calls{S}>": 120035, END: 120025}

    def convert_tokens_to_ids(self, t):
        return self.ids.get(t)


# ------------------------------------------------------------------------------------------------ render
def test_oracle_covers_the_cases():
    o = _oracle()
    names = [c["name"] for c in o["cases"]]
    assert len(names) >= 20 and len(set(names)) == len(names) and len(o["template"]) == 17187
    hf = [c["hf"] for c in o["cases"]]
    assert any(h["tools"] for h in hf) and any(m["role"] == "tool" for h in hf for m in h["messages"])
    assert any(m.get("tool_calls") for h in hf for m in h["messages"]) and any(m.get("reasoning_content") for h in hf for m in h["messages"])
    assert {h["kwargs"].get("reasoning_effort") for h in hf} >= {None, "no_think", "low", "high", "max"}
    assert sum(1 for h in hf if sum(m["role"] == "user" for m in h["messages"]) >= 2) >= 4                   # multi-turn


@pytest.mark.parametrize("conv", _cases())
def test_render_equals_the_template(conv):
    r = _family().render(conv["body"])
    assert r.prompt == conv["expected"]
    assert r.thinking_first == conv["expected"].endswith(THINK)


@pytest.mark.parametrize("conv", _cases())
def test_render_with_object_arguments(conv):
    body = dict(conv["body"], messages=conv["hf"]["messages"])            # arguments already objects (what a template-side client sends)
    assert _family().render(body).prompt == conv["expected"]


def test_oracle_matches_live_jinja():
    rj = pytest.importorskip("transformers.utils.chat_template_utils").render_jinja_template
    o = _oracle()
    for c in o["cases"]:
        out, _ = rj(conversations=[c["hf"]["messages"]], tools=c["hf"]["tools"], chat_template=o["template"], add_generation_prompt=True, **c["hf"]["kwargs"])
        assert out[0] == c["expected"], c["name"]


def test_template_from_the_serve_dir(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": _oracle()["template"]}))
    c = _oracle()["cases"][0]
    assert C.Hy4Family(None, str(tmp_path)).render(c["body"]).prompt == c["expected"]
    with pytest.raises(C.ChatError):
        C.Hy4Family(None, str(tmp_path / "missing"))


def test_generation_prompt_cut():
    for c in _oracle()["cases"]:
        r = _family().render(c["body"])
        if c["hf"]["messages"][-1]["role"] == "assistant":
            continue
        tail = START + "assistant" + MIDDLE + THINK + (UNTHINK if r.opts["reasoning_effort"] == "no_think" else "")
        assert r.prompt.endswith(tail), c["name"]
        assert r.prompt[:-len(tail)] in r.cut_texts, c["name"]                      # the next turn starts with exactly that text
        assert all(r.prompt.startswith(t) and t != r.prompt for t in r.cut_texts)
    ids = lambda t: [ord(ch) for ch in t]                                           # noqa: E731  (a 1:1 encoder: the cut lands at the char position)
    r = _family().render(_oracle()["cases"][1]["body"])
    assert C.token_cuts(r.cut_texts, ids(r.prompt), ids) == sorted(len(t) for t in r.cut_texts)


# ------------------------------------------------------------------------------------------------ reasoning control
def _effort(body, defaults=None):
    r = _family(defaults).render({"messages": [{"role": "user", "content": "hi"}], **body})
    mode = r.prompt.split("reasoning_effort:")[1].split(END)[0]
    return mode, r.thinking_first, r.prompt.endswith(THINK + UNTHINK)


@pytest.mark.parametrize("body, want", [
    ({}, "high"),                                                       # the publisher's default: the template's own (reasoning_effort left undefined)
    ({"reasoning_effort": "low"}, "low"), ({"reasoning_effort": "high"}, "high"), ({"reasoning_effort": "max"}, "max"),
    ({"reasoning_effort": "no_think"}, "no_think"), ({"reasoning_effort": "none"}, "no_think"), ({"reasoning_effort": "minimal"}, "low"),
    ({"reasoning_effort": "medium"}, "high"), ({"reasoning_effort": "xhigh"}, "max"),
    ({"enable_thinking": False}, "no_think"), ({"chat_template_kwargs": {"enable_thinking": False}}, "no_think"),
    ({"thinking": {"type": "disabled"}}, "no_think"), ({"reasoning": {"enabled": False}}, "no_think"), ({"reasoning": {"effort": "none"}}, "no_think"),
    ({"reasoning": {"effort": "max"}}, "max"), ({"reasoning": {"effort": "ultra"}}, "high"),           # an unknown reasoning.effort is ignored
    ({"enable_thinking": True, "reasoning_effort": "none"}, "high"),   # thinking asked for: the template's default level
    ({"enable_thinking": False, "reasoning_effort": "max"}, "no_think"),
    ({"chat_template_kwargs": {"reasoning_effort": "low"}, "reasoning_effort": "max"}, "low"),
    ({"chat_template_kwargs": {"reasoning_effort": None}, "reasoning_effort": "max"}, "max"),
])
def test_reasoning_effort(body, want):
    mode, first, closed = _effort(body)
    assert mode == want and first == (want != "no_think") and closed == (want == "no_think")


def test_server_defaults():
    assert _effort({}, {"thinking": False})[0] == "no_think"           # --thinking off (the family has no row: the key is "thinking")
    assert _effort({}, {"reasoning_effort": "low"})[0] == "low"         # --reasoning-effort low
    assert _effort({"reasoning_effort": "max"}, {"reasoning_effort": "low"})[0] == "max"


@pytest.mark.parametrize("body", [{"reasoning_effort": "ultra"}, {"reasoning_effort": 3}, {"enable_thinking": "no"},
                                  {"chat_template_kwargs": "x"}, {"reasoning": "high"}, {"preserved_thinking": "yes"}])
def test_bad_options_are_a_400(body):
    with pytest.raises(C.ChatError):
        _effort(body)


# ------------------------------------------------------------------------------------------------ tool calls
def _assistant_turns(prompt):
    """The assistant turn bodies of a rendered prompt (between START assistant MIDDLE and END)."""
    head = START + "assistant" + MIDDLE
    return [p.split(END)[0] for p in prompt.split(head)[1:] if END in p]


def _call_cases():
    return [pytest.param(c, id=c["name"]) for c in _oracle()["cases"] if any(m.get("tool_calls") for m in c["hf"]["messages"])]


def _stream(parser, text, sizes):
    ev, i = [], 0
    for n in sizes:
        ev += parser.feed(text[i:i + n])
        i += n
    return ev + parser.feed(text[i:]) + parser.finish()


def _message(ev):
    msg = {"content": "".join(e[1] for e in ev if e[0] == "content"), "reasoning": "".join(e[1] for e in ev if e[0] == "reasoning")}
    args = {e[1]: e[2] for e in ev if e[0] == "tool_args"}
    msg["calls"] = [(e[3], json.loads(args[e[1]])) for e in ev if e[0] == "tool_start"]
    return msg


@pytest.mark.parametrize("conv", _call_cases())
def test_tool_call_round_trip(conv):
    """Each rendered assistant turn with tool calls, as the model would write it after the generation prompt's open think tag, parses back to the
    request's own reasoning, content and tool_calls: whole completion and streamed (one char at a time, seeded random pieces)."""
    tools = conv["body"].get("tools")
    turns = [m for m in conv["hf"]["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    texts = [t for t in _assistant_turns(conv["expected"]) if f"<tool_calls{S}>" in t]
    assert len(texts) == len(turns) and texts
    for m, t in zip(turns, texts):
        assert t.startswith(THINK)
        out = t[len(THINK):]
        want_calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m["tool_calls"]]
        whole = C.hy4_parse_completion(out, tools)
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in whole["tool_calls"]] == want_calls
        assert all(c["id"].startswith("call_") and c["type"] == "function" for c in whole["tool_calls"])
        rng = random.Random(len(out))
        for sizes in ([1] * len(out), [rng.randint(1, 9) for _ in range(len(out))]):
            got = _message(_stream(C.Hy4Parser(True, True, tools), out, sizes))
            assert got["calls"] == want_calls
            assert got["content"] == (m.get("content") or "") and got["reasoning"] == out.split(UNTHINK)[0]     # as rendered (history thinking may be dropped)
        assert out.split(UNTHINK)[0] in ("", m.get("reasoning_content"))


def test_typed_arguments():
    tools = next(c["body"]["tools"] for c in _oracle()["cases"] if c["body"].get("tools"))
    t = (f"plan{UNTHINK}<tool_calls{S}><tool_call{S}>search<arg_key{S}>query</arg_key{S}><arg_value{S}>42</arg_value{S}>"
         f"<arg_key{S}>limit</arg_key{S}><arg_value{S}>42</arg_value{S}><arg_key{S}>other</arg_key{S}><arg_value{S}>[1, 2]</arg_value{S}></tool_call{S}></tool_calls{S}>")
    m = C.hy4_parse_completion(t, tools)
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"query": "42", "limit": 42, "other": [1, 2]}   # string stays text, integer typed
    assert m["reasoning_content"] == "plan" and m["content"] == ""


def test_parser_edges():
    call = f"<tool_call{S}>get_weather<arg_key{S}>city</arg_key{S}><arg_value{S}>Paris</arg_value{S}></tool_call{S}>"
    m = C.hy4_parse_completion(f"r{UNTHINK}Checking.{call}")                          # bare call, no <tool_calls> wrapper
    assert m["content"] == "Checking." and json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    m = C.hy4_parse_completion(f"r{UNTHINK}<tool_calls{S}>{call}")                    # wrapper never closed (eos right after the call): the call counts
    assert len(m["tool_calls"]) == 1 and m["content"] == ""
    m = C.hy4_parse_completion(f"r{UNTHINK}<tool_calls{S}><tool_call{S}>get_weather<arg_key{S}>city")   # cut-off call: no call, no tag text as content
    assert "tool_calls" not in m and m["content"] == ""
    m = C.hy4_parse_completion(f"thinking about {call} here{UNTHINK}done")           # a call inside the think block is reasoning text
    assert "tool_calls" not in m and call in m["reasoning_content"] and m["content"] == "done"
    p = C.Hy4Parser(True, False, None)                                                # tool_choice none: the tags are plain content
    ev = p.feed(f"r{UNTHINK}{call}") + p.finish()
    assert "".join(e[1] for e in ev if e[0] == "content") == call and not any(e[0] == "tool_start" for e in ev)
    m = C.hy4_parse_completion(f"{THINK}own block{UNTHINK}answer", reasoning_first=False)   # a reply that opens its own think block
    assert m["reasoning_content"] == "own block" and m["content"] == "answer"
    m = C.hy4_parse_completion("no tags at all <think> stays </think> text", reasoning_first=False)   # the plain tags mean nothing to Hy4
    assert m["content"] == "no tags at all <think> stays </think> text"


def test_glm_parser_unchanged():
    t = "<think>r</think>ok<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
    m = C.glm_parse_completion(t)
    assert m["reasoning_content"] == "r" and m["content"] == "ok" and json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert "tool_calls" not in C.glm_parse_completion(t.replace("<tool_call>", f"<tool_call{S}>").replace("</tool_call>", f"</tool_call{S}>"))


# ------------------------------------------------------------------------------------------------ think tag outside the parser
def test_think_tags_of_families():
    assert C.think_tags(_family()) == (THINK, UNTHINK)
    assert C.think_tags(C.GlmFamily(None, HERE)) == ("<think>", "</think>")


def test_resume_reads_the_family_close_tag():
    body = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "reasoning_content": "Let me"}], "continue_final_message": True}
    r = C.render_resume(_family(), body)
    assert r.prompt.endswith(THINK + "Let me")
    with pytest.raises(C.ChatError):                                                 # no_think: the block is already closed
        C.render_resume(_family(), dict(body, reasoning_effort="no_think"))


def test_loop_guard_sees_the_hy4_think_block():
    p = C.Hy4Parser(True, True, None)
    p.feed("still thinking")
    assert LG.in_think(p)
    p.feed(f" done{UNTHINK}answer")
    assert not LG.in_think(p)


def test_spec_phase_think_ids():
    """server.wire_think_ids (was server.py:841, which looked up "<think>" / "</think>"): Hy4 gets its suffixed tags' ids, so spec_typical_phase
    stays on; a family whose tags the tokenizer does not hold leaves think_ids alone."""
    S_ = pytest.importorskip("kyojin_mlx.serve.server")

    class Eng:
        think_ids = None
    e = Eng()
    S_.wire_think_ids(e, _Tok(), _family())
    assert e.think_ids == (120029, 120030)
    e = Eng()
    S_.wire_think_ids(e, _Tok(), C.GlmFamily(None, HERE))                           # this vocab has no plain <think>
    assert e.think_ids is None
    assert C.think_token_ids(_Tok(), _family()) == (120029, 120030)


def test_server_picks_hy4():
    S_ = pytest.importorskip("kyojin_mlx.serve.server")
    from kyojin_mlx import modeldesc
    assert S_._family_name(HERE, {"model_type": "kyojin_skeleton", "gguf_architecture": "hy_v4_flash"}) == "hy4"
    assert C.FAMILIES["hy4"] is C.Hy4Family
    assert modeldesc.stop_tokens_of("hy4") == (END,)
    assert S_.stop_token_ids(HERE, {}, type("T", (_Tok,), {"eos_token_id": 120025})(), "hy4") == {120025}


# ------------------------------------------------------------------------------------------------ the real file (headers / vocab only)
def test_real_gguf(tmp_path, monkeypatch):
    p = os.environ.get("KYOJIN_TEST_HY4_GGUF")
    if not p or not os.path.exists(p):
        pytest.skip("KYOJIN_TEST_HY4_GGUF not set (a Hy4-flash GGUF file or directory; only its headers and vocab are read)")
    transformers = pytest.importorskip("transformers")
    from kyojin_mlx import foreign_names as FN
    monkeypatch.setenv("KYOJIN_SKELETON", "1")
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    out = FN.gguf_serve_dir(p, cache=str(tmp_path))
    tok = transformers.AutoTokenizer.from_pretrained(out)
    fam = C.Hy4Family(tok, out)
    assert fam.template == _oracle()["template"]                                     # the stored template is the file's
    assert C.think_token_ids(tok, fam) == (120029, 120030) and C.think_token_ids(tok, C.GlmFamily(None, HERE)) is None
    c = [c for c in _oracle()["cases"] if c["name"] == "tool_call_and_result"][0]
    ids = tok.encode(fam.render(c["body"]).prompt, add_special_tokens=False)
    for t in (THINK, UNTHINK, f"<tool_calls{S}>", f"<tool_call{S}>", f"<arg_key{S}>", f"<arg_value{S}>", f"<tool_response{S}>", START, END):
        i = tok.convert_tokens_to_ids(t)
        assert isinstance(i, int) and i in ids, t                                    # every tag is one token in the encoded prompt
    S_ = pytest.importorskip("kyojin_mlx.serve.server")
    assert 120025 in S_.stop_token_ids(out, json.load(open(os.path.join(out, "config.json"))), tok, "hy4")
