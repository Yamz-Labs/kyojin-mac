"""Loop resume: a request may carry kyojin_flags (per-request levers) and continue a reasoning prefix (continue_final_message).

Render tests use the chat templates only (no model). The HTTP tests run the real server and Engine on the scripted
backend of test_hermes_conformance, on the CPU device.
"""
import json
import os

import mlx.core as mx
import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx.serve import ds41_chat as C
from kyojin_mlx.serve.chat import ChatError, Ds41Family, render_resume, split_resume
from tests import test_hermes_conformance as H

PREFIX = "The user wants the weather. First I list what I know."
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Weather in Paris?"}]
RESUME = MSGS + [{"role": "assistant", "content": "", "reasoning_content": PREFIX}]


class FakeTok:
    chat_template = None


@pytest.fixture(autouse=True)
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


# ------------------------------------------------------------------------------------------------ rendering
def test_resume_prompt_ends_with_the_prefix_inside_the_open_think_block():
    fam = Ds41Family(FakeTok(), "", None, {})
    plain = fam.render({"messages": MSGS, "tools": H.TOOLS})
    r = render_resume(fam, {"messages": RESUME, "tools": H.TOOLS, "continue_final_message": True})
    assert plain.prompt.endswith(C.ASSISTANT_SP + C.THINK_OPEN) and r.thinking_first is True
    assert r.prompt == plain.prompt + PREFIX
    assert r.prompt.rindex(C.THINK_OPEN) < len(r.prompt) - len(PREFIX) and C.THINK_CLOSE not in r.prompt[r.prompt.rindex(C.THINK_OPEN):]
    assert r.cut_texts == plain.cut_texts and all(r.prompt.startswith(c) for c in r.cut_texts)


def test_without_the_flag_the_trailing_assistant_message_is_not_a_resume():
    fam = Ds41Family(FakeTok(), "", None, {})
    body = {"messages": RESUME}
    assert split_resume(body) == (body, None)


@pytest.mark.parametrize("last, extra", [
    ({"role": "user", "content": "x"}, {}),
    ({"role": "assistant", "content": "answer", "reasoning_content": PREFIX}, {}),
    ({"role": "assistant", "content": "", "reasoning_content": ""}, {}),
    ({"role": "assistant", "content": "", "reasoning_content": PREFIX}, {"chat_template_kwargs": {"thinking": False}}),
])
def test_resume_refusals(last, extra):
    fam = Ds41Family(FakeTok(), "", None, {})
    with pytest.raises(ChatError):
        render_resume(fam, {"messages": MSGS + [last], "continue_final_message": True, **extra})


# ------------------------------------------------------------------------------------------------ per-request flags
def test_request_flags_are_whitelisted():
    assert FL.check_request_flags(None) == {}
    assert FL.check_request_flags({"spec_typical": 0, "spec_lookup": False}) == {"spec_typical": "0", "spec_lookup": "0"}
    for bad in ({"engram_gate": 0}, {"nope": 1}, {"moefuse": 1}, [1], {"spec_typical": "maybe"}):
        with pytest.raises(ValueError):
            FL.check_request_flags(bad)


def test_override_is_restored_and_nested_holders_share_one_baseline(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    gen = FL.GEN
    off = FL.check_request_flags({"spec_typical": 0, "spec_lookup": 0})
    with FL.request_override(off):
        assert os.environ["KYOJIN_SPEC_TYPICAL"] == "0" and os.environ["KYOJIN_SPEC_LOOKUP"] == "0"
        with FL.request_override(off):
            pass
        assert os.environ["KYOJIN_SPEC_TYPICAL"] == "0"          # the inner holder leaving does not restore
    assert os.environ["KYOJIN_SPEC_TYPICAL"] == "1" and os.environ["KYOJIN_SPEC_LOOKUP"] == "1" and FL.GEN == gen
    with pytest.raises(RuntimeError):
        with FL.request_override(off):
            raise RuntimeError
    assert os.environ["KYOJIN_SPEC_TYPICAL"] == "1"


# ------------------------------------------------------------------------------------------------ HTTP, scripted engine
CONT = " Then I answer.</think>It is sunny in Paris."


class ResumeScript(H.Script):
    """Plays CONT after whatever follows the last <think> of the prompt that is the PREFIX; records the levers each round saw."""

    def __init__(self, tok, replies):
        super().__init__(tok, replies)
        self.seen = []
        self.tail = None

    def _next(self, h):
        self.seen.append((os.environ.get("KYOJIN_SPEC_TYPICAL"), os.environ.get("KYOJIN_SPEC_LOOKUP")))
        text = self.tok.decode(h)
        if C.THINK_OPEN not in text:                                     # a prefill chunk that stops before the generation header
            return self.eos
        after = text[text.rindex(C.THINK_OPEN) + len(C.THINK_OPEN):]
        self.tail = after
        if after.startswith(PREFIX):
            out = self.tok.encode(CONT) + [self.eos]
            n = len(self.tok.encode(after[len(PREFIX):]))
        else:
            out = self.tok.encode("Plain.</think>Fresh.") + [self.eos]
            n = len(self.tok.encode(after))
        return out[n] if n < len(out) else self.eos


def make_rig(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "1")
    orig = H.Script
    H.Script = ResumeScript
    try:
        r = H.Rig([])
    finally:
        H.Script = orig
    r.app.log = print
    return r


def test_resume_over_http_streams_the_continuation_only_with_the_override(monkeypatch):
    r = make_rig(monkeypatch)
    try:
        body = H.hermes_body(RESUME, tools=None, tool_choice=None, continue_final_message=True,
                             kyojin_flags={"spec_typical": 0, "spec_lookup": 0})
        body = {k: v for k, v in body.items() if v is not None}
        status, raw = r.post(body)
        assert status == 200, raw
        msg = H.assemble(H.sse(raw))
        assert msg["reasoning_content"] == " Then I answer." and msg["content"] == "It is sunny in Paris."
        assert PREFIX not in raw                                         # the prefix is not echoed back: Hermes joins it
        assert set(r.engine.backend.seen) == {("0", "0")}                # every round of that request ran with the override
        assert os.environ["KYOJIN_SPEC_TYPICAL"] == "1" and os.environ["KYOJIN_SPEC_LOOKUP"] == "1"   # restored for the next request
        r.engine.backend.seen.clear()
        status, raw = r.post(H.hermes_body(MSGS, tools=None))
        assert status == 200 and set(r.engine.backend.seen) == {("1", "1")}
    finally:
        r.close()


def test_bad_request_flags_are_a_400(monkeypatch):
    r = make_rig(monkeypatch)
    try:
        assert r.post(H.hermes_body(MSGS, kyojin_flags={"engram_gate": 0}))[0] == 400
        assert r.post(H.hermes_body(MSGS, continue_final_message=True))[0] == 400     # last message is a user turn
    finally:
        r.close()
