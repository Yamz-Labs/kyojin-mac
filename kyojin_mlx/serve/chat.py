# SPDX-License-Identifier: Apache-2.0
"""Chat templates and reply parsing per model family, shared with the box servers so Hermes tool calls behave the same.

    qwen  Qwen3.5 / Qwen3-Next: the model's chat_template.jinja; reasoning in <think>, XML tool calls          (tools/qwen/serve.py, box repo db6103b)
    glm   GLM-5.3: glm53-template-medium.jinja (assets/); reasoning in <think>, <tool_call> with arg_key / arg_value   (tools/glm/serve.py, db6103b)
    ds41  DeepSeek-V4.1: ds41_chat.py / ds41_parse.py, DSML tool calls (copied byte for byte from release 1.7.3, 4b684cd)

A family renders an OpenAI request into a prompt string and gives a stream parser. Every stream parser has the same interface:
    feed(text) -> events, finish() -> events, with events
    ("reasoning", text) ("content", text) ("tool_start", index, call_id, name, namespace) ("tool_args", index, json_fragment) ("tool_end", index)
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2 import nodes
from jinja2.ext import Extension

from . import ds41_chat, ds41_parse
from .. import ds41_vision
from .toolargs import coerce as _coerce

ASSETS = Path(__file__).with_name("assets")


class ChatError(ValueError):
    """The request cannot be rendered (HTTP 400)."""


@dataclass
class Rendered:
    prompt: str
    thinking_first: bool = False                    # the prompt ends inside a reasoning block
    parse_tools: bool = True
    tools: list | None = None
    opts: dict = field(default_factory=dict)
    cut_texts: list = field(default_factory=list)   # text prefixes of `prompt` that are safe reuse points (see token_cuts)
    images: list = field(default_factory=list)      # DS4.1: image records in prompt order, one placeholder each in `prompt`


def split_resume(body: dict) -> tuple[dict, str | None]:
    """Resume a reasoning block (body "continue_final_message": true): the last message is an assistant turn holding only `reasoning_content` (no
    content, no tool calls). Returns (body without that turn, the reasoning prefix); (body, None) when the flag is off."""
    if body.get("continue_final_message") is not True:
        return body, None
    msgs = body.get("messages")
    last = msgs[-1] if isinstance(msgs, list) and msgs else None
    if not isinstance(last, dict) or last.get("role") != "assistant":
        raise ChatError("continue_final_message needs an assistant message last")
    prefix = last.get("reasoning_content")
    if not isinstance(prefix, str) or not prefix or last.get("content") or last.get("tool_calls"):
        raise ChatError("continue_final_message: the last assistant message must hold only a non-empty reasoning_content")
    return {**body, "messages": msgs[:-1]}, prefix


def render_resume(family, body: dict) -> Rendered:
    """family.render(body), or for a resume request the prompt of the conversation so far with the reasoning prefix right after the open think tag
    (the reply streams the continuation only). The cut texts stay valid: they are prefixes of the prompt before the prefix."""
    body, prefix = split_resume(body)
    r = family.render(body)
    if prefix is None:
        return r
    if not r.thinking_first or r.prompt.rstrip().endswith(think_tags(family)[1]):
        raise ChatError("continue_final_message: this request does not end inside an open think block (thinking is off?)")
    r.prompt += prefix
    return r


def think_tags(family) -> tuple:
    """(open, close) think tags of a chat family: `think_tags` on the family (Hy4: <think:6124c78e>), else <think> / </think> (qwen, glm, ds41, generic)."""
    return tuple(getattr(family, "think_tags", None) or ("<think>", "</think>"))


def think_token_ids(tokenizer, family) -> tuple | None:
    """(open id, close id) of the family's think tags when the tokenizer holds each as one token, else None (flag spec_typical_phase)."""
    ids = []
    for t in think_tags(family):
        try:
            i = tokenizer.convert_tokens_to_ids(t)
        except Exception:                                                            # noqa: BLE001
            return None
        if not isinstance(i, int) or i < 0 or i == getattr(tokenizer, "unk_token_id", None):
            return None
        ids.append(i)
    return tuple(ids)


# ------------------------------------------------------------------------------------------------ templates (qwen, glm)
class _GenerationTag(Extension):
    """{% generation %}...{% endgeneration %} (transformers' AssistantTracker): marks the assistant span for training masks; rendering keeps the body as is."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


def _template_env() -> Any:
    """Same environment as transformers' apply_chat_template (and the box servers)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", _GenerationTag])
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
        json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise ChatError(str(msg))
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: __import__("datetime").datetime.now().strftime(fmt)     # transformers' global (gpt-oss harmony template)
    return env


_COMPILED: dict = {}


def _compiled(template: str):
    hit = _COMPILED.get(template)
    if hit is None:
        hit = _COMPILED[template] = _template_env().from_string(template)
    return hit


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _text(content: Any) -> str:
    """OpenAI content (string, null or a list of parts) -> text. This server is text only."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                out.append(part.get("text", ""))
            else:
                raise ChatError("only text content parts are supported (no images on this server)")
        return "".join(out)
    raise ChatError("message content must be a string or an array of parts")


def split_images(messages: Any) -> tuple[Any, list]:
    """Flag visiongen: image parts of the messages -> (messages whose image parts are {"type": "image"} so the pack's own template places its
    placeholder, the image records in message / part order). Messages without an image part are returned as they are."""
    from .ds41_chat import IMAGE_PART_TYPES, ChatFormatError, _image_record
    if not isinstance(messages, list):
        return messages, []
    out, records = [], []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES for p in c):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES:
                    try:
                        records.append(_image_record(p))
                    except ChatFormatError as exc:
                        raise ChatError(str(exc)) from exc
                    parts.append({"type": "image"})
                elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif isinstance(p, str):
                    parts.append({"type": "text", "text": p})
                else:
                    raise ChatError("only text and image content parts are supported")
            m = dict(m, content=parts)
        out.append(m)
    return out, records


def _has_image(content: Any) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image" and "text" not in p for p in content)


def render_jinja(template: str, messages: list, tools: list | None, generation: bool = True, parts: bool = False, none_tools: bool = False, **kwargs: Any) -> str:
    """The model's own Jinja chat template. Tool-call arguments go to the template as dicts (OpenAI sends JSON strings)."""
    if not isinstance(messages, list) or not messages:
        raise ChatError("messages must be a non-empty array")
    fixed = []
    for m in messages:
        if not isinstance(m, dict):
            raise ChatError("each message must be an object")
        if not (parts and _has_image(m.get("content"))):                             # parts (visiongen): a message with an image keeps its part list for the template
            m = dict(m, content=_text(m.get("content")))
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            m["tool_calls"] = [
                dict(c, function=dict(c["function"], arguments=_json_value(c["function"]["arguments"])))
                if isinstance(c.get("function", {}).get("arguments"), str) else c
                for c in m["tool_calls"]]
        fixed.append(m)
    tl = (tools or None) if none_tools else (tools or [])          # none_tools: no tools = None, as transformers' apply_chat_template (Llama 3 tests `tools is not none`)
    return _compiled(template).render(messages=fixed, tools=tl, add_generation_prompt=generation, **kwargs)


def jinja_cut_texts(template: str, messages: list, tools: list | None, prompt: str, **kw: Any) -> list:
    """Reuse points of a jinja render: the leading system / developer messages (+ tools) shared by every session of a client, and the whole
    conversation without the generation prompt (the next turn starts with exactly that text). Only texts that are a prefix of `prompt` count."""
    lead = 0
    while lead < len(messages) and isinstance(messages[lead], dict) and messages[lead].get("role") in ("system", "developer"):
        lead += 1
    out = []
    for msgs in ([messages[:lead]] if 0 < lead < len(messages) else []) + [messages]:
        try:
            t = render_jinja(template, msgs, tools, generation=False, **kw)
        except Exception:                                                            # noqa: BLE001  (a template that needs a user turn: no cut)
            continue
        if t and t != prompt and prompt.startswith(t):
            out.append(t)
    return out


def token_cuts(cut_texts: list, ids: list, encode) -> list:
    """Token positions of the cut texts: sorted, only those whose token ids are an exact proper prefix of `ids` (a merge at the boundary drops the cut)."""
    out = set()
    for t in cut_texts:
        try:
            c = encode(t)
        except Exception:                                                            # noqa: BLE001
            continue
        if 0 < len(c) < len(ids) and list(ids[:len(c)]) == list(c):
            out.add(len(c))
    return sorted(out)


def _complete_calls(index0: int, calls: list) -> list:
    """OpenAI tool-call dicts -> stream events."""
    ev = []
    for n, c in enumerate(calls, index0):
        ev.append(("tool_start", n, c["id"], c["function"]["name"], None))
        ev.append(("tool_args", n, c["function"]["arguments"]))
        ev.append(("tool_end", n))
    return ev


# ------------------------------------------------------------------------------------------------ qwen
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CHANNEL_OPEN, CHANNEL_CLOSE = "<|channel>thought\n", "<channel|>"             # Gemma 4's thinking block
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\n?(.*?)\n?</parameter>", re.S)
EFFORT_ALIASES = {"high": "xhigh", "minimal": "low"}


def _partial_tag(buf: str, tags: list) -> int:
    """Length of the longest suffix of buf that is a proper prefix of one of the tags."""
    best = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(buf)), best, -1):
            if tag.startswith(buf[-n:]):
                best = n
                break
    return best


class Splitter:
    """Incremental split of Qwen text into reasoning, content and tool-call text (tools/qwen/serve.py).

    `reasoning_first`: the generation prompt ended inside a <think> block, so the text starts as reasoning until </think>.
    From <tool_call> to </tool_call> everything is held for parse_tool_calls(); text after the close is content again.
    Whitespace at the edges of a piece (the newlines around </think> and <tool_call>) is dropped."""

    def __init__(self, reasoning_first: bool, think_open: bool = False, think_tags: tuple = (THINK_OPEN, THINK_CLOSE), tool_tags: dict | None = None):
        self.think_open = think_open                       # the model itself opens the <think> block (SmolLM3 /think, Gemma 4 channel; no <think> in the generation prompt)
        self.opens = {think_tags[0]: think_tags[1]}        # think open tag -> its close tag (Hy4: the suffixed tags)
        if tuple(think_tags) == (THINK_OPEN, THINK_CLOSE):
            self.opens[CHANNEL_OPEN] = CHANNEL_CLOSE
        self.tool_tags = dict(tool_tags or {TOOL_OPEN: TOOL_CLOSE})    # tool block open tag -> its close tag
        self.tool_close = TOOL_CLOSE
        self.closers = [think_tags[1]]
        self.mode = "reasoning" if reasoning_first else "content"
        self.buf = ""
        self.ws = ""
        self.lead = reasoning_first
        self.tool_text = ""
        self.tool_from = 0
        self.said = self.after_call = False
        self.calls = True                                  # False: <tool_call> is plain text (tool_choice none)
        self.calls_in_reasoning = True                     # False: a <tool_call> inside a think block is reasoning text (GLM)

    def _emit(self, seg: str, out: list) -> None:
        seg = self.ws + seg
        self.ws = ""
        if self.lead:
            seg = seg.lstrip()
            self.lead = not seg
        body = seg.rstrip()
        self.ws = seg[len(body):]
        if body:
            if self.mode == "content":
                if self.after_call and self.said:
                    body = "\n" + body
                self.said, self.after_call = True, False
            out.append((self.mode, body))

    def feed(self, text: str, final: bool = False) -> list:
        out: list = []
        self.buf += text
        while True:
            if self.mode == "tool":
                self.tool_text += self.buf
                self.buf = ""
                end = self.tool_text.find(self.tool_close, self.tool_from)
                if end < 0:
                    return out
                end += len(self.tool_close)
                self.buf, self.tool_text = self.tool_text[end:], self.tool_text[:end]
                self.mode, self.lead, self.after_call = "content", True, True
                continue
            opens = self.think_open and self.mode == "content" and not self.said and not self.tool_text    # only as the very first thing the model says
            tags = (list(self.tool_tags) if self.calls and (self.calls_in_reasoning or self.mode != "reasoning") else []) \
                + (self.closers if self.mode == "reasoning" else []) + (list(self.opens) if opens else [])
            hits = [(self.buf.find(t), t) for t in tags if t in self.buf and (t not in self.opens or not self.buf[:self.buf.find(t)].strip())]
            if hits:
                pos, tag = min(hits)
                self._emit(self.buf[:pos], out)
                self.ws = ""
                self.buf = self.buf[pos + len(tag):]
                if tag in self.tool_tags:
                    self.mode, self.tool_from, self.tool_close = "tool", len(self.tool_text), self.tool_tags[tag]
                    self.tool_text += tag
                elif tag in self.opens:
                    self.mode, self.lead, self.think_open = "reasoning", True, False
                    self.closers = [self.opens[tag]]
                else:
                    self.mode, self.lead = "content", True
                continue
            keep = 0 if final else _partial_tag(self.buf, tags)
            self._emit(self.buf[:len(self.buf) - keep], out)
            self.buf = self.buf[len(self.buf) - keep:]
            return out

    def finish(self) -> list:
        out = self.feed("", final=True)
        self.ws = ""
        return out


def parse_tool_calls(tool_text: str, tools: list | None) -> list:
    """Qwen XML tool calls (<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>) to OpenAI."""
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties", {})
    calls = []
    for block in re.findall(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), tool_text, re.S):
        fm = _FUNC_RE.search(block)
        if fm:
            name, props = fm.group(1), schemas.get(fm.group(1), {})
            args = {k: _coerce(v, props.get(k) if props else None) for k, v in _PARAM_RE.findall(fm.group(2))}
        else:
            try:
                obj = json.loads(block.strip())
                name, args = obj["name"], obj.get("arguments", obj.get("parameters", {}))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
    return calls


class QwenParser:
    def __init__(self, reasoning_first: bool, tools: list | None, parse_tools: bool = True, think_open: bool = False):
        self.sp, self.tools, self.parse_tools = Splitter(reasoning_first, think_open), tools, parse_tools

    def feed(self, text: str) -> list:
        return [(k, t) for k, t in self.sp.feed(text)]

    def finish(self) -> list:
        ev = [(k, t) for k, t in self.sp.finish()]
        if self.parse_tools:
            ev += _complete_calls(0, parse_tool_calls(self.sp.tool_text, self.tools))
        return ev


# Tool-call text of families with no row of their own: Mistral / Ministral "[TOOL_CALLS]name[ARGS]{json}" (or a JSON array), Llama 3.x "<|python_tag|>{json}" or a bare
# JSON object, Gemma 4 "<|tool_call>call:name{key:<|"|>text<|"|>}<tool_call|>". The text is held back from the first marker on and read once at the end; text that is not a call of
# one of the request's tools goes out as content, unchanged.
_CALL_MARKERS = ("[TOOL_CALLS]", "<|python_tag|>", "<|tool_call>")


def _json_calls(text: str, names: set) -> list:
    """{"name"|"function": ..., "arguments"|"parameters": ...} objects (one, several separated by ';' or newlines, or a JSON array) -> [(name, args)]."""
    dec, i, out = json.JSONDecoder(), 0, []
    text = text.strip()
    while i < len(text):
        if text[i] in "; \n\t":
            i += 1
            continue
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            return []
        for o in obj if isinstance(obj, list) else [obj]:
            if not isinstance(o, dict):
                return []
            fn = o.get("function")
            name = o.get("name") or (fn if isinstance(fn, str) else (fn or {}).get("name") if isinstance(fn, dict) else None)
            args = o.get("arguments", o.get("parameters", (fn or {}).get("arguments", {}) if isinstance(fn, dict) else {}))
            if name not in names:
                return []
            out.append((name, args))
    return out


def _gemma_value(text: str):
    """Gemma 4 call arguments `{k:<|"|>text<|"|>,n:3,flag:true,l:[1,2]}` -> python value (JSON once keys and strings are quoted)."""
    strs: list = []

    def keep(m):
        strs.append(m.group(1))
        return f"\x00{len(strs) - 1}\x00"
    t = re.sub(r'<\|"\|>(.*?)<\|"\|>', keep, text, flags=re.S)
    t = re.sub(r'([{,]\s*)([A-Za-z_][\w.-]*)\s*:', r'\1"\2":', t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: json.dumps(strs[int(m.group(1))], ensure_ascii=False), t)
    return json.loads(t)


def parse_format_calls(held: str, names: set) -> list:
    """The held text (from the first marker or bare '{') to [(name, args)]; [] when it is not a call of one of the tools."""
    calls: list = []
    if held.startswith("[TOOL_CALLS]"):
        for part in held.split("[TOOL_CALLS]")[1:]:
            part = part.strip()
            if "[ARGS]" in part:
                name, _, a = part.partition("[ARGS]")
                try:
                    args = json.loads(a.strip())
                except json.JSONDecodeError:
                    return []
                if name.strip() not in names:
                    return []
                calls.append((name.strip(), args))
            else:
                got = _json_calls(part, names)
                if not got:
                    return []
                calls += got
    elif held.startswith("<|tool_call>"):
        for part in held.split("<|tool_call>")[1:]:
            m = re.match(r"\s*call:([^\s{]+)\s*(\{.*\})\s*(?:<tool_call\|>)?\s*$", part, re.S)
            if not m or m.group(1) not in names:
                return []
            try:
                calls.append((m.group(1), _gemma_value(m.group(2))))
            except (json.JSONDecodeError, ValueError):
                return []
    else:
        body = held[len("<|python_tag|>"):] if held.startswith("<|python_tag|>") else held
        calls = _json_calls(body, names)
    return calls


class FormatQwenParser(QwenParser):
    """QwenParser plus the call formats above, for the fallback (no row) chat family."""

    def __init__(self, reasoning_first: bool, tools: list | None, parse_tools: bool = True, think_open: bool = False):
        super().__init__(reasoning_first, tools, parse_tools, think_open)
        self.names = {t.get("function", t).get("name") for t in (tools or [])} if parse_tools else set()
        self.buf, self.held, self.said = "", None, False

    def _content(self, text: str, out: list) -> None:
        if self.held is not None:
            self.held += text
            return
        self.buf += text
        hits = [(self.buf.find(m), m) for m in _CALL_MARKERS if m in self.buf]
        if hits:
            pos = min(hits)[0]
            self._flush(self.buf[:pos], out)
            self.held, self.buf = self.buf[pos:], ""
        elif not self.said and self.buf.lstrip().startswith("{"):             # Llama 3.x answers a tool request with a bare JSON object
            self.held, self.buf = self.buf, ""
        else:
            keep = _partial_tag(self.buf, list(_CALL_MARKERS))
            self._flush(self.buf[:len(self.buf) - keep], out)
            self.buf = self.buf[len(self.buf) - keep:]

    def _flush(self, text: str, out: list) -> None:
        if text:
            out.append(("content", text))
            self.said = self.said or bool(text.strip())

    def feed(self, text: str) -> list:
        ev = super().feed(text)
        if not self.names:
            return ev
        out: list = []
        for k, t in ev:
            if k == "content":
                self._content(t, out)
            else:
                out.append((k, t))
        return out

    def finish(self) -> list:
        ev = super().finish()
        if not self.names:
            return ev
        out: list = []
        for e in ev:
            if e[0] == "content":
                self._content(e[1], out)
            else:
                out.append(e)
        held, tail = self.held, self.buf
        self.held, self.buf = None, ""
        if held is not None:
            calls = parse_format_calls(held.strip(), self.names)
            if calls:
                n0 = sum(1 for e in out if e[0] == "tool_start")
                out += _complete_calls(n0, [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                                              "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)}} for n, a in calls])
            else:
                self._flush(held, out)
        self._flush(tail, out)
        return out


class QwenFamily:
    name = "qwen"

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        path = Path(model_dir) / "chat_template.jinja"
        self.template = template or (path.read_text() if path.is_file() else getattr(tokenizer, "chat_template", None))
        if not self.template:
            raise ChatError(f"no chat template in {model_dir}")
        self.defaults = defaults or {}
        self.vision = None                                                   # visiongen: the served model's GenVision (set by the server), else images are rejected

    def template_kwargs(self, body: dict) -> dict:
        kw: dict = {}
        ctk = body.get("chat_template_kwargs") or {}
        if not isinstance(ctk, dict):
            raise ChatError("chat_template_kwargs must be an object")
        for key in ("enable_thinking", "reasoning_effort"):
            for src in (ctk, body, self.defaults):
                if src.get(key) is not None:
                    kw[key] = src[key]
                    break
        effort = kw.get("reasoning_effort")
        if effort == "none":
            kw.pop("reasoning_effort")
            kw["enable_thinking"] = False
        elif isinstance(effort, str):
            kw["reasoning_effort"] = EFFORT_ALIASES.get(effort, effort)
        if "enable_thinking" in kw and not isinstance(kw["enable_thinking"], bool):
            raise ChatError("enable_thinking must be boolean")
        return kw

    def render(self, body: dict) -> Rendered:
        tools = body.get("tools")
        parse_tools = body.get("tool_choice") != "none"
        if not parse_tools:
            tools = None
        kw = self.template_kwargs(body)
        msgs, images = body.get("messages"), []
        if self.vision is not None:
            msgs, images = split_images(msgs)
            if images:
                kw["parts"] = True
        prompt = render_jinja(self.template, msgs, tools, **kw)
        cuts = jinja_cut_texts(self.template, msgs, tools, prompt, **kw)
        return Rendered(prompt, prompt.rstrip().endswith("<think>"), parse_tools, tools, cut_texts=cuts, images=images)

    def parser(self, r: Rendered):
        return QwenParser(r.thinking_first, r.tools, r.parse_tools)


# ------------------------------------------------------------------------------------------------ harmony (gpt-oss)
H_MODES = {"<|channel|>analysis<|message|>": "reasoning", "<|channel|>final<|message|>": "content", "<|channel|>commentary<|message|>": "content"}
H_DROP = ("<|end|>", "<|start|>assistant", "<|return|>", "<|call|>")


class HarmonyParser:
    """gpt-oss harmony output: the analysis channel streams as reasoning, the final (and commentary) channel as content; the turn markers
    (<|end|>, <|start|>assistant, <|return|>, <|call|>) are dropped. A marker split over two pieces is held until it completes."""

    def __init__(self, *_):
        self.mode, self.buf, self.lead = "content", "", True
        self.tags = list(H_MODES) + list(H_DROP)

    def _emit(self, seg: str, out: list) -> None:
        if self.lead:
            seg = seg.lstrip()
            self.lead = not seg
        if seg:
            out.append((self.mode, seg))

    def feed(self, text: str, final: bool = False) -> list:
        out: list = []
        self.buf += text
        while True:
            hits = [(self.buf.find(t), t) for t in self.tags if t in self.buf]
            if not hits:
                keep = 0 if final else _partial_tag(self.buf, self.tags)
                self._emit(self.buf[:len(self.buf) - keep], out)
                self.buf = self.buf[len(self.buf) - keep:]
                return out
            pos, tag = min(hits)
            seg = self.buf[:pos]
            self._emit(seg.rstrip() if tag in H_MODES or tag == "<|end|>" else seg, out)
            self.buf = self.buf[pos + len(tag):]
            if tag in H_MODES:
                self.mode, self.lead = H_MODES[tag], True

    def finish(self) -> list:
        return self.feed("", final=True)


class HarmonyFamily(QwenFamily):
    """QwenFamily rendering (the pack's own jinja template) with the harmony output parser."""
    name = "harmony"

    def parser(self, r: Rendered):
        return HarmonyParser()


# ------------------------------------------------------------------------------------------------ glm
_GLM_NAME_RE = re.compile(r"\s*([^\s<]+)")
GLM_TAG_SUFFIX = ""                                                  # GLM-5: <tool_call> <arg_key> <arg_value> as they are; Hy4: every tag carries HY4_SUFFIX
_GLM_RES: dict = {}


def _glm_res(suffix: str) -> tuple:
    """(call block regex, key / value regex) of the GLM-shaped tool-call tags with `suffix` ("" = GLM-5, ":6124c78e" = Hy4)."""
    hit = _GLM_RES.get(suffix)
    if hit is None:
        t = lambda name, close=False: re.escape(f"<{'/' if close else ''}{name}{suffix}>")     # noqa: E731
        hit = _GLM_RES[suffix] = (re.compile(t("tool_call") + r"(.*?)" + t("tool_call", True), re.S),
                                  re.compile(t("arg_key") + r"(.*?)" + t("arg_key", True) + r"\s*" + t("arg_value") + r"(.*?)" + t("arg_value", True), re.S))
    return hit


def glm_parse_tool_calls(tool_text: str, tools: list | None = None, suffix: str = GLM_TAG_SUFFIX) -> list:
    """GLM tool calls (<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>, one block per call) to OpenAI tool_calls.
    A value is typed by the parameter's schema when the request declares the tool (a string parameter keeps its text exactly, "3" stays "3");
    otherwise it is JSON when it parses and text when not. A block whose close tag never came is not a call. `suffix`: the tag suffix (Hy4)."""
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    call_re, arg_re = _glm_res(suffix)
    calls = []
    for body in call_re.findall(tool_text):
        name_match = _GLM_NAME_RE.match(body)
        if not name_match:
            continue
        name = name_match.group(1)
        props = schemas.get(name) or {}
        args: dict = {}
        for k, v in arg_re.findall(body):
            k = k.strip()
            schema = props.get(k)
            args[k] = _coerce(v, schema) if schema else _json_value(v.strip())
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def glm_parse_completion(text: str, tools: list | None = None, parser=None) -> dict:
    """A whole GLM completion (the text after the generation prompt's open <think>, or with its own <think> ...) -> OpenAI message fields.
    `parser`: another GLM-shaped stream parser (Hy4Parser), default GlmParser(False, True, tools)."""
    p = parser if parser is not None else GlmParser(False, True, tools)
    ev = p.feed(text) + p.finish()
    message: dict = {"role": "assistant", "content": "".join(e[1] for e in ev if e[0] == "content")}
    reasoning = "".join(e[1] for e in ev if e[0] == "reasoning")
    if reasoning:
        message["reasoning_content"] = reasoning
    calls = [e for e in ev if e[0] == "tool_start"]
    if calls:
        args = {e[1]: e[2] for e in ev if e[0] == "tool_args"}
        message["tool_calls"] = [{"id": e[2], "type": "function", "function": {"name": e[3], "arguments": args.get(e[1], "{}")}} for e in calls]
    return message


class GlmParser(QwenParser):
    """GLM-5 reply: reasoning up to </think> (reasoning_content), prose (content) and every <tool_call>...</tool_call> block, held and parsed once at
    the end (one tool_calls entry per block, in order). `<tool_call>` inside a think block is reasoning text; with tool_choice "none" it is content.
    Edge whitespace (around </think> and the calls) is dropped, as the Qwen parser does."""

    def __init__(self, reasoning_first: bool, parse_tools: bool = True, tools: list | None = None):
        super().__init__(reasoning_first, tools, parse_tools, think_open=True)      # think_open: a reply that opens its own <think> (thinking off) is split too
        self.sp.calls, self.sp.calls_in_reasoning = parse_tools, False

    def finish(self) -> list:
        ev = list(self.sp.finish())
        if self.parse_tools:
            ev += _complete_calls(0, glm_parse_tool_calls(self.sp.tool_text, self.tools))
        return ev


def _glm_messages(messages):
    """OpenAI messages -> what the GLM template reads: `reasoning` (the OpenRouter spelling) read as reasoning_content, tool-call arguments as objects."""
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            m = dict(m)
            if not isinstance(m.get("reasoning_content"), str) and isinstance(m.get("reasoning"), str):
                m["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    fn = dict(c.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = _json_value(args) if args.strip() else {}
                    if args is None:
                        args = {}
                    if not isinstance(args, dict):
                        raise ChatError("tool_calls: function.arguments must be a JSON object")
                    calls.append(dict(c, function=dict(fn, arguments=args)))
                m["tool_calls"] = calls
        out.append(m)
    return out


class GlmFamily:
    name = "glm"
    DEFAULTS = {"reasoning_effort": "medium"}

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        self.template = template or (ASSETS / "glm53-template-medium.jinja").read_text()
        self.defaults = {**self.DEFAULTS, **(defaults or {})}

    def _options(self, body: dict, ctk: dict) -> tuple[bool, Any]:
        """(thinking, reasoning effort) of a request, read like the DS4.1 family reads Hermes: chat_template_kwargs, then body fields, then `reasoning`
        ({"enabled", "effort"}), then the server default. `thinking` / `enable_thinking` are a bool or {"type": "enabled"|"disabled"};
        reasoning_effort "none" turns thinking off. The GLM template knows low / high (anything else renders as max, medium is this server's own level)."""
        try:
            effort = ctk.get("reasoning_effort")
            if effort is None:
                effort = body.get("reasoning_effort")
            r_switch, r_effort = ds41_chat._reasoning_object(body)
            if effort is None:
                effort = r_effort
            thinking = ds41_chat._switch(ctk)
            if thinking is None:
                thinking = ds41_chat._switch(body)
        except ds41_chat.ChatFormatError as e:
            raise ChatError(str(e)) from None
        if thinking is None:
            thinking = r_switch
        if thinking is None:
            thinking = False if effort == "none" else bool(self.defaults.get("thinking", True))
        if effort in (None, "none") or not isinstance(effort, str):
            effort = self.defaults["reasoning_effort"]
        return thinking, {"minimal": "low"}.get(effort, effort)

    def render(self, body: dict) -> Rendered:
        tools = body.get("tools")
        parse_tools = body.get("tool_choice") != "none"
        if not parse_tools:
            tools = None
        ctk = body.get("chat_template_kwargs") or {}
        if not isinstance(ctk, dict):
            raise ChatError("chat_template_kwargs must be an object")
        pick = lambda key, default=None: ctk[key] if ctk.get(key) is not None else body.get(key, default)   # noqa: E731
        thinking, effort = self._options(body, ctk)
        kw = {"clear_thinking": pick("clear_thinking")}
        if effort is not None:
            kw["reasoning_effort"] = effort
        if pick("effort_guidance", os.environ.get("KYOJIN_EFFORT_GUIDANCE") == "1"):   # opt-in medium guidance text, off by default
            kw["effort_guidance"] = True
        messages = _glm_messages(body.get("messages"))
        prompt = render_jinja(self.template, messages, tools, **kw)
        cuts = jinja_cut_texts(self.template, messages, tools, prompt, **kw)
        if not thinking and prompt.endswith("<think>"):          # the template always opens a think block: close it empty (how it renders earlier turns)
            prompt += "</think>"
        return Rendered(prompt, prompt.rstrip().endswith("<think>"), parse_tools, tools, cut_texts=cuts)

    def parser(self, r: Rendered):
        return GlmParser(r.thinking_first, r.parse_tools, r.tools)


# ------------------------------------------------------------------------------------------------ hy4 (Hy4-flash, GGUF hy_v4_flash)
HY4_SUFFIX = ":6124c78e"                                             # every control tag of the Hy4 template carries it; each tag is one token
HY4_THINK_OPEN, HY4_THINK_CLOSE = f"<think{HY4_SUFFIX}>", f"</think{HY4_SUFFIX}>"              # 120029 / 120030
HY4_CALLS_OPEN, HY4_CALLS_CLOSE = f"<tool_calls{HY4_SUFFIX}>", f"</tool_calls{HY4_SUFFIX}>"    # 120035 / ..
HY4_CALL_OPEN, HY4_CALL_CLOSE = f"<tool_call{HY4_SUFFIX}>", f"</tool_call{HY4_SUFFIX}>"        # 120033 / ..
HY4_END = f"<｜hy_end{HY4_SUFFIX}｜>"                                 # 120025: end of every turn, the file's eos (llama.cpp special_eog_ids = {it})
HY4_EFFORTS = ("no_think", "low", "high", "max")                     # what the template accepts (any other value raises in it)
HY4_EFFORT_ALIASES = {"none": "no_think", "minimal": "low", "medium": "high", "xhigh": "max"}   # OpenAI / Hermes spellings -> the template's


class Hy4Parser(GlmParser):
    """Hy4-flash reply: GlmParser with the suffixed tags. Reasoning up to </think:6124c78e>, then content; the <tool_calls:6124c78e> ...
    </tool_calls:6124c78e> block (or a bare <tool_call:6124c78e> block when the model leaves the wrapper out) is held and parsed once at the end,
    one tool_calls entry per <tool_call:6124c78e> in it."""

    def __init__(self, reasoning_first: bool, parse_tools: bool = True, tools: list | None = None):
        QwenParser.__init__(self, reasoning_first, tools, parse_tools)
        self.sp = Splitter(reasoning_first, think_open=True, think_tags=(HY4_THINK_OPEN, HY4_THINK_CLOSE),
                           tool_tags={HY4_CALLS_OPEN: HY4_CALLS_CLOSE, HY4_CALL_OPEN: HY4_CALL_CLOSE})
        self.sp.calls, self.sp.calls_in_reasoning = parse_tools, False

    def finish(self) -> list:
        ev = list(self.sp.finish())
        if self.parse_tools:
            ev += _complete_calls(0, glm_parse_tool_calls(self.sp.tool_text, self.tools, HY4_SUFFIX))
        return ev


def hy4_parse_completion(text: str, tools: list | None = None, reasoning_first: bool = True) -> dict:
    """A whole Hy4 completion (by default the text after the generation prompt's open <think:6124c78e>) -> OpenAI message fields."""
    return glm_parse_completion(text, tools, Hy4Parser(reasoning_first, True, tools))


class Hy4Family:
    """Hy4-flash: the GGUF's own tokenizer.chat_template (tokenizer_config.json of the serve directory), rendered as transformers does.
    Reasoning: the template's `reasoning_effort` no_think / low / high / max (default: left to the template, which picks high); thinking off
    (enable_thinking / thinking false, reasoning_effort none, reasoning.enabled false) = no_think. Tool calls are GLM-shaped with suffixed tags."""
    name = "hy4"
    think_tags = (HY4_THINK_OPEN, HY4_THINK_CLOSE)

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        cfg = Path(model_dir) / "tokenizer_config.json"
        self.template = template or getattr(tokenizer, "chat_template", None) or (json.loads(cfg.read_text()).get("chat_template") if cfg.is_file() else None)
        if not isinstance(self.template, str) or not self.template:
            raise ChatError(f"no chat template in {model_dir}")
        self.defaults = dict(defaults or {})
        self.vision = None

    def _effort(self, value: Any, where: str) -> str:
        e = HY4_EFFORT_ALIASES.get(value, value) if isinstance(value, str) else value
        if e not in HY4_EFFORTS:
            raise ChatError(f"{where}: {value!r}, should be no_think / low / high / max")
        return e

    def options(self, body: dict) -> dict:
        """The template kwargs of a request: chat_template_kwargs, then body fields, then `reasoning` ({"enabled", "effort"}), then the server
        defaults. `preserved_thinking` (or its negation `clear_thinking`) passes through; the template defaults it to `tools` being declared."""
        ctk = body.get("chat_template_kwargs") or {}
        if not isinstance(ctk, dict):
            raise ChatError("chat_template_kwargs must be an object")
        try:
            thinking = ds41_chat._switch(ctk)
            if thinking is None:
                thinking = ds41_chat._switch(body)
        except ds41_chat.ChatFormatError as e:
            raise ChatError(str(e)) from None
        pick = lambda key: ctk[key] if ctk.get(key) is not None else body.get(key)     # noqa: E731
        effort = pick("reasoning_effort")
        r = body.get("reasoning")
        if r is not None and not isinstance(r, dict):
            raise ChatError("reasoning must be an object")
        r = r or {}
        if thinking is None and isinstance(r.get("enabled"), bool):
            thinking = r["enabled"]
        if effort is None and HY4_EFFORT_ALIASES.get(r.get("effort"), r.get("effort")) in HY4_EFFORTS:    # an unknown reasoning.effort is ignored, never a 400
            effort = r["effort"]
        if thinking is None:
            thinking = ds41_chat._switch(self.defaults)
        if effort is None:
            effort = self.defaults.get("reasoning_effort")
        kw: dict = {}
        if effort is not None:
            kw["reasoning_effort"] = self._effort(effort, "reasoning_effort")
        if thinking is False:
            kw["reasoning_effort"] = "no_think"
        elif thinking is True and kw.get("reasoning_effort") == "no_think":     # thinking asked for explicitly: the template's default level
            kw.pop("reasoning_effort")
        keep = pick("preserved_thinking")
        if keep is None and pick("clear_thinking") is not None:
            keep = not pick("clear_thinking")
        if keep is not None:
            if not isinstance(keep, bool):
                raise ChatError("preserved_thinking / clear_thinking must be boolean")
            kw["preserved_thinking"] = keep
        return kw

    def render(self, body: dict) -> Rendered:
        tools = body.get("tools")
        parse_tools = body.get("tool_choice") != "none"
        if not parse_tools:
            tools = None
        kw = self.options(body)
        messages = _glm_messages(body.get("messages"))
        prompt = render_jinja(self.template, messages, tools, **kw)
        cuts = jinja_cut_texts(self.template, messages, tools, prompt, **kw)
        return Rendered(prompt, prompt.endswith(HY4_THINK_OPEN), parse_tools, tools, {"reasoning_effort": kw.get("reasoning_effort", "high")}, cut_texts=cuts)

    def parser(self, r: Rendered):
        return Hy4Parser(r.thinking_first, r.parse_tools, r.tools)


# ------------------------------------------------------------------------------------------------ ds41
class Ds41Family:
    name = "ds41"

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        d = defaults or {}
        self.thinking = d.get("thinking", True)
        self.effort = d.get("reasoning_effort", ds41_chat.DEFAULT_EFFORT)
        self.vision = ds41_vision.Vision(model_dir)

    def render(self, body: dict) -> Rendered:
        tools = body.get("tools")
        parse_tools = body.get("tool_choice") != "none"
        if not parse_tools:
            tools = None
        images = []
        try:
            opts = ds41_chat.chat_options(body, self.thinking, self.effort)
            kw = dict(thinking=opts.thinking, reasoning_effort=opts.effort, drop_thinking=opts.drop_thinking, history_thinking=opts.history_thinking,
                      effort_placement=opts.effort_placement)
            if self.vision.available():
                prompt, images = ds41_chat.render_chat_vl(body.get("messages"), tools, **kw)
            else:
                prompt = ds41_chat.render_chat(body.get("messages"), tools, **kw)
        except ds41_chat.ChatFormatError as exc:
            raise ChatError(str(exc)) from exc
        return Rendered(prompt, opts.thinking, parse_tools, tools, {"thinking": opts.thinking}, self._cut_texts(body, tools, opts, prompt), images)

    @staticmethod
    def _cut_texts(body: dict, tools, opts, prompt: str) -> list:
        """The system block (leading system messages + tools) and the conversation without the generation header (ASSISTANT_SP + think tag)."""
        msgs, out = body.get("messages"), []
        lead = 0
        while lead < len(msgs) and isinstance(msgs[lead], dict) and msgs[lead].get("role") in ("system", "developer"):
            lead += 1
        if 0 < lead < len(msgs):
            try:                                                       # the renderer wants a user turn last: add an empty one, cut before it
                t = ds41_chat.render_chat(msgs[:lead] + [{"role": "user", "content": ""}], tools, thinking=opts.thinking, reasoning_effort=opts.effort,
                                          drop_thinking=opts.drop_thinking, history_thinking=opts.history_thinking,
                                          effort_placement=opts.effort_placement)
                out.append(t[:t.rindex(ds41_chat.USER_SP)])
            except (ds41_chat.ChatFormatError, ValueError):
                pass
        for tag in (ds41_chat.THINK_OPEN + ds41_chat.THINK_CLOSE, ds41_chat.THINK_OPEN, ds41_chat.THINK_CLOSE):
            if prompt.endswith(ds41_chat.ASSISTANT_SP + tag):
                head = prompt[:-len(ds41_chat.ASSISTANT_SP + tag)]
                tail_line = ds41_chat.SYSTEM_SP + ds41_chat.effort_line(opts.effort)
                if opts.thinking and head.endswith(tail_line):         # effort at the tail: the stable prefix ends before it
                    head = head[:-len(tail_line)]
                out.append(head)
        return [t for t in out if t and t != prompt and prompt.startswith(t)]

    def parser(self, r: Rendered):
        return ds41_parse.StreamParser(r.thinking_first, r.parse_tools, r.tools)


class JinjaFamily(QwenFamily):
    """KYOJIN_CHAT_FALLBACK: a model_type with no modeldesc row. The pack's own chat template with the tokenizer's special tokens (bos_token, eos_token ...) as template
    variables, the way transformers' apply_chat_template passes them: the server encodes a rendered prompt without special tokens, so a template that prints
    {{ bos_token }} (Gemma, Llama, Mistral) must see it or the model never reads a BOS. Tool calls and <think> are parsed as the Qwen family does."""
    name = "jinja"

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        try:
            super().__init__(tokenizer, model_dir, template, defaults)
        except ChatError as e:                       # a base pack without a chat template still serves raw completions; chat requests get the error
            self.template, self.defaults, self.vision, self.no_template = None, defaults or {}, None, str(e)
        self.tokens = {k: v for k, v in (getattr(tokenizer, "special_tokens_map", None) or {}).items() if isinstance(v, str)}

    def render(self, body: dict) -> Rendered:
        if self.template is None:
            raise ChatError(f"{self.no_template}: /v1/chat/completions needs one, /v1/completions works")
        return super().render(body)

    def template_kwargs(self, body: dict) -> dict:
        # none_tools: a request without tools renders `tools=None`, as transformers' apply_chat_template does (Llama 3.x tests `tools is not none` and would
        # otherwise put its tool-calling system prompt into every plain chat: the model then answers with a <|python_tag|>{"name": ...} call)
        return {**self.tokens, **super().template_kwargs(body), "none_tools": True}

    def parser(self, r: Rendered):
        return FormatQwenParser(r.thinking_first, r.tools, r.parse_tools, think_open=True)


class GenericFamily(QwenFamily):
    """A bare GGUF of an architecture with no family of its own (llama, mistral3, gemma, glm4, glm4_moe, deepseek_v3 ...): the prompt is the file's own
    tokenizer.chat_template (tokenizer_config.json written from the metadata), stop ids come from the file (generation_config.json), nothing is assumed about the
    reply format: reasoning tags and tool calls are NOT parsed (the template still renders the tools it is given), the reply comes back as text."""
    name = "generic"

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        super().__init__(tokenizer, model_dir, template, defaults)
        self.tokens = {"bos_token": getattr(tokenizer, "bos_token", None) or "", "eos_token": getattr(tokenizer, "eos_token", None) or ""}   # llama-2 / gemma templates print them
        bos = getattr(tokenizer, "bos_token_id", None)
        try:                                                                                     # the tokenizer adds the BOS itself: a template that prints it too must not double it
            self.strip_bos = bool(self.tokens["bos_token"]) and bos is not None and bos in tokenizer.encode("", add_special_tokens=True)
        except Exception:                                                                        # noqa: BLE001
            self.strip_bos = False

    def render(self, body: dict) -> Rendered:
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        kw = {**self.tokens, **self.template_kwargs(body), "none_tools": True}       # the file's template is written for apply_chat_template
        prompt = render_jinja(self.template, body.get("messages"), tools, **kw)
        cuts = jinja_cut_texts(self.template, body.get("messages"), tools, prompt, **kw)
        b = self.tokens["bos_token"]
        if self.strip_bos and prompt.startswith(b):                                              # llama.cpp does the same (common_chat_templates_apply: add_bos + prompt starts with bos)
            prompt = prompt[len(b):]
            cuts = [c[len(b):] if c.startswith(b) else c for c in cuts]
        return Rendered(prompt, False, False, tools, cut_texts=cuts)


FAMILIES = {"qwen": QwenFamily, "glm": GlmFamily, "ds41": Ds41Family, "generic": GenericFamily, "jinja": JinjaFamily, "harmony": HarmonyFamily, "hy4": Hy4Family}
GGUF_ARCH_FAMILIES = {"hy_v4_flash": "hy4"}                          # a serve dir whose config.json names this gguf_architecture gets that family


def family_of_gguf(model_type: str) -> str:
    """Chat family of a bare GGUF file: the Qwen family keeps its qwen2 / qwen2_moe files, every other mapped architecture gets the generic one
    (family_of would hand glm4 / glm4_moe the GLM-5 family and refuse llama / gemma)."""
    from .. import modeldesc
    row = modeldesc.chat_desc(model_type)
    return "qwen" if row is not None and row.server == "qwen" else "generic"


def family_of(model_type: str) -> str:
    from .. import modeldesc
    row = modeldesc.chat_desc(model_type)
    if row is not None:
        return row.server
    if os.environ.get("KYOJIN_CHAT_FALLBACK", "1") != "0":              # no row: the pack's own template (with the tokenizer's special tokens) + the <tool_call> parser
        return "jinja"
    raise ValueError(f"no chat family for model_type {model_type!r}")
