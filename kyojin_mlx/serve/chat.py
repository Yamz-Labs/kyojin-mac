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
    if not r.thinking_first or r.prompt.rstrip().endswith("</think>"):
        raise ChatError("continue_final_message: this request does not end inside an open think block (thinking is off?)")
    r.prompt += prefix
    return r


# ------------------------------------------------------------------------------------------------ templates (qwen, glm)
def _template_env() -> Any:
    """Same environment as transformers' apply_chat_template (and the box servers)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
        json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise ChatError(str(msg))
    env.globals["raise_exception"] = raise_exception
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
THINK_CLOSE = "</think>"
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

    def __init__(self, reasoning_first: bool):
        self.mode = "reasoning" if reasoning_first else "content"
        self.buf = ""
        self.ws = ""
        self.lead = reasoning_first
        self.tool_text = ""
        self.tool_from = 0
        self.said = self.after_call = False

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
                end = self.tool_text.find(TOOL_CLOSE, self.tool_from)
                if end < 0:
                    return out
                end += len(TOOL_CLOSE)
                self.buf, self.tool_text = self.tool_text[end:], self.tool_text[:end]
                self.mode, self.lead, self.after_call = "content", True, True
                continue
            tags = [TOOL_OPEN] + ([THINK_CLOSE] if self.mode == "reasoning" else [])
            hits = [(self.buf.find(t), t) for t in tags if t in self.buf]
            if hits:
                pos, tag = min(hits)
                self._emit(self.buf[:pos], out)
                self.ws = ""
                self.buf = self.buf[pos + len(tag):]
                if tag == TOOL_OPEN:
                    self.mode, self.tool_from = "tool", len(self.tool_text)
                    self.tool_text += TOOL_OPEN
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
    def __init__(self, reasoning_first: bool, tools: list | None, parse_tools: bool = True):
        self.sp, self.tools, self.parse_tools = Splitter(reasoning_first), tools, parse_tools

    def feed(self, text: str) -> list:
        return [(k, t) for k, t in self.sp.feed(text)]

    def finish(self) -> list:
        ev = [(k, t) for k, t in self.sp.finish()]
        if self.parse_tools:
            ev += _complete_calls(0, parse_tool_calls(self.sp.tool_text, self.tools))
        return ev


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


# ------------------------------------------------------------------------------------------------ glm
def glm_parse_completion(text: str) -> dict:
    """Split GLM think blocks and tool calls into OpenAI fields (tools/glm/serve.py parse_completion)."""
    calls: list = []
    content_parts: list = []
    reasoning_parts: list = []
    cursor = 0
    pattern = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
    for match in pattern.finditer(text):
        visible = text[cursor:match.start()]
        if "<think>" in visible:
            before, after = visible.split("<think>", 1)
            content_parts.append(before)
            thinking, _, visible = after.partition("</think>")
            reasoning_parts.append(thinking)
        content_parts.append(visible)
        body = match.group(1)
        name_match = re.match(r"\s*([^\s<]+)", body)
        if not name_match:
            continue
        args: dict = {}
        for part in re.finditer(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.S):
            args[part.group(1).strip()] = _json_value(part.group(2).strip())
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name_match.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
        cursor = match.end()
    tail = text[cursor:]
    if "<think>" in tail:
        before, after = tail.split("<think>", 1)
        content_parts.append(before)
        thinking, _, tail = after.partition("</think>")
        reasoning_parts.append(thinking)
    content_parts.append(tail)
    message: dict = {"role": "assistant", "content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts)
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


class GlmParser:
    """Reasoning and prose stream as they decode; everything from <tool_call> on is held and parsed once at the end
    (the box GLM server does the same)."""

    def __init__(self, reasoning_first: bool, parse_tools: bool = True):
        self.text = "<think>" if reasoning_first else ""
        self.done = len(self.text)
        self.mode = "reasoning" if reasoning_first else "content"
        self.parse_tools = parse_tools
        self.hold = max(len("</think>"), len(TOOL_OPEN)) - 1

    def _flush(self, final: bool) -> list:
        out: list = []
        while self.mode != "tool":
            tags = ["</think>"] if self.mode == "reasoning" else ["<think>", TOOL_OPEN]
            hits = [(self.text.find(t, self.done), t) for t in tags if self.text.find(t, self.done) >= 0]
            end = min(hits)[0] if hits else (len(self.text) if final else max(self.done, len(self.text) - self.hold))
            piece = self.text[self.done:end]
            if piece:
                out.append(("reasoning" if self.mode == "reasoning" else "content", piece))
            self.done = end
            if not hits:
                break
            pos, tag = min(hits)
            self.done = pos + len(tag)
            self.mode = {"</think>": "content", "<think>": "reasoning", TOOL_OPEN: "tool"}[tag]
        return out

    def feed(self, text: str) -> list:
        self.text += text
        return self._flush(False)

    def finish(self) -> list:
        out = self._flush(True)
        if self.parse_tools:
            out += _complete_calls(0, glm_parse_completion(self.text).get("tool_calls", []))
        return out


class GlmFamily:
    name = "glm"
    DEFAULTS = {"reasoning_effort": "medium"}

    def __init__(self, tokenizer, model_dir: str, template: str | None = None, defaults: dict | None = None):
        path = Path(model_dir) / "chat_template.jinja"
        self.template = template or (ASSETS / "glm53-template-medium.jinja").read_text()
        self.defaults = {**self.DEFAULTS, **(defaults or {})}

    def render(self, body: dict) -> Rendered:
        tools = body.get("tools")
        parse_tools = body.get("tool_choice") != "none"
        if not parse_tools:
            tools = None
        ctk = body.get("chat_template_kwargs") or {}
        if not isinstance(ctk, dict):
            raise ChatError("chat_template_kwargs must be an object")
        pick = lambda key, default=None: ctk[key] if ctk.get(key) is not None else body.get(key, default)   # noqa: E731
        kw = {"clear_thinking": pick("clear_thinking")}
        effort = pick("reasoning_effort", self.defaults["reasoning_effort"])
        if effort is not None:
            kw["reasoning_effort"] = effort
        if pick("effort_guidance", os.environ.get("KYOJIN_EFFORT_GUIDANCE") == "1"):   # opt-in medium guidance text, off by default
            kw["effort_guidance"] = True
        prompt = render_jinja(self.template, body.get("messages"), tools, **kw)
        cuts = jinja_cut_texts(self.template, body.get("messages"), tools, prompt, **kw)
        return Rendered(prompt, prompt.rstrip().endswith("<think>"), parse_tools, tools, cut_texts=cuts)

    def parser(self, r: Rendered):
        return GlmParser(r.thinking_first, r.parse_tools)


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


FAMILIES = {"qwen": QwenFamily, "glm": GlmFamily, "ds41": Ds41Family, "generic": GenericFamily}


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
    if os.environ.get("KYOJIN_CHAT_FALLBACK", "0") != "0":              # no row: the pack's own template + the <tool_call> parser (QwenFamily)
        return "qwen"
    raise ValueError(f"no chat family for model_type {model_type!r}")
