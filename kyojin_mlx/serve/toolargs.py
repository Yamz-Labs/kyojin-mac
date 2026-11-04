"""Tool-call argument values: schema coercion and a light JSON repair, shared by the Qwen and DS4.1 parsers."""
from __future__ import annotations

import ast
import json
import math
from typing import Any


def schema_types(schema: dict | None) -> list[str]:
    kind = (schema or {}).get("type")
    return [k for k in (kind if isinstance(kind, list) else [kind]) if isinstance(k, str)]


def coerce(value: str, schema: dict | None) -> Any:
    """Typed value of one parameter written as text; the schema type wins (tools/qwen/serve.py)."""
    kinds = schema_types(schema)
    text = value.strip()
    if kinds and "string" in kinds and not (text == "null" and "null" in kinds):
        return value
    if "boolean" in kinds and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if "null" in kinds and text in ("null", "None"):
        return None
    for want, conv in (("integer", int), ("number", float)):
        if want in kinds:
            try:
                got = conv(text)
            except ValueError:
                continue
            if isinstance(got, float) and not math.isfinite(got):
                continue
            return got
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if schema is None and not isinstance(parsed, (dict, list)):
        return value
    return parsed


def _close_open(raw: str) -> str:
    """Close the strings, arrays and objects a cut-off JSON text left open."""
    stack, in_str, esc = [], False, False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}" and stack:
            stack.pop()
    out = raw + ('\\' if esc else "") + ('"' if in_str else "")
    out = out.rstrip()
    while out.endswith(","):
        out = out[:-1].rstrip()
    return out + "".join(reversed(stack))


def repair_json(raw: str) -> tuple[bool, Any]:
    """(True, value) when `raw` is JSON, a Python literal (True/None/'single quotes') or cut-off JSON; else (False, None)."""
    text = raw.strip()
    if not text:
        return False, None
    try:
        return True, json.loads(text)
    except ValueError:
        pass
    try:
        value = ast.literal_eval(text)
        json.dumps(value)
        return True, value
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    if text[0] in "[{":
        try:
            return True, json.loads(_close_open(text))
        except ValueError:
            pass
    return False, None
