#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer-generic calibration for the EXL3 conversion: a RAW text set (data/calib_text.jsonl, no token ids) that is tokenized by the model's OWN
tokenizer (and chat template) at run time.

  calib_text.py build  CORPUS_DIR OUT.jsonl [--seed 0]          one-off: cut the raw set from the qualify corpus (held out from the parity docs)
  calib_text.py tokens SRC_DIR --text CALIB_TEXT.jsonl [--rows 128] [--cols 1024] --out CAL.safetensors   -> {"input_ids": int32 [rows, cols]}

Recipe mix (METHOD.md / RECIPE_v1.md): tool JSON, code, fr / en chat + prose, technical text, and long rows (a quarter of the rows are packed across
domains). Rows are contiguous token streams, so a row holds several short documents. Jsonl line = {"domain", "text"} or, for chat / tool rows,
{"domain", "messages", "tools"} rendered with the tokenizer's chat template (a plain-JSON rendering when the tokenizer has none).
Held out: the character spans the parity / KV docs of tools/qualify/docs.py read (EVAL_SPANS) are never used here."""
from __future__ import annotations

import argparse, json, random, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# char spans of the corpus files read by tools/qualify/docs.py (short docs up to 8192 tokens * 8 chars, long doc 3 chars / token): 80 K each, generous
EVAL_SPANS = {"wiki.utf8": [(990_000, 1_070_000), (1_890_000, 2_100_000)], "code.utf8": [(690_000, 770_000), (890_000, 970_000)],
              "multilingual.utf8": [(20_000, 100_000), (140_000, 220_000)], "technical.utf8": [(190_000, 270_000)], "c4.utf8": []}
MIX = {"tool": 0.20, "code": 0.22, "en": 0.18, "fr": 0.08, "multi": 0.06, "tech": 0.10, "chat": 0.16}
LONG_FRAC = 0.25
FR_WORDS = re.compile(r"\b(?:les|des|est|une|dans|pour|avec|qui|sont)\b")
OTHER_WORDS = re.compile(r"\b(?:della|che|il|di|per|nel|und|der|los|del)\b")


# ---------------------------------------------------------------------------------------------- build (one-off, from the corpus)
def allowed(name: str, n: int) -> list:
    cuts, lo = [], 0
    for a, b in sorted(EVAL_SPANS[name]):
        if a > lo:
            cuts.append((lo, min(a, n)))
        lo = max(lo, b)
    if lo < n:
        cuts.append((lo, n))
    return [(a, b) for a, b in cuts if b - a > 4000]


def pieces(text: str, name: str, size: int, budget: int, rng: random.Random, split: str) -> list:
    """Pieces of ~size chars cut at `split` boundaries inside the allowed spans, random order, up to `budget` chars."""
    out = []
    for a, b in allowed(name, len(text)):
        seg = text[a:b]
        parts, cur = [], ""
        for p in seg.split(split):
            cur += p + split
            if len(cur) >= size:
                parts.append(cur.strip()); cur = ""
        out += [p for p in parts[1:] if len(p) > size // 2]       # parts[0] may start mid-sentence
    rng.shuffle(out)
    got, res = 0, []
    for p in out:
        if got >= budget:
            break
        res.append(p); got += len(p)
    return res


VERBS = ["get", "search", "create", "delete", "update", "list", "send", "convert", "calculate", "book", "cancel", "find", "fetch", "translate", "schedule"]
NOUNS = ["order", "invoice", "flight", "contact", "event", "file", "ticket", "stock_price", "recipe", "route", "user", "message", "playlist", "document", "product"]
PARAMS = [("query", "string"), ("limit", "integer"), ("id", "string"), ("date", "string"), ("amount", "number"), ("currency", "string"), ("language", "string"),
          ("include_details", "boolean"), ("name", "string"), ("email", "string"), ("sort", "string"), ("page", "integer"), ("origin", "string"), ("destination", "string")]
VALUES = {"string": ["alpha", "Lyon", "2026-11-03", "report.pdf", "marie@example.org", "EUR", "fr", "name_asc", "A-4821", "Berlin", "hello world"],
          "integer": [1, 3, 5, 10, 25, 100], "number": [9.99, 120.5, 3.14, 1500.0], "boolean": [True, False]}
USER_EN = ["Please {v} the {n} for me ({a}).", "Can you {v} a {n}? Use {a}.", "I need to {v} my {n}, details: {a}.", "{v} {n} with {a}, thanks."]
USER_FR = ["Peux-tu faire l'action « {v} » sur {n} avec {a} ?", "Je voudrais {v} un {n} ({a}).", "Merci de {v} le {n} : {a}.", "Lance {v} {n} avec {a} s'il te plaît."]
DONE_EN = ["Done. The {n} operation returned: {r}.", "The {v} call succeeded and the service answered {r}."]
DONE_FR = ["C'est fait. L'opération sur {n} a renvoyé : {r}.", "L'appel {v} a réussi, le service a répondu {r}."]


def tool_conversations(rng: random.Random, n: int) -> list:
    rows = []
    for i in range(n):
        fr = i % 3 == 0
        tools, calls = [], []
        for _ in range(rng.choice([1, 2, 2, 3, 4])):
            v, noun = rng.choice(VERBS), rng.choice(NOUNS)
            ps = rng.sample(PARAMS, rng.randint(1, 4))
            props = {p: {"type": t, "description": f"The {p.replace('_', ' ')} of the {noun.replace('_', ' ')}."} for p, t in ps}
            for p, t in ps:
                if t == "string" and rng.random() < 0.3:
                    props[p]["enum"] = rng.sample(VALUES["string"], 3)
            tools.append({"type": "function", "function": {"name": f"{v}_{noun}", "description": f"{v.capitalize()} a {noun.replace('_', ' ')}.",
                                                              "parameters": {"type": "object", "properties": props, "required": [p for p, _ in ps[:max(1, len(ps) - 1)]]}}})
        for t in rng.sample(tools, rng.randint(1, min(2, len(tools)))):
            f = t["function"]
            args = {}
            for p, spec in f["parameters"]["properties"].items():
                args[p] = rng.choice(spec["enum"]) if "enum" in spec else rng.choice(VALUES[spec["type"]])
            calls.append((f["name"], args))
        tgt = calls[0]
        v, noun = tgt[0].split("_", 1)
        user = rng.choice(USER_FR if fr else USER_EN).format(v=v, n=noun.replace("_", " "), a=json.dumps(tgt[1], ensure_ascii=False))
        results = [{"status": "ok", "id": f"X-{rng.randint(1000, 9999)}", "echo": a, "count": rng.randint(0, 40)} for _, a in calls]
        done = rng.choice(DONE_FR if fr else DONE_EN).format(n=noun.replace("_", " "), v=v, r=json.dumps(results[0], ensure_ascii=False))
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": nm, "arguments": a}} for nm, a in calls]}]
        msgs += [{"role": "tool", "content": json.dumps(r, ensure_ascii=False)} for r in results]
        msgs.append({"role": "assistant", "content": done})
        rows.append({"domain": "tool", "messages": msgs, "tools": tools})
    return rows


PROMPTS_EN = ["Explain this in a few sentences.", "Summarize the following topic for me.", "What should I know about this?", "Give me a short overview.", "Tell me more about it."]
PROMPTS_FR = ["Explique-moi cela en quelques phrases.", "Résume ce sujet pour moi.", "Que faut-il savoir à ce propos ?", "Donne-moi un bref aperçu.", "Dis-m'en plus."]


def chat_rows(rng: random.Random, en_pieces: list, fr_pieces: list, n: int) -> list:
    rows = []
    for i in range(n):
        fr = i % 2 == 0 and fr_pieces
        body = rng.choice(fr_pieces if fr else en_pieces)[:900]
        rows.append({"domain": "chat", "messages": [{"role": "user", "content": rng.choice(PROMPTS_FR if fr else PROMPTS_EN)}, {"role": "assistant", "content": body}]})
    return rows


def build(corpus: Path, out: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rd = lambda f: (corpus / f).read_text(encoding="utf8", errors="replace")      # noqa: E731
    rows = []
    wiki = pieces(rd("wiki.utf8"), "wiki.utf8", 2400, 110_000, rng, "\n\n")
    c4 = pieces(rd("c4.utf8"), "c4.utf8", 2400, 110_000, rng, ". ")
    code = pieces(rd("code.utf8"), "code.utf8", 2400, 260_000, rng, "\n")
    multi = pieces(rd("multilingual.utf8"), "multilingual.utf8", 900, 90_000, rng, "\n\n")
    tech = pieces(rd("technical.utf8"), "technical.utf8", 2400, 110_000, rng, "\n\n")
    is_fr = lambda p: len(FR_WORDS.findall(p)) >= 3 and len(FR_WORDS.findall(p)) > len(OTHER_WORDS.findall(p))      # noqa: E731
    fr = [p for p in multi if is_fr(p)]
    for dom, ps in (("en", wiki[: len(wiki) // 2] + c4), ("code", code), ("fr", fr), ("multi", [p for p in multi if not is_fr(p)]), ("tech", tech)):
        rows += [{"domain": dom, "text": p} for p in ps]
    rows += tool_conversations(rng, 260)
    rows += chat_rows(rng, wiki[len(wiki) // 2:], fr, 140)
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf8")
    cnt = {}
    for r in rows:
        cnt[r["domain"]] = cnt.get(r["domain"], 0) + 1
    return {"rows": len(rows), "by_domain": cnt, "bytes": out.stat().st_size, "fr_pieces": len(fr)}


# ---------------------------------------------------------------------------------------------- run time (the model's own tokenizer)
def render(tok, r: dict) -> str:
    if "text" in r:
        return r["text"]
    try:
        return tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False, add_generation_prompt=False)
    except Exception:                                   # no jinja template or the template rejects the doc: plain JSON of the same content
        return json.dumps({"tools": r.get("tools"), "messages": r["messages"]}, ensure_ascii=False)


def tokenize_rows(tok, text_path: Path, rows: int, cols: int, seed: int = 0) -> tuple:
    import numpy as np
    rng = random.Random(seed)
    docs = {}
    for line in text_path.read_text(encoding="utf8").splitlines():
        r = json.loads(line)
        docs.setdefault(r["domain"], []).append(r)
    eos = [tok.eos_token_id] if getattr(tok, "eos_token_id", None) is not None else []
    streams = {}
    for dom, rs in docs.items():
        rng.shuffle(rs)
        ids = []
        for r in rs:
            ids += tok.encode(render(tok, r), add_special_tokens=False) + eos
        streams[dom] = ids
    n_long = int(round(rows * LONG_FRAC))
    plan = {d: int(rows * (1 - LONG_FRAC) * w) for d, w in MIX.items() if d in streams}
    plan["_mix"] = rows - sum(plan.values())
    # mixed (long) rows: documents of every domain interleaved in doc-sized strides
    mixed, order = [], list(streams)
    cursors = {d: 0 for d in order}
    while sum(len(m) for m in mixed) < plan["_mix"] * cols + cols:
        d = order[len(mixed) % len(order)]
        s = streams[d]; c = cursors[d] % max(1, len(s) - 400)
        mixed.append(s[c:c + 400]); cursors[d] = c + 400
    streams["_mix"] = [t for m in mixed for t in m]
    out = []
    for dom, k in plan.items():
        s = streams[dom]
        if not s:
            continue
        need = k * cols
        if need == 0:
            continue
        if len(s) < need:
            s = (s * (need // max(1, len(s)) + 1))
        for j in range(k):
            out.append(s[j * cols:(j + 1) * cols])
    arr = np.array(out, np.int32)
    perm = np.random.default_rng(seed).permutation(len(arr))
    return arr[perm], {d: int(k) for d, k in plan.items()}, n_long


def tokens_cli(src: str, text: Path, rows: int, cols: int, out: str, seed: int = 0) -> dict:
    from transformers import AutoTokenizer
    import mlx.core as mx
    tok = AutoTokenizer.from_pretrained(src)
    arr, plan, _ = tokenize_rows(tok, text, rows, cols, seed)
    vocab = len(tok)
    assert arr.shape == (rows, cols) and arr.min() >= 0 and arr.max() < vocab, (arr.shape, arr.max(), vocab)
    mx.save_safetensors(out, {"input_ids": mx.array(arr)})
    return {"rows": rows, "cols": cols, "plan": plan, "vocab": vocab, "tokenizer": type(tok).__name__}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("corpus"); b.add_argument("out"); b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("tokens"); t.add_argument("src"); t.add_argument("--text", required=True, help="raw calibration text, one JSON object per line (not shipped; build it with the `build` command)"); t.add_argument("--rows", type=int, default=128)
    t.add_argument("--cols", type=int, default=1024); t.add_argument("--out", required=True); t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(Path(a.corpus).expanduser(), Path(a.out), a.seed)))
    else:
        print(json.dumps(tokens_cli(a.src, Path(a.text), a.rows, a.cols, a.out, a.seed)))


if __name__ == "__main__":
    main()
