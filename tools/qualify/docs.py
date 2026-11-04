"""Eval docs for the parity / KV stages, generic: domain texts cut from the corpus + chat / tool docs rendered with the PACK's own chat template
(adapted from the CUDA/ROCm docs.py / nanb mk_docs.py). Tokenization uses the pack's tokenizer (transformers AutoTokenizer, in the venv). Output is JSON
(no torch on the Mac): {"names": [...], "ids": [[...], ...]}. Run: docs.py short|long PACK OUT.json [N_TOKENS]."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from load import corpus_dir, load_corpus

DOMAINS = {"wiki": ("wiki.utf8", 1_900_000), "code": ("code.utf8", 700_000), "fr": ("multilingual.utf8", 28_000), "tech": ("technical.utf8", 240_000)}
CHAT_FR = [{"role": "user", "content": "Écris une fonction Python qui fusionne des intervalles qui se chevauchent, avec une courte explication en français."},
           {"role": "assistant", "content": "Trie les intervalles par début, puis parcours-les une fois en étendant le dernier intervalle fusionné.\n\n```python\ndef merge(iv):\n    out = []\n    for s, e in sorted(iv):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n```\n\nComplexité O(n log n) à cause du tri."}]
CHAT_EN = [{"role": "user", "content": "Explain in three sentences why the sky is blue, then give a one-line analogy."},
           {"role": "assistant", "content": "Sunlight contains all colors. Air molecules scatter short wavelengths far more than long ones (Rayleigh scattering), so blue light reaches your eye from every direction. It is like a sieve that lets red through and bounces blue around."}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}}}]
TOOL_MSGS = [{"role": "user", "content": "What's the weather in Paris and Tokyo, in Celsius?"},
             {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}}, {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "C"}}}]},
             {"role": "tool", "content": "{\"city\": \"Paris\", \"temp\": 21}"}, {"role": "tool", "content": "{\"city\": \"Tokyo\", \"temp\": 17}"},
             {"role": "assistant", "content": "Paris is at 21 °C and Tokyo at 17 °C."}]


def _tok(pack: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    p = Path(pack)
    if (p / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(str(p))
    fallback = Path.home() / "kyojin-mlx/models/tok_ds41"          # slice packs / DS4.1 reference dirs carry a bare tokenizer.json elsewhere
    return PreTrainedTokenizerFast(tokenizer_file=str((p if (p / "tokenizer.json").exists() else fallback) / "tokenizer.json"))


def _text(name: str) -> str:
    f = corpus_dir() / name
    return f.read_text(encoding="utf8", errors="replace") if f.exists() else load_corpus()


def make_docs(pack: str, n_tokens: int = 1024) -> dict:
    """{'names': [...], 'ids': [[...]]}: chat_fr, chat_en, tool (when the pack has a jinja template), wiki, code, fr, tech."""
    tok = _tok(pack)
    enc = lambda s: tok.encode(s, add_special_tokens=False)    # noqa: E731
    docs = {}
    for name, msgs, tools in (("chat_fr", CHAT_FR, None), ("chat_en", CHAT_EN, None), ("tool", TOOL_MSGS, TOOLS)):
        try:
            docs[name] = enc(tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False))
        except Exception as e:             # no jinja template (DS4.1 uses its own encoder) or the template rejects the doc: the domain is not measured
            print(f"docs: {name} skipped ({repr(e)[:100]})", file=sys.stderr)
    for name, (f, off) in DOMAINS.items():
        t = _text(f)
        off = off if off + n_tokens * 8 < len(t) else max(0, len(t) // 2)
        docs[name] = enc(t[off:off + n_tokens * 8])[:n_tokens]
    docs = {k: v[:n_tokens] for k, v in docs.items() if len(v) >= 16}
    return dict(names=list(docs), ids=list(docs.values()))


def make_long_doc(pack: str, n_tokens: int = 4096) -> dict:
    """One long doc for the KV stage (wiki + code + fr + tech slices; late offsets: held out in practice, not by construction)."""
    tok = _tok(pack)
    ids = []
    for f, off in [("wiki.utf8", 1_000_000), ("code.utf8", 900_000), ("multilingual.utf8", 150_000), ("technical.utf8", 200_000)]:
        t = _text(f)
        off = off if off + n_tokens * 3 < len(t) else max(0, len(t) // 3)
        ids += tok.encode(t[off:off + n_tokens * 3], add_special_tokens=False)[: n_tokens // 4 + 1]
    return dict(names=["long"], ids=[ids[:n_tokens]])


if __name__ == "__main__":
    kind, pack, out = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else (1024 if kind == "short" else 4096)
    d = make_docs(pack, n) if kind == "short" else make_long_doc(pack, n)
    Path(out).write_text(json.dumps(d))
    print({k: len(i) for k, i in zip(d["names"], d["ids"])})
