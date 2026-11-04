import json, os, sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import calib_text as C

TEXT = Path(os.environ.get("KYOJIN_CALIB_TEXT", "calib_text.jsonl"))
if not TEXT.exists():
    pytest.skip("set KYOJIN_CALIB_TEXT to a calibration jsonl (not shipped)", allow_module_level=True)


class FakeTok:
    """Any tokenizer: bytes -> ids offset by 3, no chat template (the JSON fallback), eos 2."""
    eos_token_id = 2

    def encode(self, s, add_special_tokens=False):
        return [3 + b for b in s.encode("utf8")]

    def apply_chat_template(self, *a, **k):
        raise ValueError("no template")

    def __len__(self):
        return 259


def test_rows_use_the_given_tokenizer_and_mix():
    arr, plan, _ = C.tokenize_rows(FakeTok(), TEXT, 32, 256)
    assert arr.shape == (32, 256) and arr.min() >= 2 and arr.max() < 259       # ids come from THIS tokenizer's vocabulary
    assert sum(plan.values()) == 32 and plan["tool"] > 0 and plan["code"] > 0 and plan["_mix"] >= 6


def test_tool_rows_are_json_and_fr_en():
    rows = [json.loads(l) for l in TEXT.read_text().splitlines()]
    tools = [r for r in rows if r["domain"] == "tool"]
    assert len(tools) >= 200 and all(r["tools"] and r["messages"][1]["tool_calls"] for r in tools)
    assert any(r["messages"][0]["content"].startswith(("Peux", "Je voudrais", "Merci", "Lance")) for r in tools)


def test_calibration_text_is_disjoint_from_the_parity_docs():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools/qualify"))
    import docs
    cal = TEXT.read_text()
    corp = Path.home() / "qualify_data"
    if not corp.exists():
        return
    for dom, (f, off) in docs.DOMAINS.items():
        t = (corp / f).read_text(encoding="utf8", errors="replace")
        for o in (off, 1_000_000 if dom == "wiki" else off):
            for k in range(0, 12000, 600):                                       # 120-char windows across the eval docs
                w = t[o + k:o + k + 120]
                assert len(w) < 120 or json.dumps(w, ensure_ascii=False)[1:-1] not in cal, (dom, o + k)
