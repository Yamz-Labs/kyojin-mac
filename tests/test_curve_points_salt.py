# SPDX-License-Identifier: Apache-2.0
"""curve_points repeat protocol (CURVE_SALT): every decode request builds the same prompt, whatever token counts the server answers; without it the reps differ. No server."""
import hashlib
import importlib.util
import os

SPEC = importlib.util.spec_from_file_location("curve_points", os.path.join(os.path.dirname(__file__), "..", "tools", "bench", "curve_points.py"))
CP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CP)


class A:
    ctx = 0


def run(monkeypatch, salt):
    seen, counts = [], iter(range(1000, 2000, 37))
    def ask(a, prompt, max_tokens, ignore_eos=False):
        seen.append(hashlib.md5(prompt.encode()).hexdigest())
        return {"prompt_tokens": next(counts), "decode_tok_s": 1.0}
    monkeypatch.setattr(CP, "ask", ask)
    if salt: monkeypatch.setenv("CURVE_SALT", salt)
    else: monkeypatch.delenv("CURVE_SALT", raising=False)
    corpus, st = "lorem ipsum dolor sit amet. " * 4000, {"cpt": 3.5}
    for k in range(3):
        CP.fit(A, st, 4096, lambda n, k=k: CP.make_text(corpus, n, CP.rep_salt(f"dec4096x_r{k}")) + "\nGo.", 8)
    return seen


def test_same_prompt_in_every_rep(monkeypatch):
    assert len(set(run(monkeypatch, "fixed"))) == 1


def test_reps_differ_without_the_salt(monkeypatch):
    assert len(set(run(monkeypatch, ""))) == 3
