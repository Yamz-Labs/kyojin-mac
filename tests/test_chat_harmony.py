"""gpt-oss harmony output split: analysis -> reasoning, final -> content, turn markers dropped, markers split across pieces held."""
from kyojin_mlx.serve.chat import HarmonyParser

TEXT = ("<|channel|>analysis<|message|>User wants a hash map.\nThink.<|end|><|start|>assistant<|channel|>final<|message|>"
        "A hash map stores pairs.<|return|>")


def _run(pieces):
    p = HarmonyParser()
    ev = []
    for x in pieces:
        ev += p.feed(x)
    ev += p.finish()
    out = {"reasoning": "", "content": ""}
    for k, t in ev:
        out[k] += t
    return out


def test_whole_text():
    assert _run([TEXT]) == {"reasoning": "User wants a hash map.\nThink.", "content": "A hash map stores pairs."}


def test_char_by_char_same_split():
    assert _run(list(TEXT)) == _run([TEXT])


def test_final_only():
    assert _run(["<|channel|>final<|message|>Hi there.<|return|>"]) == {"reasoning": "", "content": "Hi there."}
