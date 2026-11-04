"""Flag tok_inc: the spliced ids of a long prompt equal the ids of a whole tokenize, at any edit point."""
import os
import json
import random
from pathlib import Path

import pytest

MODEL = Path(os.path.expanduser("~/ds41/model/tokenizer.json"))
CORPUS = Path.home() / "kyojin-work/hermes-test/corpus1m.json"
pytestmark = pytest.mark.skipif(not MODEL.exists() or not CORPUS.exists(), reason="needs the DS4.1 tokenizer and the corpus")


@pytest.fixture(scope="module")
def tok():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=str(MODEL))


@pytest.fixture(scope="module")
def text():
    c = json.loads(CORPUS.read_text())
    s = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return s[:300000]


def whole(tok, t):
    return tok.encode(t, add_special_tokens=False)


def test_a_common_prefix_is_the_first_difference():
    from kyojin_mlx.serve.tokinc import common_prefix as C, STEP
    assert C("", "") == 0 and C("a", "") == 0 and C("abc", "abd") == 2 and C("abc", "abc") == 3 and C("abc", "abcd") == 3
    a = "x" * (STEP * 3 + 17)
    for k in (0, 1, STEP - 1, STEP, STEP + 1, STEP * 2 + 5, len(a) - 1):
        assert C(a, a[:k] + "y" + a[k + 1:]) == k


TAILS = ["\n\nuser: and then?", " le café coûte 3,50 €… 日本語のテキスト 🙂🙂", "<｜end▁of▁sentence｜>", "def f(x):\n    return x**2  # 123456789\n", "   \t\n\n\n", "a" * 3000, "12345678901234567890"]


def test_a_conversation_grows_with_exact_ids(tok, text):
    from kyojin_mlx.serve.tokinc import IncTok, HITS
    inc = IncTok(tok)
    t = text[:100000]
    before = HITS["inc"]
    for extra in TAILS * 2:
        assert inc.encode(t) == whole(tok, t)
        t += extra
    assert HITS["inc"] - before >= len(TAILS)


def test_random_edit_points_give_exact_ids(tok, text):
    from kyojin_mlx.serve.tokinc import IncTok, HITS
    rng = random.Random(7)
    before = HITS["inc"]
    n = 40
    for _ in range(n):
        inc = IncTok(tok)
        inc.encode(text)
        p = rng.randrange(40000, len(text) - 1)
        cut = rng.randrange(0, 4)
        t = text[:p] + (rng.choice(TAILS) if cut != 3 else "") + (text[p + rng.randrange(0, 50):] if cut != 2 else "")
        assert inc.encode(t) == whole(tok, t), p
    assert HITS["inc"] - before >= n * 3 // 4


def test_a_cut_of_the_prompt_is_exact_and_the_prompt_stays_kept(tok, text):
    from kyojin_mlx.serve.tokinc import IncTok
    inc = IncTok(tok)
    inc.encode(text)
    for p in (100000, 200001, 299000):
        assert inc.encode(text[:p]) == whole(tok, text[:p])
    assert any(e.text == text for e in inc.entries)
    assert inc.encode(text) == whole(tok, text)


def test_short_text_and_flag_off_take_the_plain_path(tok, text):
    from kyojin_mlx.serve import tokinc as T
    inc = T.IncTok(tok)
    assert inc.encode("hello") == whole(tok, "hello") and not inc.entries
    T.ON, on = False, T.ON
    try:
        assert inc.encode(text) == whole(tok, text) and not inc.entries
    finally:
        T.ON = on


def test_the_store_stays_bounded(tok, text):
    from kyojin_mlx.serve import tokinc as T
    inc = T.IncTok(tok)
    for i in range(T.KEEP + 3):
        inc.encode(f"{i} " + text[:70000])
    assert len(inc.entries) == T.KEEP


def test_no_sync_point_falls_back_to_a_whole_tokenize(tok, text):
    from kyojin_mlx.serve import tokinc as T
    inc = T.IncTok(tok)
    t = text[:70000] + "z" * 200000                       # one regex segment longer than any window
    inc.encode(t)
    u = t[:-5] + "y" + t[-4:]
    assert inc.encode(u) == whole(tok, u)


def test_long_segments_give_exact_ids(tok, text):
    """Runs a window can restart inside: one long word, digits, CJK, emoji, base64-like, repeats. The edit lands inside or just after the run."""
    from kyojin_mlx.serve.tokinc import IncTok
    rng = random.Random(11)
    letters = "abcdefghijklmnopqrstuvwxyz"
    runs = ["".join(rng.choice(letters) for _ in range(6000)), "".join(rng.choice("0123456789") for _ in range(6000)), "".join(chr(rng.randrange(0x4e00, 0x9fa5)) for _ in range(4000)),
            "🙂" * 3000, "".join(rng.choice(letters + letters.upper() + "0123456789+/") for _ in range(8000)), "abc" * 3000, " " * 5000, "\n" * 5000, "-" * 5000, "the " * 2000]
    head = text[:70000]
    for run in runs:
        for k in range(6):
            t = head + run + text[70000:72000]
            inc = IncTok(tok)
            inc.encode(t)
            p = len(head) + rng.randrange(0, len(run) + 1500)
            u = t[:p] + rng.choice(TAILS[:4]) + t[p + rng.randrange(0, 4):]
            assert inc.encode(u) == whole(tok, u), (run[:8], p)
