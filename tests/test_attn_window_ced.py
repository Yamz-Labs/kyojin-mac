"""pf_attn_lazymask (E6) must not bypass the CED window restart: a tail chunk with `cache._ced_nohist` takes the attention's own (overridden) `_window_part`."""
import types

import pytest

pytest.importorskip("mlx.core")

from kyojin_mlx import attn_prefill as AP          # noqa: E402


class Attn:
    def __init__(self):
        self.calls = []

    def _window_part(self, x, start_pos, cache):
        self.calls.append("method")
        return "kv", "mask"


@pytest.fixture
def lazy(monkeypatch):
    seen = []
    monkeypatch.setattr(AP, "window_part", lambda a, x, s, c, need_mask=True: seen.append(need_mask) or ("kv", None))
    return seen


def test_lazy_mask_on_uses_the_mask_free_window(monkeypatch, lazy):
    monkeypatch.setattr(AP, "LAZY_MASK", True)
    a = Attn()
    assert AP.window_for(a, None, 8, types.SimpleNamespace(_ced_nohist=False)) == ("kv", None)
    assert AP.window_for(a, None, 8, types.SimpleNamespace()) == ("kv", None)
    assert lazy == [False, False] and a.calls == []


def test_ced_tail_chunk_keeps_the_override_with_the_flag_on(monkeypatch, lazy):
    monkeypatch.setattr(AP, "LAZY_MASK", True)
    a = Attn()
    assert AP.window_for(a, None, 8, types.SimpleNamespace(_ced_nohist=True)) == ("kv", "mask")
    assert a.calls == ["method"] and lazy == []


def test_flag_off_uses_the_method(monkeypatch, lazy):
    monkeypatch.setattr(AP, "LAZY_MASK", False)
    a = Attn()
    assert AP.window_for(a, None, 8, types.SimpleNamespace()) == ("kv", "mask")
    assert a.calls == ["method"] and lazy == []
