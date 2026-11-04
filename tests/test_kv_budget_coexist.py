from kyojin_mlx.serve import engine as E

import pytest

G = E.GIB


@pytest.fixture(autouse=True)
def cache_limit_2g():
    old = E.mx.set_cache_limit(2 * G)                                            # the server sets it before sizing the KV
    yield
    E.mx.set_cache_limit(old)


def test_default_kv_budget_shrinks_beside_a_resident_lane(monkeypatch):
    monkeypatch.setattr(E, "kv_budget_for", lambda *a: 180 * G)                 # fixed terms see only this process
    monkeypatch.setattr(E, "free_unified_bytes", lambda: 40 * G)                # a 189 GiB lane holds the rest
    assert E.default_kv_budget() == 37 * G


def test_default_kv_budget_keeps_fixed_terms_on_a_quiet_machine(monkeypatch):
    monkeypatch.setattr(E, "kv_budget_for", lambda *a: 20 * G)
    monkeypatch.setattr(E, "free_unified_bytes", lambda: 200 * G)
    assert E.default_kv_budget() == 20 * G
    monkeypatch.setattr(E, "free_unified_bytes", lambda: None)
    assert E.default_kv_budget() == 20 * G
