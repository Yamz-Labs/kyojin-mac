"""rope_fma (KYOJIN_ROPE_FMA, default 0): the fused rope kernels' pair rotation as a fma, and the kv-chain ring map. No GPU needed.

The kvchain / ropefuse kernels run under fp contract(off); mx.fast.rope may contract its products. These tests pin what can be proved
without Metal: the default text is untouched, the other forms rewrite every rope-tail product of every fused kernel, the cache keys and
kernel names separate the forms, the numpy emulation of the forms differs in fp16 on about 1 element in 10^4 (the size of a md5 drift over a few
hundred tokens), and the dense `wmap` of the kv-chain kernel equals the `overlay` scatter of the stock chain for any session layout."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import dsv4_core as C, flags as FL, stagek2 as SK2  # noqa: E402
from kyojin_mlx.ref_numpy import rope_pairs_ref  # noqa: E402

SOURCES = ("_KV_CHAIN", "_ATTN_Q", "_ATTN_PART_Q", "_ATTN_MERGE_Q")


def test_default_is_auto_and_form_0_keeps_the_old_text():
    assert FL.FLAGS["rope_fma"].default == "3" and FL.FLAGS["rope_fma"].kind == "int" and C.ROPE_FMA in (0, 1, 2)
    for name in SOURCES:
        src = getattr(C, name)
        assert C.rope_src(src, 0) is src                                        # form 0: the old text, same object


@pytest.mark.parametrize("name", SOURCES)
@pytest.mark.parametrize("form", [1, 2])
def test_forms_rewrite_every_rope_product_and_nothing_else(name, form):
    src = getattr(C, name)
    n = src.count(C._ROPE_PLAIN[0])
    assert n >= 1 and src.count(C._ROPE_PLAIN[1]) == n
    out = C.rope_src(src, form)
    a, b = C._ROPE_FORMS[form]
    assert C._ROPE_PLAIN[0] not in out and C._ROPE_PLAIN[1] not in out
    assert out.count(a) == n and out.count(b) == n
    assert out.replace(a, C._ROPE_PLAIN[0]).replace(b, C._ROPE_PLAIN[1]) == src      # only the products moved


def test_stage_qkv_source_inherits_the_kv_chain_rope_text():
    assert C._ROPE_PLAIN[0] in SK2.SOURCE and C.rope_src(SK2.SOURCE, 1).count("fma(x1, c, -(x2 * s))") == C._KV_CHAIN.count(C._ROPE_PLAIN[0])


def test_cache_keys_and_names_separate_the_forms(monkeypatch):
    names = set()
    for f in (0, 1, 2):
        monkeypatch.setattr(C, "ROPE_FMA", f)
        names.add(C._fk())
    assert names == {"", "_fma1", "_fma2"}


def test_flag_switches_the_module_attribute_at_runtime():
    old = C.ROPE_FMA
    try:
        FL.set("rope_fma", "2")
        assert C.ROPE_FMA == 2
        FL.set("rope_fma", "0")
        assert C.ROPE_FMA == 0
    finally:
        C.ROPE_FMA = old


def _tab(n, rng):
    ang = rng.uniform(0, 2 * np.pi, (n, 32))
    t = np.empty((n, 64), np.float32)
    t[:, 0::2], t[:, 1::2] = np.cos(ang), np.sin(ang)
    return t


def test_emulated_forms_differ_in_fp16_at_the_expected_rate():
    """The mechanism: same inputs, plain vs fma form -> fp16 results differ on ~1 element in 10^4 (so a few thousand elements can never see it)."""
    rng = np.random.default_rng(3)
    n = 40000                                                              # x 64 = 2.56M elements
    x = (rng.standard_normal((n, 64)) * 0.5).astype(np.float16)
    t = _tab(n, rng)
    p, f1, f2 = (rope_pairs_ref(x, t, f) for f in (0, 1, 2))
    r1, r2 = (float((p != f).mean()) for f in (f1, f2))
    assert 1e-5 < r1 < 1e-2 and 1e-5 < r2 < 1e-2, (r1, r2)
    assert (p[:2000] == f1[:2000]).mean() > 0.99                           # and 2k elements (the Metal tests' sample) usually agree


def test_emulation_is_exact_for_the_nope_identity():
    """cos = 1, sin = 0 (the nope dims of the 512-wide stock rope): every form returns x exactly."""
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((8, 64)) * 3).astype(np.float16)
    t = np.zeros((8, 64), np.float32)
    t[:, 0::2] = 1.0
    for f in (0, 1, 2):
        assert np.array_equal(rope_pairs_ref(x, t, f), x)


# ------------------------------------------------------------------------------------------------ the kv chain's ring map
def _ring_overlay(plan, base, kv, m):
    """The stock chain: base[session] with the overlay scatter (what dsv4_core.core builds without kvchain)."""
    ov = plan.overlay(m)
    win = base[np.array(plan.sess)].copy()
    rows, slots = np.array(ov.rows), np.array(ov.slots)
    src = rows if ov.src is None else np.array(ov.src)
    win[rows, slots] = kv[src]
    return win


def _ring_wmap(plan, base, kv, m):
    """The kv-chain kernel's rule: slot s of row b holds kv[wmap[b, s]] when >= 0, else the base ring of b's session."""
    wm = np.array(plan.wmap(m)).reshape(plan.n, m)
    out = np.empty((plan.n, m, base.shape[-1]), base.dtype)
    for b in range(plan.n):
        for s in range(m):
            out[b, s] = base[plan.sess[b], s] if wm[b, s] < 0 else kv[wm[b, s]]
    return out


@pytest.mark.parametrize("pos,counts", [
    ([5], None), ([5, 130], None), ([40, 41, 42, 43], [4]), ([40, 41, 90, 91], [2, 2]), ([126, 127, 128, 129, 130], [5]),
    ([250, 251, 252, 253, 7, 8], [4, 2]), ([0, 1, 2, 3, 4, 5], [6]), ([127, 128, 255, 256, 257], [2, 3]),
])
def test_wmap_equals_the_overlay_scatter(pos, counts):
    m, D = 128, 8
    plan = C.Plan(pos, counts)
    rng = np.random.default_rng(len(pos))
    base = rng.standard_normal((plan.S, m, D)).astype(np.float32)
    kv = rng.standard_normal((plan.n, D)).astype(np.float32)
    assert np.array_equal(_ring_overlay(plan, base, kv, m), _ring_wmap(plan, base, kv, m))
