"""cmerge: the eight merged lever branches stay default-off, compose, and keep the fp16 saturation. CPU only (kernel sources are strings)."""
import ast
import collections
import itertools
import os

import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL, metal_nax as NX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# every flag the eight branches added: (name, expected default). All are "0" except the numeric floors, which only act when their lever is on.
NEW = {"pf_unroll2": "0", "pf_bmu16": "0", "pf_tail_c4": "0", "pf_nosync_rowend": "0", "pf_moe_epi_cap": "0", "spec_couple": "0", "dec_filter": "0",
       "pf_idx_bound": "0", "kv_digest_inc": "0", "mix_decode": "0", "mix_slice_ms": "250", "pf_share": "0", "pf_path": "0", "kv_admit": "0"}
NUMERIC = {"dec_filter_min": "131072", "dec_filter_budget": "2", "pf_idx_bound_min": "131072", "pf_idx_bound_sb": "64", "pf_idx_bound_maxpct": "25"}


def test_new_flags_default_off_and_documented():
    levers = open(os.path.join(ROOT, "docs", "cloud", "LEVERS.md")).read()
    for name, d in {**NEW, **NUMERIC}.items():
        f = FL.FLAGS[name]
        assert f.default == d, name
        assert f.env in levers or name in levers, f"{name}: neither {f.env} nor the admin name is in docs/cloud/LEVERS.md"       # the numeric floors are described inside their lever's row
    for name in NEW:
        assert f"`{FL.FLAGS[name].env}`" in levers, f"{name}: no row in docs/cloud/LEVERS.md"


def test_flags_table_has_no_duplicate_key_or_env():
    """A dict literal keeps the last of two equal keys silently: an auto-merge that doubled a flag would drop one of them."""
    tree = ast.parse(open(os.path.join(ROOT, "kyojin_mlx", "flags.py")).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "FLAGS":
            keys = [k.value for k in n.value.keys]
            envs = [v.args[0].value for v in n.value.values]
            assert [k for k, c in collections.Counter(keys).items() if c > 1] == []
            assert [k for k, c in collections.Counter(envs).items() if c > 1] == []
            assert len(keys) == len(FL.FLAGS)
            return
    pytest.fail("FLAGS not found")


KS = [(3,), (2, 3, 5), (4, 6)]
CFGS = [(32, 2, 32, 64), (16, 4, 32, 64), (32, 1, 32, 32), (32, 2, 16, 64)]


@pytest.mark.parametrize("Ks,cfg", list(itertools.product(KS, CFGS)))
def test_rowend_and_unroll_compose(Ks, cfg):
    """cmoe8k (rowend, persist gate / up) and cpftrellis (unroll) each added a trailing parameter of coop_source; the merge keeps both. Every combination builds,
    each lever's own text appears exactly when asked, the default text does not move, and braces balance."""
    bmu, nmb, bns, bks = cfg
    build = lambda persist, rowend, unroll, pipe=False: NX.coop_source(Ks, "mul1r2", bmu, nmb, bns, bks, True, "full", 2, True, "", False, persist, pipe, 0, rowend, unroll)
    base = NX.coop_source(Ks, "mul1r2", bmu, nmb, bns, bks, True, "full", 2, True, "", False, True)
    assert build(True, False, False) == base                       # rowend / unroll off: the text of the call without them
    assert "const uint zo" not in base and "wb[" not in base
    seen = set()
    for persist, rowend, unroll in itertools.product([False, True], [False, True], [False, True]):
        if rowend and not persist:
            continue                                               # rowend is a persist-only form
        src = build(persist, rowend, unroll)
        seen.add(src)
        assert src.count("{") == src.count("}")
        assert ("const uint zo" in src) == rowend
        assert ("wb[" in src) == unroll, (persist, rowend, unroll)
        assert ("for (uint mg = threadgroup_position_in_grid.y" in src) == persist
    assert len(seen) == 6                                          # (persist, rowend, unroll): 6 distinct sources


def test_rowend_needs_two_projections():
    with pytest.raises(AssertionError):
        NX.coop_source((3,), "mul1r2", 32, 2, 32, 64, True, "full", 1, True, "", False, True, False, 0, True, False)


def test_default_pick_is_unchanged():
    assert NX.UNROLL2 is False and NX.BMU16 is False
