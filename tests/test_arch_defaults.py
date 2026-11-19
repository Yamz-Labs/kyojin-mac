"""Op-named flags, their model-named aliases and ArchSpec.defaults (kyojin_mlx.opflags). Pure python: no mlx, no GPU.

Core assertion: the GLM-5.3 lane's flags.env (snapshot tests/data/glm_lane_flags.env, optionally the live file named by KYOJIN_LANE_FLAGS_ENV) resolves to the same
flag dict as an EMPTY environment plus GLM5.defaults, except KYOJIN_GLM_DENSE_Q, which stays env-only."""
import os

import pytest

from kyojin_mlx import archs, opflags as OF

SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "glm_lane_flags.env")
ENV_ONLY = {"KYOJIN_GLM_DENSE_Q"}                       # second resident copy of the dense weights: never an arch default (owner rule 16/11)


def read_env_file(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def lane_files():
    files = [SNAPSHOT]
    live = os.environ.get("KYOJIN_LANE_FLAGS_ENV", "")
    if live and os.path.isfile(live):
        files.append(live)
    return files


@pytest.mark.parametrize("path", lane_files())
def test_empty_env_with_glm_defaults_reproduces_the_lane(path):
    lane = read_env_file(path)
    assert "KYOJIN_GLM_DENSE_Q" in lane or path != SNAPSHOT
    with_file = OF.resolve({k: v for k, v in lane.items() if k not in ENV_ONLY})
    only_defaults = OF.resolve({}, archs.GLM5.defaults)
    assert with_file == only_defaults
    assert OF.canonical("KYOJIN_GLM_DENSE_Q") not in only_defaults                 # env-only, never defaulted
    assert not (set(OF.spellings("KYOJIN_GLM_DENSE_Q")) & set(archs.GLM5.defaults))


def test_defaults_cover_exactly_the_served_set():
    lane = read_env_file(SNAPSHOT)
    served = {OF.canonical(k) for k in lane if k not in ENV_ONLY}
    assert {OF.canonical(k) for k in archs.GLM5.defaults} == served and len(served) == len(archs.GLM5.defaults)


def test_default_keys_are_op_names_not_aliases():
    for spec in archs.ARCHS.values():
        assert OF.validate_defaults(spec.defaults) == []
    assert OF.validate_defaults({"KYOJIN_GLM_VROWS": "1"}) and OF.validate_defaults({"X": "1"}) and OF.validate_defaults({"KYOJIN_A": 1})


def test_other_architectures_default_nothing():
    assert archs.DS41.defaults == {}
    assert OF.apply_arch_defaults(archs.DS41, environ={}, log=None) == {}


def test_every_glm_named_lever_of_the_served_set_that_has_an_op_name_keeps_its_alias():
    for old, new in {"KYOJIN_GLM_VROWS": "KYOJIN_VERIFY_ROWS_GEMV", "KYOJIN_GLM_HOSTGAP": "KYOJIN_HOST_EARLY_SUBMIT", "KYOJIN_GLM_DENSE_GEMV": "KYOJIN_DENSE_GEMV_1B",
                     "KYOJIN_GLM_PF_UNTILE": "KYOJIN_PF_UNTILE", "KYOJIN_GLM_MOE_E": "KYOJIN_MOE_EXPERT_E", "KYOJIN_GLM_FUSE_DSA": "KYOJIN_DSA_FUSE",
                     "KYOJIN_GLM_KDA_PF": "KYOJIN_KDA_PF", "KYOJIN_KQ_GMM_GLM": "KYOJIN_KQ_GMM_STORED", "KYOJIN_GLM_HOSTGAP_ROWS": "KYOJIN_HOST_EARLY_SUBMIT_ROWS"}.items():
        assert OF.canonical(old) == new and old in OF.spellings(new) and OF.spellings(old)[0] == new


def test_names_are_unique_and_op_names_are_not_model_named():
    seen = []
    for op in OF.OPS.values():
        seen += [op.env, *op.aliases]
        assert "GLM" not in op.env and op.env.startswith("KYOJIN_")
    assert len(seen) == len(set(seen))


def test_sync_mirrors_alias_to_op_and_op_to_alias():
    e = {"KYOJIN_GLM_VROWS": "1"}
    OF.sync_env(e)
    assert e["KYOJIN_VERIFY_ROWS_GEMV"] == "1"
    e = {"KYOJIN_HOST_EARLY_SUBMIT": "1", "KYOJIN_HOST_EARLY_SUBMIT_ROWS": "3"}
    OF.sync_env(e)
    assert e["KYOJIN_GLM_HOSTGAP"] == "1" and e["KYOJIN_GLM_HOSTGAP_ROWS"] == "3"


def test_op_name_wins_a_conflict_and_it_is_logged():
    e, lines = {"KYOJIN_GLM_VROWS": "0", "KYOJIN_VERIFY_ROWS_GEMV": "1"}, []
    OF.sync_env(e, lines.append)
    assert e["KYOJIN_GLM_VROWS"] == "1" and len(lines) == 1 and "op name wins" in lines[0]


def test_apply_defaults_only_where_env_is_unset_and_logs_them():
    lines, e = [], {"KYOJIN_ARCH_DEFAULTS": "1", "KYOJIN_GLM_HOSTGAP": "0", "KYOJIN_GLM_SPEC_ROWS": "2"}                  # the user set one lever by its old name, one by a plain name
    got = OF.apply_arch_defaults(archs.GLM5, environ=e, log=lines.append)
    assert "KYOJIN_HOST_EARLY_SUBMIT" not in got and "KYOJIN_GLM_SPEC_ROWS" not in got        # kept: env wins, whatever the spelling
    assert e["KYOJIN_GLM_HOSTGAP"] == "0" and e["KYOJIN_HOST_EARLY_SUBMIT"] == "0" and e["KYOJIN_GLM_SPEC_ROWS"] == "2"
    assert e["KYOJIN_VERIFY_ROWS_GEMV"] == "1" and e["KYOJIN_GLM_VROWS"] == "1"               # defaulted on BOTH spellings
    assert len(got) == len(archs.GLM5.defaults) - 2
    assert lines[0].startswith("[kyojin] arch glm5_next defaults (env unset): ")
    assert all("WARNING arch defaults reached modules already imported" in l for l in lines[1:])   # earlier tests may have imported lever modules
    assert all(f"{k}={v}" in lines[0] for k, v in got.items()) and "kept from env:" in lines[0]
    assert "DENSE_Q" not in e                                                                  # env-only


def test_a_value_set_by_apply_defaults_is_not_a_user_choice(monkeypatch):
    monkeypatch.setattr(OF, "_FILLED", {"KYOJIN_GLM_HC_FUSE": "0"})
    monkeypatch.setenv("KYOJIN_GLM_HC_FUSE", "0")                                               # filled by flags.apply_defaults at import
    monkeypatch.setenv("KYOJIN_GLM_MOE", "0")                                                   # really set by the user
    monkeypatch.setenv("KYOJIN_ARCH_DEFAULTS", "1")
    assert not OF.user_set("KYOJIN_GLM_HC_FUSE") and OF.user_set("KYOJIN_GLM_MOE")
    lines = []
    got = OF.apply_arch_defaults(archs.GLM5, log=lines.append)
    try:
        assert got["KYOJIN_GLM_HC_FUSE"] == "1" and "KYOJIN_GLM_MOE" not in got and os.environ["KYOJIN_GLM_MOE"] == "0"
    finally:
        for sp in {s for k in archs.GLM5.defaults for s in OF.spellings(k)}:
            os.environ.pop(sp, None)                                                            # monkeypatch only restores the two it knows


def test_flags_registry_set_writes_every_spelling(monkeypatch):
    from kyojin_mlx import flags
    monkeypatch.setenv("KYOJIN_GLM_VROWS", "0")
    monkeypatch.setenv("KYOJIN_VERIFY_ROWS_GEMV", "0")
    flags.set("glm_vrows", 1)
    assert os.environ["KYOJIN_GLM_VROWS"] == "1" and os.environ["KYOJIN_VERIFY_ROWS_GEMV"] == "1"


def test_package_import_mirrors_an_op_name_set_in_the_environment():
    import subprocess, sys
    env = dict(os.environ, KYOJIN_HOST_EARLY_SUBMIT="1", PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    for k in ("KYOJIN_GLM_HOSTGAP",):
        env.pop(k, None)
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", "import os, kyojin_mlx; print(os.environ.get('KYOJIN_GLM_HOSTGAP'))"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "1", out.stderr[-400:]


def test_load_backend_applies_the_arch_defaults_for_a_glm_row():
    src = open(os.path.join(os.path.dirname(__file__), "..", "kyojin_mlx", "serve", "backends.py")).read()
    assert "opflags.apply_arch_defaults(_archs.ARCHS[row.arch]" in src
    from kyojin_mlx import modeldesc
    assert modeldesc.backend_desc("glm5_next").arch == "glm5_next" and "glm5_next" in archs.ARCHS


def test_master_switch_off_by_default_applies_nothing():
    e = {"KYOJIN_GLM_HOSTGAP": "0"}
    assert OF.apply_arch_defaults(archs.GLM5, environ=e, log=None) == {} and e == {"KYOJIN_GLM_HOSTGAP": "0"}
