"""kyojin_mlx.applegpu: eval-thread QoS and the residency refresh env. No Metal, no mlx: a fake libSystem records the calls."""
import ctypes
import sys
import threading

import pytest

from kyojin_mlx import applegpu as AG
from kyojin_mlx import flags as FL


class FakeLibc:
    def __init__(self, qos=AG.QOS_USER_INITIATED, set_rc=0, get_rc=0):
        self.qos, self.set_rc, self.get_rc, self.calls = qos, set_rc, get_rc, []

    def pthread_self(self):
        return 1234

    def pthread_get_qos_class_np(self, th, cls, prio):
        self.calls.append(("get", th))
        cls.contents.value = self.qos
        return self.get_rc

    def pthread_set_qos_class_self_np(self, qos, prio):
        self.calls.append(("set", qos, prio))
        if self.set_rc == 0:
            self.qos = qos
        return self.set_rc


@pytest.fixture(autouse=True)
def fresh_state():
    AG._state.clear()
    yield
    AG._state.clear()


def sets(lib):
    return [c for c in lib.calls if c[0] == "set"]


def test_flag_off_makes_no_call():
    lib = FakeLibc()
    assert AG.sync_qos({"KYOJIN_EVAL_QOS": "0"}, lib, "darwin") is None
    assert AG.sync_qos({}, lib, "darwin") is None
    assert lib.calls == []


def test_flag_on_sets_user_interactive_once_and_logs_before_after():
    lib, env = FakeLibc(), {"KYOJIN_EVAL_QOS": "1"}
    msg = AG.sync_qos(env, lib, "darwin")
    assert sets(lib) == [("set", 0x21, 0)]
    assert "USER_INITIATED -> USER_INTERACTIVE" in msg
    n = len(lib.calls)
    assert AG.sync_qos(env, lib, "darwin") is None and len(lib.calls) == n      # a second tick costs no call


def test_switching_off_restores_the_original_class():
    lib, env = FakeLibc(qos=AG.QOS_DEFAULT), {"KYOJIN_EVAL_QOS": "1"}
    AG.sync_qos(env, lib, "darwin")
    env["KYOJIN_EVAL_QOS"] = "0"
    msg = AG.sync_qos(env, lib, "darwin")
    assert sets(lib) == [("set", 0x21, 0), ("set", AG.QOS_DEFAULT, 0)] and lib.qos == AG.QOS_DEFAULT
    assert "USER_INTERACTIVE -> DEFAULT" in msg


def test_already_user_interactive_still_reports_it():
    lib = FakeLibc(qos=AG.QOS_USER_INTERACTIVE)
    msg = AG.sync_qos({"KYOJIN_EVAL_QOS": "1"}, lib, "darwin")
    assert "USER_INTERACTIVE -> USER_INTERACTIVE" in msg


def test_non_darwin_is_a_no_op(monkeypatch):
    lib, env = FakeLibc(), {"KYOJIN_EVAL_QOS": "1"}
    monkeypatch.setattr(AG, "_libc", lambda: pytest.fail("libSystem must not load off macOS"))
    assert "ignored" in AG.sync_qos(env, None, "linux")
    assert AG.sync_qos(env, None, "linux") is None                              # said once
    env["KYOJIN_EVAL_QOS"] = "0"
    assert AG.sync_qos(env, None, "linux") is None and lib.calls == []


def test_failure_is_logged_once_and_not_retried():
    lib, env = FakeLibc(set_rc=1), {"KYOJIN_EVAL_QOS": "1"}
    assert "failed" in AG.sync_qos(env, lib, "darwin")
    n = len(lib.calls)
    assert AG.sync_qos(env, lib, "darwin") is None and len(lib.calls) == n


def test_state_is_per_thread():
    lib, env, out = FakeLibc(), {"KYOJIN_EVAL_QOS": "1"}, []
    AG.sync_qos(env, lib, "darwin")
    t = threading.Thread(target=lambda: out.append(AG.sync_qos(env, lib, "darwin")))
    t.start(); t.join()
    assert out[0] is not None and len(sets(lib)) == 2                           # the other thread was raised on its own


def test_get_qos_reads_through_pointers():
    assert AG.get_qos(FakeLibc(qos=0x19)) == 0x19
    with pytest.raises(OSError):
        AG.get_qos(FakeLibc(get_rc=22))


@pytest.mark.skipif(sys.platform != "darwin", reason="real libSystem")
def test_real_libsystem_roundtrip():
    lib = AG._libc()
    before = AG.get_qos(lib)
    AG.set_qos(lib, AG.QOS_USER_INTERACTIVE)
    assert AG.get_qos(lib) == AG.QOS_USER_INTERACTIVE
    AG.set_qos(lib, before)


def test_refresh_env():
    env = {}
    assert AG.apply_residency_refresh_env(env) and env["MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"] == "1000"
    env = {"KYOJIN_RESIDENCY_REFRESH_MS": "0"}
    assert not AG.apply_residency_refresh_env(env) and "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS" not in env
    env = {"KYOJIN_RESIDENCY_REFRESH_MS": "250"}
    assert AG.apply_residency_refresh_env(env) and env["MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"] == "250"
    env = {"KYOJIN_RESIDENCY_REFRESH_MS": "250", "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS": "50"}
    AG.apply_residency_refresh_env(env)
    assert env["MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"] == "50"               # an explicit MLX value wins
    assert not AG.apply_residency_refresh_env({"KYOJIN_RESIDENCY_REFRESH_MS": "x"})


def test_flag_table_entries():
    q, r = FL.FLAGS["eval_qos"], FL.FLAGS["residency_refresh_ms"]
    assert (q.env, q.default, q.runtime, q.kind) == (AG.QOS_ENV, "0", True, "bool")
    assert (r.env, r.default, r.runtime, r.kind) == (AG.REFRESH_ENV, AG.REFRESH_DEFAULT, False, "int") and r.reason
    with pytest.raises(PermissionError):
        FL.set("residency_refresh_ms", 100)
    old = FL.os.environ.get(AG.QOS_ENV)
    try:
        FL.set("eval_qos", 1)
        assert FL.value("eval_qos") == "1" and FL.os.environ[AG.QOS_ENV] == "1"
    finally:
        FL.os.environ.pop(AG.QOS_ENV, None) if old is None else FL.os.environ.__setitem__(AG.QOS_ENV, old)


def test_package_import_leaves_the_defaults(monkeypatch):
    monkeypatch.delenv("KYOJIN_EVAL_QOS", raising=False)
    monkeypatch.delenv("KYOJIN_RESIDENCY_REFRESH_MS", raising=False)
    FL.apply_defaults()
    assert FL.value("eval_qos") == "0" and FL.value("residency_refresh_ms") == "1000"
