import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx.serve.kltap import Tap  # noqa: E402


def test_ref_then_cmp_identical_logits_give_zero_kl_and_full_agreement():
    mx.random.seed(0)
    steps = [mx.random.normal((1, 50)) for _ in range(6)]
    ref = Tap("ref")
    for lg in steps:
        ref(lg)
    cmp_ = Tap("cmp", ref)
    for lg in steps:
        forced = cmp_(lg)
    r = cmp_.result()
    assert r["steps"] == 6 and r["kl_mean"] < 1e-6 and r["top1"] == 1.0
    assert int(forced.item()) == int(ref.toks[-1].item())


def test_cmp_detects_a_perturbed_step():
    mx.random.seed(1)
    steps = [mx.random.normal((1, 50)) for _ in range(4)]
    ref = Tap("ref")
    for lg in steps:
        ref(lg)
    cmp_ = Tap("cmp", ref)
    for lg in steps:
        cmp_(lg + 0.5 * mx.random.normal(lg.shape))
    r = cmp_.result()
    assert r["kl_mean"] > 1e-3 and r["kl_max"] >= r["kl_mean"]
