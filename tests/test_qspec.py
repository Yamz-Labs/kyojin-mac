"""Qwen3.5 MTP speculative decode: greedy output equals plain decode, verify rows equal solo steps bit for bit.
Needs a Qwen3.5 EXL3 pack and the bf16 checkpoint (MTP head): KYOJIN_QSPEC_PACK, KYOJIN_QSPEC_BF16."""
import os
import pytest
import mlx.core as mx

PACK, BF16 = os.environ.get("KYOJIN_QSPEC_PACK"), os.environ.get("KYOJIN_QSPEC_BF16")
pytestmark = pytest.mark.skipif(not (PACK and BF16), reason="KYOJIN_QSPEC_PACK / KYOJIN_QSPEC_BF16 not set")


@pytest.fixture(scope="module")
def setup():
    from kyojin_mlx import exl3_linear as EL, kernels as KN, qspec as QS
    EL.ROW_SPLIT_MAX = 16
    KN.MAX_ROWS = 8
    EL.GEMV_MAX_ROWS = 8
    model, _ = EL.load_exl3(PACK)
    QS.row_linears(model)
    mtp = QS.load_mtp(model, BF16, quant_bits=4)
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(PACK).encode("Explain how a hash map handles collisions and compare chaining with open addressing.")
    return QS, model, mtp, ids


@pytest.mark.parametrize("depth", [1, 3, 7])
def test_spec_equals_plain(setup, depth):
    QS, model, mtp, ids = setup
    QS.uninstall_capture()
    ref = QS.plain_generate(model, ids, 40)
    sp = QS.QSpec(model, mtp, depth=depth)
    sp.prefill(ids)
    assert sp.generate(40) == ref


def test_rows_bitwise(setup):
    from mlx_lm.models.cache import make_prompt_cache
    QS, model, mtp, ids = setup
    lm = model.language_model

    def fresh():
        c = make_prompt_cache(model)
        lm.model(mx.array(ids)[None], c)
        return c
    toks = QS.plain_generate(model, ids, 9)
    c = fresh()
    solo = []
    for t in toks[:8]:
        solo.append(lm.lm_head(lm.model(mx.array([t])[None], c))[0, 0])
        mx.eval(solo[-1])
    c = fresh()
    lg = lm.lm_head(lm.model(mx.array(toks[:8])[None], c))
    mx.eval(lg)
    assert all(bool(mx.array_equal(lg[0, j], solo[j])) for j in range(8))
