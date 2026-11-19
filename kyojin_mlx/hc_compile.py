"""KYOJIN_HC_COMPILE: mx.compile of the gated hyper-connection module's whole __call__ at decode (small row counts).

Keyed on structure, not on a model name: a module with hc_norm + input_mix_weight_down / input_mix_weight_up (+ optional block_inject_weight) and hc_count / hidden_size
(mlx-vlm qwen4_exp `Qwen4ExpGatedResidual`; its pointwise parts are already compiled one by one, so a step builds ~7 graph nodes per call on the host).
One compiled function per module (weights passed as compile inputs through module.state): the host builds one cached call instead of norm + 3 linears + 3 pointwise.
Same primitives in the same order, so the result is expected bit-identical: proven by md5 on greedy tokens (cell c31b), never assumed. Rows above MAX_ROWS take the stock call.
Not the GLM mHC (sinkhorn, glm_hc.py): different math. Default off. Read at load."""
import os
import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_HC_COMPILE", "0") not in ("0", "")
ACTIVE = ENABLED                                   # runtime toggle of the installed wrapper (A/B cells); install itself is at load
MAX_ROWS = int(os.environ.get("KYOJIN_HC_COMPILE_ROWS", "16"))


def is_hc(m) -> bool:
    return all(hasattr(m, a) for a in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "hc_count", "hidden_size")) and not hasattr(m, "layers")


def _wrap(mod):
    cls = mod.__class__
    stock = cls.__call__
    comp = mx.compile(lambda x: stock(mod, x), inputs=mod.state, outputs=mod.state)

    def call(self, x):
        rows = 1
        for d in x.shape[:-1]:
            rows *= d
        return comp(x) if (ACTIVE and rows <= MAX_ROWS) else stock(self, x)

    mod.__class__ = type("Hcc" + cls.__name__, (cls,), {"__call__": call, "_hc_compiled": True})


def install(model, log=print, force=False) -> int:
    global ACTIVE
    if not (ENABLED or force):
        return 0
    ACTIVE = True
    n = 0
    for _, m in model.named_modules():
        if is_hc(m) and not getattr(m.__class__, "_hc_compiled", False):
            _wrap(m)
            n += 1
    if n:
        log(f"[foreign] hc_compile: {n} hyper-connection modules compiled (rows <= {MAX_ROWS})")
    return n
