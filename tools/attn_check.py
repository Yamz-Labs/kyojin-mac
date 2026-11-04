"""End-to-end check of a decode cut against the path without it on a real pack: same cache state, both paths, logits and tokens.
usage: attn_check.py PACK [--prompt 150] [--steps 40] [--seeds 2] [--switch fuse_attn|fuse_layer] [--noise]   (Mac, mlx)
  --switch   the module whose ENABLED attribute turns the cut on (fuse_attn = attention kernels vs the stock composite, fuse_layer = compiled regions vs eager)
  --noise    reference noise: the stock attention against itself with the keys of the sparse attention in reverse order (equal maths, other rounding)
  teacher forced  both paths decode from forks of one cache and both are fed the STOCK argmax: per step rel L2 of the logits, argmax agreement
  free run        each path follows its own argmax: step of the first token difference
The switch is a module attribute of kyojin_mlx (ENABLED), flipped in one process (the model is the same object)."""
import argparse, importlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--prompt", type=int, default=150); ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--switch", default="fuse_attn")
ap.add_argument("--noise", action="store_true", help="instead of the cut, compare the stock path with itself under a reordering of the keys of the sparse attention (equal maths, other rounding): the noise floor of the model")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
sw = importlib.import_module("kyojin_mlx." + a.switch)
V = cfg["text_config"]["vocab_size"]


def decode(cache, tok):
    return R.step(model, cache, mx.array([tok]))[0]


if a.noise:
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    _orig = L._sparse_pooled_attention

    def _reordered(q, local_kv, pooled, topk, local_mask, pooled_mask, scale, sinks):
        return _orig(q, local_kv[..., ::-1, :], pooled, topk[..., ::-1], local_mask[..., ::-1], pooled_mask[..., ::-1], scale, sinks)

    class _Sw:
        ENABLED = False
        def __setattr__(self, k, on):
            if k == "ENABLED":
                L._sparse_pooled_attention = _reordered if on else _orig
    importlib.import_module("kyojin_mlx.fuse_attn").ENABLED = False             # both runs on the stock code
    sw = _Sw()
    sw.ENABLED = False


def run(on, cache, tok):
    sw.ENABLED = on
    return decode(cache, tok)


tot = dict(rel=[], agree=[], first=[])
for seed in range(a.seeds):
    ids = np.random.default_rng(100 + seed).integers(16, V - 16, a.prompt).tolist()
    cache0, lg0 = R.prefill(model, ids)
    tok = int(mx.argmax(lg0[0]).item())
    cs, cf = R.fork(cache0), R.fork(cache0)
    rels, agree = [], 0
    ts = tok
    for t in range(a.steps):
        ls = run(False, cs, ts); lf = run(True, cf, ts); mx.eval(ls, lf)
        ls_, lf_ = np.array(ls), np.array(lf)
        rels.append(float(np.linalg.norm(lf_ - ls_) / np.linalg.norm(ls_)))
        agree += int(lf_.argmax() == ls_.argmax())
        ts = int(ls_.argmax())
    # free run
    fs, ff = R.fork(cache0), R.fork(cache0)
    a_tok = b_tok = tok; seq_s, seq_f = [], []
    for t in range(a.steps):
        a_tok = int(run(False, fs, a_tok).argmax().item()); b_tok = int(run(True, ff, b_tok).argmax().item())
        seq_s.append(a_tok); seq_f.append(b_tok)
    first = next((i for i, (x, y) in enumerate(zip(seq_s, seq_f)) if x != y), None)
    print(f"seed {seed}: teacher-forced rel L2 logits mean {np.mean(rels):.2e} max {np.max(rels):.2e}, argmax agree {agree}/{a.steps}; free run first difference: {first}", flush=True)
    tot["rel"] += rels; tot["agree"].append(agree); tot["first"].append(first)
sw.ENABLED = True
print(f"TOTAL rel L2 mean {np.mean(tot['rel']):.2e} max {np.max(tot['rel']):.2e}; agree {sum(tot['agree'])}/{a.seeds * a.steps}; first diffs {tot['first']}")
