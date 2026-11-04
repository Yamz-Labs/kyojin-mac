# SPDX-License-Identifier: Apache-2.0
"""Conformance checks a model must pass before any speed lever is looked at: load, prefill, decode, KV snapshot and restore, each against a slow
reference. CPU, any MLX backend, no Metal. The checks take a loaded model (`lm`, `make_cache`, the family's fork) and never name an architecture:
a new model passes them by being described (modeldesc row, ArchSpec), not by a branch here.

Reference: "slow" = no cache is reused. The logits of the last position of a sequence come from ONE forward over the whole sequence on a fresh cache;
decode step i is compared with that for the sequence so far. What is compared with what:

    prefill_vs_ref     chunked prefill (chunks of c tokens) vs the one-shot forward                       tolerance TOL (fp32 summation order differs)
    decode_vs_ref      every greedy decode step vs the one-shot forward of the sequence so far, and the argmax at every step     TOL + exact argmax
    snapshot           fork the cache, decode the same tokens on the original and on two forks, in turn             BITWISE equal
    disk               KvDisk.save at a block boundary, lookup, prefill the rest, decode                             BITWISE equal to the in-memory run
    ced                (CED layouts only) CED prefill with the receptive-field tail vs the exact prefill             tolerance CED_TOL; the default tail is reported, not asserted

Each check returns its numbers (max abs logit difference); the caller asserts, or prints them for a report."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

import mlx.core as mx

TOL = 2e-3                     # fp32 logits, hidden 64: chunk-order summation differences are ~1e-6..1e-5 (see docs/reports/cgeneric_REPORT.md for the measured values)
CED_TOL = 5e-3


def seq(n: int, vocab: int, seed: int = 1) -> list:
    import numpy as np
    return [int(x) for x in np.random.default_rng(seed).integers(3, vocab - 4, n)]


def last_row(out) -> mx.array:
    """[V] float32 logits of the last position of a model output (mlx-vlm returns an object with .logits, mlx-lm the array)."""
    lg = getattr(out, "logits", out)
    return lg[0, -1].astype(mx.float32)


def forward(lm, make_cache, ids: list, cache=None):
    cache = make_cache() if cache is None else cache
    lg = last_row(lm(mx.array([ids]), cache=cache))
    mx.eval(lg)
    return cache, lg


def reference_last(lm, make_cache, ids: list) -> mx.array:
    """The slow reference: one forward over the whole sequence on a fresh cache."""
    return forward(lm, make_cache, ids)[1]


def maxdiff(a, b) -> float:
    return float(mx.abs(a - b).max().item())


def bitwise(a, b) -> bool:
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def prefill(lm, make_cache, ids: list, chunk: int, cache=None):
    cache = make_cache() if cache is None else cache
    lg, s = None, 0
    while s < len(ids):
        lg = last_row(lm(mx.array([ids[s:s + chunk]]), cache=cache))
        mx.eval(lg)
        s += chunk
    return cache, lg


def decode(lm, cache, first: int, steps: int, forced: list | None = None):
    """Greedy decode (or teacher-forced on `forced`): [(token fed, logits after it)] for `steps` steps, starting by feeding `first`."""
    out, tok = [], first
    for i in range(steps):
        lg = last_row(lm(mx.array([[tok]]), cache=cache))
        mx.eval(lg)
        out.append((tok, lg))
        tok = forced[i] if forced is not None else int(mx.argmax(lg).item())
    return out


# ------------------------------------------------------------------------------------------------------------------ checks
def check_prefill_decode(lm, make_cache, vocab: int, n: int = 37, chunk: int = 8, steps: int = 6) -> dict:
    ids = seq(n, vocab)
    ref = reference_last(lm, make_cache, ids)
    cache, lg = prefill(lm, make_cache, ids, chunk)
    res = {"prefill_vs_ref": maxdiff(lg, ref), "prefill_argmax_equal": int(mx.argmax(lg).item()) == int(mx.argmax(ref).item())}
    toks, d_max, argmax_ok = list(ids), 0.0, True
    tok = int(mx.argmax(lg).item())
    for _ in range(steps):
        toks.append(tok)
        step = decode(lm, cache, tok, 1)[0][1]
        r = reference_last(lm, make_cache, toks)
        d_max = max(d_max, maxdiff(step, r))
        argmax_ok &= int(mx.argmax(step).item()) == int(mx.argmax(r).item())
        tok = int(mx.argmax(step).item())
    res.update(decode_vs_ref=d_max, decode_argmax_equal=bool(argmax_ok))
    return res


def check_snapshot(lm, make_cache, fork, vocab: int, n: int = 37, chunk: int = 8, steps: int = 5) -> dict:
    """A fork taken after the prefill continues bit for bit like the cache it came from, in either order, and the original is untouched by the forks."""
    ids = seq(n, vocab, seed=2)
    cache, lg = prefill(lm, make_cache, ids, chunk)
    first = int(mx.argmax(lg).item())
    s1, s2 = fork(cache), fork(cache)
    a = decode(lm, cache, first, steps)
    forced = [t for t, _ in a[1:]] + [0]
    b = decode(lm, s1, first, steps, forced)
    c = decode(lm, s2, first, steps, forced)
    return {"snapshot_fork1_bitwise": all(bitwise(x[1], y[1]) for x, y in zip(a, b)), "snapshot_fork2_bitwise": all(bitwise(x[1], y[1]) for x, y in zip(a, c)),
            "snapshot_tokens_equal": [t for t, _ in a] == [t for t, _ in b] == [t for t, _ in c]}


def check_disk(lm, make_cache, vocab: int, namespace: dict, n: int = 40, block: int = 16, steps: int = 4, root: str | None = None) -> dict:
    """KvDisk round trip: save at the block boundary, look the prefix up, prefill the remainder, decode; bitwise the in-memory run."""
    from . import kvdisk as K
    ids = seq(n, vocab, seed=3)
    own = root is None
    root = root or tempfile.mkdtemp(prefix="kyojin_conf_")
    try:
        return _disk(lm, make_cache, vocab, namespace, n, block, steps, root, ids)
    finally:
        if own:
            shutil.rmtree(root, ignore_errors=True)


def _disk(lm, make_cache, vocab, namespace, n, block, steps, root, ids) -> dict:
    from . import kvdisk as K
    kv = K.KvDisk(root, namespace, make_cache, block=block)
    cache, _ = prefill(lm, make_cache, ids[:block], block)
    wrote = kv.save(ids, block, cache)
    cache, lg = prefill(lm, make_cache, ids[block:], block, cache)
    mem = decode(lm, cache, int(mx.argmax(lg).item()), steps)
    b, got, _ = kv.lookup(ids)
    if got is None:
        return {"disk_saved_bytes": wrote, "disk_hit_at": 0, "disk_bitwise": False}
    got, lg2 = prefill(lm, make_cache, ids[b:], block, got)
    disk = decode(lm, got, int(mx.argmax(lg2).item()), steps)
    return {"disk_saved_bytes": int(wrote), "disk_hit_at": int(b), "disk_bitwise": bool(bitwise(lg, lg2) and all(bitwise(x[1], y[1]) for x, y in zip(mem, disk)))}


def check_ced(model, lm, spec, n: int = 100, vocab: int = 96) -> dict:
    """CED prefill vs the exact one on a model whose layout supports it. tail = receptive_rows makes the last-row logits exact in exact arithmetic."""
    from . import ced as C
    ids = seq(n, vocab, seed=4)
    mk = lm.make_cache
    exact = forward(lm, mk, ids)[1]
    out = {"ced_receptive_rows": C.receptive_rows(C.layout(lm.config))}
    C.enable(model, spec, tail="exact")
    st = C.state(model)
    cut = forward(lm, mk, ids)[1]
    out["ced_exact_tail_diff"] = maxdiff(cut, exact)
    out["ced_exact_tail_rows_cut"] = int(st.stats.get("rows_cut", 0))
    C.enable(model, spec, tail=None)
    approx = forward(lm, mk, ids)[1]
    out["ced_default_tail_diff"] = maxdiff(approx, exact)               # the approximation CED makes on purpose (reported, not asserted)
    out["ced_default_tail_argmax_equal"] = int(mx.argmax(approx).item()) == int(mx.argmax(exact).item())
    return out


@dataclass
class Result:
    shape: str
    numbers: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


def run_shape(shape, workdir: str | None = None) -> Result:
    """Everything for one `synth.Shape`: write the pack, check it against its descriptor, load it through the loader, run the checks above."""
    from . import archs, packcheck, synth
    res = Result(shape.name)
    fail = lambda m: res.failures.append(m)                            # noqa: E731
    own = workdir is None
    d = workdir or tempfile.mkdtemp(prefix=f"kyojin_{shape.name}_")
    try:
        return _run_shape(shape, d, res, fail)
    finally:
        if own:
            shutil.rmtree(d, ignore_errors=True)


def _run_shape(shape, d, res, fail) -> Result:
    from . import archs, packcheck, synth
    ref_model, ref_lm, ref_mk = synth.write_pack(shape, d)
    ced_asked = shape.ced
    findings = packcheck.check(d, ced=ced_asked)
    res.numbers["packcheck_findings"] = len(findings)
    if findings:
        fail("packcheck: " + "; ".join(map(str, findings)))
    model, lm, mk = synth.load_pack(shape, d)
    vocab = int(shape.text["vocab_size"])
    probe = seq(9, vocab, seed=5)
    res.numbers["load_vs_source_diff"] = maxdiff(forward(lm, mk, probe)[1], forward(ref_lm, ref_mk, probe)[1])
    if res.numbers["load_vs_source_diff"] != 0.0:
        fail(f"loaded model differs from the model the pack was written from: {res.numbers['load_vs_source_diff']}")
    r = check_prefill_decode(lm, mk, vocab)
    res.numbers.update(r)
    if r["prefill_vs_ref"] > TOL or r["decode_vs_ref"] > TOL or not r["decode_argmax_equal"] or not r["prefill_argmax_equal"]:
        fail(f"prefill/decode vs reference: {r}")
    fork = _fork_of(shape)
    s = check_snapshot(lm, mk, fork, vocab)
    res.numbers.update(s)
    if not all(s.values()):
        fail(f"snapshot: {s}")
    ns = {"arch": shape.name, "pack": os.path.basename(d)}
    k = check_disk(lm, mk, vocab, ns)
    res.numbers.update(k)
    if not (k["disk_hit_at"] > 0 and k["disk_bitwise"]):
        fail(f"disk: {k}")
    if shape.family == "ds":
        from . import ced as C
        why = C.explain(lm.config)
        res.numbers["ced_supported"] = why is None
        if (why is None) != shape.ced:
            fail(f"shape says ced={shape.ced} but ced.explain says {why!r}")
        if why is None:
            c = check_ced(model, lm, archs.ARCHS[archs.arch_of(shape.config).name], vocab=vocab)
            res.numbers.update(c)
            if c["ced_exact_tail_diff"] > CED_TOL:
                fail(f"ced exact tail: {c}")
    return res


def _fork_of(shape):
    if shape.family == "ds":
        from . import rows
        return rows.fork
    from . import multiuser
    return multiuser.fork_cache
