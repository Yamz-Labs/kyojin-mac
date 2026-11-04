"""Fail-fast lever check: whole-model greedy decode of a DS4.1 / GLM pack with every Mac lever ON (defaults) vs every lever OFF (KYOJIN_* off-switches).
Decode of the OFF run is greedy; the ON run is teacher-forced on the OFF tokens, so the per-step KL(off || on) is meaningful even after a flipped argmax.
Gate: mean KL <= KL_MEAN and max KL <= KL_MAX (env DAY1_LEVER_KL_MEAN / DAY1_LEVER_KL_MAX). Mismatch: bisect, ONLY lever X on, the others off, and print the levers that break the gate.
usage: lever_check.py PACK [PACK ...] [--new 32] [--prompt 160]       rc 0 = gate passed on every pack, rc 1 = a lever broke it (named on stdout)"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

SWITCHES = {      # lever -> env that turns it off (feat/ultra-day1)
    "moe_prefill_nax": {"KYOJIN_PF_NAX": "0"},
    "ds_prefill_attn": {"KYOJIN_DS_PF": "0"},
    "ds_prefill_indexer": {"KYOJIN_DS_PF_IDX": "0"},
    "ds_prefill_mhc": {"KYOJIN_DS_PF_MHC": "0"},
    "dsa_decode_nax": {"KYOJIN_DSA_NAX": "0"},
    "dsa_decode_glm_indexer": {"KYOJIN_DSA_INDEX_GLM": "0"},
    "attend_mma": {"KYOJIN_ATTEND_MMA": "0"},
    "attend_split": {"KYOJIN_ATTEND_SPLIT": "0"},
    "gdn_fast": {"KYOJIN_GDN_FAST": "0"},
    "step_regions": {"KYOJIN_STEP_REGIONS": "0"},
}
OPTIN = {        # opt-in lever (not bit-identical, off by default) -> env that turns it on; checked by --optin (default vs default + lever)
    "moe_affold": {"KYOJIN_MOE_AFFOLD": "1"},
}
KL_MEAN = float(os.environ.get("DAY1_LEVER_KL_MEAN", "2e-3"))
KL_MAX = float(os.environ.get("DAY1_LEVER_KL_MAX", "2e-2"))
TOP1_MIN = float(os.environ.get("DAY1_LEVER_TOP1_MIN", "0.99"))


def worker(pack, out, new, plen, force, block=0):
    sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
    import mlx.core as mx
    from kyojin_mlx import patch as P
    if json.load(open(os.path.join(pack, "config.json"))).get("model_type") == "qwen3_5":      # Qwen3.5 EXL3 (rivals packs): no DSA patch, plain exl3 loader
        from kyojin_mlx import exl3_linear as EL, qspec as QS
        model, _ = EL.load_exl3(pack); _inner = model.language_model; vocab = 4096
        class _W:
            make_cache = staticmethod(lambda: QS.make_prompt_cache(model))
            def __call__(self, x, cache=None):
                class R: pass
                r = R(); r.logits = _inner.lm_head(_inner.model(x, cache)); return r
        lm = _W()
    else:
        model, spec, cfg = P.build_model(pack)
        P.patch_model(model, pack, spec, cfg)
        if os.environ.get("LEVER_ROWS") == "1":                      # the serving decode path: rows.enable installs the fused attention / compiled regions (plain patch_model decode does not)
            from kyojin_mlx import rows as R
            R.enable(model, spec)
            vocab = int(cfg["text_config"]["vocab_size"]); ids = np.random.default_rng(0).integers(100, min(vocab, 4000), plen).tolist()
            cache, lg = R.prefill(model, ids); lg = lg.astype(mx.float32).reshape(1, -1)
            toks, lps = [], []
            for i in range(new):
                lp = lg - mx.logsumexp(lg, -1, keepdims=True); mx.eval(lp); lps.append(np.array(lp[0]))
                t = force[i] if force is not None else int(mx.argmax(lp, -1).item()); toks.append(t)
                lg = R.step(model, cache, mx.array([t]))[0].astype(mx.float32).reshape(1, -1)
            np.savez(out, toks=np.array(toks), lps=np.stack(lps)); return
        lm = model.language_model
        vocab = int(getattr(cfg, "vocab_size", 0) or cfg.get("vocab_size", 4096) if isinstance(cfg, dict) else getattr(cfg, "vocab_size", 4096))
    ids = np.random.default_rng(0).integers(100, min(vocab, 4000), plen).tolist()
    cache = lm.make_cache()
    lg = lm(mx.array([ids]), cache=cache).logits[:, -1].astype(mx.float32)
    toks, lps = [], []
    if block:                                  # verify-like: teacher-forced blocks of `block` rows per forward (the rows MMA_ROWS..16 path), lps[i] = distribution before token i
        lps = [np.array((lg - mx.logsumexp(lg, -1, keepdims=True))[0])]
        for i in range(0, new - 1, block):
            blk = [int(t) for t in force[i:min(i + block, new - 1)]]
            lgb = lm(mx.array([blk]), cache=cache).logits[0].astype(mx.float32)
            lpb = lgb - mx.logsumexp(lgb, -1, keepdims=True); mx.eval(lpb)
            lps += [np.array(r) for r in lpb]
        np.savez(out, toks=np.array(force), lps=np.stack(lps)); return
    for i in range(new):
        lp = lg - mx.logsumexp(lg, -1, keepdims=True); mx.eval(lp)
        lps.append(np.array(lp[0]))
        t = force[i] if force is not None else int(mx.argmax(lp, -1).item())
        toks.append(t)
        lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
    np.savez(out, toks=np.array(toks), lps=np.stack(lps))


def run(pack, tag, env, new, plen, force=None, block=0):
    out = os.path.join(tempfile.mkdtemp(prefix="lvchk"), tag + ".npz")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", pack, out, "--new", str(new), "--prompt", str(plen)]
    if block:
        cmd += ["--block", str(block)]
    if force is not None:
        cmd += ["--force", json.dumps([int(x) for x in force])]
    r = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    if r.returncode:
        print(f"  {tag}: worker rc {r.returncode}\n{r.stderr[-1500:]}", flush=True)
        raise SystemExit(2)
    return np.load(out)


def kl(ref, got):
    p = np.exp(ref["lps"]); k = (p * (ref["lps"] - got["lps"])).sum(-1)
    return float(k.mean()), float(k.max())


def main():
    if "--worker" in sys.argv:
        ap = argparse.ArgumentParser(); ap.add_argument("--worker", action="store_true"); ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--new", type=int); ap.add_argument("--prompt", type=int); ap.add_argument("--force"); ap.add_argument("--block", type=int, default=0)
        a = ap.parse_args(); worker(a.pack, a.out, a.new, a.prompt, json.loads(a.force) if a.force else None, a.block); return
    ap = argparse.ArgumentParser(); ap.add_argument("packs", nargs="+"); ap.add_argument("--new", type=int, default=32); ap.add_argument("--prompt", type=int, default=160)
    ap.add_argument("--mma-rows", type=int, default=0, help="instead of the lever sweep: KYOJIN_MMA_ROWS=<n> vs 0 on teacher-forced 8-row blocks (Qwen EXL3 packs)")
    ap.add_argument("--block", type=int, default=8, help="rows per teacher-forced block of --mma-rows (5..8 MMA GEMV, 9..16 k-split NAX; -1 as --mma-rows = the defaults vs both levers off)")
    ap.add_argument("--solo", default="", help="comma list of levers: each one OFF vs default (all ON), isolates its own KL")
    ap.add_argument("--optin", default="", help="comma list of OPTIN levers: each one ON vs the defaults (teacher-forced), its own KL")
    ap.add_argument("--plain", action="store_true", help="decode through patch_model only (no rows.enable): the old behaviour")
    a = ap.parse_args()
    if not a.plain:
        os.environ["LEVER_ROWS"] = "1"
    if a.optin:
        rc = 0
        for pack in a.packs:
            name = os.path.basename(pack.rstrip("/"))
            ref = run(pack, "def", {}, a.new, a.prompt)
            for lever in a.optin.split(","):
                on = run(pack, lever, OPTIN[lever], a.new, a.prompt, force=ref["toks"])
                m, mx_ = kl(ref, on); t1 = float((ref["lps"].argmax(-1) == on["lps"].argmax(-1)).mean())
                ok = m <= KL_MEAN and mx_ <= KL_MAX and t1 >= TOP1_MIN; rc |= not ok
                print(f"{name}: opt-in {lever} ON vs defaults, prompt {a.prompt} +{a.new}: KL mean {m:.2e} max {mx_:.2e} top1 agree {t1:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(rc)
    if a.solo:
        rc = 0
        for pack in a.packs:
            name = os.path.basename(pack.rstrip("/"))
            for lever in a.solo.split(","):
                ref = run(pack, "off", SWITCHES[lever], a.new, a.prompt)
                on = run(pack, "on", {}, a.new, a.prompt, force=ref["toks"])
                m, mx_ = kl(ref, on); t1 = float((ref["lps"].argmax(-1) == on["lps"].argmax(-1)).mean())
                ok = m <= KL_MEAN and mx_ <= KL_MAX and t1 >= TOP1_MIN; rc |= not ok; verdict = "PASS" if ok else "FAIL"
                print(f"{name}: {lever} ON vs OFF (others ON), prompt {a.prompt} +{a.new}: KL mean {m:.2e} max {mx_:.2e} top1 agree {t1:.3f} -> {verdict}", flush=True)
        sys.exit(rc)
    if a.mma_rows:
        rc = 0
        for pack in a.packs:
            name = os.path.basename(pack.rstrip("/"))
            ref = run(pack, "seq", {}, a.new, a.prompt)
            off = run(pack, "blk0", {"KYOJIN_MMA_ROWS": "0", "KYOJIN_NAX_KS": "0"}, a.new, a.prompt, force=ref["toks"], block=a.block)
            on = run(pack, "blkmma", {"KYOJIN_MMA_ROWS": str(a.mma_rows)}, a.new, a.prompt, force=ref["toks"], block=a.block)
            m, mx_ = kl(off, on); t1 = float((off["lps"].argmax(-1) == on["lps"].argmax(-1)).mean())
            ok = m <= KL_MEAN and mx_ <= KL_MAX and t1 >= TOP1_MIN; rc |= not ok
            verdict = "PASS" if ok else "FAIL"
            print(f"{name}: MMA_ROWS={a.mma_rows} vs 0 (k-split off), {a.block}-row blocks, prompt {a.prompt} +{a.new}: KL mean {m:.2e} max {mx_:.2e} top1 agree {t1:.3f} -> {verdict}", flush=True)
        sys.exit(rc)
    off = {k: v for d in SWITCHES.values() for k, v in d.items()}
    bad = []
    for pack in a.packs:
        name = os.path.basename(pack.rstrip("/"))
        ref = run(pack, "off", off, a.new, a.prompt)
        on = run(pack, "on", {}, a.new, a.prompt, force=ref["toks"])
        m, mx_ = kl(ref, on)
        t1 = float((ref["lps"].argmax(-1) == on["lps"].argmax(-1)).mean()); same = t1 == 1.0
        ok = m <= KL_MEAN and mx_ <= KL_MAX and t1 >= TOP1_MIN
        print(f"{name}: all levers ON vs OFF, {a.new} tokens: KL mean {m:.2e} max {mx_:.2e} argmax {'equal' if same else 'DIFFERS'} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if ok:
            continue
        for lever, sw in SWITCHES.items():            # bisect: only this lever on
            env = {k: v for k, v in off.items() if k not in sw}
            g = run(pack, lever, env, a.new, a.prompt, force=ref["toks"])
            lm_, lx = kl(ref, g)
            hit = lm_ > KL_MEAN or lx > KL_MAX
            print(f"  only {lever:24s} on: KL mean {lm_:.2e} max {lx:.2e} {'<-- BREAKS THE GATE' if hit else 'ok'}", flush=True)
            if hit:
                bad.append(f"{name}:{lever}")
        if not any(b.startswith(name + ":") for b in bad):
            bad.append(f"{name}:combination")
            print("  no single lever breaks the gate: an interaction between levers", flush=True)
    if bad:
        print("LEVER CHECK FAILED:", ", ".join(bad), flush=True)
        sys.exit(1)
    print("LEVER CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
