"""pf_idx_heads recall on the real dumps in /tmp/idxcap: per plain indexer layer, the exact all-head top-k of a chunk row against what the lever returns.

What the lever does per row, mirrored here in numpy: pick the H heads of largest |w| * ||q_h|| (pf_flat.head_sel), take the max over each 8-key block of the
partial score (the int8 block-max kernel, `bmax_i8_rows`), keep the M best blocks (pf_idx_heads_m), rescore their keys on all 32 heads and take the exact
top-k. `--exact-keys` uses the fp32 subset scores; the default mirrors the int8 pack (one code per key / per query head, the kernel's operands) exactly as
pf_flat.i8pack rounds them, so the served first pass's ranking is reproduced (its int32 product and fp32 head sum are then exact for the packed values).

One .npz (q [T, 32, 128] fp16, w [T, 32], idx [T, topk] int32, start, ratio) and one keys .npy (Tc, 128) at a time: < 1 GiB, no Metal.

    python tools/pfheads_recall.py                      # 3 layers x 32 rows, H = 8 / 16, M = 256 .. 4096
    python tools/pfheads_recall.py --layers 1 --rows 8   # a quick pass
"""
import argparse
import glob
import time

import numpy as np

D = "/tmp/idxcap"
BS = 8


def pack8(a):
    """pf_flat.i8pack in numpy: (codes int8, scale float32) with a[r] == codes[r] * scale[r] on the grid, else rounded."""
    x = a.astype(np.float32)
    m = np.abs(x).max(-1)
    e = np.where(m > 0, np.ceil(np.log2(np.where(m > 0, m, 1.0) / 127.0)), 0.0)
    return np.clip(np.rint(x * np.exp2(-e)[:, None]), -127, 127), np.exp2(e).astype(np.float32)


def head_sel(q, w, h):
    """pf_flat.head_sel: the h heads of largest |w| * ||q_h|| of a row."""
    return np.argsort(-(np.abs(w) * np.linalg.norm(q, axis=1)))[:h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--heads", default="8,16")
    ap.add_argument("--m", default="256,512,544,1024,2048,4096")
    ap.add_argument("--exact-keys", action="store_true", help="fp32 subset scores instead of the int8 pack's operands")
    a = ap.parse_args()
    hs = [int(x) for x in a.heads.split(",")]
    ms = [int(x) for x in a.m.split(",")]
    files = sorted(glob.glob(D + "/idx_*.npz"), key=lambda f: f[-6:])
    for L in range(a.layers):
        f = files[L]
        d = np.load(f)
        K0 = np.load(f.replace("idx_", "keys_").replace(".npz", ".npy")).astype(np.float32)
        T = int(d["q"].shape[0])
        start, ratio, topk = int(d["start"]), int(d["ratio"]), int(d["topk"])
        Tc = K0.shape[0]
        rr = np.linspace(7, T - 3, a.rows).astype(int)
        t0 = time.time()
        E = {}
        for t in rr:
            clen = (start + t + 1) // ratio
            q = np.array(d["q"][t]).astype(np.float32)
            w = np.array(d["w"][t]).astype(np.float32)
            s = (np.maximum(q @ K0[:clen].T, 0.0) * w[:, None]).sum(0)
            E[t] = (clen, s, q, w, np.sort(np.argpartition(-s, topk)[:topk]))
        print(f"L{L} Tc={Tc} rows={len(rr)} topk={topk} start={start} ratio={ratio} "
              f"({'fp32 subset scores' if a.exact_keys else 'int8 pack mirror'}) exact reference {time.time() - t0:.0f}s", flush=True)
        for h in hs:
            line = []
            for m in ms:
                r5 = []
                for t in rr:
                    clen, s, q, w, tt = E[t]
                    nb = clen // BS
                    hd = head_sel(q, w, h)
                    qq, ww, kk = q[hd], w[hd], K0[:nb * BS]
                    if not a.exact_keys:                                    # the operands the kernel multiplies: one packed query head, one packed key
                        qc, qs = pack8(qq)
                        kc, ks = pack8(K0[:nb * BS])
                        qq, kk = qc * qs[:, None], kc.astype(np.float32) * ks[:, None]
                    sb = (np.maximum(qq @ kk.T, 0.0) * ww[:, None]).sum(0)
                    sbm = sb.reshape(nb, BS).max(1)
                    ids = np.argpartition(-sbm, min(m, nb - 1))[:m]
                    keep = np.zeros(clen, bool)
                    keep[(ids[:, None] * BS + np.arange(BS)).ravel()] = True
                    got = np.sort(np.flatnonzero(keep)[np.argsort(-s[keep], kind="stable")[:topk]])
                    r5.append(np.isin(got, tt).sum() / topk)
                line.append(f"M{m}: {100 * np.mean(r5):5.2f}")
            print(f"  heads {h:2d} " + "  ".join(line), flush=True)
        del d, K0, E


if __name__ == "__main__":
    main()
