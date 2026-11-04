"""Helpers shared by the NAX probe and tests: small stacks cut from a real pack (partial reads, < 100 MB), k-permutation of A."""
import glob, os
import numpy as np
import mlx.core as mx
from kyojin_mlx import moe as M, metal_nax as NX


def mini_stack(dirpath: str, proj: str, experts, cb: str = "mul1") -> M.Stack:
    """Stack of the listed experts of `proj` (e.g. 'w1') of the stacked layer file in `dirpath`, offsets rebased. Reads only those ranges."""
    from safetensors import safe_open
    f = sorted(glob.glob(os.path.join(dirpath, "stacked.*.safetensors")))[0]
    with safe_open(f, "np") as s:
        keys = [k for k in s.keys() if k.endswith(f".experts.{proj}.trellis")]
        pre = keys[0][:-len(".trellis")]
        off = s.get_tensor(f"{pre}.offset")
        kk = s.get_tensor(f"{pre}.k")
        shp = s.get_tensor(f"{pre}.shape")
        tr = s.get_slice(f"{pre}.trellis")
        suh, svh = s.get_tensor(f"{pre}.suh"), s.get_tensor(f"{pre}.svh")
        words, noff = [], [0]
        for e in experts:
            w = tr[int(off[e]):int(off[e + 1])]
            words.append(w); noff.append(noff[-1] + len(w))
    return M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(noff, "<u4")), mx.array(kk[list(experts)]),
                   mx.array(suh[list(experts)]), mx.array(svh[list(experts)]), int(shp[0]), int(shp[1]), cb)


def synth_stack(kin, nout, ks, cb="mul1", seed=1) -> M.Stack:
    from bench_moe import synth_stack as ss
    return ss(kin, nout, ks, cb, seed)


def kperm_cols(a: np.ndarray) -> np.ndarray:
    return NX.permute_k(a)
