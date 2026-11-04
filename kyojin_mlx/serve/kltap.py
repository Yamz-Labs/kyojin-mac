"""Teacher-forced KL probe for lever A/B on one resident server (plain decode only). POST /admin/tap arms it for the next request:
  mode "ref": greedy decode; every step's log-softmax and token are kept (the reference arm, levers OFF);
  mode "cmp": the steps are forced to the reference tokens and each step gives KL(ref || now) and whether the argmax equals the reference token.
GET /admin/tap returns the result. Runs on the engine thread (lazy arrays stay there); scalars are read after the request ends."""
from __future__ import annotations

import mlx.core as mx


class Tap:
    def __init__(self, mode: str, ref: "Tap | None" = None):
        self.mode, self.ref, self.i = mode, ref, 0
        self.lps: list = []              # ref: [1, V] log-probs per step
        self.toks: list = []             # tokens produced (ref) or forced (cmp), lazy scalars until read
        self.kl: list = []
        self.agree: list = []

    def __call__(self, lg):
        x = lg.astype(mx.float32)
        lp = x - mx.logsumexp(x, axis=-1, keepdims=True)                   # [1, V]
        arg = mx.argmax(lp, axis=-1).astype(mx.int32)
        if self.mode == "ref":
            self.lps.append(lp)
            self.toks.append(arg)
            mx.async_eval(lp, arg)
            return arg
        j = min(self.i, len(self.ref.lps) - 1)
        rlp, rtok = mx.array(self.ref.lps[j]), mx.array(self.ref.toks[j])
        kl = mx.sum(mx.exp(rlp) * (rlp - lp), axis=-1)
        self.kl.append(kl)
        self.agree.append((arg == rtok).astype(mx.float32))
        self.i += 1
        mx.async_eval(kl, self.agree[-1])
        return rtok

    def save(self, path: str) -> None:
        """A finished reference to disk: a later server (another spec) can compare against it."""
        import numpy as np
        np.savez(path, lps=np.concatenate([np.array(x) for x in self.lps], 0), toks=np.array([int(x.item()) for x in self.toks]))

    @classmethod
    def load(cls, path: str) -> "Tap":
        import numpy as np
        d = np.load(path)
        t = cls("ref")
        t.lps = [d["lps"][i:i + 1] for i in range(d["lps"].shape[0])]          # numpy: mlx arrays are made on the engine thread
        t.toks = [d["toks"][i:i + 1].astype(np.int32) for i in range(d["toks"].shape[0])]
        return t

    def result(self) -> dict:
        n = len(self.kl) if self.mode == "cmp" else len(self.toks)
        out = {"mode": self.mode, "steps": n}
        if self.mode == "cmp" and n:
            n = min(n, len(self.ref.lps))
            kl = [float(x.item()) for x in self.kl[:n]]
            ag = [float(x.item()) for x in self.agree[:n]]
            out.update(steps=n, kl_mean=sum(kl) / n, kl_max=max(kl), top1=sum(ag) / n, kl_steps=[round(x, 4) for x in kl], ref_toks=[int(x.item()) for x in self.ref.toks[:n]])
        return out
