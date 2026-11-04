"""KL of the fused NAX path vs decode+MLX matmul (and the chunk-size floor) on real text, all positions of the last chunk. Same model, fp16 logits -> fp32 log-softmax slices.
usage: kl88.py EXL3_DIR T chunk [offset=20000]   arms: A = fused (fused_max 1024), B = decode + matmul (fused_max 0), A2 = fused at chunk/2 (floor: same math, other cut)"""
import os
import sys, json, time, mlx.core as mx
exl3, T, ch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); off = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
mx.set_cache_limit(256 * 2 ** 20)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json")))[off:off + T]
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import exl3_linear as EL, wcache as WC
be, _ = load_backend(exl3)
LINS = [q for q in be.model.modules() if type(q) is EL.Exl3Linear]
def logits(fmax, chunk):
    for q in LINS: q.fused_max = fmax
    c = be.make_cache(); outs = []
    for i in range(0, T, chunk):
        x = mx.array(ids[i:i + chunk])[None]
        lg = be.model.language_model(x, cache=c)[0].astype(mx.float16)          # every row (no last_row context)
        mx.eval(lg, *be.state(c)); outs.append(lg)
    return mx.concatenate(outs, 0)
def kl(p, q, lo):
    tot, n, agree = 0.0, 0, 0
    for s in range(lo, T, 64):
        a = p[s:s + 64].astype(mx.float32); b = q[s:s + 64].astype(mx.float32)
        la, lb = a - mx.logsumexp(a, -1, keepdims=True), b - mx.logsumexp(b, -1, keepdims=True)
        tot += float(mx.sum(mx.exp(la) * (la - lb))); n += a.shape[0]; agree += int(mx.sum(mx.argmax(a, -1) == mx.argmax(b, -1)))
    return tot / n, agree / n
A = logits(1024, ch); B = logits(0, ch); A2 = logits(1024, max(ch // 2, 128)); B2 = logits(0, max(ch // 2, 128))
lo = max(ch // 2, 128)
print(f"T={T} chunk={ch}  (positions {lo}..{T})")
print("KL fused(chunk) || decode+mm(chunk)      : %.3e  top1 agree %.4f" % kl(A, B, lo))
print("KL fused(chunk) || fused(chunk/2)  [floor]: %.3e  top1 agree %.4f" % kl(A, A2, lo))
print("KL dec+mm(chunk) || dec+mm(chunk/2) [floor]: %.3e  top1 agree %.4f" % kl(B, B2, lo))
print("KL dec+mm(chunk) || fused(chunk/2)        : %.3e  top1 agree %.4f" % kl(B, A2, lo))
print(f"peak {mx.get_peak_memory()/2**30:.2f} GiB")
