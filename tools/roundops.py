"""roundops.py [ROWS ...]: count the graph ops of one decode / verify round of a hybrid foreign (GGUF) Qwen3.5 model, per op kind. CPU-side: the lazy graph is
built and counted (mx.export_to_dot), nothing is evaluated for the count; a synthetic GGUF with the dims of the 27B stack (3 gated-delta + 1 attention
pattern, mixed Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0 types) is written once to the temp dir. Run from the repo root with PYTHONPATH=$PWD.
KYOJIN_ROUNDOPS_LAYERS=N sets the layer count (default 8); counts scale linearly with it. Env flags (KYOJIN_DECFUSE*, KYOJIN_CFMTSPEED ...) are read as the server reads them."""
from __future__ import annotations

import collections
import os
import re
import sys
import tempfile

import numpy as np

if "--gpu" in sys.argv:                                  # exact dispatches by Metal function name (tools/cbcount), one per command buffer
    os.environ.update(KYOJIN_GPU_KERN="1", MLX_MAX_OPS_PER_BUFFER="1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx  # noqa: E402

import test_gguf_arch as T  # noqa: E402
from kyojin_mlx import foreign_linear as FL, foreign_synth as SY  # noqa: E402

LAYERS = int(os.environ.get("KYOJIN_ROUNDOPS_LAYERS", "8"))
DIMS = dict(H=1024, V=48, I=2048, nh=8, nkv=2, hd=128, nk=2, nv=4, kd=128, vd=128, K=4, L=LAYERS, rd=64, sec=[11, 11, 10, 0])
TYPES = {"attn_qkv": "Q4_K", "attn_gate": "Q4_K", "ssm_alpha": "Q8_0", "ssm_beta": "Q8_0", "ssm_out": "Q5_K", "attn_q": "Q4_K", "attn_k": "Q4_K", "attn_v": "Q6_K",
         "attn_output": "Q4_K", "ffn_gate": "Q4_K", "ffn_up": "IQ4_XS", "ffn_down": "Q6_K"}


VIEWS = ("Reshape", "Squeeze", "Broadcast", "Transpose", "Slice", "Split", "ExpandDims", "AsStrided", "Flatten", "Unflatten")


def build(path):
    T.C.update(DIMS)
    T.C["sec"] = list(DIMS["sec"])
    hf = T._hf_qwen35(0, mtp=False)
    g = T._convert_qwen35(hf, dict(T.C), T.C["L"])
    tl = []
    for n, a in g.items():
        base = n.split(".")[-2] if n.endswith(".weight") else ""
        if a.ndim == 2 and base in TYPES and not n.startswith("token_embd"):
            n_, k_ = a.shape
            tl.append((n, (k_, n_), TYPES[base], SY.random_blocks(TYPES[base], n_, k_, np.random.default_rng(len(n)))))
        elif n.startswith("token_embd"):                                  # fp16 table like the real file's dequantized embedding: keeps the stream in fp16
            tl.append((n, tuple(reversed(a.shape)), "F16", np.ascontiguousarray(a, np.float16)))
        else:
            tl.append((n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a, np.float32)))
    SY.write_gguf(path, T._kv35(False, **{"qwen35.block_count": T.C["L"]}), tl)


def load():
    path = os.path.join(tempfile.gettempdir(), f"roundops_{LAYERS}.gguf")
    if not os.path.exists(path):
        build(path)
    return FL.load_foreign(path, log=lambda *a: None)


def graph(out):
    """(ops {id: label}, producer {array: op id}, inputs {op id: [array]}) of the graph reachable from `out`."""
    f = tempfile.NamedTemporaryFile("w+", suffix=".dot", delete=False)
    f.close()
    mx.export_to_dot(f.name, out)
    txt = open(f.name).read()
    os.unlink(f.name)
    ops = dict((i, lab) for i, lab in re.findall(r'\{ (\d+) \[label ="([^"]*)", shape=rectangle', txt))
    producer, inputs = {}, collections.defaultdict(list)
    for a, b in re.findall(r'^"?(\w+)"? -> "?(\w+)"?$', txt, re.M):
        if a in ops:
            producer[b] = a
        elif b in ops:
            inputs[b].append(a)
    return ops, producer, inputs


def count(out):
    """{op label: count} of the graph reachable from `out` (a node is counted once)."""
    return collections.Counter(graph(out)[0].values())


def view_copies(out):
    """Reshape / Flatten / Unflatten ops whose input is a strided view (Slice, Transpose, Broadcast, Split): MLX copies there when the graph is evaluated."""
    ops, producer, inputs = graph(out)
    n = collections.Counter()
    for i, lab in ops.items():
        if lab in ("Reshape", "Flatten", "Unflatten", "Squeeze", "ExpandDims") and inputs[i]:
            src = ops.get(producer.get(inputs[i][0]))
            if src in ("Slice", "Transpose", "Broadcast", "Split"):
                n[f"{src}->{lab}"] += 1
    return n


def dispatches(out, R):
    """Evaluate `out` under the cbcount probe: {Metal function name: (dispatches, GPU ms)} of the round."""
    from kyojin_mlx import gpuspan as GS
    GS.install()
    GS.begin()
    mx.eval(out)
    st = {}
    GS.end(st, R)
    return {k[len("gpu_kc_"):-len(f"_r{R}")]: (v, st.get(f"gpu_k_{k[len('gpu_kc_'):]}", 0.0)) for k, v in st.items() if k.startswith("gpu_kc_")}


GPU = "--gpu" in sys.argv


def main(rows):
    from mlx_lm.models.cache import make_prompt_cache
    from kyojin_mlx import qspec as QS
    if GPU:
        from kyojin_mlx import gpuspan as GS
        GS.install()                                       # before any pipeline is built, so every dispatch gets its name
    model, cfg = load()
    QS.install_capture()
    for R in rows:
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache))
        QS.CAP = []
        out = model(mx.array([[(i % 20) + 1 for i in range(R)]]), cache=cache)
        QS.CAP = None
        if GPU:
            d = dispatches(out, R)
            print(f"R={R}: {sum(v[0] for v in d.values())} dispatches, {sum(v[1] for v in d.values()):.2f} ms (1 op per buffer)")
            for k, (n, ms) in sorted(d.items(), key=lambda kv: -kv[1][0])[:int(os.environ.get("KYOJIN_ROUNDOPS_TOP", "16"))]:
                print(f"   {n:5d}  {ms:7.3f} ms  {k}")
            continue
        c = count(out)
        vc = view_copies(out)
        views = sum(c[k] for k in VIEWS)
        print(f"R={R}: nodes {sum(c.values())}, launches (nodes minus view ops {', '.join(sorted(VIEWS))}) {sum(c.values()) - views}")
        print(f"   possible strided-view copies (view op -> reshape): {sum(vc.values())} {dict(vc)}")
        for k, v in c.most_common(int(os.environ.get("KYOJIN_ROUNDOPS_TOP", "16"))):
            print(f"   {v:5d}  {k}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 6, 16])
