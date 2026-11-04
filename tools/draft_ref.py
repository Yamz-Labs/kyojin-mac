#!/usr/bin/env python3
"""BOX side of the draft numerics check: the checkpoint's own model.py (DSparkBlock.forward_embed / forward / forward_head, Gate, MoE, Expert,
Block with mHC, apply_rotary_emb) run on CPU in torch, with the pack's EXL3 weights dequantized by the pure-numpy reference (kyojin_mlx.ref_numpy),
against the arrays dumped by the Mac drafter (tools/draft_dump.py). Only the tilelang kernels are replaced (kernel.py semantics, in torch):
act_quant (fp8 e4m3 round trip, blocks of 32, ue8m0 scale), fp8_gemm / fp4_gemm (dequantized activations @ the dense weight), sparse_attn
(gather + softmax with the per-head sink), hc_split_sinkhorn. Every Linear keeps its fp8 weight dtype (empty placeholder + `.dense` fp32 weight), so
linear() applies the same activation round trip as the checkpoint does; wo_a is bf16 in the checkpoint (explicit act_quant before the einsum).

usage: draft_ref.py PACK DUMP.npz [DUMP2.npz ...] [--model-dir ~/kyojin-work/ds41-ref/inference] [--negs] [--noise 4.9e-4 ...] [--json OUT]
Prints, per dump: relative error of the main-kv rows of every stage, of the collapsed pre-norm state, of the 5 logits (with the markov bias), of the
confidence logits, and whether the 5 drafted ids agree (with the top-1/top-2 gap of the reference when they do not)."""
import argparse
import json
import os
import struct
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kyojin_mlx import ref_numpy as RN                           # numpy only (no mlx import)

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("dumps", nargs="+")
ap.add_argument("--model-dir", default=os.path.expanduser("~/kyojin-work/ds41-ref/inference"))
ap.add_argument("--negs", action="store_true", help="negative controls: another pending token, the final call one position later (both must differ from the Mac dump)")
ap.add_argument("--noise", type=float, nargs="*", default=[], help="noise floor: reference rerun with this relative noise on the taps (fp16 half ulp = 4.9e-4), compared with the reference")
ap.add_argument("--json")
args_cli = ap.parse_args()
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))


# ------------------------------------------------------------------------------------------------ tilelang kernels in torch (kernel.py semantics)
def _ceil_log2(v):
    m, e = torch.frexp(v)
    return torch.where(m == 0.5, e - 1, e)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    xf = x.float()
    shp = xf.shape
    xb = xf.reshape(*shp[:-1], shp[-1] // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = amax * (1.0 / 448.0)
    if scale_fmt is not None:
        s = torch.ldexp(torch.ones_like(s), _ceil_log2(s))
    q = ((xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s).reshape(shp)
    if inplace:
        x.copy_(q.to(x.dtype))
        return x
    return q.to(x.dtype), s


def fp8_gemm(x, s, weight, weight_scale, *a, **k):
    return (x.float() @ weight.dense.T).to(x.dtype)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv.float()[torch.arange(b)[:, None, None], idx.clamp(min=0)]                         # [b, m, k, d]
    sc = torch.einsum("bmhd,bmkd->bmhk", q.float(), g) * softmax_scale
    sc = sc.masked_fill(~valid[:, :, None, :], float("-inf"))
    mx = sc.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(sc - mx)
    den = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", p / den, g)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    mixes = mixes.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m


def _none_cls(name):
    return type(name, (), {"from_args": classmethod(lambda cls, a: None)})


_stub("kernel", act_quant=act_quant, fp4_act_quant=None, fp4_gemm=fp8_gemm, fp8_gemm=fp8_gemm, hc_split_sinkhorn=hc_split_sinkhorn, sparse_attn=sparse_attn)
_stub("engram", EngramLayout=_none_cls("EngramLayout"), NgramHashState=_none_cls("NgramHashState"))
_stub("image_processor", IMAGE=0, IMAGE_END=1, IMAGE_NEW_LINE=2, IMAGE_START=3)
_stub("vision", Aligner=_none_cls("Aligner"), ViT=_none_cls("ViT"))
sys.path.insert(0, args_cli.model_dir)
import model as M                                                                          # noqa: E402  (the checkpoint's reference code)


# ------------------------------------------------------------------------------------------------ pack reading
class Pack:
    DT = {"F16": np.float16, "F32": np.float32, "I16": np.int16, "U32": np.uint32, "U8": np.uint8, "I32": np.int32, "BF16": np.uint16}

    def __init__(self, d):
        self.d, self.where = d, {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                with open(os.path.join(d, f), "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    h = json.loads(fh.read(n))
                h.pop("__metadata__", None)
                for k, m in h.items():
                    self.where[k] = (f, 8 + n, m)
        self.cfg = json.load(open(os.path.join(d, "config.json")))

    def raw(self, name):
        f, off, m = self.where[name]
        a, b = m["data_offsets"]
        mm = np.memmap(os.path.join(self.d, f), dtype=np.uint8, mode="r", offset=off + a, shape=(b - a,))
        return np.frombuffer(mm, dtype=self.DT[m["dtype"]]).reshape(m["shape"]), m["dtype"]

    def t(self, name):
        """float32 torch tensor of a plain (non-EXL3) tensor."""
        a, dt = self.raw(name)
        if dt == "BF16":
            return (torch.from_numpy(np.ascontiguousarray(a).astype(np.int32) << 16).view(torch.float32)).clone()
        return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))

    def exl3(self, prefix):
        """dense torch weight [out, in] float32 of an EXL3 linear (numpy reconstruct: y = x @ W)."""
        tr, _ = self.raw(prefix + ".trellis"); su, _ = self.raw(prefix + ".suh"); sv, _ = self.raw(prefix + ".svh")
        return torch.from_numpy(np.ascontiguousarray(RN.reconstruct(np.asarray(tr), np.asarray(su), np.asarray(sv)).T))

    def expert(self, stage, e, proj):
        pre = f"mtp.{stage}.ffn.experts.{proj}"
        words, _ = self.raw(pre + ".trellis"); off, _ = self.raw(pre + ".offset"); k, _ = self.raw(pre + ".k")
        shp, _ = self.raw(pre + ".shape"); su, _ = self.raw(pre + ".suh"); sv, _ = self.raw(pre + ".svh")
        w = np.ascontiguousarray(words[off[e]:off[e + 1]]).view(np.int16).reshape(int(shp[0]), int(shp[1]), 16 * int(k[e]))
        return torch.from_numpy(np.ascontiguousarray(RN.reconstruct(w, np.asarray(su[e]), np.asarray(sv[e])).T))

    def slices(self, prefix):
        tr, _ = self.raw(prefix + ".trellis"); su, _ = self.raw(prefix + ".suh"); sv, _ = self.raw(prefix + ".svh")
        return torch.cat([torch.from_numpy(np.ascontiguousarray(RN.reconstruct(np.asarray(tr[g]), np.asarray(su[g]), np.asarray(sv[g])).T)) for g in range(tr.shape[0])], 0)


pk = Pack(args_cli.pack)
tc = pk.cfg["text_config"]
nst = int(tc["num_nextn_predict_layers"])
args = M.ModelArgs(
    max_batch_size=1, max_seq_len=1024, temperature=0, vocab_size=tc["vocab_size"], dim=tc["hidden_size"], moe_inter_dim=tc["moe_intermediate_size"],
    n_layers=tc["num_hidden_layers"], n_mtp_layers=nst, n_heads=tc["num_attention_heads"], n_routed_experts=tc["n_routed_experts"], n_shared_experts=1,
    n_activated_experts=tc["num_experts_per_tok"], score_func=tc["scoring_func"], norm_topk_prob=tc["norm_topk_prob"], route_scale=tc["routed_scaling_factor"],
    swiglu_limit=tc["swiglu_limit"], q_lora_rank=tc["q_lora_rank"], head_dim=tc["head_dim"], rope_head_dim=tc["qk_rope_head_dim"], norm_eps=tc["rms_norm_eps"],
    o_groups=tc["o_groups"], o_lora_rank=tc["o_lora_rank"], window_size=tc["sliding_window"],
    compress_ratios=tuple(tc["compress_ratios"]) + (0,) * nst, kv_source_layers=(), index_source_layers=(), rope_theta=tc["rope_theta"],
    hc_mult=tc["hc_mult"], hc_sinkhorn_iters=tc["hc_sinkhorn_iters"], hc_eps=tc["hc_eps"],
    dspark_block_size=tc["dspark_block_size"], dspark_noise_token_id=tc["dspark_noise_token_id"], dspark_target_layer_ids=tuple(tc["dspark_target_layer_ids"]),
    dspark_markov_rank=tc["dspark_markov_rank"], dspark_n_routed_experts=tc["dspark_n_routed_experts"] if "dspark_n_routed_experts" in tc else tc["n_routed_experts"],
    dspark_n_activated_experts=tc["dspark_num_experts_per_tok"], expert_dtype=None)
print(f"reference: dim {args.dim} heads {args.n_heads} stages {nst} experts {args.dspark_n_routed_experts} top {args.dspark_n_activated_experts} block {args.dspark_block_size} "
      f"taps {args.dspark_target_layer_ids} eps {args.norm_eps}", flush=True)

torch.set_default_dtype(torch.float32)
M.default_dtype = torch.float8_e4m3fn
t0 = time.time()
embed = M.ParallelEmbedding(args.vocab_size, args.dim)
embed.weight.data.copy_(pk.t("embed.weight"))
head = M.ParallelHead(args.vocab_size, args.dim)
head.weight.data.copy_(pk.exl3("head"))
blocks = []
for s in range(nst):
    blk = M.DSparkBlock(args.n_layers + s, args)
    p = f"mtp.{s}"
    a, f = blk.attn, blk.ffn
    for nm in ("wq_a", "wq_b", "wkv", "wo_b"):
        getattr(a, nm).weight.dense = pk.exl3(f"{p}.attn.{nm}")
    a.wo_a.weight = torch.nn.Parameter(pk.slices(f"{p}.attn.wo_a.slices"), requires_grad=False)
    for nm in ("q_norm", "kv_norm"):
        getattr(a, nm).weight.data.copy_(pk.t(f"{p}.attn.{nm}.weight"))
    a.attn_sink.data.copy_(pk.t(f"{p}.attn.attn_sink"))
    blk.attn_norm.weight.data.copy_(pk.t(f"{p}.attn_norm.weight")); blk.ffn_norm.weight.data.copy_(pk.t(f"{p}.ffn_norm.weight"))
    for nm in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
        getattr(blk, nm).data.copy_(pk.t(f"{p}.{nm}"))
    f.gate.weight.data.copy_(pk.t(f"{p}.ffn.gate.weight")); f.gate.bias.data.copy_(pk.t(f"{p}.ffn.gate.bias"))
    for nm in ("w1", "w2", "w3"):
        getattr(f.shared_experts, nm).weight.dense = pk.exl3(f"{p}.ffn.shared_experts.{nm}")

    def lazy(ex, stage=s, e=0):
        def fwd(x, weights=None, _ex=ex, _orig=M.Expert.forward):
            if not hasattr(_ex.w1.weight, "dense"):
                for nm in ("w1", "w2", "w3"):
                    getattr(_ex, nm).weight.dense = pk.expert(stage, e, nm)
            return _orig(_ex, x, weights)
        ex.forward = fwd
    for e, ex in enumerate(f.experts):
        lazy(ex, s, e)
    if s == 0:
        blk.main_proj.weight.dense = pk.exl3(f"{p}.main_proj")
        blk.main_norm.weight.data.copy_(pk.t(f"{p}.main_norm.weight"))
    if s == nst - 1:
        blk.norm.weight.data.copy_(pk.t(f"{p}.norm.weight"))
        blk.markov_head.embed.weight.data.copy_(pk.t(f"{p}.markov_head.embed.weight")); blk.markov_head.head.weight.data.copy_(pk.t(f"{p}.markov_head.head.weight"))
        blk.confidence_head.proj.weight.data.copy_(pk.t(f"{p}.confidence_head.proj.weight"))
    blk.embed, blk.head = embed, head
    blocks.append(blk)
print(f"weights bound (experts load on first use) in {time.time() - t0:.0f}s", flush=True)


def forward_spec(input_ids, main_hidden, start_pos):
    """Transformer.forward_spec of model.py, on this list of stages."""
    h, main_x = blocks[0].forward_embed(main_hidden, input_ids)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    for b in blocks:
        h, pre_mix = b(h, start_pos, pre_mix, main_x)
    if start_pos == 0:
        return None
    return blocks[-1].forward_head(h, pre_mix, input_ids)


def append_only(main_hidden, start_pos):
    """The window write of a decode call (DSparkAttention.forward up to the cache), without the block / MoE work."""
    main_x = blocks[0].main_norm(blocks[0].main_proj(main_hidden))
    dummy = torch.zeros(1, args.dspark_block_size, args.dim)
    for b in blocks:
        b.attn(dummy, start_pos, main_x)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)), float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def run_variant(taps, split, pending, pend_shift=0, pos_shift=0):
    """Reference forward: prefill call (start_pos 0) with `split` rows, decode-style appends for the rest, final full call with the block."""
    S, win = taps.shape[0], args.window_size
    for b in blocks:
        b.attn.window_kv_cache.zero_()
    forward_spec(torch.tensor([0]), taps[None, :split], 0)                              # prefill: seeds the window caches
    for i in range(split, S - 1):
        append_only(taps[None, i:i + 1], i)
    cap = []
    hooks = [b.attn.register_forward_hook(lambda m, i, o: cap.append(o[0].detach().numpy().copy())) for b in blocks]
    out_ids, logits, conf = forward_spec(torch.tensor([pending + pend_shift]), taps[None, S - 1:S], S - 1 + pos_shift)
    for h in hooks:
        h.remove()
    n = min(S, win)
    pos = (np.arange(S - n, S)) % win
    return dict(rows=[b.attn.window_kv_cache[0].numpy()[pos].copy() for b in blocks], logits=logits[0].numpy().copy(), conf_logit=conf[0].numpy().copy(),
                ids=out_ids[0, 1:].tolist(), attn=np.stack(cap))


def compare(got, ref, label):
    row = {"cmp": label}
    for s_, (g, r) in enumerate(zip(got["rows"], ref["rows"])):
        row[f"rows{s_}"] = rel(g, r)
    if got.get("attn") is not None and ref.get("attn") is not None:
        for s_ in range(len(ref["attn"])):
            row[f"attn{s_}"] = rel(got["attn"][s_], ref["attn"][s_])
    row["logits"] = rel(got["logits"], ref["logits"])
    row["conf_logit"] = rel(got["conf_logit"], ref["conf_logit"])
    row["ids_equal"] = got["ids"] == ref["ids"]
    print(f"  [{label}] " + "  ".join(f"{k} {row[k][0]:.2e}" for k in row if k.startswith(("rows", "attn")) or k in ("logits", "conf_logit"))
          + f"   ids {got['ids']} vs {ref['ids']} {'AGREE' if row['ids_equal'] else 'DIFFER'}", flush=True)
    return row


results = []
with torch.inference_mode():
    for path in args_cli.dumps:
        z = np.load(path)
        taps = torch.from_numpy(z["taps"].astype(np.float32))
        S, split, pending = taps.shape[0], int(z["split"]), int(z["pending"])
        mac = dict(rows=[z[f"rows{s_}"] for s_ in range(len(blocks))], attn=z["attn"] if "attn" in z.files else None, logits=z["logits"], conf_logit=z["conf_logit"], ids=z["out_ids"].tolist())
        print(f"{os.path.basename(path)}: {S} positions, prefill call {split} rows, pending {pending}", flush=True)
        t1 = time.time()
        base = run_variant(taps, split, pending)
        res = {"dump": os.path.basename(path), "S": S, "split": split, "secs": round(time.time() - t1, 1)}
        res["mac_vs_ref"] = compare(mac, base, "mac vs reference")
        gap = np.sort(base["logits"], axis=-1)
        res["ref_top_gap"] = [round(float(g), 3) for g in (gap[:, -1] - gap[:, -2])]
        res["ref_ids"], res["mac_ids"] = base["ids"], mac["ids"]
        res["mac_logit_at_ref_ids"] = [round(float(mac["logits"][i, base["ids"][i]]), 2) for i in range(len(base["ids"]))]
        res["ref_logit_at_ref_ids"] = [round(float(base["logits"][i, base["ids"][i]]), 2) for i in range(len(base["ids"]))]
        print(f"  reference top1-top2 gaps {res['ref_top_gap']}; logit at the reference ids: mac {res['mac_logit_at_ref_ids']} ref {res['ref_logit_at_ref_ids']}", flush=True)
        if args_cli.noise:
            g = torch.Generator().manual_seed(11)
            for amp in args_cli.noise:
                noisy = taps * (1 + amp * torch.randn(taps.shape, generator=g))
                res[f"noise{amp}"] = compare(run_variant(noisy, split, pending), base, f"reference with taps +{amp:g} relative noise vs reference")
        if args_cli.negs:
            res["neg_pending"] = compare(mac, run_variant(taps, split, pending, pend_shift=1), "NEG pending+1: mac vs reference")
            res["neg_pos"] = compare(mac, run_variant(taps, split, pending, pos_shift=1), "NEG position+1: mac vs reference")
        results.append(res)
if args_cli.json:
    json.dump(results, open(args_cli.json, "w"), indent=1)
