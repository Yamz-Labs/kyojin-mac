import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
d = be.new_draft(); h = mx.random.normal((1, 600, 2560)).astype(mx.float16)
be._mtp_rows(d, mx.arange(600, dtype=mx.int32) + 1000, h); mx.eval(d.mc.keys)
print("DT", d.mc.keys.dtype, d.mc.keys.shape, type(be.mtp.layer).__name__, [ (k, v.dtype) for k, v in list(__import__("mlx.utils").utils.tree_flatten(be.mtp.parameters()))[:6]])
a = be.mtp.layer.self_attn
print("ATT", a.num_attention_heads, a.num_key_value_heads, a.head_dim)
x = be.mtp.fc(mx.concatenate([be.mtp.pre_fc_norm_embedding(be.lm.model.embed_tokens(mx.arange(4, dtype=mx.int32))[None]), be.mtp.pre_fc_norm_hidden(h[:, :4])], axis=-1))
print("X", x.dtype, be.lm.model.embed_tokens(mx.arange(4, dtype=mx.int32)).dtype, be.mtp.layer.input_layernorm(x).dtype)
l = be.mtp.layer; print("L", [(k, v.dtype) for k, v in __import__("mlx.utils").utils.tree_flatten(l.parameters()) if v.dtype not in (mx.uint32,)])
