import sys, os
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, "tools")
from test_engram_rows import spec_with_rows, token_map
from count_launches import graph_census
import mlx_vlm.models.deepseek_v41.engram as eng
sp, rows = spec_with_rows(); tm = token_map()
vcfg = SimpleNamespace(engram_layer_ids=list(sp.layer_ids), engram_max_ngram_size=4, engram_n_heads=8, engram_vocab_size=sp.vocab_size,
                       engram_num_embeddings=list(rows), engram_head_dim=256, engram_pad_token_id=2, engram_compressed_vocab_size=300)
h = eng.NgramHashState(vcfg, eng.EngramLayout.from_config(vcfg), token_map=[int(x) for x in tm])
st = SimpleNamespace(engram=None)
ids = np.random.default_rng(3).integers(0, 2000, 57)
out = h(mx.array(ids[None]), 0, st); mx.eval(out, st.engram)
o = h(mx.array([[5]]), 57, st)
c = graph_census([o, st.engram]); print(c["dispatches"], c["by_type"])
