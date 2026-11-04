import sys, mlx.core as mx
from kyojin_mlx import tq, kvwin as KW
import kvwin_kl as K
from kyojin_mlx.exl3_linear import load_exl3
from mlx_lm.models import qwen3_next as QN
from transformers import AutoTokenizer
mdir = sys.argv[1]
model, _ = load_exl3(mdir); tok = AutoTokenizer.from_pretrained(mdir)
lm = model.language_model
cache = K.cache_for(model, [6, 6, 0, 4, 256])
ids = K.tokens(tok, 600, 0)
h = lm.model(mx.array(ids)[None], cache); mx.eval(h)
seen = {}
orig = tq.TQKVCache.attend
def spy(self, qs, scale):
    seen.setdefault("q", (qs.dtype, qs.shape, self.sk.dtype, self.rk.dtype, self.fused))
    return orig(self, qs, scale)
tq.TQKVCache.attend = spy
y = mx.argmax(K.head(lm)(h[:, -1:])[0], axis=-1)
h2 = lm.model(y[None], cache); mx.eval(h2)
print(seen)
