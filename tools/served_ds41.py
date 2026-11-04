import kyojin_mlx  # noqa: F401  (first: sets the one-buffer MLX env for a DS4.1 pack before mlx is imported)
import os, sys, time, json, threading, faulthandler
if os.environ.get('FH'):
    faulthandler.dump_traceback_later(int(os.environ['FH']), repeat=True)
import mlx.core as mx
from transformers import PreTrainedTokenizerFast
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine, default_kv_budget
# usage: served_ds41.py DEPTH [NEW] [PACK]   (a PACK argument also lets `import kyojin_mlx` set the Engram-gate MLX env before mlx is imported)
PACK = os.path.expanduser(sys.argv[3] if len(sys.argv) > 3 else '~/ds41/model'); DEPTH = int(sys.argv[1]); NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128
tok = PreTrainedTokenizerFast(tokenizer_file=PACK + '/tokenizer.json')
be, cfg = load_backend(PACK, mtp=DEPTH > 0, ced=True, log=lambda *a: None)
print('loaded', be.kind, 'supports_spec', be.supports_spec, flush=True)
eng = Engine(be, sessions=1, ctx=8192, chunk=2048, block=2048, prefix_bytes=1024 << 20, mtp_depth=DEPTH, kv_budget=default_kv_budget(),
             row_cost=getattr(be, 'row_cost', 0.4) or 0.4, draft_cost=getattr(be, 'draft_cost', 0.15) or 0.15, log=print if os.environ.get('KYOJIN_LOG') else (lambda *a: None),
             depth_override=int(os.environ['KYOJIN_MTP_FIXED']) if os.environ.get('KYOJIN_MTP_FIXED') else None)
P = {'prose': 'Write a long, detailed essay about the history of the Roman Empire, its institutions, its economy and its decline. ',
     'code': 'Write a complete Python implementation of a red-black tree with insert, delete and search, with comments and tests. '}
def run(ids):
    h = eng.submit(ids, max_new=NEW)
    out = []
    while True:
        ev = h.events.get()
        if ev[0] == 'tok': out.append(ev[1])
        if ev[0] == 'done':
            import hashlib
            print('  tokens md5', hashlib.md5(str(out).encode()).hexdigest()[:12], len(out), ev[1], flush=True)
            return ev[2]
def client():
    for rep in range(2):
        for k, p in P.items():
            ids = tok.encode(p + str(rep))
            s0 = dict(eng.stats); st = run(ids); s1 = eng.stats
            n = st.get('n_gen', NEW)
            dec = st['total_s'] - st['ttft_s']
            d = {k2: s1[k2] - s0.get(k2, 0) for k2 in s1 if k2.startswith('spec_') and not k2.startswith('spec_t')}
            print(f'{k} rep{rep} depth{DEPTH} fixed={os.environ.get("KYOJIN_MTP_FIXED")}: {(NEW-1)/dec:.1f} tok/s ttft {st["ttft_s"]:.2f}', d, flush=True)
    print({k: round(v, 3) for k, v in eng.stats.items() if k.startswith("spec_t")}, flush=True)
    print("DBG row", eng.row_cost, 'draft', eng.draft_cost, 'k', eng.spec_k, eng._k_seen, 'plain_ms', eng.plain_ms, 'warm', getattr(eng,'_k_warm',None), flush=True)
    eng.stop()
threading.Thread(target=client, daemon=True).start()
eng.run()
