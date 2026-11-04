# Live ICB replay of the DS verify pass: measured facts and build plan (macspecr, ds2m, N=1 D=3)

Tools: tools/rp_verify.py (tape: record the served rows.verify once per token list, replay), tools/rp_verify_live.py (dump consecutive live rounds with commit), tools/rp_ident.py, tools/rp_affine.py.

## Measured
- Tape replay of the verify pass (152 dispatches): bit-exact (0/18), CPU per verify min 799 -> 26 us, GPU unchanged (7.23 -> 7.31 ms min).
- Between live rounds: 1505 bindings change buffer, 1067 are in-round scratch (private in replay), 40 (17 distinct buffers) are external state; 62-64 constant/grid fields vary.
- Those fields are periodic-affine in p0 (period = compress ratio; 40 consecutive positions, 62/62 fit, 5435 constant). Not affine in p0 under mixed accept (they depend on (p+1)//ratio).
- Shapes are capacity-bucketed (cmp 1x256x512, keys 1x256x128, win 4|1 x128x512, pend fixed).
- 156 vs 160 commands = the accept commit gather (non-identity pick), not the forward.

## What a live replay needs (not built)
1. rp.mm: per-field table T[(p - p_ref) mod L] + slope * ((p - p_ref) div L) for arena words, grid, threadgroup; applied on the ICB commands and arena of the slot before each execute (the CPU may edit an ICB command that is not in flight). L = lcm of the ratios.
2. Per-round host inputs the forward builds from Plan (pos_arr, sess_arr, emit rows / widx / lens, overlay rows / slots, cs tables) are small host-constant buffers whose contents are also functions of p0: fit them with the same table (no Plan code in the replay path); token ids and Engram hashes come as real input arrays.
3. State: next round win / kvs / scs = previous replay pend buffer at row offset a * rowbytes (rebind with offset); cmp / keys scatter in place. Identity commit (a = 0) keeps 4-row state, a pick makes 1-row state: two shape signatures. st.pend / taps / cmp / keys are rebuilt from outputs.
4. Fallback to eager at capacity growth (need > cap), at p < p_ref, on any signature change; shadow check on the first replays, bitwise.
5. Gates: serve_spec_check ALL EXACT, digests equal N=1/N=4, then one served bench.
Estimate 2-3 engineer-days. No MLX API blocks it.
