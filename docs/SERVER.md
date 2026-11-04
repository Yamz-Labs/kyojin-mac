
## Multi-session server (`--sessions N`)

An OpenAI-compatible server for several users at once on one Mac (Hermes main agent plus sub-agents on one local endpoint):
`kyojin_mlx/serve/` (stdlib HTTP, one engine thread owns every MLX call), started by `tools/mac_serve.sh`.

    tools/mac_serve.sh ~/kyojin-mlx/models/q35-2b-exl3-2.10 --sessions 4 --ctx 32768 --port 8080
    curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
         -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

- API: `POST /v1/chat/completions` (stream and not; `tools`, `tool_choice`, `reasoning_content`, `stop`, `seed`, `temperature` / `top_p` / `top_k` / `min_p`,
  `max_tokens` / `max_completion_tokens`, `stream_options.include_usage`), `POST /v1/completions`, `GET /v1/models`, `/health`, `/v1/stats`.
  `usage` carries `prompt_tokens_details.cached_tokens` (prefix hit); replies carry a small `timings` object. `n > 1`, penalties, logprobs and images are not supported (HTTP 400 or ignored).
- Continuous batching: requests join and leave the batched decode between steps, each with its own KV cache. Every decode op is row invariant
  (`multiuser.py`, `rows.py`), so a request gives the same tokens alone or in a crowd, greedy and seeded sampling alike.
- Prefill is chunked on a fixed grid of absolute positions (`--prefill-chunk`, default 256) and interleaved with decode steps, so a long prompt never freezes the others.
- Prefix sharing: snapshots every `--prefix-block` tokens (512) keyed by a hash of the token prefix, LRU within `--prefix-cache-mb` (1024; 0 = off).
  A new request forks the longest cached prefix of its prompt; because of the grid a hit is bit-identical to a miss.
- Without `max_tokens` a request may use the context: alone it runs to the end of `--ctx`; when others wait for a slot it yields after its
  share (`ctx / sessions`, floor 2048 tokens, as the box servers' `rq_budget`) and resumes later.
- Chat templates and tool-call parsers are the box Kyojin ones: Qwen3.5 (model template, XML tool calls), GLM-5.3 (`assets/glm53-template-medium.jinja`),
  DS4.1 (`ds41_chat.py` / `ds41_parse.py`, copied byte for byte from release 1.7.3). `--family`, `--thinking on|off`, `--reasoning-effort` set defaults.
- Checks: `tests/test_serve_chat.py`, `tests/test_serve_engine.py` (fake backend), `tools/serve_inproc.py` (staggered churn on the real model, tokens equal to solo),
  `tools/serve_check.py speed|burst|tool|hol|stream` (HTTP: 1 vs 4 streams, shared-prefix burst, tool round trip, head of line, API edge cases).

Our packs (GLM / DS4.1 / Qwen3.8) are all `mul1`; the real GLM slices in `data/` cover that path with real data.
Public EXL3 packs use the `mcg` codebook, e.g. https://huggingface.co/UnstableLlama/Qwen3.5-2B-exl3-2.10bpw
(single `model.safetensors`, mcg, K2 x140, K3 x10, K6 x1; the 4.00bpw sibling is K4 x150). On a real small model the Mac
therefore exercises the mcg variant (`cb="mcg"`). A small mul1 test model needs one local quantisation run.

### Grouped routed experts (`KYOJIN_ROWS_GROUPED=1`, `rows.enable(grouped=True)`)

Opt-in. Rows >= 2 of one MoE call (N sessions x (1 + D) rows) read the trellis of each DISTINCT expert once for all the rows routed to it, instead of once per (row, expert) pair.
`MoeGather.plan` groups the pairs by expert on the GPU (one threadgroup, no host sync): per expert its member pairs in ascending order, in classes by member count k
(1 / 2 / 3-4 / 5+ rows per pass, `KYOJIN_GCLASSES`). `grouped_gemv` decodes the expert once per pass and applies it to 1, 2, 4 or 8 member rows with their activations in registers
(k > 8 loops over passes); the partials land where the per-pair path puts them, so rot / mid / final (the scatter-add of the routing weights in row order) are unchanged.
Same fma order per row: bitwise equal to the per-row path for any k (tests/test_moe_grouped.py: 206 cases, rows 2-42, both codebooks, no sharing to every row on one expert), a k = 1 group
is the one-row kernel (no loss without sharing, unlike the old `union`: that one scanned all pairs in every threadgroup, held 8 members at most and loaded x on demand). 1 row keeps the plain kernels.
mu_check, spec_check (ds2, ds2m, ds2e), serve_spec_check exact and the whole suite pass with it on.

`KYOJIN_ROWS_GROUPED=nax` (`grouped="nax"`): groups of `KYOJIN_GROUP_HOTK` (4) or more rows go through the fused trellis-decode + NAX GEMM (`metal_nax.py`, from feat/nax) with a device-side layout (hot rows group
after group, half activations): rounding class (layer rel L2 2e-3 - 5e-3 with the fp8 fake quant), greedy tokens equal to plain on ds2 / ds2m / ds2e, not bitwise (spec_check reports the rounding).
It pays only when many rows hit the same expert; with 50 % overlap over 384 experts it is no better than the GEMV groups, so it stays off unless the routing is that concentrated.

Routed experts of one layer, ms (ds2m layer 0 real shapes 5120 x 2304 top-6, Air, interleaved, min over 7-9 rounds averaged over 4 routings, quiet GPU; bench/bench_mu_experts.py):

| routing | R = 4 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|
| real routing of the slice (32 experts, 4 consecutive tokens x N users), per pair | 6.65 | 12.07 | 19.58 | 25.98 | 67.79 |
| same, old union | 3.54 | 6.55 | 17.89 | 23.59 | 67.30 |
| same, grouped (x per pair) | 2.96 (2.25) | 4.83 (2.50) | 8.29 (2.36) | 10.20 (2.55) | 27.10 (2.50) |
| same, grouped + NAX k >= 4 (x per pair) | 2.81 (2.36) | 4.16 (2.91) | 5.88 (3.33) | 6.02 (4.32) | 14.14 (4.79) |
| 384 experts, independent rows: grouped x per pair | 1.10 | 1.07 | 1.04 | 1.06 | 1.16 |
| 384 experts, each row copies half of the previous row's experts: grouped x per pair | 1.40 | 1.55 | 1.46 | 1.42 | 1.43 |
| same, NAX k >= 4 x per pair | 1.36 | 1.33 | 1.38 | 1.36 | 1.37 |

(The slice has 32 experts: its routing is concentrated, 6 / 10 / 13 / 18 distinct experts for the 24 / 48 / 96 / 192 pairs of R = 4 / 8 / 16 / 32, far above what the 384-expert model shares; the 384-expert rows tile the same 32 experts, so the work per pair is the real one.)
Verify step of the 2-layer slice (tools/mu_step_bench.py, N sessions x (1 + D) rows, ms per pass, per pair -> grouped -> grouped + NAX): 4 rows (N=1 D=3) 21.7 -> 16.0 -> 16.4; 8 rows (N=2 D=3) 38.4 -> 23.9 -> 21.6;
16 rows (N=4 D=3) 71.8 -> 44.6 -> 35.5; 32 rows (N=8 D=3) 135.8 -> 77.7 -> 58.7; logits bitwise equal to the per-pair step. serve_spec_check bench (batched core verify + commit, N sessions x (1 + D)): N=4 D=3 73.2 -> 45.3 ms,
N=2 D=3 38.2 -> 25.7, N=1 D=3 21.7 -> 17.4; the batched draft pass for 4 sessions 93.3 -> 68.0 ms (the draft head's routed experts group too).


### Speculative decode in the batch (`--mtp-depth D`)

For a pack with a draft head (DS4.1: `mtp.*` tensors, `ArchSpec.mtp`; nothing in the server names a model) every round has two passes for ALL active sessions together:

1. One draft pass (`dspark.Drafter.draft_many`, rows = sessions x block of 5). The row-wise ops (EXL3 linears, mHC mixes, routers, routed experts, markov head) give a row the bits it has alone; each
   session attends over its own draft ring. A session's proposal does not depend on the others, on their number or on their order.
2. One verify pass through the batched DS4.1 core (`rows.verify`, the code of a plain decode step): the pending token and the drafts of every session are rows of the pass
   (rows = sum over sessions of 1 + D_i, at most 24), with per-session positions and causal masks.

Each session keeps the longest draft prefix the trunk agrees with plus the trunk's own next token; its cache, Engram history and draft ring roll back to that point
(`kyojin_mlx/rows.py`, `serve/backends.py::BatchedPass`, `serve/engine.py::_spec_tick`). The replay of a rebuilt session runs on the same core.

    tools/mac_serve.sh ~/ds41/model --sessions 4 --mtp-depth 3 --ctx 131072

- Exactness. Plain decode and speculative decode are the same code, so `--mtp-depth` does not change a token: a session's output equals its plain solo run bit for bit, whatever the depth,
  the acceptance, the batch mates, joins, leaves, yields or rebuilds (the old chained verify differed from plain decode on 2 of 8 slice prompts: gone). No rounding class is left between the two
  modes for the trunk. One remains inside the draft head: its row-wise dot products sum in another order than the stock matmul of `tools/spec.py`, so the PROPOSALS can differ from the stock drafter
  at rounding level (`tools/mdraft_check.py`: batched equals one-session passes bit for bit; the stock drafter agrees on a part of the proposals on the noise head of the slice).
  Proposals only decide how much work is wasted: the trunk's verify decides every token.
- Launches. The EXL3 GEMV, the grouped `wo_a`, rowdot and the MoE take the row count in their parameters (row groups of 8 on the grid), so a pass launches the same kernels for 4 and for 16 rows:
  97 custom kernel calls at 16 rows against 95 at 4 (was 278), 300 graph dispatches for 16 rows against 3029 with the chained verify.
- Cost on the 2-layer slice (ds2m, quiet Air, min of 3 runs of 20 reps, ms per verify + commit; old = chained verify of the previous commit):

  | sessions N | rows 1 + D per session = 1 / 2 / 3 / 4 |
  |---|---|
  | 1 (rows 1-4) | old 4.5 / 6.5 / 8.4 / 10.9, now 4.5 / 6.1 / 7.8 / 9.9 |
  | 2 (rows 2-8) | old 6.5 / 10.9 / 16.7 / 21.2, now 6.1 / 9.9 / 14.8 / 18.3 |
  | 4 (rows 4-16) | old 10.7 / 21.3 / 31.9 / 42.9, now 10.3 / 18.8 / 29.3 / 36.7 |

  A plain step costs 4.5 / 6.2 / 10.3 ms for N = 1 / 2 / 4. The routed experts stream their weights once per (row, expert) pair, so the cost per extra row stays: 0.40 solo steps at N=1,
  0.49 at N=4 (16 rows = 3.6 x the 4-row plain step, not 1.5 x: the launch overhead is gone, the bandwidth of the routed experts is not). The draft pass for 1 / 2 / 4 sessions costs
  14.6 / 26.8 / 47.8 ms (the serial calls: 13.8 / 28.4 / 58.3). The slice's 3-block draft head costs 3 slice steps because the trunk has 2 layers; on the 40-layer model it is about 0.15 of a step.
- The depth policy weighs tokens per round against a cost model in solo steps: `--row-cost` (default 0.4 per extra verify row, measured 0.40-0.49 on the slice; the bandwidth model of the Ultra gives 0.34)
  and `--draft-cost` (default 0.15 per draft call; the batched pass costs less per session than a serial call only on the dense part). A plain row of a pass shared by n sessions costs
  `row_cost + (1 - row_cost) / n` solo steps (the dense weights are read once for all rows; measured on the slice: a 4-session plain step is 0.57 of four solo steps), so with several
  sessions a draft has to pay against that cheaper baseline: at acceptance 0.735 one session drafts D = 2, four sessions draft nothing (D = 0, probing with D = 1 every 16 rounds) and
  start to draft again above about 0.9. They are priors: set both from the `tools/spec_gen.py` timings (ms per round for draft and verify against the plain step) before first use
  (`DAY1_ROW_COST`, `DAY1_DRAFT_COST` for `tools/mac_serve.sh`).
- Ultra estimates at 70 tok/s solo, acceptance 0.735 / 0.85 (`tools/spec_est.py` models, and the same arithmetic for several users). Bandwidth model, each distinct routed expert read once for all rows
  (optimistic: the server does not share experts, `union` is off): one user 88 / 104, two users 109 / 126 aggregate, four users 140 / 145. The same with every (row, expert) pair read separately
  (what the server does): 87 / 102, 106 / 121, 138 / 138 (plain batched decode: 70, 104, 138). Air row costs (chained, measured): one user 83 / 93. With four users the rows of the other sessions
  already fill the bandwidth, so speculation adds nothing below an acceptance of about 0.9.
- D adapts per session from its acceptance (moving average, prior 0.7): the engine picks the D with the best expected tokens per unit of cost, D = 0 is allowed
  (a plain 1-row step), and a session at D = 0 probes with D = 1 every 16 rounds. Sampled sessions (temperature > 0) speculate too, with their own key; a pack sets its sampling defaults in `generation_config.json`.
- The real draft head of the 2-layer slice is noise (it accepts nothing), so the exactness runs replace its tokens by a per-session oracle (the true continuation with
  wrong tokens injected: every accepted count 0..D, mixed depths in one pass); the real head still runs.

### Memory budget (`--kv-budget GiB`)

Session caches (active and parked), the batched cache and prefix snapshots count against the budget (default: free unified memory after the load minus 1 GiB;
`/v1/stats` shows `kv_gb`, `kv_peak_gb`, `kv_drops`, `rebuilds`). Over the budget the engine drops the KV of the OLDEST parked (yielded) session, then prefix snapshots;
active sessions are never dropped and no new request starts while the caches are over the budget and something else runs. A dropped session is rebuilt when it resumes:
its prompt is prefilled on the same chunk grid (prefix cache reused), then the tokens it had produced are replayed teacher forced through the decode path, so it continues
exactly where it stopped (never a prefill of prompt + reply, which rounds differently). The budget bounds the tracked KV; the process also holds the activations of the prefill chunk or step in flight (on the 2-layer slice: up to 250 MB at the end of a tick, 4-9 MB once the GPU is synchronized), which the margin has to cover: not measured at DS4.1 size, check on day 1 as `active_gb` minus the weights against `kv_gb` in `/v1/stats`.

Checks: `tests/test_serve_spec.py` (fake backend: spec == plain for every accept pattern, row cap, stop and length inside a round, sampled sessions, depth falling to 0
and probing back, prefix hit with a draft state, budget churn of 8 sessions, negative controls), `tests/test_mrows.py` + `tools/mrows_check.py` (several rows per session in one pass equal the
plain steps, on 9 pack layouts), `tests/test_rows_flat.py` (5..40 rows: same bits as 4-row chunks, 3 launches per EXL3 linear), `tools/mdraft_check.py PACK` (batched draft equals one-session
drafts, with a negative control), `tools/serve_spec_check.py PACK` (the real slice, see its header; `--quick` is the short check).
