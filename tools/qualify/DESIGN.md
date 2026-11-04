# qualify --backend mlx — one command: converted Kyojin-MLX pack -> verdict + qualify.json

NORTH STAR pillar 2 on the Mac. Port of the CUDA/ROCm `tools/qualify` (branch feat/qualify c7bccf239 b1dfbc1f3; same stage names, same `<out>/<stage>.json`
resume files, same `qualify.json` schema 1 and `REPORT.md`, so one report format for CUDA / ROCm / MLX). Planner, rules, max-ctx bisection and
report are copied unchanged (plan.py differs only by per-scale estimates and the skip reasons; rules.py / search.py are byte-identical).

    tools/qualify.sh --pack PACK [--ref BF16_DIR] [--dry-run] [--stage a,b] [--redo a] [--scale air|ultra] [--sessions 1,4] [--mem-cap-gib N]

Output `<repo>/qualify_out/<pack>/mlx/{REPORT.md,qualify.json,<stage>.json,qualify.log,srv_*.log}`. `--scale` defaults by installed memory (< 64 GB = air).
Every GPU job (parity, kv, maxctx, serve+smoke) takes `~/kyojin-mlx/.gpu.lock` (flock = the same lock as `lockf -k`; ONE heavy job at a time), checks free
memory (`memory_pressure`) before it starts, keeps the Mac awake with `caffeinate -ims -w <server pid>` (never -d / -u), samples the server footprint with
`top` (never ps RSS), and fails a probe that exceeds `--mem-cap-gib` (13 on the Air = the macguard rule). Servers are started and killed by PID only;
readiness = `GET /health` == 200; a probe the guard TERMed is reported as "killed by macguard", not as a model failure.

## Stages and which MLX tool each one reuses
| stage | what | MLX implementation (reused tool) | generic / per-arch |
|---|---|---|---|
| fetch | shards vs index (or non-empty `*.safetensors` for single-file packs), license, disk floor, optional `--hf` | pack.py | generic |
| meta | arch family, layers / KV layers / heads, chat template source (jinja, tokenizer_config, DS4.1 built-in encoder), MTP depth, serve command + KV env, machine info (chip, GiB, AC / battery, low-power mode, `nax_status()` when the branch has it) | pack.py `FAMILIES` | generic code, **one table row per arch** |
| autotune | time the kernel candidates of every dense EXL3 linear shape of the pack on this device (decode k-splits, prefill NAX tile and fused-vs-matmul crossover), write `<pack>/tuning/<device>.json`; `load_exl3` / `patch.build_model` read it (no file, other device or kernel version = the constants, bit-identical). Interleaved, min, 5 % gain confirmed twice (plateau rule); the pack is not loaded (random weights of the header shapes, < 1 GiB, ~3 min Air) | `autotune_mlx.py` + `kyojin_mlx/tuning.py`; inventory and limits in `AUTOTUNE_MLX.md` | generic (shapes from the header); MoE expert stacks and multi-user row thresholds not tuned |
| parity | KL (renormalised top-256) / top-1 / dNLL per domain (chat fr / en, tool, wiki, code, fr, tech) of the pack vs the BF16 teacher; converter report summary | macair41's `eval_kl.py` maths (conv_work, feat/conv-mlx), verified against his 16 x 1023 held-out rows (KL .24-.27, top-1 .80-.82 on the 0.8B pack, same as his 0.268 / .8175) | **per-arch**: `refs/<adapter>.py` (qwen35 = mlx-lm load of the BF16 checkpoint, must fit under the cap). None for ds41 / glm5_next: stage skipped, "parity unmeasured" (the DS4.1 check is `ced_gate` + the box reference, DAY1_RUNBOOK 4 / 6) |
| kv | KL of k8v8 / k6v6 / k4v4 (affine pool + fp16 sink 4 / window 128, the server's `KYOJIN_KVWIN`) vs the fp16 cache of the same pack, on the last positions of a long doc, every position decode-faithful; choice = smallest bits >= 6 with KL vs k8v8 <= 0.02 (rules.choose_kv, unchanged) | macair37/43 `kvwin_kl.py` (`cache_for`, `run_logits`, `kl_top1`) | **per-arch**: `kv_mode` kvwin (Qwen3.5 hybrid, full-attention layers only); `native` (DS4.1 ring + compressed pools, GLM MLA): no bits to choose, stage skipped, verdict does not count it unmeasured |
| maxctx | per session count: upper bound from the KV formula (`pack.kv_bytes_per_token`) or `--ctx-cap` for native caches, then `search.bisect_max` (grid 2048): start the server at `--ctx`, 2 distinct near-full prompts (97 %, sized with the server's own `/v1/tokenize`), `sessions` concurrent shares twice, footprint <= cap, no OOM marker, not killed by the guard | `kyojin_mlx.serve.server` | generic (KV formula per arch) |
| serve | ONE load at the chosen `-c` / KV: template conformance (tools before system, rendered with the pack's tokenizer), decode 128 tok x 3, prefill, curve 2K..0.9 ctx, N = 1..sessions aggregate + TTFT, greedy sha x 3 prompts run twice (self-equality), OOM scan; speeds are the server's own `timings` | `serve/server.py` + load.py | generic |
| smoke | real-use scenario B on the same server: main agent + N subs sharing a prefix, tool turns; 3 checks: completed, outputs agree with solo (>= 0.9 before first difference), sub aggregate >= solo | macair43 `realuse_b.py --subs` | generic. Hermes CLI tasks are not available on the Mac: day-1 item |
| report | REPORT.md + qualify.json (adds `machine`, `scale`, `serve.env` = the KV choice as `KYOJIN_KVWIN=k:v:64:4:128`) | report.py | generic |

Gates (advisory, `--gates` JSON overrides): mean KL <= 0.15, domain top-1 >= 0.75, KV >= Q6, max ctx(1 session) >= 32768 (air: min(32768, ctx cap)), sha self-equal,
no OOM, conformance, smoke >= 80 %. Unmeasured items are listed, never counted as pass.

## Generic vs per-arch (summary)
Generic: planner, resume keys, locks / memory guard / PID lifecycle, docs builder, KL maths, KV rule, bisection, client / matrix / sha, smoke, report.
Per arch (one small adapter each): the `FAMILIES` row (server family, KV mode, vision, ref adapter), the BF16 reference forward (`refs/`), the KV formula
(`kv_bytes_per_token`; None = measured), the chunk (`stages.chunk_for`: 2048 for ds41 so CED applies, 512 otherwise) and the MTP depth (`--mtp-depth 3`
only for ds41 packs with `mtp.*`; the Qwen server path has no speculative decode).

## Not in v1 (explicit)
- A streamed (layer by layer) BF16 reference: the qwen35 adapter loads the whole BF16 model, so it is skipped above the cap (Air: ~10 GiB). Needed for any pack whose BF16 is larger than the cap; DS4.1 uses the box reference instead.
- `decode_curve.py` (the flat DS4.1 curve to 256K, min / median / max) is not called by the serve stage: the served curve uses the server's prefill / decode timings. `tools/day1_run.sh` runs it as cell `m.curve`.
- Hermes CLI smoke, GLM-5.3 reference, a truly held-out parity set (the docs use late corpus offsets, as in the CUDA/ROCm version).
