"""Lever flags readable and switchable at call time (one loaded model, A/B without a reload): the serving admin endpoint POST /admin/flags
(kyojin_mlx.serve.server) calls set(). Each lever keeps its KYOJIN_* environment variable as the start value; set() writes the variable back
(the levers that read os.environ per call see it at once) and updates the module attributes the code reads per call.

Compiled decode regions (fuse_layer, the DSpark draft) trace the switches they see: every set() bumps GEN and those regions retrace on their
next call. Not switchable at runtime (listed in docs/cloud/LEVERS.md): the Engram gate (MLX buffer env read when mlx is imported), the
Qwen step regions (built once at install) and KYOJIN_VERIFY_REPLAY (recorded command buffers)."""
from __future__ import annotations

import contextlib
import importlib
import os
import threading
from dataclasses import dataclass, field

GEN = 0          # bumped by every set(): compiled regions compare it with the value they were traced at


@dataclass(frozen=True)
class Flag:
    env: str
    default: str
    doc: str
    runtime: bool = True
    reason: str = ""                       # why not runtime
    kind: str = "bool"                     # "bool" ("0" / "1"), "int", "float", or "str" (free text the flag's own parser validates, "" = off), or "mode" (one of the words in `choices`)
    choices: tuple = ()                    # kind "mode": the accepted values
    attrs: tuple = field(default_factory=tuple)      # (module, attribute, fn(value str) -> attribute value)


def _on(v: str) -> bool:
    return v not in ("0", "")


def _rope_fma(v: str) -> int:
    from . import dsv4_core
    return dsv4_core.resolve_rope_fma(int(v))


def _bind(fn: str):
    """vendor_bind: the archs pin function of that name (it returns the attribute set() writes)."""
    def pin(v: str):
        from . import archs as A
        return getattr(A, fn)(v)
    return pin


def _ablate(v: str):
    from . import ablate
    return ablate.configure(v)


def _tail_reads(v: str):
    from . import metal_src as S
    return S.TAIL_VEC if v == "1" else S.TAIL_ATOM


FLAGS: dict[str, Flag] = {
    "engram_gate": Flag("KYOJIN_ENGRAM_GATE", "auto", "Engram gated decode (1 = on, auto / 0 = off) + one command buffer per step on a DS4.1 pack", runtime=False,
                        reason="sets MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER, which MLX reads once when mlx is imported"),
    "decode_buffers": Flag("KYOJIN_DECODE_BUFFERS", "1", "MLX command buffers of 200 ops / 400 MB instead of 50 / ~50 (KYOJIN_BUFFER_OPS / KYOJIN_BUFFER_MB), except on a DS4.1 pack and on a machine under 24 GiB of RAM (MLX defaults there: the larger buffers hold the prefill temporaries, warm-up out of memory on a 16 GB Air); "
                           "0 = MLX defaults, 2 = also on a small machine",
                           runtime=False, reason="MLX reads MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER once when mlx is imported"),
    "lean_malloc": Flag("KYOJIN_LEAN_MALLOC", "1", "a serve process on macOS re-executes itself once with MallocSpaceEfficient=1: libmalloc returns freed large blocks at once (0.9 GB less footprint after a GGUF load on a 16 GB Air); 0 = off",
                        runtime=False, reason="libmalloc reads MallocSpaceEfficient at process start"),
    "cache_tight": Flag("KYOJIN_CACHE_TIGHT", "1", "MLX buffer cache limit at its floor (working set / 32, at least 256 MB) when the weights leave under 4 GiB below the GPU ceiling "
                        "(16 GB Air, 27B GGUF: 0.66 GB less footprint); 0 = weights / 10 always",
                        runtime=False, reason="the cache limit is set once at load"),
    "desc_strict": Flag("KYOJIN_DESC_STRICT", "0", "loader checks (packcheck.py): a pack that lacks what its modeldesc row / ArchSpec promises (layers, experts, hc, indexer, Engram, "
                        "draft head, CED layout, vendor classes, tokenizer, stop tokens, chat template). 0 = every finding is logged and the load goes on, 1 = PackError "
                        "listing all of them. Header-only reads, no weight is touched", runtime=False, reason="read once, before the model is built"),
    "vendor_bind": Flag("KYOJIN_VENDOR_BIND", "0", "generic engine probe: the per-call vendor lookups of the served hot paths (the fp8 round trip of a MoE / grouped "
                       "linear input, the block collapse of the draft pass, the cache-name test) bind the model's own function and constant once at set() "
                       "instead of resolving archs.sub / archs.cls at every call. 0 = resolve per call, which is what the generic tables do by default. "
                       "Both forms call the same object: no arithmetic, no output change",
                       attrs=(("archs", "_FQ", _bind("pin_fq")), ("archs", "_FQ_DISABLE", _bind("pin_fq_disable")),
                              ("archs", "_HC_PRE", _bind("pin_hc_pre")), ("archs", "_CACHE_NAMES", _bind("pin_cache_names")))),
    "verify_async": Flag("KYOJIN_VERIFY_ASYNC", "0", "served MTP round: one packed readback, async settle / flush, one commit index array "
                         "(hostsync; exact, fewer forced syncs; the compiled rows regions need KYOJIN_ROWS_REGIONS=1)"),
    "gated_verify": Flag("KYOJIN_GATED_VERIFY", "0", "served MTP round: the lazy verify pass goes through the Engram commit gate (backends._verify_gated), so its "
                       "graph is built and encoded while the drafter still runs and the Engram rows are read once, instead of the build stopping at the first "
                       "Engram layer for the hash ids; needs the gate calibrated by the gated plain steps (no gate yet = the pass runs ungated and "
                       "engage.backends.gated_verify_nogate counts it; the bits are the ungated pass's)",
                       attrs=(("serve.backends", "GATED_VERIFY", _on),)),
    "rate_pair": Flag("KYOJIN_RATE_PAIR", "0", "served solo depth rule from the PAIRED measured rate: tokens and ms summed over the same last drafted rounds "
                     "against the same over the last plain rounds (engine._plan_pair), instead of the token mean of the rounds the drafter filled times a "
                     "separate ms median. Both sides are engine measurements, so a faster engine moves both and the ratio decides; a probe adds its round "
                     "to the window, so one unlucky probe cannot re-decide from two rounds"),
    "nondet_fix": Flag("KYOJIN_NONDET_FIX", "0", "run-to-run reproducibility of a T=0 request: the served solo depth policy may not read a measured wall time. "
                       "engine._plan_depth returns the depth cap (the room of the reply, the model's mtp_depth / spec_chain and max_spec_rows) before "
                       "spec_rate / spec_fit / dspol / the measured ladder price a round, so the per-round verify width is a function of the request and the "
                       "acceptance alone. Why it matters: a row's arithmetic moves with the width of the pass (metal_src.HALF_ROWS, Exl3Linear.GEMV_MAX_ROWS = 4; "
                       "open fault, docs/cloud/LEVERS.md spec_ladder), and a measured-ms rule flips that width between 1 and cap from noise, so two runs of the "
                       "same prompt emit different tokens. Cost: the cap is the most rows per round, so the tok/s of the measured policies is given up "
                       "(MODEL: not measured); default 0 keeps today's plan",
                       attrs=(("serve.engine", "NONDET_FIX", _on),)),
    "oracle_draft": Flag("KYOJIN_ORACLE_DRAFT", "0", "serve probe: every session drafts the next `oracle_rows` - 1 tokens of a recorded greedy continuation of its own "
                        "prompt (scratch/oracle.json: {ph: prompt tag, ids: [...]}) instead of the draft head, so the round is a verify pass at that width with "
                        "perfect drafts and no draft cost. The verify, the accept loop and the emitted tokens are the served ones; default 0, a measurement "
                        "of the verify path only (also raises max_spec_rows for its width)"),
    "oracle_rows": Flag("KYOJIN_ORACLE_ROWS", "6", "oracle_draft: verify rows per round (drafts = rows - 1)", kind="int"),
    "spec_chain": Flag("KYOJIN_SPEC_CHAIN", "1", "served MTP round: draft blocks per round. 1 = today. N > 1 chains N block forwards per round: the next "
                       "forward starts from the previous one's last proposal (position p0 + its length) and runs only when that forward filled its plan, "
                       "so a chained round widens a confident run and the confidence gate still cuts it. The drafted prefix is not in the draft ring "
                       "(the block attention reads the ring the last commit left). The verify decides every token", kind="int"),
    "round_log": Flag("KYOJIN_ROUND_LOG", "0", "served decode: one JSON line per speculative / plain round with the depth policy's inputs, its decision, "
                     "the proposals, the gate cut, the accepted rows, the emitted token ids and the round ms (kyojin_mlx/serve/roundlog.py). "
                     "sum(tok)/sum(ms) of the lines is the decode tok/s of the traced rounds, so a plan change can be priced from the round. "
                     "Bounded: KYOJIN_ROUND_LOG_MAX lines (20000) into KYOJIN_ROUND_LOG_PATH (scratch/roundlog.jsonl)"),
    "lazy_fit": Flag("KYOJIN_LAZY_FIT", "0", "served MTP round: the lazy draft decision from the two measured sides (engine, read per round) instead of the "
                     "drafter's cut-rate threshold — lazy when the draft wait the verify build hides (the draft section's off-CPU ms) is larger than the "
                     "rows the confidence gate cuts times the measured GPU ms per row. The verify rows, the accept loop and the tokens are the ones the "
                     "planned depth gives, so the text is the eager path's (spec_lfit_save_ms / spec_lfit_waste_ms in /v1/stats)"),
    "rows_regions": Flag("KYOJIN_ROWS_REGIONS", "0", "compiled mHC / FFN regions of the batched rows pass (fuse_layer; explicit opt-in, not implied by verify_async)",
                         attrs=(("fuse_layer", "ROWS", _on),)),
    "rope_pos_hint": Flag("KYOJIN_ROPE_POS_HINT", "0", "served MTP round: the rope-positions hint of the compressor (`fuse_attn._impl_compress_part`) also on a verify pass (rows 2..64), "
                       "so `language._apply_rope_at_positions` builds its cos/sin table from the python ints that made the positions instead of `int(mx.max(positions).item())` "
                       "— the host no longer drains the whole queue at every closing compression group of every kv-source layer. Same table rows, same values: exact",
                       attrs=(("fuse_attn", "ROPE_POS_HINT", _on),)),
    "glue_fuse": Flag("KYOJIN_GLUE_FUSE", "1", "mHC mix vector + mix args cuts, mix+collapse fold off (gluefuse)"),
    "glue_fuse_norm": Flag("KYOJIN_GLUE_FUSE_NORM", "0", "collapse + RMSNorm + sinkhorn in one launch (glue_fuse), needs glue_fuse. Off: +6 ms per plain step on the M5 Ultra (bare7 30/10, 21.5 -> 27.6 ms; verify2 31.0 -> 38.8)"),
    "alu_mixbar": Flag("KYOJIN_ALU_MIXBAR", "0", "cplain: gluefuse.mixvec (mHC mix vector) row reduction ends in one barrier instead of three: every simdgroup runs the final simd_sum "
                       "simdgroup 0 ran, `inv` stays in registers. Exact (same values, same lanes, same tree); the kernel is rebuilt on the next retrace"),
    "alu_mixpf": Flag("KYOJIN_ALU_MIXPF", "0", "cplain: gluefuse.mixvec loads the weight row one 8-wide step ahead (the first before the row reduction). Loads reordered only: exact"),
    "alu_router1": Flag("KYOJIN_ALU_ROUTER1", "0", "cplain: fuse.router_select top-k with one barrier per selection round (merge of the four simdgroup winners in every thread, "
                        "owner thread retires the pick, double-buffered winners). Same selection order and weights: exact"),
    "alu_noy32": Flag("KYOJIN_ALU_NOY32", "0", "cplain: fuse.mhc_expand does not store the fp32 copy of its output (dead under glue_fuse=1; the stock mix vector casts the fp16 array "
                      "instead, the same value). Exact"),
    "port_suglue": Flag("KYOJIN_PORT_SUGLUE", "0", "fused hyper-connection stages A/B/E: 3 launches per layer instead of 10 (suglue, solo decode). Off: +6 ms per plain step on the M5 Ultra (bare7 30/10, 21.5 -> 27.0 ms; verify2 31.0 -> 38.7)",
                        attrs=(("suglue", "ENABLED", _on),)),
    "launchcut": Flag("KYOJIN_LAUNCHCUT", "0", "fewer launches per token: suglue stages in the chained verify, MoE combine in stage E, router GEMV + top-k in one launch (launchcut; groups via KYOJIN_LAUNCHCUT_SKIP)",
                      attrs=(("launchcut", "ENABLED", _on),)),
    "kvchain": Flag("KYOJIN_KV_CHAIN", "1", "DS4.1 batched core: kv_norm + rope + fp8 round trip + window ring scatter of a pass in ONE launch (dsv4_core). Bit-identical to the stock ops only with rope_fma 1 or 3 (default 3); NOT at rope_fma=0 (the rope pair is two roundings, the stock mx.fast.rope fuses it)",
                    attrs=(("dsv4_core", "KV_CHAIN", _on),)),
    "rows_core_fuse": Flag("KYOJIN_ROWS_CORE_FUSE", "0", "DS4.1 served verify pass (RowsCache with a pass open) through the fused row core: row j of the pass reads the "
                        "session's committed window ring and pool plus the pass's own kv rows (fuse_attn.attend_rows_m / kv_apply_m, the kernels the chained verify of "
                        "the draft head already runs one ring at a time), so no per-row copy of the ring is built for the attention and the ring a commit keeps is built "
                        "once, from the base ring, at the commit. The attention summation order becomes the solo core's (not bit-identical to the batched core: KL / "
                        "error bound); plain decode is untouched",
                        attrs=(("dsv4_core", "ROWS_CORE_FUSE", _on),)),
    "prefix_compact": Flag("KYOJIN_PREFIX_COMPACT", "0", "serve prefix snapshots (memory tier): every array of the snapshot and of its drafter ring is copied into a buffer of exactly its size "
                           "(rows.snapshot / rows.compact, fp4pool.own). A slice of an evaluated array keeps its whole parent alive while nbytes counts the slice: the packed pool / "
                           "index codes `codes[:, :n]` pin the buffer capacity (1/8 slack + step), the drafter ring `rows[:, -w:]` pins window + chunk rows per stage. Same bits; "
                           "the first append after a hit regrows the layer. 0 = the snapshot of 7860fc7",
                           attrs=(("rows", "PREFIX_COMPACT", _on),)),
    "rows_state_nocopy": Flag("KYOJIN_ROWS_STATE_NOCOPY", "0", "DS4.1 served verify pass: no per-row snapshot of the compressor state (kvs / scs). The pool of a "
                          "closing group is read as base state + the pass's own compressor rows in one kernel (dsv4_core.pool_overlay) and the rows a commit keeps are "
                          "written into the state once, at the commit (bit-identical to the snapshot + overlay + pick it replaces)",
                          attrs=(("dsv4_core", "STATE_NOCOPY", _on),)),
    "ropefuse": Flag("KYOJIN_ROPE_FUSE", "1", "DS4.1 batched core: q rope in the attention kernel's q load, inverse rope in kyojin_dsv4_attn_merge (dsv4_core). Bit-identical to the stock ops only with rope_fma 1 or 3 (default 3); NOT at rope_fma=0",
                     attrs=(("dsv4_core", "ROPE_FUSE", _on),)),
    "attn_merge_tail": Flag("KYOJIN_ATTN_MERGE_TAIL", "0", "DS4.1 split decode attention: the merge (sink + inverse rope) rides in the tail of kyojin_dsv4_attn_part (the last threadgroup of a (row, head group) merges, "
                            "arrival counter reset by it): one launch instead of two, bit-identical (dsv4_core.merge_tail_source)",
                            attrs=(("dsv4_core", "ATTN_MERGE_TAIL", _on),)),
    "rope_fma": Flag("KYOJIN_ROPE_FMA", "3", "kvchain / ropefuse / dec_qkv kernels: rope-tail products as fma instead of two roundings. The fused kernels run under fp contract(off); the shipped "
                     "mx.fast.rope is contracted (its AIR: fmuladd(x1, c, (-s) * x2), fmuladd(x1, s, c * x2)). 0 = plain products (the old text, drifts from the stock op in the last "
                     "fp32 bit); 1 = first product fused = the stock form (bit-equal to mx.fast.rope); 2 = second product fused (probe only); 3 = auto: the form the stock op is measured to use "
                     "on this device at the time the flag is read (detect_rope_form; 0 when none matches). tools/kvchain_bits.py checks the result op by op on Metal",
                     kind="int", attrs=(("dsv4_core", "ROPE_FMA", lambda v: _rope_fma(v)),)),
    "dec_stage": Flag("KYOJIN_DEC_STAGE", "0", "DS4.1 served decode block as six stages (decstage; block path for 1..6 rows, bit-identical, falls back outside its shapes); the four sub-flags bisect a stage",
                      attrs=(("decstage", "ENABLED", _on),)),
    "dec_stage_in": Flag("KYOJIN_DEC_STAGE_IN", "0", "dec_stage sub-flag: the `in` stage (mHC mix + attn_norm, one same-input wq_a | wkv group with the fp8 inside the rot, wq_b)",
                         attrs=(("decstage", "IN", _on),)),
    "dec_stage_attn": Flag("KYOJIN_DEC_STAGE_ATTN", "0", "dec_stage sub-flag: the `attn` stage (kv chain + q / inverse rope fuse + fused index score / top-k in the core)",
                         attrs=(("decstage", "ATTN", _on),)),
    "dec_stage_ffn": Flag("KYOJIN_DEC_STAGE_FFN", "0", "dec_stage sub-flag: the `ffn` stage (mHC mix + ffn_norm + sinkhorn in one launch, then router / experts / expand)",
                        attrs=(("decstage", "FFN", _on),)),
    "dec_stage_moe": Flag("KYOJIN_DEC_STAGE_MOE", "0", "dec_stage sub-flag: the `moe` stage (routed MoE decode launches of the FFN stage)",
                         attrs=(("decstage", "MOE", _on),)),
    "dec_stage_glue": Flag("KYOJIN_DEC_STAGE_GLUE", "1", "dec_stage sub-flag: the one-launch mix and expand stages (stagek: 24 mix rows + collapse/norm in one launch, sinkhorn + expand in one; f32, one rounding, row-independent; "
                           "self-checked at first use, off with a printed line if they do not compile or agree). Read only when dec_stage is on; 0 = the lane's glue chain around the stage",
                           attrs=(("decstage", "GLUE", _on),)),
    "dec_stage_rows": Flag("KYOJIN_DEC_STAGE_ROWS", "24", "dec_stage: most rows of a pass the stage path serves (plain step or verify block); wider passes run the chain", kind="int",
                           attrs=(("decstage", "MAX_ROWS", int),)),
    "dec_qkv": Flag("KYOJIN_DEC_QKV", "0", "dec_stage k1-k2 sub-flag: q_norm and the kv chain (kv_norm + rope tail + fp8 + window ring scatter) in ONE launch (stagek2.qkv_stage; the two are independent, "
                    "so no role waits for another; kv part bit-identical to kv_chain, q part f32 with one rounding). Read only when dec_stage, dec_stage_in and the kv chain (dec_stage_attn) are on; self-checked at first use",
                    attrs=(("decstage", "QKV", _on),)),
    "dec_idxq": Flag("KYOJIN_DEC_IDXQ", "0", "dec_stage k1-k2 sub-flag: the attention wq_b and the indexer wq_b read the same fp8 q latent: one same-input group launch instead of two (bit-identical "
                     "where the chain runs the same GEMV kernel, decstage.group_rows_ok). Read only when dec_stage and dec_stage_in are on",
                     attrs=(("decstage", "IDXQ", _on),)),
    "dec_cproj": Flag("KYOJIN_DEC_CPROJ", "0", "decode core, kv-source layers with a compressor ratio > 1: the f32 cast of x and the two f32 compressor projections (wkv, wgate) as ONE launch (stagek2.xgemv: a simdgroup per weight row, "
                      "the weight read in its stored dtype with no f32 copy; f32 FMAs, lane-partial sum order, so within f32 tolerance of the matmul, not bit-equal); self-checked at first use",
                      attrs=(("dsv4_core", "CPROJ", _on),)),
    "idx_fast": Flag("KYOJIN_IDX_FAST", "1", "multi-threadgroup exact indexer top-k + fused candidate mask (idxfast)",
                     attrs=(("fuse_index", "IDX_FAST", _on), ("dsv4_core", "IDX_FAST", _on), ("idxfast", "ENABLED", _on))),
    "idx_fuse": Flag("KYOJIN_IDX_FUSE", "1", "indexer scores + exact top-k + candidate mask in one launch up to 64K keys (idxfuse)",
                     attrs=(("dsv4_core", "IDX_FUSE", _on), ("idxfuse", "ENABLED", _on))),
    "idx_score_fast": Flag("KYOJIN_IDX_SCORE_FAST", "1", "indexer scores on a lane-per-head grid (idxfuse.scores_fast), bit-identical, then the usual exact select "
                         "(decode core when the fused select is gated off)",
                         attrs=(("dsv4_core", "IDX_SCORE_FAST", _on),)),
    "idx_short": Flag("KYOJIN_IDX_SHORT", "1", "decode core, indexer layers whose pooled keys all fit in index_topk (max lens over the rows <= 512 on DS4.1, i.e. a context under ~512 / 1024 tokens): "
                      "the exact selection is the identity, so the wq_b, head weights, query rotary, scores and select are skipped and the index row is built from lens (idxshort); "
                      "bit-identical, per layer, off under the explicit state / verify replay",
                      attrs=(("idxshort", "ENABLED", _on),)),
    "depthflat": Flag("KYOJIN_DEPTHFLAT", "1", "decode cost independent of context: dense mirror of the fp4-packed pool / index keys (O(new rows) per read), 1/8 slack growth (depthflat)",
                      attrs=(("depthflat", "ENABLED", _on),)),
    "kvseg": Flag("KYOJIN_KVSEG", "8192", "prefix snapshots of the fp4 pool / index keys share their full segments: N > 0 = segment of N rows (a multiple of 256 is natural; 0 = off). A snapshot "
                  "copies the segments completed since the last one plus the open tail instead of aliasing the live buffer, whose next append would then copy the whole capacity; the live "
                  "buffer, and so every decode / prefill kernel binding, is unchanged; a hit rebuilds one contiguous buffer. Prefix-cache bytes bill a shared segment once (fp4pool)",
                  kind="int", attrs=(("fp4pool", "SEG", int),)),
    "stackpack": Flag("KYOJIN_STACKPACK", "0", "the served batched decode stacks the fp4-packed pool and index keys as packed codes + scales (stackpack.PackedStack, 288 B / 68 B per row) "
                      "instead of the dense fp16 [S, cap, D] buffers: stack_rows reads the sessions' packed layers (no unpack, no depthflat mirror) and releases them while stacked "
                      "(sync_rows writes them back), _put packs the new rows, the commit-back hands the codes to the session, the attention reads the top set gathered from the codes "
                      "and the index score kernels load keys from the codes. Only for 2 or more sessions (one session keeps the exact current path); a layer that is not packed stays dense. "
                      "Exact bits: same rows, same order",
                      attrs=(("stackpack", "ENABLED", _on),)),
    "extract_raw": Flag("KYOJIN_EXTRACT_RAW", "1", "a one-session cache joins a batch with its packed pool and key layers as they are, no dense read and no re-pack (the O(prompt) pause before the second token)"),
    "tick_agg": Flag("KYOJIN_TICK_AGG", "0", "stderr sum of the sampled engine-thread stacks over a prefill (observability only)"),
    "pf_idx_i8": Flag("KYOJIN_PF_IDX_I8", "1", "prefill plain indexer layers: the block-max first pass runs on int8 operands (FP4 values with one power of two per key and per query head), twice the half matmul2d rate. "
                      "The picked blocks are rescored by the exact half kernel, so scores stay exact; the set could differ only if a rounded (row, head) pushes a true top-k block past the margin. 0 = today",
                      attrs=(("pf_flat", "I8", _on),)),
    "pf_idx_i8_src": Flag("KYOJIN_PF_IDX_I8_SRC", "1", "prefill candidate source layer: the block-max first pass runs on int8 operands as pf_idx_i8. It names pf_idx_i8_src_margin more blocks than "
                          "the source keeps; the kept blocks and the top-k come from the exact scores of those. 0 = half first pass", attrs=(("pf_flat", "I8_SRC", _on),)),
    "pf_idx_i8_src_margin": Flag("KYOJIN_PF_IDX_I8_SRC_MARGIN", "128", "pf_idx_i8_src: blocks picked beyond the candidate count and rescored exactly", kind="int", attrs=(("pf_flat", "I8_SRC_MARGIN", int),)),
    "pf_idx_i8_reg": Flag("KYOJIN_PF_IDX_I8_REG", "1", "pf_idx_i8: the head sum stays in registers (the piece layout of a lane is probed at first use, another layout keeps the slow post)", attrs=(("pf_flat", "I8_REG", _on),)),
    "pf_idx_i8_tpt": Flag("KYOJIN_PF_IDX_I8_TPT", "32", "pf_idx_i8: 128-key tiles per threadgroup", kind="int", attrs=(("pf_flat", "I8_TPT", int),)),
    "pf_idx_i8_margin": Flag("KYOJIN_PF_IDX_I8_MARGIN", "32", "pf_idx_i8: blocks picked beyond topk and rescored exactly", kind="int", attrs=(("pf_flat", "I8_MARGIN", int),)),
    "pf_idx_heads": Flag("KYOJIN_PF_IDX_HEADS", "8", "prefill indexer first pass of a plain layer and of the candidate source: N query heads per row (the N of largest "
                        "|w| * ||q_h|| of that row) instead of all 32, and pf_idx_heads_m blocks kept for the exact half rescoring that follows. Only the candidate set is "
                        "approximate: a kept key's score is the exact all-head one. N in {4, 8, 16} (N of 32 slots, so 32 / N query rows share one matmul2d tile: the tile is "
                        "priced per tile, a narrower one costs the same). Measured on real dumps at Tc = 495616 (idxan3): 16 heads = 50 % of the first pass, top-512 recall "
                        "98.8 / 95.8 / 99.6 % after rescoring M = 2048 candidates; 8 heads = 25 %, 95.9 / 89.5 / 91.2 % at M = 4096. Needs pf_idx_i8 (the subset lives in "
                        "the int8 first pass; off, the pass keeps all 32 heads). 0 = all 32 heads, today",
                        kind="int", attrs=(("pf_flat", "HEADS", int),)),
    "pf_idx_heads_m": Flag("KYOJIN_PF_IDX_HEADS_M", "1024", "pf_idx_heads: 8-key blocks the approximate first pass keeps per row, all of them rescored by the exact all-head "
                         "kernel (0 = the exact path's own topk + margin count)", kind="int", attrs=(("pf_flat", "HEADS_M", int),)),
    "pf_idx_heads_check": Flag("KYOJIN_PF_IDX_HEADS_CHECK", "0", "pf_idx_heads: also run the exact all-head pass on the same chunk and report the top-k recall of what was "
                             "served, per layer (pf_flat.hits heads_hit_l<layer> / heads_total_l<layer>, percent under /v1/stats engage, one stderr line per layer). "
                             "Diagnostic: one extra indexer pass per layer, same bits as the served indices",
                             attrs=(("pf_flat", "HEADS_CHECK", _on),)),
    "pf_idx_capture": Flag("KYOJIN_PF_IDX_CAPTURE", "0", "dump q, keys, w and the exact top-k of plain indexer layers at >= 400K keys for offline analysis (observability only, pf_flat)",
                           attrs=(("pf_flat", "CAPTURE", _on),)),
    "tick_trace": Flag("KYOJIN_TICK_TRACE", "0", "stderr line per decode epoch: batch, graph build, the stacks forced apart, first round (observability only, forces the stack eval)"),
    "cdecpack": Flag("KYOJIN_CDECPACK", "0", "decode reads the fp4-packed pool and index keys, gathers the selected top set (cdecpack): `stack_rows` of a one-session "
                     "decode keeps the session's own codes (no depthflat mirror, no dense stack, a mirror the prefill read built is dropped), `_put` packs the rows an "
                     "emitting pass adds into those codes, the attention gathers the K rows the top-k names and remaps the indices, and the index score kernels load "
                     "the key codes (a reader without a packed form gets a transient unpack, never a mirror). Same bits per row, same order: exact. Removes the "
                     "3.2 KB/token of dense pool + key copies at 1M (~3 GiB), costs one gather + unpack of K rows per layer and step",
                     attrs=(("cdecpack", "ENABLED", _on),)),
    "mirrorq": Flag("KYOJIN_MIRRORQ", "0", "the dense depthflat mirror of a one-session decode is gone at steady state (mirrorq): `stack_rows` builds the dense key stack the "
                    "core reads by unpacking the codes 32K rows at a time straight into it and drops the key mirror, keeps the pool as the session's own packed codes "
                    "(`fp4pool.PackedView`, the cdecpack view contract: `_put` packs, the commit settles, nothing is written back) and drops the pool mirror, and the "
                    "attention decodes the K pool rows its top-k names from the codes with one fused kernel into a slab of idx.size rows, remapping the indices; the "
                    "key write-back takes no read-back, so an append never rebuilds a mirror. Same rows, same bits, same per-step reads: the dense form the attention "
                    "reads is the same array the flag 0 path builds, only no longer retained beside the stack. Prefill keeps the dense form (a whole-width gather) and "
                    "the next stack drops it. Removes the 3.54 GB at 983K (2.82 GB pool + 0.71 GB key mirror), at ~0.65 MB per kv-source layer per row of decoded slab",
                    attrs=(("mirrorq", "ENABLED", _on),)),
    "dec_keys_codes": Flag("KYOJIN_DEC_KEYS_CODES", "0", "decode index scores of a packed-key stack (cdecpack view, stackpack stack: any session count, mu_keep slots) read the fp4 codes in the score "
                           "kernel (dsa_decode.scores_rows_codes: the 8-key tile is unpacked in threadgroup memory, 68 B per key, masked blocks skipped before their codes are read) instead of "
                           "a whole-layer fp4pack.unpack per index layer and step (~5 KB of traffic per key: depthflat.py, 650 MB per 128K-key layer, then the 256 B read). From "
                           "iscore_mma_min keys, fp16 / bf16, 32 heads x 128. Same values as scores_rows(nax=False) over the unpacked keys, bit for bit (self-checked at first use, off with "
                           "a printed line if it does not compile or agree); the flag-0 call takes the NAX form when the device has it, so the head-sum order differs: rounding-level drift. "
                           "Needs a packed key layer (cdecpack / stackpack); a dense stack never reaches it. 0 = today",
                           attrs=(("keyscodes", "ENABLED", _on),)),
    "pf_no_mirror": Flag("KYOJIN_PF_NO_MIRROR", "0", "prefill without the pool mirror: the pooled KV rows the prompt attention gathers are decoded from their fp4 codes in the same "
                      "copy (K1's 4-half row read, K3's 8-dim copy), the append of the chunk takes no read-back (`defer_dense`) and `_compress_part` hands the layer back as a "
                      "`fp4pool.PackedView` (`PackedSlots.lazy`), so a prompt chunk no longer builds the 2.82 GB dense pool mirror of a 983K-token session (the 3.54 GB depthflat "
                      "mirror is 2.82 GB pool + 0.71 GB index keys). Same rows, same bits (tests/test_pf_no_mirror.py asserts the prefill output equal bit for bit); the window "
                      "rows and the stall of the codes are untouched. The key column stays dense here (ds_pf_idx / cpackscan)",
                      attrs=(("pfnomirror", "ENABLED", _on),)),
    "moe_fastdec": Flag("KYOJIN_MOE_FASTDEC", "0", "mul1fd routed gather GEMV decode (not bit-identical, KL-gated)",
                        attrs=(("moe", "FASTDEC", lambda v: v == "1"),)),
    "dense_views": Flag("KYOJIN_DENSE_VIEWS", "0", "dense (non-routed) trellis GEMV / fused / grouped / NAX-verify kernels bind a prefix view of each weight array of 1 MiB or more "
                        "(KYOJIN_DENSE_VIEW_MIN_MB): MLX bills a command buffer for the data_size of every bound array, so the dense weights of a step "
                        "force tens of extra commits; same buffer, bit-identical (cbill)", attrs=(("cbill", "DENSE_VIEWS", lambda v: v == "1"),)),
    "wired_fit": Flag("KYOJIN_WIRED_FIT", "0", "after the weights are loaded, wire the live set plus KYOJIN_WIRED_FIT_MB (default 1024) instead of the whole recommended working set: "
                      "per-step transients no longer grow the MLX residency set (memfit)", runtime=False,
                      reason="applied once after the model load (mx.set_wired_limit)"),
    "cache_fit": Flag("KYOJIN_CACHE_FIT", "0", "after the load, lower the MLX buffer-cache limit to the room under 0.95 of the recommended working set after the live set and "
                      "KYOJIN_CACHE_FIT_MB (default 2048), so the allocator's gc line is not crossed inside malloc (memfit)", runtime=False,
                      reason="applied once after the model load (mx.set_cache_limit); the admin cache_limit_mb sets it at runtime"),
    "wired_fit_mb": Flag("KYOJIN_WIRED_FIT_MB", "1024", "wired_fit: MiB wired above the live set (0 = the rival's zero headroom; raise it for a long context whose KV is allocated after the load)",
                         kind="int", runtime=False, reason="read when the model load finishes"),
    "cache_fit_mb": Flag("KYOJIN_CACHE_FIT_MB", "2048", "cache_fit: MiB kept free under the 0.95 gc line for transients and KV growth", kind="int", runtime=False,
                         reason="read when the model load finishes"),
    "attn_fp16": Flag("KYOJIN_ATTN_FP16", "0", "Absorbed-MLA attention of the skeleton (any model whose activations are fp32, Hy4): fp16 KV buffer in a NEW fixed-state cache (rows saturated to +-65504), attention projections written as fp16, "
                      "q / latent / k_pe / k_b and v_b absorbs / scores (fp32 accumulate) / P @ latent / pooled-key reads on fp16 data, softmax fp32; the hidden stream stays fp32. Half the KV and absorb bytes; "
                      "values differ at fp16 rounding (gate with the model's noise band). Prefill (stock KV cache): the same fp16 path from the first call. Needs a fresh cache after switching. Metal. Read per cache creation / call",
                      attrs=(("attn_fp16", "ON", _on),)),
    "moe_views": Flag("KYOJIN_MOE_VIEWS", "1", "routed MoE kernels bind a prefix view of each expert bank: fewer command-buffer commits, bit-identical (moe)",
                      attrs=(("moe", "VIEWS", lambda v: v == "1"),)),
    "moeroof": Flag("KYOJIN_MOEROOF", "1", "routed MoE decode GEMVs: cheaper integer decode + next-tile prefetch, bit-identical to the stock/FASTDEC decode (moeroof)",
                    attrs=(("moe", "ROOF", _on),)),
    "moeroof_dec": Flag("KYOJIN_MOEROOF_DEC", "1", "KYOJIN_MOEROOF mul1 decode variant: 1 = lane sum by imad, 2 = plus byte pairs by subtract-imad", kind="int",
                        attrs=(("moe", "ROOF_DEC", int),)),
    "moeroof_pf": Flag("KYOJIN_MOEROOF_PF", "1", "KYOJIN_MOEROOF: prefetch the next tile's trellis words in the per-pair gather GEMV",
                       attrs=(("moe", "ROOF_PF", _on),)),
    "moe_tgt": Flag("KYOJIN_MOE_TGT", "0", "routed MoE decode GEMV grid target: threadgroups pick_tgk fills before it stops k-splitting (0 = 480). Another split adds the f32 partials in another order (moe)", kind="int",
                    attrs=(("moe", "TGT", int),)),
    "moe_ksplit": Flag("KYOJIN_MOE_KSPLIT", "0", "per-pair gather GEMV (rows = 1) splits k by the expert's bitrate, equal bytes per threadgroup: 1 = lower bitrates split less, 2 = higher bitrates split more (moe)", kind="int",
                       attrs=(("moe", "KSPLIT", int),)),
    "moe_chain": Flag("KYOJIN_MOE_CHAIN", "1", "routed MoE decode chain in 3 launches per layer: mid rides the tail of the gate/up gather GEMV and the combine the tail of the down gather GEMV "
                      "(last-arriving threadgroup of a 128-column block runs the stand-alone mid / final statements; per-pair gather path, no union / KSPLIT); bit-identical (moe, moe_sumoe)",
                      attrs=(("moe", "CHAIN", lambda v: v == "1"),)),
    "gather_roof": Flag("KYOJIN_GATHER_ROOF", "0", "routed MoE decode gate/up gather GEMV at rows = 1 as 128-thread threadgroups (one column half each, 2x the threadgroups): the decode is ALU-bound and the 256-thread launch "
                        "loses ~7 % to its tail wave; same partials bit for bit (moe)",
                        attrs=(("moe", "GATHER_ROOF", lambda v: v == "1"),)),
    "routed_fx": Flag("KYOJIN_ROUTED_FX", "0", "routed MoE decode chain in 2 launches per layer: the rot (sign, 128-Hadamard, fp8 fake quant) runs in the prologue of the gate/up gather GEMV, each threadgroup rotating its own k slice into threadgroup memory; "
                      "falls back to the 3-launch chain when the slice does not start on a 128-block; bit-identical (moe)",
                      attrs=(("moe", "ROUTED_FX", lambda v: v == "1"),)),
    "moe_finpar": Flag("KYOJIN_MOE_FINPAR", "0", "routed MoE combine launch: one simdgroup per routed pair (threadgroup per 128-block), pair terms added in order j by simdgroup 0; bit-identical to the serial loop (moe, moe_sumoe)",
                       attrs=(("moe", "FINPAR", lambda v: v == "1"),)),
    "dense_fast": Flag("KYOJIN_DENSE_FAST", "1", "fused dense tail reads one float4 per k-split",
                       attrs=(("kernels", "TAIL_READS", _tail_reads),)),
    "skinny_rows": Flag("KYOJIN_SKINNY_ROWS", "0", "skinny products of a few-row pass (hyper-connection mixes, MoE router logits of the generic foreign skeleton; the chained verify rows of a speculation round): "
                        "2 <= rows <= this value = ONE rowinv.rowdot launch (fp32 accumulate, row-invariant) instead of mx.matmul, which switches from the gemv kernel (1 row) to the steel GEMM + split-K accumulate pair (2+ rows). 0 = off, one row and prompt chunks keep mx.matmul",
                        kind="int", attrs=(("rowinv", "SKINNY_ROWS", int),)),
    "dense_dec": Flag("KYOJIN_DENSE_DEC", "0", "dense (non-routed) trellis GEMV decode cut, read per kernel build: 1 = lane sum by one imad (dec_mul1r1, "
                      "the same fp16 value bit for bit), 2 = dec_mul1fd bare byte sum with the affine folded out (fast, NOT bit-identical; KL gate)",
                      kind="int", attrs=(("metal_src", "DENSE_DEC", int),)),
    "gemv_pf2": Flag("KYOJIN_GEMV_PF2", "0", "decode GEMV prefetch depth: depth 2 keeps TWO k tiles of the weight in flight (tile i+2 issued before tile i decodes) instead of one, "
                     "in the dense trellis GEMV (gemv_source) and the routed MoE per-pair gather GEMV; only the loads are reordered, bit-identical (read at kernel build, part of the cache keys)",
                     attrs=(("metal_src", "PF2", _on),)),
    "moe_sgred": Flag("KYOJIN_MOE_SGRED", "0", "grouped routed MoE GEMV (rows >= 2, verify): each simdgroup stores its own k-split partial straight to the partial buffer, no threadgroup "
                      "reduction buffer and no barrier; mid / final sum TGK * 4 partials (moe; fp32 summation order only, not bit-identical, f32-reference gated)",
                      attrs=(("moe", "SGRED", lambda v: v == "1"),)),
    "fuse_rows": Flag("KYOJIN_FUSE_ROWS", "8", "rows of the one-launch fused dense linears (rot + GEMV + finalize, same-input groups, grouped wo_a): 0 = 4 as today, 5 .. 8 = a "
                      "verify block of that many rows takes the same fused launch as a plain step instead of the three-launch many-rows path, the S5 groups stay grouped and the "
                      "8 wo_a slices are one grouped launch set (kernels, fused_group, grouped, patch; a fused block keeps the row FMA class of its rows, so with HALF_ROWS 4 the 5..8 row "
                      "launch runs the packed-fp16 row FMAs of the 4-row class; read per call)", kind="int",
                      attrs=(("kernels", "FUSE_ROWS", int),)),
    "port_sumoe": Flag("KYOJIN_PORT_SUMOE", "0", "shared expert rides the routed MoE decode launches, 5 launches per layer (moe_sumoe)", attrs=(("moe_sumoe", "SUMOE", lambda v: v == "1"),)),
    "moefuse": Flag("KYOJIN_MOEFUSE", "0", "routed MoE decode in fused launches (moe_fused)", attrs=(("moe_fused", "MOEFUSE", lambda v: v == "1"),)),
    "moegrp": Flag("KYOJIN_MOEGRP", "0", "grouped MoE GEMV: one packed slot range and a body per member count, so a group of 5..7 rows no longer "
                   "pays the idle rows of the 8-row pass and the class ranges launch no empty threadgroups (bit-identical)",
                   attrs=(("moe", "MOEGRP", _on),)),
    "rows_ugrouped": Flag("KYOJIN_ROWS_UGROUPED", "3", "rows of a batched / verify pass up to here run the routed experts through the per-pair split-K gather GEMV "
                        "(the 1-row kernel, pairs of all rows in one launch) instead of the grouped GEMV: an expert that receives only one or two rows has nothing to share, "
                        "and the grouped body's threadgroup reduction costs more than the k-split. Same pairs, same tgk (pin_rows=1) and same summation order, so the bits are "
                        "the grouped path's (tests/test_moe_grouped.py::test_grouped_equals_gather); 0 = off", kind="int",
                        attrs=(("moe", "UGROUPED", int),)),
    "pf_glu": Flag("KYOJIN_PF_GLU", "1", "prefill MoE: device sort + double copy (moe_prefill)", attrs=(("moe_prefill", "GLU", _on),)),
    "pf_split_gu": Flag("KYOJIN_PF_SPLIT_GU", "0", "prefill MoE: split gate/up GEMM (measured slower end to end)",
                        attrs=(("moe_prefill", "SPLIT_GU", _on),)),
    "pf_moe_rowmap": Flag("KYOJIN_PF_MOE_ROWMAP", "0", "prefill MoE: gate/up GEMM reads x through the slot -> token map with the input transform "
                         "(suh, H128, k permutation) in the tile load, no [2P, K] fp16 rot copy (moe_prefill, metal_nax.coop_rm_source)",
                         attrs=(("moe_prefill", "MOE_ROWMAP", _on),)),
    "pf_moe_epi": Flag("KYOJIN_PF_MOE_EPI", "0", "prefill MoE: gate + up + SwiGLU + routing weight in one kernel epilogue, no [2P, inter] product "
                      "written then read (metal_nax_gu)", attrs=(("moe_prefill", "MOE_EPI", _on),)),
    "pf_moe_nosync": Flag("KYOJIN_PF_MOE_NOSYNC", "0", "prefill MoE: no per-layer `cum` readback (the sort's expert offsets stay on the device, the "
                        "row-block GEMMs loop over their expert's rows and the per-class buffers are sized from the pair count, not from the exact "
                        "per-class counts); bit-identical to the sync path, same tiles", attrs=(("moe_prefill", "MOE_NOSYNC", _on),)),
    "pf_route": Flag("KYOJIN_PF_ROUTE", "0", "prompt-path router (rows.py): the gate matmul then ONE selection launch (fuse.router_select) instead of "
                    "the stock sqrt / softplus / bias / neg / argpartition / take / sum / div chain, about 8 launches per layer fewer; same ids, "
                    "same weights bit for bit", attrs=(("rows", "PF_ROUTE", _on),)),
    "glm_dsa_pf": Flag("KYOJIN_GLM_DSA_PF", "0", "GLM-5.x prompt chunks (> 64 rows, batch 1): the DSA attention runs in latent space on the DS4.1 NAX gathered-key kernels (attn_prefill, one shared "
                       "K = V table of 512, q_lat = embed_q(q), unembed_out after the sum) instead of projecting K / V for all 64 heads and a SIMT kernel (stock: re-projected from the whole latent above 32768 tokens). "
                       "Same top-k ids and scale; fp16 probabilities, fp32 accumulate (rel ~1e-3). Needs a NAX GPU (KYOJIN_ATTN_PF). Read at install (load time)", runtime=False, reason="class-level install at load"),
    "glm_dsa_fused": Flag("KYOJIN_GLM_DSA_FUSED", "1", "GLM-5.x DSA prompt attention (with glm_dsa_pf on): ONE fused flash-style NAX kernel (adapted from mlx-serve, MIT, itself from oMLX, Apache-2.0: scores, online exp2 softmax and P x V in one "
                        "launch, key rows read through the ids, whole dead tiles skipped) instead of the two attn_prefill kernels with a [rows x heads, keys] fp16 P matrix in between. 0 = the two-kernel path. Read at import", runtime=False, reason="module constant"),
    "glm_pf_hc": Flag("KYOJIN_GLM_PF_HC", "0", "GLM-5.x prompt chunks (> 8 rows, batch 1) mHC glue: the collapse mix (fp32 copy + rms_norm + 32 tiled fp32 matmuls of 24 columns) is ONE simdgroup-MMA kernel reading x in fp16 "
                      "(weights split hi / lo half, fp32 accumulate, ~1e-6 relative), the expand is the DS4.1 prompt kernel (pf_glue.expand, saturates at the fp16 range). KYOJIN_GLM_PF_HC_MIX=0 / _EXPAND=0 switch one seam off. Read at install (load time)",
                      runtime=False, reason="class-level install at load"),
    "glm_pf_idx": Flag("KYOJIN_GLM_PF_IDX", "0", "GLM-5.x prompt chunks (> 64 rows, batch 1, fp16): the indexer's pool selection (`_exact_pool_select`) runs the DS4.1 fused NAX score kernel (32 heads x 128, product kept fp32, fp32 head "
                      "weights) and the exact multi-threadgroup radix top-k (idxfast.topk_mt) instead of [512, 32, pools] fp32 temporaries + argpartition. Same selected pools up to ties within fp32 accumulation order. Read at install (load time)",
                      runtime=False, reason="module-level install at load"),
    "glm_pf_untile": Flag("KYOJIN_GLM_PF_UNTILE", "0", "GLM-5.x prompt chunks (> 256 rows, batch 1): mlx_vlm's `tiled_linear` (256-row slices, one tiny GEMM each, for a fixed reduction geometry) is one call for the five projections that use it "
                          "(MoE router fp32 [4096 -> 288], indexer wk / pool gate [4096 -> 128], wq_b, weights_proj): 32 GEMMs of 256 rows at 1-13 TFLOPS become one (G5g, bench 8192 rows: router 2.0 -> 0.5 ms, wk 2.2 -> 0.24, "
                          "pool gate 1.9 -> 0.3). Same K order per row; outputs within fp32 / fp16 rounding (cut greedy check: argmax identical). Read at install (load time)", runtime=False, reason="class-level install at load"),
    "glm_pf_moe": Flag("KYOJIN_GLM_PF_MOE", "0", "GLM-5.x MoE blocks on prompt chunks (> 16 rows): the router select (sigmoid + bias + top-k + normalise, stock = argpartition + take_along_axis + sum + divide) "
                       "is one launch and the combine (stock = multiply pass writing [T, K, H] fp16, sum pass reading it back, + shared) is one launch (glm_moe.combine, the decode kernel). Same "
                       "fp16 roundings as the stock `_combine`; read per call."),
    "glm_pf_q8": Flag("KYOJIN_GLM_PF_Q8", "0", "GLM-5.3 MoE prompt chunks (>= KYOJIN_GLM_PF_Q8_MIN slots, default 16384; needs KYOJIN_KQ_GMM + KYOJIN_KQ_GMM_GLM): the routed-expert GEMMs (gate, up, down) run int8 x int8 -> int32 on the "
                      "NAX (matmul2d): activations quantized per row and per 128-wide K block (fp32 partial sums scaled per block), the NVFP4 weight decoded to int8 per K step in threadgroup memory (one scale per "
                      "output row, round half even). Stock = the fp16 kqgmm path. Falls back (fired.fb \"pf_q8\": flag_off / no_swiglu_limit / kq_gmm_glm_off / <proj>:not_experts, fmt=, rowgs, k%256, n%64 / "
                      "E>1024 / slots<min) to it. NOT validated by a full-model PPL / KL gate; read per call."),
    "pf_gemm_i8": Flag("KYOJIN_PF_GEMM_I8", "0", "Prefill GEMM numerics, routed experts (op name of KYOJIN_GLM_PF_Q8, which stays an alias: either one on = on): GLM-5.3 MoE prompt chunks run the expert GEMMs int8 x int8 -> int32 (foreign_pfq8). Read per call: off needs no reload. Costs ~0.04-0.1 % wiki PPL."),
    "pf_gemm_i8_dense": Flag("KYOJIN_PF_GEMM_I8_DENSE", "0", "Prefill GEMM numerics, dense linears (attention / KDA projections, shared expert, dense MLP; any stored format): prompt chunks of >= KYOJIN_PF_GEMM_I8_MIN rows (default 2048) run int8 x int8 -> int32; the weight is quantized to int8 ON THE FLY per call (transient, no resident copy: foreign_pfi8). Read per call: off needs no reload. Quality cost: see STATUS_PFQ8V3.md."),
    "pf_gemm_i8_row": Flag("KYOJIN_PF_GEMM_I8_ROW", "0", "Prefill GEMM numerics, sub-flag of pf_gemm_i8: ONE activation scale per row (v1 scheme) instead of one per 128-wide K block in the routed-expert GEMMs: ~12 % faster MoE layer (30.9 vs 35.1 ms), 2-5x the error on rows with an activation outlier. Read per call."),
    "glm_kda_pf": Flag("KYOJIN_GLM_KDA_PF", "0", "GLM-5.3 prompt chunks (> 64 rows, batch 1, plain cache) of the 34 KDA linear-attention layers (G5d): the conv + silu + L2 norm of q / k prework is ONE kernel (stock: concatenate, conv1d, "
                      "silu, rms_norm x2, cache slice), the delta-rule scan is kyojin_mlx.kda_nax (chunks of 16 tokens on the matrix units, the per-channel gate evaluated in the kernel, fp32 state; stock: sequential kernel + an fp32 "
                      "[T, 64, 128] gate tensor) and o_norm * sigmoid(gate) is ONE kernel. Output within bf16 / fp16 rounding of the stock path (G5e: the matrix unit truncates fp32 operands to 10 mantissa bits, so every state-dependent MMA of the scan runs as a 3-term hi/lo product, "
                      "KYOJIN_GLM_KDA_SPLIT=127: state error vs float64 4e-6 relative instead of 1e-3). KYOJIN_GLM_KDA_PF_REC=stock keeps the stock scan, _PRE=0 the stock conv chain. Read at install (load time)", runtime=False, reason="class-level install at load"),
    "router_w16": Flag("KYOJIN_ROUTER_W16", "0", "decode router (rows.py): the fp32 gate is read through an exact bf16 / fp16 copy (lowp.exact_lowp: kept only when every value "
                       "round-trips), half the gate bytes per MoE layer, same logits bit for bit (rowdot upcasts each element to fp32 before the first fma); fp32 stays for a gate that is not exact",
                       attrs=(("rows", "ROUTER_W16", _on),)),
    "pf_rdadvise": Flag("KYOJIN_PF_RDADVISE", "0", "Engram prefill gather (engram_rows): F_RDADVISE each run of rows before the memory map take, one "
                       "advisory per run of rows within 16 of each other, skipped when a request scatters into more than 4096 runs; same bytes"),
    "gguf_fp16": Flag("KYOJIN_GGUF_FP16", "1", "foreign formats (foreign_linear.load_foreign, GGUF): 1-D F32 tensors (norm weights, biases; ggml always writes them F32) load in the model dtype, "
                      "so Q and K stay fp16: prefill attention 1.4x faster, decode reaches the fp16 flash kernel. gdn_norm's fp32 weight stays. 0 = kept F32, as before",
                      runtime=False, reason="read once, at load"),
    "act_dtype": Flag("KYOJIN_FOREIGN_ACT_DTYPE", "", "foreign formats (load_foreign, GGUF): the F32 non-quantised parameters left after the 1-D rule (conv1d weights of the gated-delta and PLE blocks, "
                      "2-D norms / projections kept dense) load in the model compute dtype (bf16 | fp16 | auto = the load dtype, fp16): one F32 weight multiplied into a half hidden state promotes it, "
                      "and every later op ran fp32. A_log / dt_bias stay F32 (the class computes the gate in fp32). Unset = off; PPL differs slightly, never bit-identical",
                      runtime=False, reason="read once, at load"),
    "gguf_generic": Flag("KYOJIN_GGUF_GENERIC", "0", "RETIRED, no effect: a bare GGUF of a mapped architecture without a modeldesc row (qwen2, qwen2_moe, llama, glm4, glm4_moe, gemma3, mistral3, deepseek_v3) is "
                         "served through the generic mlx-lm loader + Q35Backend whatever this says (only files that were refused before are affected; every pack that served is unchanged)",
                         runtime=False, reason="read once, at load"),
    "pf_warm0": Flag("KYOJIN_PF_WARM0", "0", "serve engine, Engram prefill: the rows of the FIRST chunk of a prompt are read into the page cache when the request is admitted (own worker), "
                     "as the chunk-to-chunk warm already does for chunks 1+. Chunk 0 reads its distinct hashed rows cold (content dependent: a repeated passage touches few rows, "
                     "a real document or random ids up to ~196k per layer at 8192 rows); hint only, same bytes. 0 = today"),
    "engram_adopt": Flag("KYOJIN_ENGRAM_ADOPT", "0", "serve engine, Engram prefill: when a request is admitted, the rows of its FIRST chunk (after the cached prefix) are hashed on the host (RowMap, "
                         "history = the cached tokens before the cut) and the real row jobs are submitted at once, on a worker; the chunk adopts them when its device hashes "
                         "equal the host's bit for bit (no second read, no hint), else it reads as before (other cache hit, image tokens, other chunk bounds). Replaces pf_warm0 "
                         "(a hint that competed with the chunk's own reads). Stats engram_adopt_hit / _miss / _made, t_pf_eng0_first_adopt|cold. KYOJIN_ENGRAM_ADOPT_DEBUG=1 re-reads and asserts. 0 = today"),
    "pf_dec": Flag("KYOJIN_PF_DEC", "2", "prefill NAX GEMMs (routed experts, gate/up epilogue, dense shared stacks) of an exact mul1 pack: 1 = trellis decode by dec_mul1r1 "
                  "(lane sum by one imad), 2 = dec_mul1r2 (also the byte pairs by a subtract-imad); the same fp16 weight bit for bit, fewer integer ops per decoded weight "
                  "(read at kernel build)", kind="int", attrs=(("metal_nax", "PF_DEC", int),)),
    "pf_pipe": Flag("KYOJIN_NAX_PIPE", "0", "prefill NAX GEMMs (routed experts): the trellis decode of the next k step is written into a SECOND right-input tensor and the "
                  "matmul of the current step is issued behind it, so the decode ALU runs while the tensor unit works instead of alternating with it; same tiles, same decode, "
                  "same k order into the same accumulators, bit-identical (read at kernel build)", attrs=(("metal_nax", "PIPE", _on),)),
    "pf_unroll2": Flag("KYOJIN_NAX_UNROLL2", "0", "prefill NAX GEMMs (routed experts, register path): the k loop is unrolled by two steps with two alternating trellis word sets, so the "
                      "24 register moves that carry the prefetched words into the next step disappear; same loads, same decode, same k order into the same accumulators, "
                      "bit-identical (read at kernel build)", attrs=(("metal_nax", "UNROLL2", _on),)),
    "pf_bmu16": Flag("KYOJIN_PF_BMU16", "0", "routed prefill GEMMs (plain pair only, not pf_moe_rowmap / pf_moe_epi): pick_cfg's row block is cut into 16-row matmul runs instead of 32-row ones "
                    "((32, n) becomes (16, 2n), same rows per block, same accumulator registers, same decode per block), so an expert's tail pads to 16 rows and the empty runs are "
                    "skipped; only the matmul tile M changes, md5 against the stock kernel is checked by tools/cpftrellis_gemm.py (read per pick)",
                    attrs=(("metal_nax", "BMU16", _on),)),
    "pf_blk128": Flag("KYOJIN_PF_BLK128", "0", "routed prefill GEMMs: a 128-row block (32 x 4 sub-blocks, 16-column tiles) joins pick_cfg's candidates, so an expert with 128+ rows "
                     "decodes its trellis weights half as often per row; not used with pf_moe_rowmap / pf_moe_epi; same bits, only the tiling changes (read per pick)",
                     attrs=(("metal_nax", "BLK128", _on),)),
    "pf_tail_grp": Flag("KYOJIN_PF_TAIL_GRP", "0", "prefill: a routed-MoE row band wider than one decode chunk (4 rows) and below the S3 floor runs as ONE grouped call per layer "
                      "(one trellis decode per distinct expert, one wide group plan) instead of ceil(rows / 4) four-row calls. The CED decoder tail (128 rows x top-6 = 768 pairs) "
                      "is the served case; same per-row kernel bodies, the k-split (tgk) of the wide call differs, so fp32 partial order only (moe, patch._moe_rows)",
                      attrs=(("moe", "TAIL_GRP", _on),)),
    "pf_tail_c4": Flag("KYOJIN_PF_TAIL_C4", "0", "prefill: a routed-MoE row band that the four-row decode chunks serve today (rows above the chunk step, EL.ROW_SPLIT_MAX (the multi-user / verify rows, "
                      "mtp_rows once the draft is armed) and pld.MAX_DRAFT + 1, and below the S3 floor pf_min_rows) runs as ONE grouped call per layer (one trellis decode per distinct expert, "
                      "wide group plan, up to KYOJIN_PF_TAIL_PAIRS pairs); needs a grouped model with hotk 0. Unlike pf_tail_grp it never takes a call S3 serves, so the 128-row CED tail of a "
                      "384-expert top-6 NAX pack (floor 112) is not taken; prompt calls of ROW_SPLIT_MAX + 1 .. floor - 1 rows are (a short suffix, a small last chunk), and every call where the floor "
                      "is above the band (no NAX path, NAX under 32 cores). Not exact: a pair's fp16-row body depends on the rows of its call (HALF_ROWS class, KL-gated), and a remainder chunk of "
                      "rows % step != 0 rows runs other kernels with another k-split; equal to the chunks bit for bit only if HALF_ROWS=0, rows a multiple of the step, and the Mac bench says so. "
                      "Not timed on Metal (moe, patch._moe_rows)",
                      attrs=(("moe", "TAIL_C4", _on),)),
    "cpfmoe": Flag("KYOJIN_CPFMOE", "0", "prefill MoE, needs pf_moe_nosync: the passes of a multi-pass call (the served 8192-row chunk is two 4096-row passes) are submitted with async_eval "
                  "instead of eval, so the host encodes pass k + 1 while the GPU runs pass k (a nosync pass has no readback to wait for). Same kernels, same inputs, same order: "
                  "bit-identical. About one extra pass of pair buffers in flight. Not measured (moe_prefill)", attrs=(("moe_prefill", "CPFMOE", _on),)),
    "pf_moe_hot_fused": Flag("KYOJIN_PF_HOT_FUSED", "0", "prefill MoE: an expert with more rows than KYOJIN_PF_FUSED_MAX (reachable only with pf_moe_chunk > 4096) stays in the device-plan "
                            "fused launch instead of sending the whole layer back to the host plan (moe_prefill; not bit-identical for such experts)",
                            attrs=(("moe_prefill", "HOT_FUSED", _on),)),
    "pf_moe_bail_split": Flag("KYOJIN_PF_BAIL_SPLIT", "0", "prefill MoE: a pass whose device plan finds an expert above KYOJIN_PF_FUSED_MAX rows (only a pf_moe_chunk > 4096 pass can) reruns "
                             "as passes of at most that many rows on the device plan instead of falling back to the host plan (host argsort, ids sync, group path); the passes are "
                             "exactly what pf_moe_chunk=4096 runs, so the rows keep the fused-kernel bits (moe_prefill)", attrs=(("moe_prefill", "BAIL_SPLIT", _on),)),
    "pf_nosync_y": Flag("KYOJIN_PF_NOSYNC_Y", "0", "pf_moe_nosync: threadgroups per (tile, expert) sharing that expert's row blocks in the persist GEMM (0 = 8). An expert with more than "
                       "this many blocks loops, so a hot expert under skewed routing ends the launch long after the others; threadgroups past the expert's rows exit at once. "
                       "Same tiles, same per-row arithmetic (read at kernel build and at launch)", kind="int", attrs=(("metal_nax", "NOSYNC_Y", int),)),
    "pf_nosync_rowend": Flag("KYOJIN_PF_NOSYNC_ROWEND", "0", "pf_moe_nosync: the gate / up persist GEMM closes the last gate expert's row range at its class end. The glued "
                            "range [gate | up] ends that expert at class start + P, so it walks every slot up to there (garbage rows nothing reads) on 8 threadgroups at "
                            "every class of every layer. Same real rows, same tiles, same per-row arithmetic; the garbage rows are no longer computed (moe_prefill)",
                            attrs=(("moe_prefill", "NOSYNC_ROWEND", _on),)),
    "pf_moe_epi_cap": Flag("KYOJIN_PF_EPI_CAP", "0", "pf_moe_epi: allocate the epilogue kernel's output at the bucketed row count (moe_prefill.cap_rows) like the other pair buffers "
                          "instead of the class's exact pair count. The count changes at every layer and the MLX buffer cache reuses a buffer only within 2 pages of the "
                          "request, so the exact size is a fresh allocation per class per layer. Rows past the count are never written or read: same bits",
                          attrs=(("moe_prefill", "EPI_CAP", _on),)),
    "pf_moe_rm_rows": Flag("KYOJIN_PF_RM_ROWS", "0", "pf_moe_rowmap / pf_moe_epi row map: only for a bitrate class whose experts average at most this many rows (0 = every class). "
                          "The row map redoes the input transform in every N-group threadgroup and the epilogue kernel stages one 32-row block, so the copy it saves "
                          "stops paying when experts hold many rows; above the limit the class takes the rot copy. Same bits (moe_prefill)", kind="int",
                          attrs=(("moe_prefill", "RM_ROWS", int),)),
    "dspol": Flag("KYOJIN_DSPOL", "1", "confidence-head draft width policy in solo rounds (engine reads the env per round)"),
    "dspol_v2": Flag("KYOJIN_DSPOL_V2", "0", "ONE calibrated draft policy instead of the confidence gate plus the depth cut (kyojin_mlx.dspol2), solo rounds. p_k = P(accept draft k | "
                    "earlier drafts kept), calibrated on bins of -log10(1 - conf) with a small fitted prior and per-bin forgetting; rows R = 1 (plain step) .. block + 1 "
                    "maximise E[T](R) / cost(R) with the measured width table scaled by the live plain-step ms, ties within 2 % to the cheaper R. The drafter's "
                    "threshold gate is off (every drafted row is verified and counted). 1 = R chosen at plan time from the per-position confidence track (the lazy "
                    "draft overlap is kept); 2 = also re-chosen after the draft from this round's own confidences (eager draft: the verify is built at the chosen R). "
                    "Exact for any R (the verify decides every token). Costs KYOJIN_DSPOL_V2_COSTS, prior KYOJIN_DSPOL_V2_PRIOR; round_log gains pk / R / et / cms. Read per round",
                    kind="int"),
    "spec_rate": Flag("KYOJIN_SPEC_RATE", "1", "solo depth: the cap unless its measured tokens per ms fall below plain's (0 = the predictor x cost policies; read per round)"),
    "spec_fit": Flag("KYOJIN_SPEC_FIT", "1", "solo depth policy prices a round from the measured drafted-round ms per depth (0 = the spec_k formula; read per round)"),
    "spec_full": Flag("KYOJIN_SPEC_FULL", "0", "served MTP round: accept every draft the verify pass already paid for, not only the confidence-gated prefix; the depth "
                     "policy's tokens per cap round and per-position acceptance then see every verified row (read per round)"),
    "spec_rs": Flag("KYOJIN_SPEC_RS", "0", "sampled sessions: draw the drafts from a tempered top-k draft q and accept with min(1, p/q), a rejected row emits from the "
                     "residual max(0, p - q) (kyojin_mlx.specsam; exact target distribution, acceptance 1 - TV(p, q); greedy sessions unchanged; read per round)"),
    "spec_typical": Flag("KYOJIN_SPEC_TYPICAL", "0", "sampled sessions, MTP / lookup drafts (not spec_rs): a draft row is also accepted when its target probability p >= "
                         "min(spec_typical_eps, spec_typical_delta * exp(-H)), H the row's entropy under the sampling distribution (relaxed 'typical' acceptance, Medusa; "
                         "NOT the exact target distribution; greedy sessions unchanged; the mask is built on device in the sampling graph; read per round)"),
    "spec_typical_eps": Flag("KYOJIN_SPEC_TYPICAL_EPS", "0.09", "spec_typical: absolute probability bound of the rule (read per round)", kind="float"),
    "spec_typical_delta": Flag("KYOJIN_SPEC_TYPICAL_DELTA", "0.3", "spec_typical: entropy-scaled bound delta * exp(-H) of the rule (read per round)", kind="float"),
    "spec_typical_lookup": Flag("KYOJIN_SPEC_TYPICAL_LOOKUP", "rep", "spec_typical, repetition control (read per round): rep = the relaxed rule applies to MTP drafts and to lookup drafts, but a draft token that "
                                 "would complete a 12-gram already seen twice in the last 2048 tokens of the session's OWN output is accepted by exact equality only (copies of the prompt "
                                 "and of tool results stay relaxed; stat spec_typical_rep_refused counts the refusals); 0 = lookup rounds accept by equality only (measurement arm); "
                                 "1 = the relaxed rule everywhere with no repetition check (measurement arm)", kind="mode", choices=("rep", "0", "1")),
    "spec_typical_hmax": Flag("KYOJIN_SPEC_TYPICAL_HMAX", "", "spec_typical, entropy gate (read per round): when set (nats), the relaxed rule applies only to rows whose entropy H under the sampling distribution is <= hmax; "
                              "at a higher-entropy row (a real choice point) a draft is accepted by exact equality only. Empty or inf = off (the plain rule)", kind="float"),
    "spec_typical_phase": Flag("KYOJIN_SPEC_TYPICAL_PHASE", "all", "spec_typical (read per round): answer = the relaxed rule (and its repetition check) applies only to tokens generated outside an open think block "
                               "(the answer, tool calls, code after the reasoning); inside thinking a draft is accepted by exact equality with the verify sample (the normal sampled path). "
                               "The phase is tracked per session from the prompt's last think tag and the committed tokens (O(1) per round); a model without a think block is always in the answer. "
                               "all (default) = the relaxed rule everywhere. Stat spec_typical_think_skipped counts the relaxed rounds withheld because of thinking", kind="mode", choices=("answer", "all")),
    "think_temperature": Flag("KYOJIN_THINK_TEMPERATURE", "", "sampled sessions (read per round; empty = off): tokens generated inside an open think block (the phase of spec_typical_phase) are sampled at this "
                              "temperature instead of the request's; top_p / top_k / min_p unchanged, the answer keeps the request sampling. Applied to every logits-to-token point of the session: plain "
                              "and verify sampling, the spec_rs q / rejection rule and the relaxed spec_typical acceptance, so drafts are verified against the same tempered distribution. "
                              "Greedy sessions unchanged. The phase is read at the start of the step / round", kind="float"),
    "spec_typical_cooldown": Flag("KYOJIN_SPEC_TYPICAL_COOLDOWN", "512", "spec_typical, adaptive loop control (read per round; tokens, 0 = off): when a session shows an early loop signal (a text line of >= 40 non-space chars, "
                                  "outside code fences, seen a 2nd time in the last 16K chars of its output; or a refused repetition 12-gram of spec_typical_lookup=rep) it accepts by exact equality "
                                  "(no relaxed rule, MTP and lookup alike) for the next cooldown generated tokens, then relaxes again; a re-trigger while armed or within the next window doubles "
                                  "the window (cap 16384). Stats spec_typical_cooldown_triggers / spec_typical_cooldown_tokens. The hard loop guard is unchanged", kind="int"),
    "spec_typical_rewind": Flag("KYOJIN_SPEC_TYPICAL_REWIND", "3", "spec_typical_cooldown, rewind on trigger (read per round; int = max rewinds per reply, 0 = off = cooldown only): when the loop signal fires "
                                "while the session is inside an open think block, the session is rolled back to just before the FIRST copy of the repeated span (the line, or the refused 12-gram): "
                                "those committed tokens are dropped from the model context, the KV cache is rebuilt (prompt prefix + replay of the kept output, the dropped-cache path), the sampling key "
                                "advances, and the exact-acceptance cooldown window runs from there. After the cap no more rewinds (the hard loop guard handles it). Never in the answer or a tool "
                                "call, never across a think tag. Tokens already streamed stay visible to the client (stored text = streamed text); completion_tokens counts every token generated. "
                                "Stats spec_typical_rewinds / spec_typical_rewound_tokens", kind="int"),
    "replay_pipe": Flag("KYOJIN_REPLAY_PIPE", "1", "cache rebuild by replay (after a spec_typical_rewind or a KV drop; read per call): the groups of 8 teacher-forced tokens are queued with async_eval so the host builds group n+1 while the GPU runs group n "
                        "(one group in flight behind it) instead of building and waiting group by group. Same ops, same order, same stream: the cache bits are those of the synchronous loop. 0 = evaluate each group before building the next"),
    "loop_guard": Flag("KYOJIN_LOOP_GUARD", "1", "chat replies (kyojin_mlx.serve.loopguard; read per request): a thinking block that loops (token period <= 64, repeated text, or a line of >= 40 non-space "
                       "chars seen 3 times) is closed with the template's close tag and the model writes its answer in a continuation request, once per reply; a second loop ends "
                       "the reply (finish_reason stop, text so far). A stop token inside an open think block gets the same single recovery. 0 = off (answers are never watched)"),
    "spec_ptap": Flag("KYOJIN_SPEC_PTAP", "0", "probe (log only, read per round): one JSON line per sampled session and speculative round to KYOJIN_SPEC_PTAP_OUT (a jsonl path; no path = off): "
                      "draft source, temperature / top_p / top_k, and per draft position the target probability of the draft token under the exact sampling distribution, "
                      "its entropy, whether the verify sample equalled the draft, and the number accepted (kyojin_mlx.taps). Never changes a token"),
    "router_tap": Flag("KYOJIN_ROUTER_TAP", "0", "probe: the routing weights of decoded rows (plain and verify) per MoE layer, as the engine applies them, sorted by rank: per rank sum and "
                       "sum of squares of the weight share and counts of rows with a share < 0.02 / 0.05 / 0.10 / 0.15, in /v1/stats router_tap (kyojin_mlx.taps). Never changes a routing",
                       attrs=(("taps", "ROUTER", _on),)),
    "gate_adapt": Flag("KYOJIN_GATE_ADAPT", "0", "served MTP round: the confidence-gate threshold from the round's own measured reward instead of the fixed 50 %. "
                       "Every candidate threshold (30..80 %) is priced on the round that just ran: its leading run of raw confidences says how many rows it would "
                       "have proposed, min(accepted, kept) drafts are its yield, and the measured round ms less the fitted per-row cost of the rows it would not "
                       "have built is its cost. The best tokens per ms wins, with a 4 % margin so the threshold does not flap; a session starts at 50 % and "
                       "narrows a failing run (a cut can only be seen on rows the verify already paid for). Read per round; exact for greedy and sampled "
                       "sessions (the verify still decides every emitted token)"),
    "spec_gate": Flag("KYOJIN_SPEC_GATE", "0", "confidence gate threshold from the measured P(accept | conf) and the break-even of one more verified row (slope / plain ms) "
                     "instead of the drafter's fixed 50 % (extends an under-confident head, stops an over-confident one; read per round)"),
    "mu_tuner": Flag("KYOJIN_MU_TUNER", "1", "several greedy sessions that share one verify pass: the round plan (depth 0..max_spec_rows // N - 1, or the batched "
                    "plain step) is chosen per N from the measured tick cost of each arm and the pooled acceptance (serve/mu_tuner.py), instead of the per-session "
                    "depth policy priced at the solo row cost. Off: the solo policy. N = 1 always keeps the solo policy and its measured rate rule, and a backend "
                    "that declares `multi_auto` keeps its own answer (read per tick)"),
    "spec_ladder": Flag("KYOJIN_SPEC_LADDER", "0", "verify width from a measured convex cost ladder: per depth, the median ms of the WARM rounds the engine ran "
                      "(the first round at a width builds its graph and is not a sample), interpolation between measured depths, argmax expected tokens / priced ms "
                      "over d in 1..8 (clamped by max_spec_rows, the drafter block and the room); every unpriced width is measured once, then one row past the best at "
                      "a low rate + plain-round debt after a loss; a round that leaves plain pays the drained step over the measured drafted stretch, and while "
                      "several sessions share the pass the ladder may only narrow the per-session plan; plain fallback when priced tok/ms is below the measured plain "
                      "rate (read per round)"),
    "spec_lookup": Flag("KYOJIN_SPEC_LOOKUP", "1", "served MTP round: draft from a per-session 4-gram index of the committed tokens (prompt + output) when the last tokens "
                     "plus the round's first token occurred earlier (min suffix 8, 8 candidates; no draft-head forward), gated on measured tokens per ms against the "
                     "planned MTP round; greedy and sampled sessions keep the verify's compare-sample acceptance (read per round)",
                     attrs=(("pld", "ENABLED", _on),)),
    "lookup_confirm": Flag("KYOJIN_LOOKUP_CONFIRM", "0", "spec_lookup: a strong match drafts at the row cap only after TWO consecutive lookup rounds landed every "
                          "draft; otherwise its width comes from the measured accepted drafts per lookup round (ceil(ema) + 2). One full round does not show "
                          "that the copied span continues, and the row cap is where a spent match costs most (24 rows ~140 ms for 1 token). Read per round"),
    "spec_lookup_long": Flag("KYOJIN_SPEC_LOOKUP_LONG", "0", "spec_lookup: a strong match (suffix >= 16) widens its draft one tier at a time (7, 14, row cap) only after a solo lookup round of the tier below was "
                             "measured; every unlocked width is priced from the measured lookup-round ms (interpolated / extrapolated from the measured counts, the MTP fit only before two exist) and the best "
                             "tokens per ms wins, MTP included (read per round)"),
    "spec_lookup_vote": Flag("KYOJIN_SPEC_LOOKUP_VOTE", "0", "spec_lookup: the draft is voted by up to N earlier occurrences (those agreeing with the tail within half of the best agreement, "
                             "weighted by agreement) and stops at the first position they disagree on, instead of copying the single best occurrence; 0 = off, 16 is a sensible N "
                             "(read per round; host numpy only)", kind="int"),
    "spec_lookup_adapt": Flag("KYOJIN_SPEC_LOOKUP_ADAPT", "0", "spec_lookup: agreements of 4..7 tokens (below the fixed floor of 8) are tried as lookup drafts, priced from a per-session, "
                              "per-length (4-5, 6-7) accepted-drafts EMA against the planned MTP round, so they run while they pay and stop when they do not (read per round)"),
    "copy_v2": Flag("KYOJIN_COPY_V2", "0", "served MTP round: copy path on the request's own tokens. Route per round copy, then block, then plain: a match of >= copy_v2_min agreeing tokens with "
                    "more than one draft block left drafts a copy whose length is chosen from the agreement (P(span lasts i more | lasted a) = a / (a + i), best expected tokens per priced ms, "
                    "the block / plain round as the alternative); the weaker lookup gate is bypassed. A copy round commits its taps to the draft ring lazily (the head forward is paid only "
                    "if a block round follows) and idle sessions are not flushed; a copy round that lands under half its draft halves the session's confidence (spent match), a full "
                    "landing restores it. Round trace (round_log) carries src and dlen. Verify decides every token (read per round)"),
    "copy_v2_min": Flag("KYOJIN_COPY_V2_MIN", "16", "copy_v2: agreeing tokens a match needs to be a copy (rival: 16; read per round)", kind="int"),
    "copy_v2_max": Flag("KYOJIN_COPY_V2_MAX", "0", "copy_v2: longest copy draft (0 = the verify row cap less the pending row; rival: 15; read per round)", kind="int"),
    "spec_skip": Flag("KYOJIN_SPEC_SKIP", "0", "served MTP round, eager draft path: when the confidence gate cut every proposal of the round, run the plain step instead of a one-row-per-session verify "
                      "+ commit, and rest the draft head of those sessions for 1, 2, 4 .. 16 rounds (reset by a kept draft); lookup rounds untouched. Same logits row and one key per session as the "
                      "one-row verify, so emitted tokens keep their bits (read per round)"),
    "dspark_refine": Flag("KYOJIN_DSPARK_REFINE", "0", "draft pass: a second block forward with the drafted prefix as block ids (the head sees its own draft instead of "
                       "noise) re-picks the tokens and confidences; one more draft call per round, no new weights", attrs=(("dspark", "REFINE", _on),)),
    "draft_cand": Flag("KYOJIN_DRAFT_CAND", "0", "draft head tail: the bigram-biased chain of the block reads the markov head rows of the K best base-logit tokens per position (one top-K pass, "
                       "K-wide products) instead of a full-vocab product + argmax per position; 0 = the full vocab. Needs a dense markov head; spec_rs / refine keep the exact chain. "
                       "Proposals can differ from the full chain only when the bias lifts a token out of the top K (the verify decides every emitted token); read per draft call", kind="int",
                       attrs=(("dspark", "CAND", int),)),
    "verify_prestart": Flag("KYOJIN_VERIFY_PRESTART", "0", "served MTP round, lazy-draft verify pass: the Engram hash ids of its rows are read back at the FIRST Engram layer instead of at the "
                            "pass's entry. Today `EngramAhead.start` reads them first, which waits for the whole draft pass before the host builds layer 0 of the verify: the GPU idles from the end "
                            "of the draft until that graph is built and encoded. With the flag the hash kernels are submitted at once (queued behind the draft), the pre-Engram layers are built "
                            "and submitted under the draft, and the read happens when the first Engram layer needs the rows. Same ids, same rows, same graph: exact; no commit gate or calibration "
                            "(unlike gated_verify); read per pass", attrs=(("engram_mlx", "PRESTART", _on),)),
    "engram_dec_queue": Flag("KYOJIN_ENGRAM_DEC_QUEUE", "1", "Engram rows of a decode pass: their own job queue and read pool. Today a decode round waits behind the prefill chunk's "
                             "reads and its warm job on one worker (rounds of 2.9 to 4.2 s against 0.1 s, slicetrace 01/11), and its passes wipe the chunk's job table, so the chunk "
                             "reads the later layers again. Same rows, same bytes; only the order of the host reads changes; read per pass",
                             attrs=(("engram_rows", "DEC_QUEUE", _on),)),
    "hy4_engram_pread": Flag("KYOJIN_HY4_ENGRAM_PREAD", "0", "Hy4-flash Engram rows (63 GB GGUF table, 4 layers x 16 columns): read by DEDUPED parallel preads (native rowread) instead of the memory-map gather `raw[rows]` "
                             "(one serial 16 KB page-in per cold row: cold prefill 126-435 tok/s against ~2000 warm, coldbase 19/11). The table stays on disk (no warm-up, no pinning). Same rows, same "
                             "bytes, same floats: exact; 0 = the map; read per load", attrs=(("engram_gguf", "PREAD", _on),)),
    "hy4_engram_lazy": Flag("KYOJIN_HY4_ENGRAM_LAZY", "0", "Hy4-flash Engram, prompt chunks (>= 256 positions): the layers' rows are read one layer after the other in the background and the forward waits for a "
                            "layer's rows at that layer (the layers before it run on the GPU: mx.async_eval). Same rows: exact", attrs=(("engram_gguf", "LAZY", _on),)),
    "hy4_engram_decpf": Flag("KYOJIN_HY4_ENGRAM_DECPF", "0", "Hy4-flash Engram, speculative decode rounds (needs hy4_engram_pread): the rows of the pending token are read in the background when the draft chain "
                             "starts, the rows of each draft as soon as the id is known (the draft head reads no Engram row), by parallel preads of only the page-cache-cold rows; the verify "
                             "pass takes them by row id (a miss is read at once) and waits per layer, so the layers before the first Engram layer run on the GPU meanwhile. "
                             "Rows stay on disk (no warm-up, no pinning). Same rows: exact; read per load", attrs=(("engram_gguf", "DECPF", _on),)),
    "hy4_engram_ahead": Flag("KYOJIN_HY4_ENGRAM_AHEAD", "0", "Hy4-flash Engram: the engine's next-chunk hint (Hy4Backend.warm_prompt) reads the NEXT prompt chunk's rows in the background while this one "
                             "runs (content-addressed pieces: a stale hint only misses). Same rows: exact", attrs=(("engram_gguf", "AHEAD", _on),)),
    "engram_one_batch": Flag("KYOJIN_ENGRAM_ONE_BATCH", "1", "Engram rows of a decode / verify pass (under 2048 rows): the value rows and the scale rows of a layer in ONE parallel batch of preads "
                             "(interleaved over a 32-thread pool, hot rows read on the calling thread) instead of two batches one after the other. The cold NVMe read of a layer "
                             "was 1.3-3.5 ms on the critical path of the verify pass (GPU idle at the Engram layers, decgap 04/11); now 0.2-0.4 ms. Same rows, same bytes; read per call",
                             attrs=(("engram_rows", "ONE_BATCH", _on),)),
    "spec_couple": Flag("KYOJIN_SPEC_COUPLE", "0", "sampled sessions (temperature > 0), eager draft rounds: the draft head's chain picks each proposal as argmax(draft logits + T x lambda x "
                        "Gumbel(key_i)) with the very key the verify pass uses for row i (lambda = this value / 100; 0 = off, 100 = the target's own noise). The verify's sample "
                        "at row i is argmax(p_i / T + Gumbel(key_i)), so a draft close to the target now picks the SAME token whenever the noise decides, instead of matching only "
                        "with probability p(argmax q). Every emitted token is still the verify's own sample at its row, a draft only decides whether the next row is reached, so the "
                        "output distribution is the plain sampling one (tests/test_spec_couple.py: chi-square on a fake backend). Greedy sessions and the T=0 same_bench figures are "
                        "untouched; lookup rows, spec_rs, oracle_draft, spec_chain > 1, draft_cand and the lazy pass keep the argmax proposal (read per round)", kind="int"),
    "lazy_trim": Flag("KYOJIN_LAZY_TRIM", "0", "served MTP round: the draft ask follows what the confidence gate keeps. The lazy verify is built at the planned depth before the gate "
                      "can cut it, so every row the gate then drops is a row the GPU paid ~4.9 ms for (and a high cut rate also switches the lazy overlap off for good, backends.LAZY_CUT_MAX). "
                      "A session asks the head for at most ceil(EMA of the proposals its recent drafted rounds kept) + 1 (the +1 lets the ask grow back); every 8th drafted round asks the "
                      "full plan. Accept rule untouched, the verify decides every token; the row count of the pass moves, so the bits move the way spec_skip's do. Not with spec_full / "
                      "spec_rs / oracle_draft; lookup rounds untouched (read per round)"),
    "draft_compile": Flag("KYOJIN_DRAFT_COMPILE", "0", "draft pass: the serving-path rows draft (`Drafter._rows_fn*`, what draft_many / draft_lazy launch) is ONE mx.compile region, the way the solo "
                          "draft already is, instead of a python graph build and unfused elementwise launches every round. First call per (kind, rows) eager, a lever flag switch retraces, a trace "
                          "error leaves that kind eager, KYOJIN_REPLAY keeps its own region. Same values as the eager function (tokens and confidences bitwise equal; the verify decides every "
                          "token either way); read per draft call", attrs=(("dspark", "DRAFT_COMPILE", _on),)),
    "draft_tail_cut": Flag("KYOJIN_DRAFT_TAIL_CUT", "1", "draft pass: the draft head's tail (the 5-step markov bigram chain with its full-vocab bias, the trunk head rows and the "
                          "confidence head) runs on the rows the round plans to verify - its deepest planned depth - instead of on the whole block. A proposal past the planned depth is "
                          "cut by the confidence gate and never verified, and a block row's logits come from its own state, so the rows that are kept keep their proposals, their "
                          "confidences and their bits; the rows the pass skips are padded with 0 (below every threshold, past the depth). Read per draft call; a lazy/verifed pass with "
                          "want_logits (KYOJIN_SPEC_RS) keeps the whole block", runtime=True),
    "ced_skip_dec": Flag("KYOJIN_CED_SKIP_DEC", "3", "CED prefill: a prompt chunk skips its decoder half (the midpoint's queries, window, indexer scores, attention and MoE, and the 19 layers after it, on the tail rows) "
                         "when the NEXT chunk is itself a CED chunk of more than `tail` rows, which rewrites the decoder window rings and the drafter ring whole: the final logits and every cache the "
                         "decode reads are bit for bit those of the full chunk. 1 = only where the engine takes no prefix snapshot / disk checkpoint at that boundary (--prefix-block and "
                         "--kv-disk-block above the chunk); 2 = also where it would, and the engine then takes none there (snapshots with stale decoder rings would be wrong for a short follow-up); "
                         "3 = also where it would, and the engine takes the snapshot / disk checkpoint marked stale: a restore uses it only when the prompt after it starts with a CED chunk of more than `tail` rows, "
                         "else the nearest earlier whole snapshot (or a prefill from 0); old checkpoints without the mark are whole. "
                         "Never on the last chunk or ahead of a short one", kind="int", attrs=(("ced", "SKIP_DEC", int),)),
    "ttft_floor": Flag("KYOJIN_TTFT_FLOOR", "0", "short prefill chunk (a tool round, at most ttft_floor_rows rows): the two pieces of large-chunk machinery it does not need are "
                      "skipped. (1) the layer bound evaluates every ttft_floor_bound layers instead of every KYOJIN_BOUND_EVERY (4): the bound caps the transients of a chunk "
                      "whose activations grow with its rows, and at a few hundred rows the whole chunk's transients are small, while each bound is a sync where the host waits "
                      "for the GPU. (2) the DS4.1 compressor pools its groups in one vectorized step from `ratio` rows instead of 64 (compressor_pf), so the per-token Python "
                      "loop of a 2..63-row chunk (8 launches per token per ratio-2 layer) is gone. Both change no arithmetic: bound_layers only moves where the graph is "
                      "materialised and the pooling is bit-exact against the loop (tests/test_ttftfloor.py, tests/test_compressor_pf.py). docs/reports/ttftfloor_REPORT.md",
                      attrs=(("bound_layers", "TTFT_FLOOR", _on), ("compressor_pf", "TTFT_FLOOR", _on))),
    "ttft_floor_rows": Flag("KYOJIN_TTFT_FLOOR_ROWS", "256", "flag ttft_floor: chunks up to this many rows take the wide bound stride (the default covers a tool round's "
                           "new tokens; 0 = no chunk)", kind="int", attrs=(("bound_layers", "TTFT_ROWS", int),)),
    "ttft_floor_bound": Flag("KYOJIN_TTFT_FLOOR_BOUND", "16", "flag ttft_floor: layers per bound for such a chunk, 4 = today's stride (a chunk of 40 layers then takes 10 "
                            "evals instead of 3); 0 = keep KYOJIN_BOUND_EVERY", kind="int", attrs=(("bound_layers", "TTFT_BOUND_EVERY", int),)),
    "kv_disk_off": Flag("KYOJIN_KV_DISK_OFF", "0", "bench: the disk KV tier neither answers nor writes, the same as --no-kv-disk without a model reload (tools/mu2_ab.py, tools/kvmem_ab.py)", kind="bool"),
    "kv_disk_ro": Flag("KYOJIN_KV_DISK_RO", "0", "bench: the disk KV tier answers lookups and writes nothing, so the same prompt can prefill its tail once per arm from one checkpoint (tools/depth_ab.py). Read per save"),
    "generic_cache": Flag("KYOJIN_GENERIC_CACHE", "1", "cache classes the code does not list (a model's own RotatingKVCache / CacheList / KVCache subclass / pure state cache): multiuser.fork_cache copies "
                          "them through their own state / meta_state / from_state instead of raising TypeError, and kvdisk.GenericAdapter reads the token count of a CacheList from its members and "
                          "takes the caller's b for a model whose caches carry no offset at all (a pure SSM) instead of raising 'cache holds 0 tokens'. The listed classes take the same branches "
                          "as before (mlx-lm KVCache, RotatingKVCache, ArraysCache, WinKVCache): output bit-identical. On by default; 0 = the errors. Read per call (tools/dayone)"),
    "backend_fallback": Flag("KYOJIN_BACKEND_FALLBACK", "1", "serve.backends.load_backend: a model_type with no modeldesc row is served by GenericBackend (the Q35Backend per-user cache / multiuser plumbing "
                             "on the stock mlx-lm / mlx-vlm class, no Qwen step regions, no draft head, no kvwin) instead of ValueError. Described models keep their backend: output "
                             "bit-identical. On by default; 0 = ValueError (opt-out). Read per call (tools/dayone)"),
    "chat_fallback": Flag("KYOJIN_CHAT_FALLBACK", "1", "serve.chat.family_of: a model_type with no modeldesc row gets the Jinja family (the pack's own chat template + the XML / JSON "
                          "<tool_call> parser, the \"qwen\" family) instead of ValueError, so a new family can render a prompt and parse a tool call before it has a row. Rows keep their "
                          "family: output bit-identical for every described model. On by default; 0 = ValueError (opt-out). Read per call (tools/dayone)"),
    "mtpgen": Flag("KYOJIN_MTPGEN", "0", "serve.backends.load_backend (generic fallback path): a pack whose config declares a draft head (num_nextn_predict_layers / mtp_num_hidden_layers / num_mtp_layers) gets "
                   "kyojin_mlx.mtpgen: the head is read from model_mtp.safetensors or the shards, built from the trunk's own layer class, and driven through the Backend verify / commit contract "
                   "(KV trim, or fork + rerun of the kept rows for GDN / SSM state). Refused with a logged reason when the layout is unknown or the trunk fails the verify-4 contract. "
                   "Greedy output is decided by the verify rows: tokens equal plain greedy decode within the verify-k tolerance (not bit-exact claim). 0 = no head, load_backend unchanged. Read per call"),
    "draft_vocab": Flag("KYOJIN_DRAFT_VOCAB", "0", "mtpgen.draft_many: the draft steps take their argmax over the K most frequent token ids of the output head only (K cut to a multiple of 128; "
                        "ranked by a local text sample tokenized with the model's tokenizer, cached on disk; ids < 256 and special tokens always in; the argmax is mapped back to ids). Foreign packs use a DevPack row gather, MLX quantized / dense / tied heads a row gather built once. The verify keeps the full head: "
                        "output text unchanged, only the acceptance can change. 0 = full head. Read per draft call", kind="int"),
    "visiongen": Flag("KYOJIN_VISIONGEN", "0", "generic image input in serve for a model served by GenericBackend (needs backend_fallback and chat_fallback for a model_type without a row) whose config has a "
                      "vision_config: the pack's own processor (preprocessor_config.json) makes the pixels and the image token count, the model's own get_input_embeddings makes the rows, they go through "
                      "Backend.prefill_chunk as an embedding overlay; prefix cache / disk checkpoints are keyed on virtual ids from the image bytes + processor files (repeated image: one encode, "
                      "one prefill). Trunks with multimodal rope (get_rope_index) and scaled embeddings are refused with a 400, not approximated. DS4.1 keeps ds41_vision. Without an image in the "
                      "request nothing changes: output bit-identical. 0 = image parts rejected as before. Read per call"),
    "kv_tail": Flag("KYOJIN_KV_TAIL", "1", "serve prefill: a snapshot (memory and, when a disk tier is on, a checkpoint) at the largest chunk end below the prompt's last cut (below its end without cuts), so a continuation of the prompt "
                    "(next turn, agent loop) restarts there instead of at the last memory / disk block (up to a block minus one tokens paid again). Cost per prompt: one extra file of the rows since the block "
                    "below plus the fixed state (kvdisk stats / engine kv_tail_saved, t_kv_tail_save), written between two prefill chunks, never on the decode path. 0 = off. Read per call", kind="bool"),
    "kv_handoff_free": Flag("KYOJIN_KV_HANDOFF_FREE", "1", "serve prefill: after each evaluated prompt chunk the cross-layer handoff (cache.index_k, candidates, compress_kv, topk_idxs) is dropped (kvledger.free_handoff). "
                            "Every forward pass republishes it in layer order before a consumer reads it, so it is dead between passes, yet `Backend.nbytes` counts it: ~1.28 KB per token on DS4.1 "
                            "by arithmetic (compress_kv 1 KiB + index_k 256 B per row, fp16), the gap between 2.18 and 0.89 KB/token of the 1M live-KV reading. Exact: logits and caches bit-identical "
                            "(tests/test_kvledger.py). It frees the bill and the bytes only the handoff pins (candidates, topk, a memo-backed pool); a handoff that is a slice of a depthflat mirror "
                            "leaves the mirror resident (a separate, decode-side lever). Not measured on Metal. 0 = today. Read per chunk"),
    "kv_seg_once": Flag("KYOJIN_KV_SEG_ONCE", "1", "serve memory: with kvseg > 0 a sealed segment is billed once by `Engine.kv_bytes` (the prefix ledger bills it; the live session's `seg_bytes` of the same "
                        "segment is subtracted). 0 = today (counted twice: live layer + prefix ledger). Only the bill and so the eviction point move; no array is touched. Read per call"),
    "stack_dedupe": Flag("KYOJIN_STACK_DEDUPE", "0", "batched decode keeps ONE copy of each pool / index-key layer: when the sessions are stacked (dsv4_core.stack_rows) the dense stacked buffers are the "
                         "working state, and the sessions' own packed codes + scales, dense depthflat mirror, memo and capacity buffers of the same layer (a second and third copy, 1 KiB + 256 B "
                         "per row fp16 plus 1/8 slack for the mirror, ~0.3 of that for the codes) are released as each layer is stacked. `sync_rows` (extract, state, filter, extend) writes the "
                         "layer back from the stack, re-packed on the fp4 grid: same values, same codes. No arithmetic: logits and every cache are bit-identical (tests/test_stack_dedupe.py). "
                         "Cost: leaving the batch after a stack with no decode step now re-packs the layers (a step already did). N = 1 is untouched. 0 = today. Read per stack"),
    "bound_layers": Flag("KYOJIN_BOUND_LAYERS", "1", "prompt chunks evaluated every few layers", attrs=(("bound_layers", "ENABLED", _on),)),
    "bound_every": Flag("KYOJIN_BOUND_EVERY", "4", "bound_layers: layers per evaluated group", kind="int",
                        attrs=(("bound_layers", "EVERY", int),)),
    "bound_min_rows": Flag("KYOJIN_BOUND_MIN_ROWS", "64", "bound_layers: smallest chunk that is bounded", kind="int",
                           attrs=(("bound_layers", "MIN_ROWS", int),)),
    "bound_dec_every": Flag("KYOJIN_BOUND_DEC_EVERY", "2", "bound_layers: decode / verify steps submit the graph every n layers (0 = once at the end)", kind="int",
                            attrs=(("bound_layers", "DEC_EVERY", int),)),
    "bound_dec": Flag("KYOJIN_BOUND_DEC", "1", "bound_layers: bound the whole prefill chunk, decoder half included. CED gives every layer after the midpoint its own class: only "
                       "`type(layers[0])` is wrapped today, the midpoint never reaches a wrapper (`_block_mid`), and the tail-sized `h` of the layers after it reads as a decode step below "
                       "MIN_ROWS, so the half where a chunk is expensive has no eval group / mix_slice_ms YIELD point of its own (1 bound where 2 are expected on the tiny probe). 1 wraps "
                       "every class the layer list uses and reads the chunk's rows from the CED context (`cache[0]._ced.n`) instead of `h`; the outermost eval and yield stay outside "
                       "`_block_call`, after CED restored `cache.offset`, where the vendor itself materialises between chunks, so logits and every cache buffer are bit-identical. 0 = "
                       "pass-through on those classes (today's bound count, phase, eval points and counters). Read per call", attrs=(("bound_layers", "DEC", _on),)),
    "verify_nax": Flag("KYOJIN_VERIFY_NAX", "0", "verify passes of 5+ rows use the ROWS_NAX / auto-MMA rounding class (0 = GEMV chunks: each row keeps its plain-step bits; read per pass)"),
    "rg_half": Flag("KYOJIN_RG_HALF", "0", "the many-rows GEMV of a 5..8 row block (exl3_linear._many, rows above GEMV_MAX_ROWS) FMA in packed fp16, the class the <= MAX_ROWS fused launch "
                   "takes from HALF_ROWS rows on, instead of fp32 rows: the dense EXL3 class is 28 % of a 6-row verify round and its largest slope, and only its 5..8 row band ran fp32 rows "
                   "(metal_src.gemv_rg_source half=True builds the same half4 x groups the fused prologue stages). A row's bits are then the fused launch's bits at the same row count "
                   "(the same class as shipped half_rows, so the same one-fp16-ulp drift, KL-gated); rows <= GEMV_MAX_ROWS untouched",
                   attrs=(("exl3_linear", "RG_HALF", _on),)),
    "shared_glu": Flag("KYOJIN_SHARED_GLU", "0", "decode rows of the shared expert (pf_glue._impl_shared_mlp): silu(min(gate, L)) * clip(up, -L, L), its cast to fp16 and the fp8 round trip run in the prologue of the "
                       "down EXL3 GEMV (kernels.Mul1Gemv.fused_glu) instead of their own compiled launch: one launch fewer per MoE layer; same arithmetic and rounding points, bit-identical",
                       attrs=(("pf_glue", "SHARED_GLU", _on),)),
    "half_rows": Flag("KYOJIN_HALF_ROWS", "4", "fused dense GEMV and MoE grouped GEMV of that many rows or more: packed-fp16 row FMAs (0 = fp32 FMAs; real-text NLL vs fp32 rows +0.002 nats, within noise)", kind="int",
                      attrs=(("metal_src", "HALF_ROWS", int),)),
    "pfroof2": Flag("KYOJIN_PFROOF2", "0", "prefill indexer: each query block scores only its causal columns, exact radix top-k (pfroof2; identical output)"),
    "pfroof2_topk": Flag("KYOJIN_PFROOF2_TOPK", "0", "pfroof2: use the exact multi-threadgroup top-k (0 = argsort + sort, the module default); only read when pfroof2 is on"),
    "pf_idx_cand": Flag("KYOJIN_PF_IDX_CAND", "1", "prefill indexer, candidate-consumer layers: score only the kept 8-key blocks, top-k over the compacted set (exact, ds_pf_idx.cand_topk)"),
    "longctx_rope": Flag("KYOJIN_LONGCTX_ROPE", "1", "one RoPE table grown by powers of two instead of a float64 rebuild every 256 positions (0.3-0.5 s at 1M, up to 16 GB of tables); exact"),
    "longctx_rope_pf": Flag("KYOJIN_LONGCTX_ROPE_PF", "0", "longctx_rope: build the next power-of-two table on a background thread at 3/4 use, so no decode step pays the float64 build (one extra table in memory); exact",
                            attrs=(("longctx", "ROPE_PF", _on),)),
    "idx_keys_packed": Flag("KYOJIN_IDX_KEYS_PACKED", "0", "decode index scores read the KeySlots fp4 codes (4.25 bits) and unpack in the score kernel instead of reading the fp16 mirror (E8: 84 MB/128K, 670 MB/1M per token, x MTP rows); exact",
                           attrs=(("fuse_index", "PACKED_KEYS", _on),)),
    "iscore_kb": Flag("KYOJIN_ISCORE_KB", "0", "decode index scores: register-blocked kernel (one thread per 4 keys x NH/4 heads) instead of one thread per key; exact, measured behind above 128K keys",
                      attrs=(("dsv4_core", "ISCORE_KB", _on),)),
    "iscore_kb_min": Flag("KYOJIN_ISCORE_KB_MIN", "0", "iscore_kb: keys from which the register-blocked kernel runs (0 = from the first key, i.e. short contexts)", kind="int",
                          attrs=(("dsv4_core", "ISCORE_KB_MIN", int),)),
    "iscore_mma": Flag("KYOJIN_ISCORE_MMA", "1", "batched-core indexer scores (verify rounds / multi-row passes) on the simdgroup-matrix kernel the solo decode step "
                      "already uses (dsa_decode.scores_rows: one launch for every row of the pass) instead of the one-thread-per-key fp32 chain (_ISCORE); "
                      "same arithmetic and rounding points as dsa_decode.scores, so the head sum is 4 partial sums, not the stock 32-step order: "
                      "decode text is rounding-level, drift gate", attrs=(("dsv4_core", "ISCORE_MMA", _on),)),
    "iscore_mma_min": Flag("KYOJIN_ISCORE_MMA_MIN", "262144", "iscore_mma: keys from which the simdgroup-matrix score kernel runs (shorter contexts keep _ISCORE)",
                           kind="int", attrs=(("dsv4_core", "ISCORE_MMA_MIN", int),)),
    "iscore_blocks": Flag("KYOJIN_ISCORE_BLOCKS", "0", "batched-core consumer index layers (the layers that read the source layer's block mask): score and select only the "
                        "blocks the mask kept (16384 keys of any width) instead of walking every key of the pass - the kept keys are compacted and gathered on the device and "
                        "the same matrix kernel scores them, so the selection is the masked path's selection bit for bit and the term stops growing with the context "
                        "(idxblocks; consumers only, fp16 / bf16, 32 heads x 128)", attrs=(("idxblocks", "ENABLED", _on),)),
    "iscore_blocks_min": Flag("KYOJIN_ISCORE_BLOCKS_MIN", "262144", "iscore_blocks: key width from which a consumer layer takes the gathered path (below it the full-width "
                             "kernel is already cheap and the compaction adds fixed launches)", kind="int",
                             attrs=(("idxblocks", "MIN_KEYS", int),)),
    "dec_idx_i8": Flag("KYOJIN_DEC_IDX_I8", "0", "decode index scores: a cheap int8 first pass names the candidate blocks and the exact half kernel rescores them. The block maxima of "
                      "every row come from the pf_flat int8 block-max kernel (matmul2d on the FP4 grid, twice the half rate, half the key bytes) over the keys below the row's `lens`; "
                      "the topk + margin blocks (the candidate source: max(candidate_topk_blocks, topk) + margin, last visible block pinned) are gathered per row from its own "
                      "session's buffer and scored by dsa_decode.scores_rows (nax=False, the masked path's kernel: per key, so the same values), and the top-k of those is the answer "
                      "(the source rebuilds its kept block mask from the exact block maxima). The int8 codes are held for the committed rows only, 132 B per key per index-key layer "
                      "(decidx8; fp16, 32 heads x 128, runtime width, >= dec_idx_i8_min keys). Off = today", attrs=(("decidx8", "ENABLED", _on),)),
    "dec_idx_i8_margin": Flag("KYOJIN_DEC_IDX_I8_MARGIN", "128", "dec_idx_i8: blocks picked beyond the ones the answer needs and rescored exactly (the int8 ranking's rounding "
                            "shield; ties and the source's pinned last block are exact)", kind="int", attrs=(("decidx8", "MARGIN", int),)),
    "dec_idx_i8_min": Flag("KYOJIN_DEC_IDX_I8_MIN", "262144", "dec_idx_i8: index keys from which a layer takes the int8 first pass (below it the packed keys and the fixed launches "
                         "do not pay)", kind="int", attrs=(("decidx8", "MIN_KEYS", int),)),
    "dec_blk_sum": Flag("KYOJIN_DEC_BLK_SUM", "0", "decode index scores: score one mean summary key per 8-key block (1/8 of the key bytes, the same row kernel) and rescore exactly only the "
                      "dec_blk_sum_mult x topk best blocks (gathered per row from its own session; the candidate source rebuilds its kept block mask from them). Approximate: top-k recall vs the exact "
                      "selection is reported by tests/test_dec_blk_sum.py, re-measure on real keys before serving (decblk; fp16 / bf16, 32 heads x 128, runtime width, >= dec_blk_sum_min keys). Off = today",
                      attrs=(("decblk", "ENABLED", _on),)),
    "dec_blk_sum_mult": Flag("KYOJIN_DEC_BLK_SUM_MULT", "8", "dec_blk_sum: blocks rescored exactly per block the answer needs (topk; the source: max(candidate_topk_blocks, topk)); "
                           "the recall shield, higher = closer to exact and more rescore reads", kind="int", attrs=(("decblk", "MULT", int),)),
    "dec_blk_sum_min": Flag("KYOJIN_DEC_BLK_SUM_MIN", "262144", "dec_blk_sum: index keys from which a layer takes the summary first pass", kind="int", attrs=(("decblk", "MIN_KEYS", int),)),
    "gpu_keepalive": Flag("KYOJIN_GPU_KEEPALIVE_S", "0.5", "serve: seconds between tiny GPU commands while idle (for an hour after the last work), so the next request skips the ~1.2 s GPU wake-up; 0 = off",
                          runtime=False, reason="serve.engine reads the interval once at import (a float, which a 0/1 switch cannot carry)"),
    "serve_coalesce": Flag("KYOJIN_SERVE_COALESCE", "0", "serve: the HTTP thread drains every token the engine queued meanwhile, writes the SSE frames of the whole batch with one write, "
                           "and still emits one delta per token (the frame count a client sees is unchanged; drain_events). The engine queue is a SimpleQueue (read per request)",
                           attrs=(("serve.hostloop", "COALESCE", _on),)),
    "serve_nodelay": Flag("KYOJIN_SERVE_NODELAY", "0", "serve: TCP_NODELAY on accepted sockets, so a small SSE frame waits for no delayed ACK (read per connection)",
                          attrs=(("serve.hostloop", "NODELAY", _on),)),
    "serve_gc": Flag("KYOJIN_SERVE_GC", "0", "serve: gc.freeze() of the loaded model, young generation 20000 / 20 / 200, a full collection only when the engine idles (cycles are still collected)",
                     attrs=(("serve.hostloop", "GC", _on),)),
    "serve_switch": Flag("KYOJIN_SERVE_SWITCH", "0", "serve: sys.setswitchinterval in microseconds (how long a thread that wants the GIL waits for the running one); 0 = the interpreter's own 5000",
                         kind="int", attrs=(("serve.hostloop", "SWITCH_US", int),)),
    "serve_prof": Flag("KYOJIN_SERVE_PROF", "0", "serve: host wall time per section of the engine tick, GC pause time and GIL scheduling lag into /v1/stats timing (t_hp_* / n_hp_*)",
                       attrs=(("serve.hostloop", "PROF", _on),)),
    "host_timed": Flag("KYOJIN_HOST_TIMED", "0", "serve probe: wall and thread CPU time inside every blocking call of the verify build and of the plain step "
                      "(t_hs_bc_<eval|async_eval|tolist|item>_w / _c, t_hs_pl_*), so a section's off-CPU time splits into the calls it waited in and the rest; "
                      "plus spec_rvcpu_r<rows> / spec_rpcpu_r<rows>, the thread CPU of the build, always counted",
                      attrs=(("hostsync", "TIMED", _on),)),
    "gpu_span": Flag("KYOJIN_GPU_SPAN", "0", "serve probe: GPU busy / idle of every tick from the command buffers' own GPUStartTime / GPUEndTime "
                     "(tools/cbcount, no MLX patch, nothing added to the GPU): gpu_n_r<rows> / gpu_buf_r<rows> / gpu_busy_r<rows> / gpu_idle_r<rows> / "
                     "gpu_span_r<rows> / gpu_gap_r<rows> / gpu_gapl_r<rows> (gaps > 0.5 ms) / gpu_gmax_r<rows> / gpu_starve_r<rows> (first GPU start - first "
                     "commit) / gpu_lead_r<rows> (peak committed-not-completed, MLX's MAX_ACTIVE_TASKS is 10); a prefill chunk is bracketed too and "
                     "keyed by its own rows (gpu_*_r<chunk rows>, the chunk's encoder and decoder halves)",
                     attrs=(("gpuspan", "ON", _on),)),
    "plain_split": Flag("KYOJIN_PLAIN_SPLIT", "0", "serve probe: the host split of a plain decode step against its GPU time (plainsplit; the same uptime clock "
                      "as the command buffers): ps_plan_r<rows> (the round's plan) / ps_build_r<rows> (the plain launch = the graph the host builds) / "
                      "ps_encode_r<rows> (the mx.async_eval part of that build) / ps_wait_r<rows> (the y.tolist() that waits for the step in flight) / "
                      "ps_accept_r<rows> + ps_post_r<rows> / ps_tick_r<rows> + ps_round_r<rows>, and the overlap terms ps_overlap_r<rows> (build covered "
                      "by GPU work) / ps_starve_r<rows> (GPU idle for want of a commit) / ps_cur_nd_r<rows> (dispatches of the step) / ps_cur_busy_r<rows>; "
                      "per-tick rows to KYOJIN_PLAIN_SPLIT_OUT"),
    "moe_gpf": Flag("KYOJIN_MOE_GPF", "0", "grouped routed MoE GEMV (rows >= 2): k tiles of trellis words loaded ahead of the tile being decoded "
                     "(1 = one tile, 2 = two), so a tile's HBM latency is covered by the decode of the previous one. The same words in the same order, "
                     "read earlier: bit-identical to the stock grouped kernel", kind="int",
                     attrs=(("metal_src", "GPF", int),)),
    "gpu_kern": Flag("KYOJIN_GPU_KERN", "0", "serve probe (with gpu_span): price every compute dispatch by its Metal function name, from the same command buffers "
                     "(tools/cbcount records the pipeline's function name per dispatch): gpu_k_<name>_r<rows> / gpu_kc_<name>_r<rows>, plus gpu_kspan_r<rows> "
                     "(sum of all buffer spans) / gpu_kb1_r<rows> (buffers with exactly one dispatch, exact) / gpu_kmix_r<rows> (split evenly). "
                     "MLX_MAX_OPS_PER_BUFFER=1 makes every buffer single-dispatch",
                     attrs=(("gpuspan", "KERN", _on),)),
    "ablit_fused": Flag("KYOJIN_ABLIT_FUSED", "1", "direction projection (ablit.project): clamp, the K dot products and the update of a row in ONE launch instead of ~7 "
                        "(fp16 clip pair, fp32 cast, two small fp32 matmuls, fused update); fp32 inside like the stock path, only the dot product's "
                        "summation order differs", attrs=(("ablit", "FUSED", _on),)),
    "attn_core_fuse": Flag("KYOJIN_ATTN_CORE_FUSE", "0", "DS4.1 plain decode attention core: the kv chain also writes the stacked window buffer the commit keeps, so the commit's slot scatter launch is gone "
                           "(bit-identical outputs and cache state; applies when the stack has spare slots or a non-identity slot map; 0 = the old path)",
                           attrs=(("dsv4_core", "ATTN_CORE_FUSE", _on),)),
    "ablate": Flag("KYOJIN_ABLATE", "", "serve probe (timing only, outputs become garbage): comma list of op classes of the DS4.1 decode step (1-row plain step and verify rows) "
                   "each replaced by the cheapest stand-in that keeps shapes and dtypes, to read its exposed cost as a round-time difference. Classes: moe_routed (routed experts -> zeros), "
                   "moe_shared (shared expert -> zeros), router (logits + top-k -> fixed ids / weights), moe_glue (routed GEMVs kept, rot / mid / final -> zero buffers), "
                   "mhc (mixes + sinkhorn -> constants, pre -> stream mean, expand -> add; block runs the stock chain), norm (RMSNorm -> identity; block runs the stock chain), "
                   "attn (attention core -> zeros, cache untouched), idx (indexer -> fixed first-k), ablit (direction projection -> identity); on the generic decoder skeleton (Hy4, GLM, kit models) "
                   "also attn_proj / attn_core / attn_out / mlp / hc / engram / head (kyojin_mlx/ablate_struct.py, structure only; the first ablated call of a shape runs real). \"\" = off, bit-identical",
                   kind="str", attrs=(("ablate", "ACTIVE", _ablate),)),
    "host_sample": Flag("KYOJIN_HOST_SAMPLE", "0", "serve probe: sample the engine thread's own Python stack every N microseconds (0 = off, read per loop "
                      "iteration) and dump the wall attribution per file:line to KYOJIN_HOST_SAMPLE_OUT (default ./pysample.txt) when the lever goes back to 0; "
                      "`gil_held` per site is the part of it the engine thread spent holding the GIL (MLX waits for the GPU with it), the rest of the off-CPU "
                      "time is a re-acquire",
                      kind="int"),
    "pf_idx_ktile": Flag("KYOJIN_PF_IDX_KTILE", "1", "prefill indexer scores: tile the keys (4096 columns, the vendor width) so the query micro-block stays large at long context (exact)"),
    "pf_moe_chunk": Flag("KYOJIN_PF_CHUNK", "4096", "prefill MoE: rows of one pass over the expert banks (each pass sweeps every bank once and ends with an eval; capped at 8192)", kind="int"),
    "pf_moe_hoist": Flag("KYOJIN_PF_MOE_HOIST", "0", "prefill MoE: a call of several passes (the served 8192-row chunk is two 4096-row passes) plans every pass on the device and reads all the expert "
                         "offsets back in ONE sync, then builds pass 1, submits it (async_eval) and builds pass 2 while the GPU runs pass 1; today each pass is plan -> sync -> build -> sync, "
                         "3 GPU idle gaps a layer. Same kernels on the same inputs: bit-identical. Costs about one extra pass of pair buffers in flight (~1.1 GB at 4096 rows x top-6)",
                         attrs=(("moe_prefill", "HOIST", _on),)),
    "pf_moe_one_pass": Flag("KYOJIN_PF_ONE_PASS", "0", "prefill MoE: one pass over the expert banks per call, up to the 8192-row cap, instead of splitting the call at pf_moe_chunk rows. Two passes of a "
                          "chunk decode each expert's trellis twice per row block they share and pay the plan, sort and readback twice; one pass pays them once and ends with one eval",
                          attrs=(("moe_prefill", "ONE_PASS", _on),)),
    "pf_idx_fscore": Flag("KYOJIN_PF_IDX_FSCORE", "1", "prefill indexer scores: one fused kernel per query row (32 heads x 32 keys per simdgroup tile in the threadgroup, relu x w, heads summed in order, mask), no [T*32, L] half product in memory (exact, ds_pf_idx.fscore)", attrs=(("ds_pf_idx", "FSCORE", _on),)),
    "pf_idx_qmm": Flag("KYOJIN_PF_IDX_QMM", "0", "prefill indexer scores: the q . k GEMM on the MXFP8 path (mx.quantized_matmul, keys packed once per layer) instead of the fused fp16 tile kernel, keys from 4096; lossless when the index keys are already fp4/ue8m0 quantized (ds_pf_idx.scores_qmm)"),
    "pf_idx_topk_mt": Flag("KYOJIN_PF_IDX_TOPK_MT", "1", "prefill indexer top-k: exact multi-threadgroup radix select instead of sort(argsort(-s)) over the whole key width (idxfast.topk_mt, same indices)", attrs=(("ds_pf_idx", "TOPK_MT", _on),)),
    "pf_attn_pool": Flag("KYOJIN_ATTN_PF_POOL", "1", "prefill sparse attention: window and pool as two kernel inputs, no concat([window, pool]) table per layer per chunk (exact, attn_prefill)", attrs=(("attn_prefill", "POOLIN", _on),)),
    "pf_attn_lazymask": Flag("KYOJIN_ATTN_PF_LAZYMASK", "1", "prefill attention: no [T, Tw] window mask when the gathered-key path runs (built only for the stock/dense fallback; exact)", attrs=(("attn_prefill", "LAZY_MASK", _on),)),
    "pf_attn_online": Flag("KYOJIN_ATTN_PF_ONLINE", "1", "prefill sparse attention: K2 folded into K1. K1 takes the per-head max of its 128-key tile from a threadgroup "
                        "buffer and writes P = exp(S - m_tile) fp16 plus the tile's m and sum, so the fp32 S buffer (1.34 GB a layer at 8192 rows) and K2's read of it "
                        "disappear; K3 rescales each block by exp(m_tile - M)/Z from the tile arrays. Same math, different reduction order (not bit-identical)",
                        attrs=(("attn_prefill", "ONLINE", _on),)),
    "dspark_conf_pct": Flag("KYOJIN_DSPARK_CONF_PCT", "0", "DSpark confidence gate threshold in percent (0 = the drafter's own KYOJIN_DSPARK_CONF; read per draft)", kind="int",
                            attrs=(("dspark", "CONF_OVERRIDE", lambda v: int(v) / 100 if int(v) > 0 else None),)),
    "glm_block_arm": Flag("KYOJIN_GLM_BLOCK_ARM", "", "GLM-5.3 external block drafter in use when KYOJIN_GLM_BLOCK loaded several (dflash2:DIR,dspark:DIR): '' = the first one. Switch only while idle (a session's drafter state belongs to one drafter)",
                          kind="mode", choices=("", "dflash2", "dspark")),
    "glm_block_conf": Flag("KYOJIN_GLM_BLOCK_CONF", "0", "GLM-5.3 DSpark block drafter: draft-length gate in percent on the cumulative confidence of the drafted prefix (0 = off; read per draft)", kind="int"),
    "glm_block_cand": Flag("KYOJIN_GLM_BLOCK_CAND", "0", "GLM-5.3 DSpark block drafter: Markov bias only on the top-N base-logit tokens of each position (0 = the whole vocabulary; read per draft)", kind="int"),
    "dec_topk_blocks_rows": Flag("KYOJIN_DEC_TOPK_BLOCKS_ROWS", "0", "dec_topk_blocks: only for score tiles of at most this many rows (0 = any); a prefill tile has 64-512 rows and the candidate path adds ~20 launches per tile, so 16 keeps decode / verify on it and prefill on the radix select", kind="int", attrs=(("idxblk", "MAX_ROWS", int),)),
    "dec_topk_blocks": Flag("KYOJIN_DEC_TOPK_BLOCKS", "0", "indexer top-k (decode, verify, prefill: every idxfast.topk_mt call with N*Tm >= 2^20 and >= 4k blocks): exact candidate-block select, one block-max read, top-k of the block maxima, top-k of the k picked blocks, instead of 4 full reads of the scores (idxblk, same indices)", attrs=(("idxfast", "DEC_TOPK_BLOCKS", _on),)),
    "topk_tree": Flag("KYOJIN_TOPK_TREE", "0", "indexer top-k of decode / verify rows (idxfast.topk_mt and select_fused, <= 16 rows, fp16 / bf16, from KYOJIN_TOPK_TREE_MIN keys): exact chunk-tree select in ONE launch "
                      "(topk_tree: per-chunk K best in registers, last-arriver threadgroup selects among the chunk lists, candidate-source layers add the block level and the mask) instead of the 7-launch radix chain; same indices and mask",
                      attrs=(("idxfast", "TOPK_TREE", _on),)),
    "cidxfuse": Flag("KYOJIN_CIDXFUSE", "0", "decode indexer layer at depth (<= 16 rows, fp16 / bf16, 32 index heads, from KYOJIN_TOPK_TREE_MIN keys, chunk <= 8192 keys): scores + exact chunk-tree top-k + candidate mask in ONE launch, "
                    "each (row, chunk) threadgroup scores its own chunk into threadgroup memory (no [rows, capacity] score buffer, work follows lens); same indices / mask as topk_tree over the idxfuse.scores_fast scores "
                    "(= index_scores rt bits). Takes the place of the iscore kernel + select when on, so it is NOT bit-claimed against the matrix-unit scores of iscore_mma (cidxfuse)",
                    attrs=(("dsv4_core", "CIDXFUSE", _on), ("cidxfuse", "ENABLED", _on))),
    "topk_tree_min": Flag("KYOJIN_TOPK_TREE_MIN", "16384", "topk_tree: score capacity (index keys) from which the tree runs; shorter rows keep the radix chain or the fused indexer", kind="int",
                          attrs=(("topk_tree", "MIN_KEYS", int),)),
    "dec_sel_every": Flag("KYOJIN_DEC_SEL_EVERY", "0", "decode index selection reuse (decsel): successive decode queries of a session select nearly the same keys, so the "
                      "batched-core index layers (plain and masked consumer layers, not the candidate source whose answer is a full-width block mask) score only a "
                      "candidate set - the selection of the last committed row plus the keys appended since - and select inside it. The candidate keys are gathered out "
                      "of the session's own buffer, scored by the kernel the full scan would use and selected by idxfast.topk_mt, so a reused round is bit-identical "
                      "to the full scan whenever the true top-k is inside the set. P = rounds between full scans: 0 = today (the lever is not armed at all), 1 = a full "
                      "scan every round (the bookkeeping runs, the answer is the full scan's bit for bit). The cache is written at the commit, from the row the "
                      "commit keeps, so a rejected draft never publishes", kind="int",
                      attrs=(("decsel", "EVERY", int),)),
    "dec_sel_check": Flag("KYOJIN_DEC_SEL_CHECK", "0", "dec_sel_every: also run the full scan on the reused rounds and count the top-k overlap per index layer "
                       "(/v1/stats engage: decsel.ov<layer> / decsel.slots<layer>). One host read per reused layer per round: a debug flag, not a served one",
                       attrs=(("decsel", "CHECK", _on),)),
    "dec_filter": Flag("KYOJIN_DEC_FILTER", "0", "decode indexer of the batched core (verify rounds), every index layer without a candidate mask (plain layers and the candidate source, mask included) from dec_filter_min keys: one fp16 matmul scores every key with a rigorous per-key error bound, the best blocks are rescored with the exact kernel, and a per-row certificate (round(best left-out upper bound) < k-th exact score) proves the same top-k; a failed certificate runs the stock chain (decfilter; one boolean read per filtered layer)",
                       attrs=(("decfilter", "ENABLED", _on),)),
    "dec_filter_min": Flag("KYOJIN_DEC_FILTER_MIN", "131072", "dec_filter: index-key width from which a layer takes the filter", kind="int", attrs=(("decfilter", "MIN_KEYS", int),)),
    "dec_filter_budget": Flag("KYOJIN_DEC_FILTER_BUDGET", "2", "dec_filter: blocks rescored exactly = budget x index_topk (blocks of 16 keys)", kind="int", attrs=(("decfilter", "BUDGET", int),)),
    "pf_idx_bmax": Flag("KYOJIN_PF_IDX_BMAX", "1", "prefill indexer past pf_idx_bmax_min keys (plain layers; the candidate source too with pf_idx_gscore): block-max kernel + exact block pick + rescoring of the picked blocks, no [rows, Tc] scores in memory (pf_flat; same indices)", attrs=(("pf_flat", "BMAX", _on),)),
    "pf_idx_bmax_min": Flag("KYOJIN_PF_IDX_BMAX_MIN", "65536", "pf_idx_bmax: keys from which a layer takes the block-max path", kind="int", attrs=(("pf_flat", "BMAX_MIN", int),)),
    "pf_idx_gscore": Flag("KYOJIN_PF_IDX_GSCORE", "1", "prefill indexer: rescoring of picked 8-key blocks in one kernel (blocks staged from the key array, no gathered [rows, 8K, 128] copy); also enables the block-max path of the candidate source (pf_flat)", attrs=(("pf_flat", "GSCORE", _on),)),
    "pf_cand_ids": Flag("KYOJIN_PF_CAND_IDS", "1", "prefill candidate source hands its kept block ids to the consumer layers (cumsum + scatter once) instead of each consumer re-reading the [T, Tc] mask and sorting Tc / 8 blocks per row (pf_flat; same indices)", attrs=(("pf_flat", "CAND_IDS", _on),)),
    "pf_idx_selfuse": Flag("KYOJIN_PF_IDX_SELFUSE", "1", "prefill candidate source on the mask path: idxfast.select_fused (top-k + candidate mask, one exact pass) instead of select_candidate_blocks + a second radix top-k (pf_flat; same outputs)", attrs=(("pf_flat", "SELFUSE", _on),)),
    "pf_idx_bound": Flag("KYOJIN_PF_IDX_BOUND", "0", "prefill indexer of a plain layer past pf_idx_bound_min keys: exact two-level bound. Per superblock of pf_idx_bound_sb keys a box + ball bound on every head's q . k, "
                         "a threshold from the exactly scored best superblocks, and only superblocks whose bound reaches it are scored (pf_flat gathered kernel); the stock chain runs for a query block "
                         "whose survivors pass pf_idx_bound_maxpct percent (pf_bound; same indices as the stock chain; needs pf_idx_gscore=1 on the GPU, off while pf_idx_qmm is on; the gain depends on how clustered the index keys are, not measured)",
                         attrs=(("pf_bound", "ENABLED", _on),)),
    "pf_idx_bound_min": Flag("KYOJIN_PF_IDX_BOUND_MIN", "131072", "pf_idx_bound: keys from which a plain layer takes the bound path", kind="int", attrs=(("pf_bound", "MIN_KEYS", int),)),
    "pf_idx_bound_sb": Flag("KYOJIN_PF_IDX_BOUND_SB", "64", "pf_idx_bound: keys per superblock (rounded down to a multiple of 8)", kind="int", attrs=(("pf_bound", "SB", int),)),
    "pf_idx_bound_maxpct": Flag("KYOJIN_PF_IDX_BOUND_MAXPCT", "25", "pf_idx_bound: most survivors, in percent of the superblocks (worst row of a query block), before that block runs the stock chain", kind="int",
                                attrs=(("pf_bound", "MAXPCT", int),)),
    "pf_idx_cluster": Flag("KYOJIN_PF_IDX_CLUSTER", "0", "prefill indexer of a plain layer past pf_idx_cluster_min keys: one candidate 8-key block set per query block. "
                         "A few probe rows of the block run the exact full-width selection, the union of their best keys is the only key set the rest of the "
                         "block is scored on (gather + exact rescoring, same column values), so most key blocks are never scored. APPROXIMATE for the rows "
                         "between the probes: quality, not bitwise (see docs/reports/pf1m_REPORT.md for the recall against the exact top-k, and "
                         "pf_idx_cluster_recall to measure it on the served path). Not used by the candidate source or its consumers, which already score "
                         "few rows / kept blocks (pf_cluster)",
                         attrs=(("pf_cluster", "ENABLED", _on),)),
    "pf_idx_cluster_min": Flag("KYOJIN_PF_IDX_CLUSTER_MIN", "65536", "pf_idx_cluster: keys from which a plain layer takes the union path", kind="int",
                               attrs=(("pf_cluster", "MIN_KEYS", int),)),
    "pf_idx_cluster_probes": Flag("KYOJIN_PF_IDX_CLUSTER_PROBES", "16", "pf_idx_cluster: probe rows per query block (KYOJIN_IDX_QBLOCK rows, 512 by default); "
                                 "the last row of the block is always one of them", kind="int", attrs=(("pf_cluster", "PROBES", int),)),
    "pf_idx_cluster_mult": Flag("KYOJIN_PF_IDX_CLUSTER_MULT", "4", "pf_idx_cluster: keys kept per probe = mult x index_topk (wider union, higher recall, more keys scored)",
                                kind="int", attrs=(("pf_cluster", "MULT", int),)),
    "pf_idx_cluster_cap": Flag("KYOJIN_PF_IDX_CLUSTER_CAP", "8192", "pf_idx_cluster: most 8-key blocks of one union (a wider set runs the stock block instead)",
                               kind="int", attrs=(("pf_cluster", "CAP", int),)),
    "pf_idx_cluster_maxpct": Flag("KYOJIN_PF_IDX_CLUSTER_MAXPCT", "25", "pf_idx_cluster: widest union, in percent of the keys of the row, that is still worth "
                                   "scoring; a wider one runs the stock block (and after two of them the rest of the chunk runs stock without probes)", kind="int",
                                   attrs=(("pf_cluster", "MAXPCT", int),)),
    "pf_idx_cluster_recall": Flag("KYOJIN_PF_IDX_CLUSTER_RECALL", "0", "pf_idx_cluster: measure the recall of the served path. One non-probe row per query "
                                  "block is scored at full width and its exact top-k compared with what the union path returned; pf_cluster.hits "
                                  "carries recall_hit / recall_total (percent under /v1/stats engage). Diagnostic, costs one full-width row per block",
                                  attrs=(("pf_cluster", "RECALL", _on),)),
    "pf_idx_cpack": Flag("KYOJIN_PF_IDX_CPACK", "0", "prefill indexer of a plain layer (owns its keys, not the candidate source or a consumer) past pf_idx_cpack_min keys: the exact superblock bound of pf_idx_bound with grouped "
                         "box + ball summaries (pf_idx_cpack_group dims per group, tighter than the whole-vector bound), computed from the fp4 codes of the key slots; the fp16 keys exist only as a transient decode "
                         "inside the layer call, so the key mirror is neither built nor grown by the prefill (cpackscan; same indices as the stock chain; needs pf_idx_gscore=1 on the GPU, off while pf_idx_qmm is on; "
                         "the gain depends on how clustered the index keys are, not measured)",
                         attrs=(("cpackscan", "ENABLED", _on),)),
    "pf_idx_cpack_min": Flag("KYOJIN_PF_IDX_CPACK_MIN", "131072", "pf_idx_cpack: keys from which a plain layer takes the packed bound path", kind="int", attrs=(("cpackscan", "MIN_KEYS", int),)),
    "pf_idx_cpack_group": Flag("KYOJIN_PF_IDX_CPACK_GROUP", "32", "pf_idx_cpack: dims per bound group (128, 64, 32 or 16; 128 is the pf_idx_bound bound)", kind="int", attrs=(("cpackscan", "GROUP", int),)),
    "kv_digest_inc": Flag("KYOJIN_KV_DIGEST_INC", "0", "prefill with the kv-disk tier: the checkpoint digests (of ids[:b] and of the parent length) come from the session's running blake2b instead of hashing the whole prefix twice per "
                          "checkpoint (an O(b) list-to-array copy and hash per 8192-token chunk: ~120 ms per chunk at 1M tokens on a 4-core x86 sandbox, host only, GPU idle). 1 = incremental, 2 = incremental and "
                          "checked against the one-shot hash (mismatch counted in stats kv_digest_mismatch, one-shot used). Same digests, same files (kvdisk.save dg / pdg)", kind="int"),
    "eval_qos": Flag("KYOJIN_EVAL_QOS", "0", "engine thread (the thread MLX encodes every GPU command buffer on) runs at QOS_CLASS_USER_INTERACTIVE "
                     "(pthread_set_qos_class_self_np); the engine loop applies a change at its next tick, 0 puts the thread back on the class it had; macOS only (applegpu)"),
    "residency_refresh_ms": Flag("KYOJIN_RESIDENCY_REFRESH_MS", "1000", "MLX re-requests residency of its sets every n ms on a background thread "
                                 "(MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS; macOS can drop residency after GPU idle); 0 = MLX default (off)", kind="int", runtime=False,
                                 reason="MLX reads MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS when the Metal device is built, before the first GPU command"),
    "fired": Flag("KYOJIN_FIRED", "0", "lever-fired counters (fired.HITS, listed under /v1/stats engage): one counter per lever site that took the lever's path, so a live run can show a lever ran. "
                  "Off = one global read per site. Sites inside compiled regions count at trace time only",
                  attrs=(("fired", "ON", _on),)),
    "mix_decode": Flag("KYOJIN_MIX_DECODE", "0", "serve engine, decode during prefill: percent (0..95) of the wall time given to decode rounds while a prefill chunk is pending and "
                       "sessions decode. After a chunk that took t ms the decoders are owed t * m / (100 - m) ms of rounds before the next chunk (the chunk grid is untouched). "
                       "0 = today: decode_per_chunk rounds (1) between two chunks. The prefill rate of that stretch falls to (100 - m) % by construction", kind="int"),
    "mix_slice_ms": Flag("KYOJIN_MIX_SLICE_MS", "250", "serve engine, decode during prefill: a decode round of the decoding sessions runs INSIDE a prefill chunk, at the layer bounds "
                         "(bound_layers, after the bound's eval), whenever the last round STARTED this many ms ago. 0 = off (a decoder waits a whole chunk). Needs a backend with "
                         "supports_yield (VlmBackend with bound_layers on); only a rebuild is not sliced", kind="int"),
    "pf_srpt": Flag("KYOJIN_PF_SRPT", "1", "serve engine, needs pf_fair: among sessions prefilling, the next chunk goes to the one with the fewest tokens left, with an age bonus so a "
                    "long prompt cannot starve. No text changes: every session keeps its own chunk grid and cache"),
    "pf_floor": Flag("KYOJIN_PF_FLOOR", "15", "serve engine, needs pf_fair: tok/s each decoding session keeps while another session prefills (0 = off, the old behaviour). Turns on "
                     "fewest-tokens-left-first prefill order (as pf_srpt) and replaces mix_slice_ms by decode BURSTS: the decoders are owed duty/(1-duty) of the prefill time, duty = "
                     "floor * measured round seconds / tokens per session per round, capped by pf_floor_max; a burst runs once pf_floor_period_ms of prefill has accrued that debt, "
                     "at a layer bound, and again at the chunk end for the rest. No text changes (chunk grids and caches untouched). The floor is a target: the cap wins", kind="int"),
    "pf_floor_max": Flag("KYOJIN_PF_FLOOR_MAX", "60", "serve engine, flag pf_floor: most percent (0..95) of the wall the decode bursts may take while a prefill runs", kind="int"),
    "pf_floor_period_ms": Flag("KYOJIN_PF_FLOOR_PERIOD_MS", "1000", "serve engine, flag pf_floor: prefill milliseconds between two decode bursts (a burst lasts period * duty/(1-duty))", kind="int"),
    "pf_floor_catchup": Flag("KYOJIN_PF_FLOOR_CATCHUP", "30", "serve engine, flag pf_floor: percent (0..95) of a chunk's wall the decode bursts may take to bring a decoder that is behind pf_floor over its own window "
                             "(first look with a prefill running -> now) back to the floor, on top of the duty; the first decoder meets a cold duty estimate and a feedback ramp of 1.2x per chunk "
                             "alone, and later ones inherit the learned factor. N=4 33K: lowest decoder 9.1 -> 16.6 tok/s, mean TTFT 42-46 -> 47.5 s. 0 = off", kind="int"),
    "pf_fair": Flag("KYOJIN_PF_FAIR", "1", "serve engine, several sessions prefilling: the queue takes turns by chunk (the session whose last chunk started longest ago runs next, "
                   "a session that has not run yet first) instead of strict FIFO. A short request that arrives behind a long prompt then reaches its first token in about one "
                   "chunk of the long one instead of after all of it; every session keeps its own chunk grid and cache, so no text changes. Flag pf_share holds a session back "
                   "while the running one can still hand it the shared prefix (share_held counts the held chunk turns). 0 = FIFO (read per chunk)"),
    "lk_keep": Flag("KYOJIN_LK_KEEP", "1", "serve: the prompt-lookup index of a long prompt (65536 tokens or more) is kept for the next prompt that starts with it, the same index as a fresh insert. "
                    "A fresh insert is a sort: 118 ms of restore on each turn at 983K tokens"),
    "fin_nosync": Flag("KYOJIN_FIN_NOSYNC", "1", "serve: when every member of the batch finishes, the batched cache is dropped without writing it back to the rows: nothing reads them. "
                       "The write-back cost 900 ms between the last token and done at 983K tokens"),
    "sync_tail": Flag("KYOJIN_SYNC_TAIL", "1", "stacked decode, write-back of a session that stays (a member leaves or joins, a yield): only the rows the stack appended are packed, the older codes "
                      "stay. The same codes as the whole re-pack, which costs 900 ms at 983K tokens for each write-back", attrs=(("fp4pool", "SYNC_TAIL", _on),)),
    "mu_keep": Flag("KYOJIN_MU_KEEP", "1", "serve batched decode: a member joining or leaving updates only its own rows. The stacked buffers reserve a spare session slot and each session owns one "
                  "(`stack_rows`), `Plan.sess` holds the slot of a row, so a leave drops the slot from the map after writing that one row back and a join writes the joining session's rows into a free "
                  "slot (`KYOJIN_MU_KEEP_SPARE` spare slots, default 1, else the stack grows once). The members that stay are not extracted, re-merged, re-stacked or re-packed: at 131072 tokens that "
                  "round trip is ~4 GB of state traffic per join and per leave (the 525 / 541 ms gaps). Exact: same rows, same order, same bits (tests/test_mustall.py). 0 = the dissolve and "
                  "rebuild of 7f854df. Never recasts a pass in flight, a non-slot stack or a backend without `recast` (the old path then runs unchanged)"),
    "tok_inc": Flag("KYOJIN_TOK_INC", "1", "serve: a prompt of 64K chars or more is tokenized from where it differs from a recent prompt (serve/tokinc.py), the same ids as a whole tokenize. "
                    "At 983K tokens a whole tokenize takes 0.85 s, twice per chat turn, under the tokenizer lock", attrs=(("serve.tokinc", "ON", _on),)),
    "ttft_stages": Flag("KYOJIN_TTFT_STAGES", "1", "serve engine, stats: the time to first token of every request is split into render / tokenize (server), queue wait, prefix "
                     "lookup, restore, first chunk (first eval), whole prefill, first sample, and the total, returned in the done stats and aggregated as t_ttft_<stage> "
                     "(_max for the worst). One perf_counter pair per stage, no GPU work, nothing per token: on by default (read per request)"),
    "pf_share": Flag("KYOJIN_PF_SHARE", "0", "serve engine, sub-agents on a shared prefix: a session that has not started takes the cache of a prefilling session at the last chunk end both prompts "
                     "reach by the same chunk path (the leader's real path equals the child's own grid there; a view of the cache, kept outside the prefix-cache budget but inside kv_budget, "
                     "one Seed for all takers), instead of prefilling from the last block snapshot or from 0. 0 = today"),
    "pf_cut_merge": Flag("KYOJIN_PF_CUT_MERGE", "0", "serve engine, DS4.1: the chunk that ends at the LAST chat cut runs on to the end of the prompt (the generation prompt, <= 64 tokens) as ONE pass "
                         "and the cut snapshot (memory and disk, same keys and digests) is built from the state the pass captured at the cut (the attention of each layer runs in two segments, "
                         "the MoE once over all rows: the fixed expert-decode cost is paid once). 1 = passes that are not CED passes, 2 = also CED passes (run exact), 3 = also CED passes kept CED (decoder skip kept: the decoder half runs tail + the rows past the cut, captured in two segments). Falls back to the split "
                         "path: CED pass in mode 1, TurboQuant cache, images, rebuilt session, tail > 64, a chunk end in between, a pf_share taker at the cut. 0 = today", kind="int"),
    "pf_path": Flag("KYOJIN_PF_PATH", "0", "serve engine, prefix cache: a memory snapshot carries the chunk ends that built it and is used only by a prompt whose own grid has that path "
                    "(a snapshot taken after another prompt's cut would leave other bits than the asking prompt's solo prefill; a disk checkpoint has no path: position on the grid "
                    "only). Costs reuse: the chat layer's cut at the end of a conversation is not a cut of the next turn, so the next turn restarts from an earlier snapshot. 0 = today"),
    "kv_admit": Flag("KYOJIN_KV_ADMIT", "0", "serve engine, memory (needs kv_budget): a request is admitted when the bytes reserved for the active sessions (the larger of what a cache holds and what it may "
                     "grow to in one stretch: measured bytes per token x (prompt + reply)), the prefix snapshots and the hand-off seeds plus its own need fit in kv_budget, else it waits; no second "
                     "session starts before bytes per token is measured; a request that cannot fit alone is refused (ValueError at submit once bytes per token is known, finish reason "
                     "'refused' when admitted); when the live caches go over the budget and nothing else can be dropped the youngest decoding session is parked, and its cache dropped and "
                     "rebuilt later (prompt on the grid + replay of its tokens). 0 = today: admission counts the prompt only, over budget with only active sessions is logged"),
    "pf_win_slice": Flag("KYOJIN_PF_WIN_SLICE", "0", "prefill attention, sliding-window ring (cpfbase): the history read of a chunk (mx.take, a Gather per layer) and the ring write "
                         "(a Scatter on a copy of the ring) become Slice + Concatenate of the same rows (pf_ring). Exact: copies only, bit-identical ring and window kv. 0 = today",
                         attrs=(("pf_ring", "ENABLED", _on),)),
    "pf_pool_lazy": Flag("KYOJIN_PF_POOL_LAZY", "0", "prefill, compressed KV pool append (cpfbase; the index keys are left to cpackscan): the on-grid check of an append (pack, unpack, "
                         "array_equal, read back with bool() = one host sync and a GPU drain per KV-source layer per chunk) stays a lazy array, evaluated with the state at the chunk end. "
                         "The appended rows are written to the dense mirror as they are, so every read is exact before the verdict; a layer that comes back off the grid goes dense, as today. "
                         "Needs depthflat (mirror) and an existing mirror (from the second append of a layer); else the checked append. 0 = today",
                         attrs=(("fp4pool", "LAZY_POOL", _on),)),
    "pf_keys_lazy": Flag("KYOJIN_PF_KEYS_LAZY", "0", "prefill, index-key append (fp4pool.KeySlots, cfreshgap): the on-grid check of every key append (pack, unpack, array_equal, bool() = one host sync "
                         "and a GPU drain per indexer layer per chunk, ~one per layer on top of the pool's) stays a lazy array, as pf_pool_lazy does for the pool: the appended rows go to "
                         "the dense key mirror as written, so every read is exact before the verdict, and an off-grid layer goes dense exactly as the checked append makes it. Needs depthflat "
                         "and an existing key mirror (from the second append of a layer; not with pf_idx_cpack, which builds none), else the checked append. Not measured on Metal "
                         "(tools/cfreshgap_mac.py). 0 = today",
                         attrs=(("fp4pool", "LAZY_KEYS", _on),)),
    "pf_grow": Flag("KYOJIN_PF_GROW", "0", "serve, the pool / index keys of a cache restored from a prefix snapshot or a disk checkpoint: the restore copies the rows once into a code buffer "
                    "with room for the next appends (`longctx.room`: rows + 1/16 + one step) instead of leaving a buffer of exactly the rows (or a view of the stored one), which the first append "
                    "grows with mx.zeros of the whole next capacity step and a copy of every row (985 MB of zeros + the 0.83 GiB copy at 983040 rows, per turn). The append then writes its rows "
                    "where they go. At rest +6.25 % of the codes (0.83 -> 0.88 GiB at 983040); reads are the same rows of the same codes (the capacity is only the row stride), and past the "
                    "room the growth rule is today's. 0 = today",
                    attrs=(("fp4pool", "RESTORE_ROOM", _on),)),
    "cfmtspeed": Flag("KYOJIN_CFMTSPEED", "0", "foreign formats (GGUF / NVFP4 / MXFP4 served as stored, foreign_linear; Qwen3-family mlx-lm models, not DS4.1 / GLM): at load, ForeignLinears "
                      "that read the same input (found by tracing one 1-row forward, as fused_group does: MLP gate / up, q / k / v, GDN projections) become ONE concatenated matrix "
                      "(foreign_pack.concat_rows: the members' bytes one after the other, nothing re-quantised) read by one launch, with a per-row global scale, and the E2M1 decode "
                      "is a 16-entry table read. Same float products and the same per-row accumulation order as the separate launches, so every output is bit-identical (R <= 64; "
                      "above, each member keeps its own matmul on the same decoded rows). Members of different formats, dense (FP8 / Q8_0 / Q6_K decoded to fp16) members and "
                      "ForeignSwitchGLU are left alone. Not measured on Metal (tests/test_cfmtspeed.py needs a Mac for the kernel half). 0 = today", runtime=False,
                      reason="the matrices are concatenated once at load and the member arrays freed"),
    "foreign_g2a": Flag("KYOJIN_FOREIGN_G2A", "1", "foreign formats, NVFP4 routed experts decoded one row per slot (GLM-5.3, Qwen-style NVFP4 MoE; k % 1024 == 0, k <= 12288, n % 16 == 0): the gather GEMV "
                        "is the register-fed kernel of foreign_metal.gemv_g2_source (1 output row per simdgroup, 16 simdgroups per threadgroup, every weight load of a row issued before the x barrier, "
                        "E2M1 pair decode in 5 integer ops, block scale applied to the 16-weight dot product). ForeignSwitchGLU (GLM: the mlx-vlm MoE calls it as switch_mlp) runs gate | up in one "
                        "launch, silu * up with the swiglu limit in one launch (instead of ~8 elementwise ops) and the down gather. Same products, float sum order differs (fp16 outputs within 1 ulp-ish, "
                        "tests/test_foreign_g2a.py). Micro-bench at the GLM shapes (E 288, gate / up 2048 x 4096, down 4096 x 2048, R rows x top-8, tools/g2a_gather_bench.py): ForeignSwitchGLU per layer R1 160 -> 118 us, R2 265 -> 218, R4 491 -> 418. "
                        "0 = the previous kernels (read per call)"),
    "fp8stored": Flag("KYOJIN_FP8STORED", "1", "foreign formats (foreign_linear, Qwen3-family models): FP8 E4M3 safetensors linears (per-tensor / per-channel scale; block scales, E5M2 and "
                      "experts stay decoded) and GGUF Q8_0 matrices (not stacked experts) are read as stored, one byte per weight: a GEMV kernel up to KYOJIN_FOREIGN_GEMV8_ROWS (8) rows, "
                      "a simdgroup-matrix kernel (32-row slices) up to KYOJIN_FOREIGN_DENSE8_ROWS (16), above it a dequant kernel to a transient fp16 matrix and one MLX matmul. "
                      "Value 2 (default) also keeps a resident fp16 copy of those matrices, which runs the matmul from KYOJIN_FOREIGN_F16_ROWS (16) rows: GEMV decode gain, no N=4 loss, "
                      "bytes of the 8-bit linears x3 against 1 (measured 27B NVFP4: 32.4 GB off, 22.5 GB at 1, 42.3 GB at 2). 1 = stored bytes only (22.5 GB, N=4 loses 15-35 %). "
                      "Scale applied to the sum in float, so not bit-equal to the fp16 decode at load. 0 = decoded to fp16 at load, as before", runtime=False,
                      reason="the weights are laid out once at load", kind="int"),
    "fp16pad": Flag("KYOJIN_FP16PAD", "1", "foreign formats (foreign_linear._matmul: dense linears and the resident fp16 copies): a product of fewer than 16 rows on a matrix of 8M weights or more runs "
                    "as the 16-row product over cached zero rows, then keeps its own rows. MLX's fast path starts at 16 rows: 8 x 8192^2 ms plain / padded R2 2.52/1.50, R8 2.05/1.18, R12 2.82/1.17. "
                    "Real 27B Q4_K_M: the tuner's R12 round 102 -> 82 ms at N=4, N=8 sum 159-178 -> 173-190, solo and N=4 equal. 0 = plain product. Read at import"),
    "mu_remeasure": Flag("KYOJIN_MU_REMEASURE", "1", "serve.mu_tuner: once every arm of a session count has a price, the arm measured first (on cold pipelines) is measured once more and its "
                         "price replaced. 0 = the first price stays until the neighbour probe. Read per call"),
    "mu_unseen": Flag("KYOJIN_MU_UNSEEN", "100", "serve.engine._pooled_acceptance: a draft position never tried at the current width inherits this percent of the acceptance of the one before (was 90; "
                      "real 27B conditional acceptance rises with depth). Read per call", kind="int"),
    "f4copy": Flag("KYOJIN_F4COPY", "0", "foreign formats (foreign_linear.DevPack, 4-bit stored: Q4_0, Q4_K, IQ4_NL, MXFP4, NVFP4): a resident fp16 copy of each matrix when it fits under 60 % of the working set, "
                   "one matmul on the copy from KYOJIN_FOREIGN_F4_ROWS (16) rows, stored kernels below. Trades 4x bytes for N sessions: real 27B N=4 sum tok/s Q4_0 108 -> 130, NVFP4 115 -> 131, Q4_K_M 125 -> 140, solo equal; "
                   "Q4_0 25 -> 67 GB, NVFP4 42 -> 70 GB. No copy for expert stacks. Above the copy's rows fp16 overflow is unsaturated, as on the stock dense path. 0 = stored kernels only. Read at load"),
    "decfuse": Flag("KYOJIN_DECFUSE", "1", "fewer launches per decode / verify round (decfuse.py, decrows.py, gdn_pf.py, qspec.py): master switch of the three classes below, for MLX affine packs (compiled 1-row regions) "
                    "and for foreign packs (GGUF / NVFP4) at any row count 1..128. 0 = stock op bodies. Foreign packs, R=6 round of the 27B stack: ~2750 -> ~1700 dispatches (tools/roundops.py --gpu)",
                    runtime=False, reason="read at import; the same-input groups are built once at load"),
    "decfuse_proj": Flag("KYOJIN_DECFUSE_PROJ", "1", "decfuse class 1: same-input projections (q|k|v, gate|up, qkv|z, alpha|beta) in one launch. MLX affine packs: FusedQ at load; foreign packs: "
                         "foreign_linear.group_same_input now also runs at load with this switch (before: only KYOJIN_CFMTSPEED). Rows of the concatenated matrix are independent, bit-equal",
                         runtime=False, reason="the matrices are concatenated once at load"),
    "decfuse_addnorm": Flag("KYOJIN_DECFUSE_ADDNORM", "1", "decfuse class 2: residual add + RMSNorm in one launch (decfuse.add_rms; fp16 sum saturates at +-65504). Foreign packs: decrows.enable_foreign installs the "
                            "seam forward for 1..KYOJIN_DECFUSE_ADDNORM_ROWS (128) rows. Bit-equal to the stock pair (tests/test_roundfuse.py)", runtime=False, reason="the forward is installed at load"),
    "decfuse_gdn": Flag("KYOJIN_DECFUSE_GDN", "1", "decfuse class 3: gated-delta front end (conv, silu, q/k L2 norm, g, beta) + gated output norm in 2 launches around the step recurrence. "
                        "qspec's captured call (served MTP / verify rounds of any model class with head dim 128) takes it for 1..KYOJIN_DECFUSE_GDN_ROWS (128) rows; the rollback replays on the fused g / beta. "
                        "Within fp16 rounding of the stock ops (not bit-equal)", runtime=False, reason="read at import"),
    "mqcopy": Flag("KYOJIN_FOREIGN_MQCOPY", "0", "foreign formats (foreign_linear.DevPack, 4-bit stored: Q4_0, Q4_K, MXFP4, NVFP4, and GGUF Q5_K as MLX affine bits=5; no expert stacks): a resident MLX-native quantized copy (~1.1x the stored bytes, mapped code for code: "
                   "affine g32 for Q4_0 / Q4_K, MLX modes mxfp4 / nvfp4, the nvfp4 global scale applied after) runs mx.quantized_matmul for KYOJIN_FOREIGN_MQ_MIN..MQ_MAX (12..127) rows, ahead of the fp16 copy. "
                   "MLX takes a slow path up to 32 rows, so there the copy serves only n >= 10240 with k <= 5120; from 33 rows it serves every matrix of 8M weights or more. 64 mats ms before > after, 17408x5120 "
                   "R12 8.0 > 7.1, R16 8.0 > 6.6, R24 11.4 > 7.0, R32 11.9 > 6.9, R48 19 > 10, R64 17 > 10; 5120x17408 R48 14.7 > 10.4, R64 14.5 > 10.0, R12-32 unchanged. Kept only while it fits (60 % of the working set). 0 = fmma / fp16 copy. Read at load"),
    "foreign_fmma": Flag("KYOJIN_FOREIGN_FMMA", "1", "foreign formats (foreign_linear.DevPack, stored Q4_0, Q4_K, NVFP4, MXFP4; not IQ4_NL, whose table decode loses to the old kernel): rows 5..16 run "
                         "foreign_metal.fmma_source, the q4mma3 design (half2 decode, shared transposed x, 2 row tiles per simdgroup, K split over 4 simdgroups) instead of the float MMA. "
                         "17408x5120 over 64 matrices, R8: Q4_K 456 -> 685, NVFP4 465 -> 694, MXFP4 456 -> 684, Q4_0 546 -> 745 GB/s; R16 316-360 -> 390-420 (the MMA count doubles). Needs k % 128 == 0, no expert stacks. "
                         "Weights are decoded in half (as the K formats' mmak); 0 = the float MMA. Read per call"),
    "kstored": Flag("KYOJIN_KSTORED", "1", "foreign formats (foreign_linear, GGUF files of Qwen3-family models): Q5_K, Q6_K and IQ4_XS 2D matrices are read as stored (0.53-0.83 byte per weight "
                    "instead of the 2 of the fp16 decode): a GEMV kernel (foreign_metal.gemvk_source) up to KYOJIN_FOREIGN_GEMVK_ROWS (8) rows, above it one MLX matmul on a transient decode "
                    "(dequantk_source). Value 2 also keeps a resident fp16 copy for those rows while it fits under 60 % of the working set. Row sums in float, saturated to the fp16 range. "
                    "0 = decoded to fp16 at load, as before", runtime=False, reason="the weights are laid out once at load", kind="int"),
    "legacy5_stored": Flag("KYOJIN_LEGACY5_STORED", "0", "foreign formats (foreign_linear, GGUF files): Q4_1, Q5_0 and Q5_1 2D / 3D weights (any width) are read as stored (0.625-0.75 byte per weight) "
                            "by the K-class kernels (foreign_metal q4_1 / q5_0 / q5_1: GEMV, gather, few-row MMA, transient decode above the rows) instead of the resident fp16 decode "
                            "(2 bytes per weight). Off until tests/test_legacy5_metal.py passes on the GPU. 0 = decoded to fp16 at load, as before", runtime=False,
                            reason="the weights are laid out once at load", kind="int"),
    "q2_0_stored": Flag("KYOJIN_Q2_0_STORED", "0", "foreign formats (foreign_linear, GGUF files): GGML type id 42 (Q2_0: 64 weights = fp16 d + 16 bytes of 2-bit codes, w = (q - 1) d) becomes a known weight type "
                         "(reader, exact dequant) and its 2D / 3D weights are read as stored (0.28 byte per weight) by the K-class kernels (foreign_metal q2_0: GEMV, gather, few-row MMA, transient decode above the rows). "
                         "Off until the Metal test of gpu_archkit/types2_metal.sh passes on the GPU. 0 = id 42 is an unknown type id (per-tensor failure, day0_inspect TYPE NEEDED), as before", runtime=False,
                         reason="the weights are laid out once at load"),
    "gemvk_partial": Flag("KYOJIN_GEMVK_PARTIAL", "0", "foreign formats, K-class stored kernels (foreign_metal.gemvk_source: GEMV, routed gather, glu): a stored matrix whose width k is not a multiple of the 512-weight chunk "
                          "(k % 32 == 0: Q4_1 / Q5_0 / Q5_1 / Q2_0 experts of 640 / 704, K and IQ types of 768 / 1280) takes the GEMV / gather kernel with a partial last chunk (zeros staged past k, unit index clamped to the row) "
                          "instead of the transient decode + matmul. The source of widths % 512 == 0 is unchanged. Off until the Metal tests of gpu_archkit/types2_metal.sh pass", runtime=False,
                          reason="set before the engine starts: the kernel route of a loaded layer must not change under it"),
    "moe_partialk": Flag("KYOJIN_MOE_PARTIALK", "0", "foreign formats, routed-expert stacks only: Q4_1 / Q5_0 / Q5_1 / IQ4_NL experts whose width k is not a multiple of 512 (k % 32 == 0, e.g. 640) take the partial-chunk "
                          "gather / pair GEMV (as gemvk_partial but for experts) and the grouped prefill (foreign_kqgmm kinds q4_1 / q5_0 / q5_1 / iq4_nl, k % 128 == 0). Dense matrices unchanged; the q4_1 / q5 files are kept stored "
                          "only with KYOJIN_LEGACY5_STORED (load time). Off until the GPU micro-gates (tools/moe_partialk_gate.py) are reviewed", runtime=False,
                          reason="set before the engine starts: the kernel route of a loaded layer must not change under it"),
    "moe_partialk_fp4": Flag("KYOJIN_MOE_PARTIALK_FP4", "0", "NVFP4 routed-expert stacks with k % 32 == 0, k % 1024 != 0 (Qwen3.8-Flash-Next down k = 640): the g2a gather / pair kernels in their partial-unit "
                              "text instead of the per-expert dequant loop (decode); glm_moe / glm_moe_rows keep their k % 1024 gate. Off until the GPU micro-gate (tools/moe_partialk_fp4_gate.py) is reviewed", runtime=False,
                          reason="set before the engine starts: the kernel route of a loaded layer must not change under it"),
    "arch_skeleton": Flag("KYOJIN_SKELETON", "0", "foreign formats, arch kit (foreign_archkit_cpu.load_cpu, foreign_archkit_serve.register): a GGUF architecture with no mlx-lm model class is served by the config-driven "
                          "decoder skeleton (foreign_skeleton; rows in foreign_skeleton_rows), and an architecture that has both a class and a skeleton row takes the skeleton (checks). Off until the Metal runs of "
                          "gpu_archkit/skeleton_identity.sh are reviewed. 0 = KitError for an architecture with no model class, as before", runtime=False,
                          reason="read when the file is loaded"),
    "kit_unproven_rows": Flag("KYOJIN_KIT_UNPROVEN_ROWS", "0", "foreign formats, arch kit: the rows that have no Metal identity proof yet are in force. A row marked proven=False in its table (foreign_cfgrows.ROWS / foreign_archkit.VETTED) or listed in "
                          "foreign_unproven (the model_type aliases of minimax-m3 and nemotron_h_moe, the glm5-next GGUF rows of foreign_glm_gguf) does not exist while this is 0: every load path and tools/day0_inspect.py "
                          "give the verdict they gave without it (plus one line saying the rows exist). Today: gemma4, lfm2moe, glm5-next, the two aliases. Off until the Metal runs of gpu_archkit/ identity scripts pass",
                          runtime=False, reason="read when the file is loaded / the module is imported"),
    "kit_vlm_classes": Flag("KYOJIN_KIT_VLM_CLASSES", "0", "foreign formats, arch kit (foreign_vlm_classes): an architecture for which mlx-lm has no model class takes the text model class of mlx-vlm of the same model_type "
                            "(cohere2_moe, flint); the vision tower of a wrapper is dropped when the file has no vision tensors. mlx-lm is always asked first, so no architecture that resolves today changes class. "
                            "0 = KitError as before. The rows of those architectures are also unproven (kit_unproven_rows)", runtime=False, reason="read when the file is loaded"),
    "mapped_row_table": Flag("KYOJIN_MAPPED_ROW_TABLE", "0", "foreign formats (foreign_mapped_rows): a GGUF table the model class reads by row id (qwen4exp's per-layer n-gram table, 28.8 GB) stays in the mapped file and only the "
                              "rows of the step are decoded (fp16, on the host). Off: the tensor stays in Arch.unserved and the loader raises, as before. Read at load.", runtime=False, reason="read when the model loads"),
    "mapped_row_batch": Flag("KYOJIN_MAPPED_ROW_BATCH", "0", "foreign formats (foreign_mapped_rows.RowBatch): with mapped_row_table, the n-gram rows of ALL layers of a decode step are ids-evaluated, read and decoded in ONE host call "
                              "(one GPU sync per step instead of one per PLE layer); identical rows by construction. Steps over 64 tokens keep the per-layer path. Off: per layer, as before. Read at load.", runtime=False, reason="read when the model loads"),
    "mapped_row_prefetch": Flag("KYOJIN_MAPPED_ROW_PREFETCH", "0", "foreign formats (foreign_mapped_rows.pretouch): with mapped_row_table, a background thread reads one byte of every 16 KiB page of the mapped row table so the decode steps'"
        " row gathers do not pay a first-touch page fault per row (page-cache hits that still fault, ~12 per token measured). 1 = whole table, N>1 = first N GiB (a table larger than the page cache). Read only: no value changes; clean file-backed pages. Off: first touch in the step, as before. Read at load.", runtime=False, reason="read when the model loads"),
    "mapped_row_nvfp4": Flag("KYOJIN_MAPPED_ROW_NVFP4", "0", "foreign formats (foreign_mapped_rows.MappedNvfp4Rows, foreign_qwen4exp_st): the NVFP4 n-gram table of a Qwen3.8-Flash-Next safetensors pack (128 shards, 28.8 GB) stays in the mapped files and only the "
                               "rows of the step are decoded (fp16, on the host); nothing else of the pack changes. Needs the class lookup of the unproven qwen4_exp row (KYOJIN_KIT_UNPROVEN_ROWS). Off: the shards load as 4-bit Packs, as for any pack. Read at load.", runtime=False, reason="read when the model loads"),
    "gemma4_rows": Flag("KYOJIN_GEMMA4_ROWS", "0", "foreign formats, arch kit (alias of kit_unproven_rows for gemma4 alone): the gemma4 rows (foreign_cfgrows config row, foreign_archkit VETTED line) are in force: load_cpu and the serve register build the mlx-lm gemma4_text class from the GGUF, "
                        "and tools/day0_inspect.py reads the vetted line. Off until the Metal run of gpu_archkit/gemma4_identity.sh passes. 0 = gemma4 has no config row and no vetted line, as before "
                        "(day0_inspect: the same verdict as without the rows)", runtime=False, reason="read when the file is loaded"),
    "glm_dense_rows": Flag("KYOJIN_GLM_DENSE_ROWS", "0", "GLM-5.3 verify rows (G3e2): a dense FP8 projection (qkv_proj, o_proj, f_b / g_b, q_a / kv_a, q_b, shared experts, lm_head) of 2..4 rows of one session is ONE G3 GEMV "
                        "launch that reads its weights once (needs KYOJIN_GLM_DENSE_GEMV); stock = mlx_vlm's `tokenwise` fallback, one launch and one full weight read PER ROW (cut ladder lm_head 0.56 / 1.16 / 1.59 / 2.14 ms "
                        "at R = 1..4). Row r of the result equals the R = 1 result bit for bit. Read per call",
                        attrs=(("glm_dense_rows", "ENABLED", _on),)),
    "glm_vrows": Flag("KYOJIN_GLM_VROWS", "0", "GLM-5.3 verify rows, N1-VROWS (needs KYOJIN_GLM_DENSE_GEMV; supersedes glm_dense_rows): every dense ForeignLinear (KDA in-proj / o_proj, DSA q_a / q_b / o, lm_head) of a verify block [B, L, H] "
                      "with B * L <= KYOJIN_GLM_VROWS_MAX (8) rows is ONE G3 / 6-bit (KYOJIN_GLM_DENSE_Q, fg_g6_r<R>) GEMV launch that reads its weights once. Stock = mlx_vlm's `tokenwise`: L launches of B rows "
                      "(MTP d1, 1 session = two fg_g6_r1 reads, 4 sessions = two fg_g6_r4 reads; G2h: ~3 ms of a 31.5 ms round). Row r of the result equals the per-row result bit for bit; more rows than the cap, "
                      "or the flag off = stock. Read per call",
                      attrs=(("glm_dense_rows", "VROWS", _on),)),
    "glm_compile": Flag("KYOJIN_GLM_COMPILE", "0", "GLM-5.3 decode: every linear-attention decoder layer (34 of 45) runs as one mx.compile region (conv / recurrent state in and out): the python graph build is paid once per (layer, rows) and replayed, "
                        "MLX fuses the elementwise chains. Plain decode steps only (one token per row, no mask, no speculative cache); anything else runs the stock layer (kyojin_mlx.glm_compile). Read per call",
                        attrs=(("glm_compile", "ENABLED", _on),)),
    "glm_mtpfeed": Flag("KYOJIN_GLM_MTPFEED", "0", "GLM-5.3 MTP head, PROMPT feed (P2-MTPFEED): the head's pairs of a prompt chunk are fed without the stock per-head projected K / V cache (64 x T x 256, grown by a full "
                        "copy per 512-row piece below 32768 tokens: 0.2-1.2 s of GPU idle per 8K chunk at offsets 0-24K). The head is one layer and its prompt-feed output is discarded, so its three caches "
                        "(latent, indexer keys, pool keys) are bit-identical; only the draft's own attention takes the latent path (as it does above 32768 tokens). Draft tokens verified, outputs exact "
                        "(kyojin_mlx.glm_nextn). Read per call",
                        attrs=(("glm_nextn", "MTPFEED", _on),)),
    "glm_hostgap": Flag("KYOJIN_GLM_HOSTGAP", "0", "GLM-5.3 decode (N3-HOSTGAP): mx.async_eval the layer output after the first KYOJIN_GLM_HOSTGAP_FIRST layers and then every KYOJIN_GLM_HOSTGAP_EVERY layers of a "
                        "decode step (<= KYOJIN_GLM_HOSTGAP_ROWS rows per session), so the GPU starts the step while the host is still building it instead of idling the ~3 ms the python graph build "
                        "takes after the previous step ends. Same ops, same order: only where the graph is committed moves (kyojin_mlx.glm_hostgap). Read per call",
                        attrs=(("glm_hostgap", "ENABLED", _on),)),
    "glm_hostgap_first": Flag("KYOJIN_GLM_HOSTGAP_FIRST", "1", "glm_hostgap: layers built before the first submit", kind="int", attrs=(("glm_hostgap", "FIRST", int),)),
    "glm_hostgap_every": Flag("KYOJIN_GLM_HOSTGAP_EVERY", "4", "glm_hostgap: layers between two submits", kind="int", attrs=(("glm_hostgap", "EVERY", int),)),
    "glm_hostgap_rows": Flag("KYOJIN_GLM_HOSTGAP_ROWS", "1", "glm_hostgap: most rows per session of a pass that submits early (1 = plain decode steps)", kind="int",
                             attrs=(("glm_hostgap", "ROWS", int),)),
    "glm_lag1": Flag("KYOJIN_GLM_LAG1", "0", "GLM-5.3 served MTP (G13, V1-VERIFY): the draft head's proposals stay on the device (async_eval, no host read) and the verify rows are built from them (GlmBackend.spec_verify_lazy), so the host builds the verify graph "
                     "while the GPU drafts instead of waiting for the draft first. Same graphs in the same order on one stream: the same tokens. Needs KYOJIN_GLM_CHAIN=0 (a chained / skip round reads the drafts first). Read per round"),
    "verify_poolsel": Flag("KYOJIN_VERIFY_POOLSEL", "0", "V1-VERIFY: PoolingCache.commit_speculation (any arch with pooled caches: GLM DSA layers, DS4.1 x 2 buffers) selects the remainder window with static slices + one pad instead of ~22 small mx ops "
                               "(arange / mod / clip / take_along_axis / where) on python ints. Pure data movement: bit-identical (kyojin_mlx.verify_glue). Read per call",
                               attrs=(("verify_glue", "POOLSEL", _on),)),
    "draft_head_f16": Flag("KYOJIN_DRAFT_HEAD_F16", "0", "V1-VERIFY: the MTP draft head (GLM-5.3 nextn) gets the target's float32 hidden and runs fully in float32 (weights cast f16 -> f32 on every call, 3.7 ms per round); with this the normed hidden is rounded to the "
                              "model dtype so the head runs in f16. Draft only (the verify pass is untouched, so every emitted token is the same as with any other draft; only the proposals / acceptance can move). Read per call",
                              attrs=(("verify_glue", "HEAD16", _on),)),
    "moe_rows2_unroll": Flag("KYOJIN_MOE_ROWS2_UNROLL", "0", "V2-VERIFY: the g2m routed-expert kernels (pair_glu / down, GLM-5.x MoE_E path) at T > 1 rows (MTP verify, 2-4 rows): the loops over the rows that picked an expert "
                            "(dynamic count, indexed accumulators) are unrolled over the row count so the accumulators live in registers instead of private memory. The same operations in the same order: bit-identical (tests/test_glm_moe_e.py). Read per call",
                            attrs=(("glm_moe_e", "ROWS2_UNROLL", _on),)),
    "dsa_attend_rows": Flag("KYOJIN_DSA_ATTEND_ROWS", "0", "V2-VERIFY: the DSA attend of the fused path (plain step and MTP verify rows, 1..16 (session, row) pairs; needs KYOJIN_GLM_FUSE_DSA) as scores / softmax / PV / sum launches "
                            "with thousands of independent simdgroups instead of the stock SDPA fallback (~8 launches per row, split-K P @ V ~90 us) (kyojin_mlx.dsa_attend_rows). Same rounding points as the stock path, fp32 sums in another order: 1-ulp-class "
                            "logits; every row is reduced on its own, so verify row i == plain row i with the flag on for both. Read per call",
                            attrs=(("dsa_attend_rows", "ENABLED", _on),)),
    "dsa_attend_cps": Flag("KYOJIN_DSA_ATTEND_CPS", "5", "dsa_attend_rows: 32-key chunks per P @ V split (more splits = more simdgroups, a longer fp32 sum pass)", kind="int", attrs=(("dsa_attend_rows", "CPS", int),)),
    "glm_fuse_dsa": Flag("KYOJIN_GLM_FUSE_DSA", "0", "GLM-5.3 DSA fused verify-row / batch indexer + attend kernels (kyojin_mlx.glm_fuse_dsa; with the variable unset the module follows KYOJIN_GLM_DSA_FAST at import, so this registry default is NOT applied). "
                         "The class hooks are installed at load only when the load started with it on; on such a load this switches the kernels off / on per call (a bisect on a live server); on a load started with 0 it is a no-op",
                         kind="mode", choices=("0", "1"), attrs=(("glm_fuse_dsa", "ENABLED", _on),)),
    "glm_fuse_dsa_eval": Flag("KYOJIN_GLM_FUSE_DSA_EVAL", "0", "GLM-5.3 FUSE_DSA verify rows (L > 1): async_eval the attention output of every Nth DSA layer of the pass (0 = off, 1 = every layer like the stock `_attend`, 11 = the first only). The stock path "
                              "evaluates every row output asynchronously, so the GPU starts while the host builds the rest of the pass; the fused path builds the pass ~4x faster but without this the build is serial in front of the GPU work (DSA2). Same arrays, only the host wait moves; read per call",
                              kind="int", attrs=(("glm_fuse_dsa", "EVAL", int),)),
    "glm_fuse_small": Flag("KYOJIN_GLM_FUSE_SMALL", "0", "GLM-5.3 decode: the linear-attention (KDA) body of a layer (conv + silu, f_b / g_b, q / k norms, g / beta, vector-gated delta rule, o_norm * sigmoid(gate)) runs as ONE launch "
                           "after the stock projections (kyojin_mlx.glm_small, G10b; oMLX kda_decode_step port), B >= 1, up to 8 rows; stock path for masks, prefill rows, non-plain linears. Read per call",
                           attrs=(("glm_small", "ENABLED", _on),)),
    "glm_fuse_small_spec": Flag("KYOJIN_GLM_FUSE_SMALL_SPEC", "1", "glm_fuse_small also serves the verify rows of a speculating cache (G8d): the kernel writes the post-token states of rows 0..T-2 and the conv window is "
                                "recorded through the cache's record API, so a partial accept keeps the stock states; 0 = the stock body there (G10b as shipped: every MTP verify round ran the 26 / 35 / 51-launch stock "
                                "body of the 34 layers). Only read when glm_fuse_small is on; read per call",
                                attrs=(("glm_small", "SPEC", _on),)),
    "glm_rowfold": Flag("KYOJIN_GLM_ROWFOLD", "0", "GLM-5.3 decode (N2-ROWDOT, needs glm_fuse_small + dense_q): the KDA fbg_a projection (fp16 [4096 -> 320], one rowdot2 launch per layer that G2h measured at 138 us per layer "
                        "paired with the 115 us qkv GEMV) runs as extra threadgroups of the 6-bit qkv GEMV launch (kyojin_mlx.glm_rowfold): same arithmetic as rowinv.rowdot (bit-identical), one launch less per KDA layer "
                        "(34 per decode step). Decode rows 1..4 of fp16 x only; verify rows and other types keep the two launches. Read per call",
                        attrs=(("glm_rowfold", "ENABLED", _on),)),
    "glm_plan": Flag("KYOJIN_GLM_PLAN", "0", "GLM-5.3 served MTP: the drafts of every session come from ONE per-round plan that maximises expected tokens per ms of the pass (kyojin_mlx.glm_cost: measured "
                     "verify / draft / commit ms by rows, each session's own per-position acceptance), instead of the cap (solo) or the fixed depth (several sessions); the n-gram gate prices its "
                     "rounds from the same table. A row costs ~12 ms here (a plain step 31), so depth 1 at one session, plain from ~3 sessions unless acceptance is near 1. Read per round"),
    "glm_chain": Flag("KYOJIN_GLM_CHAIN", "0", "GLM-5.3 served MTP: when the head's draft token equals the first token of the session's n-gram continuation (two independent drafters agree, even on a "
                      "4..7-token match the lookup gate refuses alone), the rest of that continuation is appended as free draft rows (no head step), as many as pay at the pass's rate. Read per round"),
    "glm_adaptive": Flag("KYOJIN_GLM_ADAPTIVE", "0", "GLM-5.3 served MTP: the drafts of every session come from an online controller (kyojin_mlx.glm_adaptive, adapted from oMLX's Lightning MTP depth controller): "
                         "per pass, the vector of depths that maximises expected tokens / measured ms, where the ms of a pass of n sessions and R rows is the live EMA of the served rounds (the glm_cost table only "
                         "fills the keys nobody measured, scaled by the measured / table ratio) and acceptance is a per-position EMA per session and text class. Rivals within 15 % are probed 4 rounds at a "
                         "duty-bounded cadence (cooldown doubles when they lose); drafts that do not beat the plain pass by 3 % park the pass. Takes precedence over glm_plan; a fixed depth (mtp_fixed) wins "
                         "over it. KYOJIN_GLM_ADAPTIVE_CLASS=0 keeps one class. Read per round"),
    "glm_multi_plain": Flag("KYOJIN_GLM_MULTI_PLAIN", "0", "Session-count depth policy of a served MTP backend (name kept from DEPTH-B, GLM-5.3 first): keep --mtp-depth 1 but run the TRUE plain step (the decode tick: lookahead, "
                            "no verify pass, no tuner) every round in which more than one session is decoding (sessions prefilling or waiting do not count); one decoding session drafts as the policy says. Measured on the HOSTGAP lane: the plain "
                            "step is 17 ms, a d1 round 31 ms, so d1 loses 5-7 % aggregate at 2-4 sessions and wins ~2 % at 1. Crowd rounds are the depth-0 step (byte-identical to plain); solo d1 rounds keep the known d1 near-tie divergence. Not default-on outside GLM-5.3 before the 2nd-arch gate (DS4.1 pack, Qwen3.5-2B vs multi_auto)."
                            " A fixed depth (mtp_fixed) wins over it. Any MTP backend (needs the plain decode tick, which every backend has); opt-in, a backend whose verify pass is shared is to be A/B-ed first. Read per round"),
    "spec_depth_auto": Flag("KYOJIN_SPEC_DEPTH_AUTO", "0", "Speculation depth that follows the recent acceptance (kyojin_mlx.spec_depth, SPEC-DEEP; any MTP backend, GLM-5.3 first): per session an EMA of the acceptance of every draft "
                            "position (only the positions a round reached), the depth = argmax over 0..mtp_depth of expected tokens / round ms (E[T] = 1 + p1 + p1 p2 + ...; ms = the measured warm round of that depth, else a prior shape "
                            "scaled by the nearest measured depth); one more row per round at most, back down at once, the plain step when the best drafted rate is below 0.9 x plain's (a d1 probe every 16 plain rounds). "
                            "Predictable text (code continuation, edits, tool output) drafts 2-4 deep, prose stays at d1. mtp_depth is the CEILING (load with --mtp-depth 4 and KYOJIN_GLM_SPEC_ROWS >= 5; flag off = the fixed depth as before). "
                            "Solo only: with 2+ sessions the engine's own rule runs capped at depth 1 and 4 verify rows (KYOJIN_GLM_MULTI_PLAIN still governs crowds). Lossless: the verify decides every token. A fixed depth (mtp_fixed) wins. "
                            "KYOJIN_SPEC_DEPTH_COST = comma list, the prior round cost per depth in plain steps. Read per round"),


    "spec_plain_min_sessions": Flag("KYOJIN_SPEC_PLAIN_MIN_SESSIONS", "0", "Session-count threshold N of the MTP depth policy (0 = off, read per round, live via /admin/flags): with N or more sessions decoding (sessions prefilling or waiting "
                                    "do not count) the round is the TRUE plain step (the decode tick: HOSTGAP, lookahead, no verify pass, no head feed, tuner not consulted); with fewer the backend drafts as its policy says. "
                                    "KYOJIN_GLM_MULTI_PLAIN=1 stays the alias for N = 2 when this flag is 0. Measured on GLM-5.3 after dec10 (cheaper d1 round): d1 beats plain at 2 sessions, plain wins at 4, so N = 3 is the "
                                    "candidate (A/B in STATUS_MPTHRESH.md). A fixed depth (mtp_fixed) wins over it. Any MTP backend; opt-in.", kind="int"),
    "glm_lookup_only": Flag("KYOJIN_GLM_LOOKUP_ONLY", "0", "GLM-5.3 served spec with the nextn head loaded: n-gram drafts only (a session without a lookup draft takes the plain step, the head runs no draft step). "
                            "The n-gram-only A/B arm without a second load. Read per round"),
    "glm_head_prof": Flag("KYOJIN_GLM_HEAD_PROF", "0", "GLM-5.3 draft head: per-op wall ms of the draft step (embed, norms, eh_proj, attention + indexer, router, shared expert, 8 routed experts, "
                          "combine, lm_head + argmax; one forced eval per op, so the sum is NOT the served step) into KYOJIN_GLM_HEAD_PROF_OUT (json). A probe. Read per draft call"),
    "glm_head_q": Flag("KYOJIN_GLM_HEAD_Q", "0", "GLM-5.3 draft head: its dense linears (eh_proj, attention, indexer, shared expert; fp16 today) re-quantised for the DRAFT only (0 off, 4 / 8 affine g64, "
                       "nvfp4, mxfp4); built on first use; the verify is untouched", kind="mode", choices=("0", "4", "8", "nvfp4", "mxfp4")),
    "glm_draft_topk": Flag("KYOJIN_GLM_DRAFT_TOPK", "0", "GLM-5.3 draft head: route to K experts instead of 8 for the draft only (0 = the checkpoint's 8). A probe: the head reads 8 of 288 experts per step "
                           "(0.11 GB of 1.1), so this moves few bytes", kind="int"),
    "idx_qblock": Flag("KYOJIN_IDX_QBLOCK", "512", "indexer query block rows in long prefill (0 = one block; read per call)", kind="int"),
    "glm_dense4": Flag("KYOJIN_GLM_DENSE4", "0", "GLM-5.3 checkpoint converted by tools/glm_dense4_convert.py (dense FP8 linears rewritten as NVFP4, an overlay on the original): the trunk reads them in place "
                       "with no flag; this one also keeps the MTP head's NVFP4 linears as stored 4-bit ForeignLinear modules in the draft head (0.12 GB read per draft step instead of the 0.37 GB fp16 decode)",
                       runtime=False, reason="read when the checkpoint is loaded"),
    "hc_compile": Flag("KYOJIN_HC_COMPILE", "0", "gated hyper-connection modules (hc_norm + input_mix_weight_down / up [+ block_inject_weight], e.g. qwen4_exp): the whole __call__ is one mx.compile'd function at decode rows (<= KYOJIN_HC_COMPILE_ROWS, 16), "
                       "fewer host graph nodes per step. Same primitives / order: md5-proven on greedy tokens (cell c31b). Read at load.", runtime=False, reason="read when the model loads"),
    "step_compile": Flag("KYOJIN_STEP_COMPILE", "0", "generic_foreign decode (any model of the kit): every decoder layer whose cache is a fixed-state ArraysCache (linear attention / SSM) is one mx.compile'd program at decode rows (<= KYOJIN_STEP_COMPILE_ROWS, 16), the cache arrays go in and out; KV layers compile only their cache-free mlp (KYOJIN_STEP_COMPILE_MLP=0 off). First call per shape stock, second traces; a shape key that cannot trace (host sync) stays stock for that key (per shape, not per layer). A flags.set retraces every program (GEN). Read at load.", runtime=False, reason="read when the model loads"),
    "verify_capture": Flag("KYOJIN_VERIFY_CAPTURE", "0", "foreign MTP / lookup speculation (serve.fnmtp): the verify pass records the state of every fixed-state layer after each row (fused gated-delta kernel steps output, conv windows, PLE windows; compiled layer programs return them as extra outputs), so a partial accept selects the state after `keep` rows instead of restoring the snapshot and running the whole trunk again on the accepted rows. A cache with a slot nobody recorded replays as before. Same tokens as the replay path (the per-row recurrence is the kernel's own).", attrs=(("verify_capture", "ACTIVE", _on),)),
    "verify_compile_draft": Flag("KYOJIN_VERIFY_COMPILE_DRAFT", "0", "foreign MTP (serve.fnmtp, needs KYOJIN_STEP_COMPILE): the draft head's decoder layer goes through step_compile like the trunk layers (its routed-expert mlp is one mx.compile'd program per row count; the chain of single-row head steps launches fewer kernels). Switchable at idle: the head layer's classes are swapped (programs retrace on a flags.set). Same tokens: the compiled mlp is the same arithmetic.", attrs=(("verify_capture", "DRAFT", _on),)),
    "step_compile_prep": Flag("KYOJIN_STEP_COMPILE_PREP", "0", "needs KYOJIN_STEP_COMPILE: decoder layers that own a host-prep module (a mapped row table: the n-gram / per-layer embedding of qwen4_exp) compile too: the host part (ids -> rows gather -> upload; the shared RowBatch when on) runs eagerly before the program and the rows go in as an input. Modules declare `step_host_prep(args)` / `step_host_feed(arrs)`; found by duck typing. Read at load.", runtime=False, reason="read when the model loads"),
    "step_compile_split": Flag("KYOJIN_STEP_COMPILE_SPLIT", "0", "needs KYOJIN_STEP_COMPILE: KV-cache (attention) layers as two compiled programs around the attention call: everything before it, and everything after it (the attention call, with its cache update and python offsets, stays eager; the pre part is recomputed inside the post program). Read at load.", runtime=False, reason="read when the model loads"),
    "hc_fuse": Flag("KYOJIN_HC_FUSE", "0", "gated hyper-connection modules (qwen4_exp) at decode rows (<= KYOJIN_HC_FUSE_ROWS, 16): norm / act / mix as 3 fused Metal kernels around the module's own linears (6 launches per call instead of 14); supersedes KYOJIN_HC_COMPILE for qualifying calls. Read at load.", runtime=False, reason="read when the model loads"),
    "hc_mix_view": Flag("KYOJIN_HC_MIX_VIEW", "0", "GLM-5.x prompt chunks: the mHC mix reads x as a reshape view ([1, S, 4, D] -> [S, 4 D]) instead of `x[0].reshape` (mlx `x[0]` is a Gather = a 268 MB copy per sub-block at 8K rows, "
                        "90 per chunk). Same values (P3-PFREST, glm_pf_glue). Read per call", attrs=(("glm_pf_glue", "VIEW", _on),)),
    "hc_mix_norm": Flag("KYOJIN_HC_MIX_NORM", "0", "GLM-5.x prompt chunks (needs glm_pf_hc): sinkhorn + collapse + the sub-block rms_norm in ONE kernel (hc_pf_fuse.collapse_norm; MLX's own single-row reduction tree: bit-identical to "
                         "the stock cast + rms_norm launches). Read per call", attrs=(("hc_pf_fuse", "NORM", _on),)),
    "hc_mix_twin": Flag("KYOJIN_HC_MIX_TWIN", "0", "with hc_mix_norm: the f32 ffn norm output also written as fp16 (= .astype(float16)) and handed to the shared / routed experts (glm_moe._pf_call), which skips two casts per MoE layer. "
                         "Bit-identical; measured no gain on the 5-layer cut (P3-PFREST). Read per call", attrs=(("hc_pf_fuse", "TWIN", _on),)),
    "hc_id_fuse": Flag("KYOJIN_HC_ID_FUSE", "0", "decode rows (<= 16 per step) of the identity-form hyper-connection (foreign_skeleton_hc, 2 hc mix rows, no comb; any stream dtype, "
                       "any hc, D % 4 == 0): per sub-layer the mix (gluefuse.mixvec), split + collapse + the pre rms_norm (fuse.mhc_pre_identity) and the sandwich post "
                       "rms_norm + expand (fuse.mhc_expand_identity) are 3 launches instead of ~19. The mix sums in another order than the stock f32 matmul (~1e-7 rel); "
                       "the rest follows the stock op sequence. Read per call", attrs=(("foreign_skeleton_hc", "IDFUSE", _on),)),
    "dsa_indexer_pf_view": Flag("KYOJIN_DSA_INDEXER_PF_VIEW", "0", "GLM-5.x prompt indexer (glm_pf_idx): q / pool keys / weights / candidate mask read as reshape views instead of `a[0]` Gathers (full copies per 512-row block: "
                                "2 x 176 + 2 x 176 launches per chunk, 42 ms at 24K, 72 ms at 120K). Same values. Read per call", attrs=(("glm_pf_idx", "VIEW", _on),)),
    "dsa_pf_view": Flag("KYOJIN_DSA_PF_VIEW", "0", "GLM-5.x prompt sparse attention (glm_dsa_fused): q_lat / latent / ids as reshape views instead of `a[0]` Gathers (q_lat is 536 MB at 8K rows, per DSA layer). Same values. Read per call",
                        attrs=(("glm_dsa_fused", "VIEW", _on),)),
    "engram_device": Flag("KYOJIN_ENGRAM_DEVICE", "0", "Hy4-form Engram (foreign_skeleton EngramParams): the kv / kv_b / gate / lookup_norm / pre_gamma / hc_scale projections stay f32 mx arrays on the device (cached per parameter object) "
                          "instead of a GPU->host numpy copy + host->GPU copy per token and layer. Same ops, same order: output bit-equal. Generic (any Engram-form model). Read per call",
                          attrs=(("engram_gguf", "DEVICE", _on),)),
    "mla_attend_fuse": Flag("KYOJIN_MLA_ATTEND_FUSE", "0", "Decode (1 .. 16 rows) attention core of an absorbed-MLA layer on the fixed-state KV buffer as TWO launches (mla_attend_fused: split-key flash scores + causal / window / keep mask + "
                            "sinks + softmax + P @ latent, then the split merge) instead of ~15 small ops; only the visible keys are read. After the stock k_b absorb, before the stock v_b absorb. Same maths, scores and P stay "
                            "fp32 (the stock path rounds them to fp16). fp16 buffer, 576-wide key rows; others take the stock core. Metal only. Read per call",
                            attrs=(("mla_attend_fused", "ON", _on),)),
    "mla_prelude_fuse": Flag("KYOJIN_MLA_PRELUDE_FUSE", "0", "Decode (1 .. 16 rows) fixed-state call of an absorbed-MLA layer: the kv_a norm of the latent, both ropes, the q split / transpose, the concat of the cache row and the saturating clamp + dtype cast in ONE launch "
                          "(mla_prelude) instead of ~8 small ones; the linears, q_a_norm, the slice_update write, the absorbs and the attend keep their own flags. Same maths in fp32 (the stock path rounds the norm output to the activation dtype first). Metal only. Read per call",
                          attrs=(("mla_prelude", "ON", _on),)),
    "attn_fp16": Flag("KYOJIN_ATTN_FP16", "0", "Absorbed-MLA attention of the skeleton (any model whose activations are fp32, Hy4): fp16 KV buffer in a NEW fixed-state cache (rows saturated to +-65504), attention projections written as fp16, "
                      "q / latent / k_pe / k_b and v_b absorbs / scores (fp32 accumulate) / P @ latent / pooled-key reads on fp16 data, softmax fp32; the hidden stream stays fp32. Half the KV and absorb bytes; "
                      "values differ at fp16 rounding (gate with the model's noise band). Needs a fresh cache after switching. Metal. Read per cache creation / call",
                      attrs=(("attn_fp16", "ON", _on),)),
    "attn_fp16_parts": Flag("KYOJIN_ATTN_FP16_PARTS", "7", "attn_fp16: which parts run in fp16 (bit mask): 1 = the KV buffer (read at cache creation), 2 = the attention projections (q_b, kv_a, index and gate linears written fp16), 4 = the k_b / v_b absorbs "
                            "(fp16 copies, q cast). 7 = all; single bits isolate the source of a quality loss (n8: +3.65 % PPL at 4096 ctx with all three)", kind="int", attrs=(("attn_fp16", "PARTS", int),)),
    "moe_union": Flag("KYOJIN_MOE_UNION", "0", "Routed-expert block (ForeignSwitchGLU, stored k-quant experts, flag expert_gateup_pair plan glu_down) of a 2 .. 8 row pass (MTP verify rows): each DISTINCT expert's bytes are read once for all "
                      "the rows that chose it (moe_union: device-side distinct-expert tables, one threadgroup per expert group running the stored format's gather GEMV with R = rows) instead of once per (row, expert) slot. "
                      "Same decoders, saturation and per-row sum order as the gather. Metal only. Read per call", attrs=(("moe_union", "UNION", _on),)),
    "split_eval": Flag("KYOJIN_SPLIT", "0", "Plain and verify passes (1 .. 8 rows): mx.async_eval(h) after layers 1 and 8, so the GPU runs the first layers while the host builds the rest of the graph. n10 Mac Studio, Hy4: with one eval at the end the GPU idled for the whole host build (2.6 ms at 1 row, 7.2 ms at 6 rows); plain p4 40.4 -> 43.9 tok/s (ids equal), verify T=6 50.3 -> 44.2 ms. "
                       "Explicit KYOJIN_SPLIT_EVAL / _ROWS / _ROWS_MIN env wins. Read per call",
                       attrs=(("foreign_skeleton", "SPLIT_ON", _on),)),
    "foreign_fmm8": Flag("KYOJIN_FOREIGN_FMM8", "0", "First row count served by the few-row simdgroup-matrix product on a stored Q8_0 linear (fmm8, ForeignLinear.apply before the NAX wide product / gemv8): "
                         "0 = off, 2 = verify rows from 2. One launch reads each weight once for 8 x NT weight rows and up to 16 rows, the simdgroups of a threadgroup split K (llama.cpp mul_mv_mma, Q8_0 from 2 rows on "
                         "M3 Ultra). fp16 x, exact int8 weights, fp32 sums (same products as gemv8, other sum order). Metal only. Read per call", kind="int", attrs=(("fmm8", "MIN_ROWS", int),)),
    "foreign_fmm8_max": Flag("KYOJIN_FOREIGN_FMM8_MAX", "8", "Last row count of the foreign_fmm8 product (1 .. 16; above it the wide NAX product / matmul route). Read per call", kind="int", attrs=(("fmm8", "MAX_ROWS", int),)),
    "head_fmm8": Flag("KYOJIN_HEAD_FMM8", "0", "Output head of a 1 .. 16 row pass (decode, verify rows) through a Q8_0 copy of the dense head matrix (symmetric int8 + fp16 scale per 32, 1.06 bytes per weight) read by the few-row "
                      "simdgroup-matrix kernel (fmm8, float32 logits) instead of the dense float32 matmul. Takes precedence over head_q. One copy shared with the MTP draft's trunk-side head. Logits differ at the "
                      "8-bit rounding: gate with the model's noise band. Metal only. Read per call", kind="int", attrs=(("headq", "FMM8", int),)),
    "mla_decode_vec": Flag("KYOJIN_HY4_ATTN_VEC", "0", "Decode / verify (1 .. 8 rows) attention core of an absorbed-MLA layer on the fixed-state KV buffer as ONE launch (mla_decode_vec: flash-attention-vec shape of llama.cpp, one threadgroup of 4 simdgroups per "
                        "head and row, online softmax + attention sink + P @ latent, V = the first 512 columns of the key row) instead of the stock chain of ~15-25 small launches, between the k_b and v_b absorb products. "
                        "1 = windowed (SWA) layers: the window is read straight from the buffer with the device offset; 2 = also kpool layers (keep mask below the gather threshold, key ids above it). The buffer and queries keep their dtype "
                        "(fp32 or fp16), scores / accumulation fp32. Metal only; rows or dtypes outside the contract run the stock core. Read per call",
                        kind="int", attrs=(("mla_decode_vec", "MODE", int),)),
    "foreign_nbr_small": Flag("KYOJIN_FOREIGN_NBR_SMALL", "0", "Output rows per simdgroup (NBR) of the 8-bit stored GEMV (gemv8, 1 .. 8 rows) on matrices under 8M weights: 0 = the global NBR (2), 1 = one row per "
                              "simdgroup = twice the threadgroups (the 96-threadgroup 3072 x 1536 / 1536 x 3072 / 1024-wide projections run at 0.5-0.6 TB/s against 1.0-1.2 for the big ones). "
                              "Same sums per output row (the row order inside a simdgroup does not change a dot product). Read per call", kind="int", attrs=(("foreign_linear", "NBR_SMALL", int),)),
    "head_q": Flag("KYOJIN_HEAD_Q", "0", "Output head of a 1 .. 16 row pass (plain decode, verify rows; prompt chunks keep the dense head) through an 8- (or 4-) bit affine group-64 copy of the dense head matrix "
                   "(mx.quantized_matmul) instead of the dense float32 matmul (Hy4: 1.6 GB, 1.3 ms of a 22.9 ms step at the bandwidth roof; Q8 reads a quarter). The packed copy is built once and "
                   "shared with the MTP draft's output projection (one low-bit copy; the dense head stays resident for the prompt chunks). Logits differ at the 8-bit rounding: gate with the model's "
                   "noise band. Read per call", kind="int", attrs=(("headq", "BITS", int),)),
    "moe_route_rs": Flag("KYOJIN_MOE_ROUTE_RS", "0", "Decode / verify (1 .. 8 rows) router of a generic MoE block with sigmoid gating (GenericMoE.route): the gate GEMV, sigmoid, selection bias, top-k, weight normalisation and scale "
                         "as ONE launch (glm_moe_rs.rowsel: rowdot + the last-arriving threadgroup selects) instead of the stock matmul + bias + sigmoid + argsort + take + sum + divide (~6 launches). "
                         "Logits = rowdot (row-invariant, so a plain step and a verify row agree); one group, no logit bias; others take the stock path. Metal only. Read per call",
                         attrs=(("foreign_moe_generic", "ROUTE_RS", _on),)),
    "moe_route_sel": Flag("KYOJIN_MOE_ROUTE_SEL", "0", "Decode / verify (1 .. 16 rows) router selection of a generic MoE block (GenericMoE.route), any gating (softmax, sigmoid, sqrt_softplus, none, softmax_weight): "
                          "gating, selection bias, group limit, top-k, weight normalisation and scale as ONE launch (moe_route_sel.select, llama.cpp topk_moe layout) on the stock router logits, instead of "
                          "~6 launches (sigmoid, + bias, argsort, take, sum, divide, scale). Logits unchanged (bit-identical indices vs stock; weights within fp32 ulps; ties: lowest index). "
                          "moe_route_rs (gate GEMV fused in too, rowdot logits) takes precedence where eligible. Metal only. Read per call",
                          attrs=(("foreign_moe_generic", "ROUTE_SEL", _on),)),
    "kpool_fuse": Flag("KYOJIN_KPOOL_FUSE", "0", "Decode (1 .. 8 query rows) pool selection of a kpool layer as ONE launch (kpool_fuse.select): indexer scores over the pooled keys + exact top-k (ties: lowest pool index, as the stable argsort) "
                       "+ the [B, 1, L, S] slot mask with the forced tail group, instead of ~50 elementwise launches. Generic by shape. Selection equal to the stock path (scores to ~1e-6 relative: plain d-loop dot). Metal only; prefill unchanged. Read per call",
                       attrs=(("kpool_fuse", "ON", _on),)),
    "kpool_score": Flag("KYOJIN_KPOOL_SCORE", "0", "kpool (block-sparse indexer) layers: the indexer scores sum_h relu(iq_h . pk_j) * w_h in ONE multi-threadgroup launch on fp32 simdgroup 8x8 tiles (kpool_score.scores, any query rows) instead of the "
                        "[B, H, L, J] fp32 temporaries (prompt chunks) or the one-threadgroup scalar loop of kpool_fuse (decode rows). Prompt chunks (batch 1, > 8 rows) then take the exact multi-threadgroup top-k (idxfast.topk_mt) instead of the full-row argsort; "
                        "decode rows keep kpool_fuse's radix select + slot mask with the scores as input. Same selected set up to the order of the fp32 d-sum (~1e-6 relative). Generic by shape. Metal only. Read per call",
                        attrs=(("kpool_score", "ON", _on),)),
    "kpool_keycache": Flag("KYOJIN_KPOOL_KEYCACHE", "0", "kpool (block-sparse indexer) layers: the pooled keys are kept per cache across tokens (extended only by the groups a step completes, a rewound cache drops "
                           "the pools past its offset) instead of recomputed from every cached value / gate row each token (gather + f32 cast + softmax + norm + rope, ~26 launches and a host sync per layer). "
                           "Same ops per pool: bit-equal. Solo int-offset caches; others take the stock path. Generic by shape. Read per call", attrs=(("foreign_skeleton_kpool", "KEYCACHE", _on),)),
    "moe_router_select_par": Flag("KYOJIN_MOE_ROUTER_SELECT_PAR", "0", "GLM-5.x prompt chunks: the MoE router's top-k select as `select_par` (one threadgroup per token, same outputs bit for bit) instead of the walking one-simdgroup kernel. "
                                  "Measured NOT faster at 8192 rows (0.45 vs 0.23 ms min, P3-PFREST): off. Read per call", attrs=(("glm_moe", "PF_SELPAR", _on),)),
    "kda_scan_nsg": Flag("KYOJIN_KDA_SCAN_NSG", "4", "KDA chunked scan (kda_nax main kernel): simdgroups per threadgroup, 4 = shipped (128 threads), 2 = 64 threads / 256 groups (bit-identical, -2 ms per KDA layer at 8K rows on the cut), "
                         "1 = 2x slower; any other value (8 builds a kernel with no gate channels) falls back to 4 and the flag audit says why. Read per prompt call", kind="int", attrs=(("kda_nax", "NSG", int),)),
    "foreign_castfree": Flag("KYOJIN_FOREIGN_CASTFREE", "0", "foreign formats (foreign_linear.ForeignLinear, any model): a one-byte stored linear (fp8 / q8_0 / fp8b) whose rows take the "
                             "stored GEMV (gemv8, decode rows <= KYOJIN_FOREIGN_GEMV8_ROWS) reads a float32 / bfloat16 input and writes a float32 output inside the kernel "
                             "(foreign_metal.gemv8_source xin / yout): the AsType launch before and after the GEMV go away (2 per call; f32-stream models). The input is "
                             "rounded to half while it is staged and the output is the fp16-rounded value widened, as the casts did: bit-identical. Read per call",
                             attrs=(("foreign_linear", "CASTFREE", _on),)),
    "shared_input_gemv": Flag("KYOJIN_SHARED_INPUT_GEMV", "0", "skeleton stored load (foreign_skeleton_stored.load_stored, any model of that loader): (1) a BF16 / F16 GGUF linear stays 16-bit on the device "
                              "(foreign_linear.DenseLinear) instead of being decoded to a float32 matrix (half the bytes: the lm_head of a 120k vocabulary is 0.74 GB, not 1.48), and decode rows "
                              "(<= KYOJIN_SHARED_GEMV_ROWS, 8) read it in one GEMV (widened exactly, float32 accumulate: same products, the sum's order differs); (2) the linears that read ONE input "
                              "object (found from the module graph by one traced 1-row forward) are one pass over x: stored-format groups (group_same_input: q8_0 / q4_k / fp8 ...) and the dense 16-bit "
                              "groups. Read at load (the layout) and per call (a flag flip then runs the old float32 matmul / the member alone: the cell toggles it on one load). 0 = today",
                              attrs=(("foreign_linear", "SHARED_GEMV", _on),)),
    "expert_gateup_pair": Flag("KYOJIN_EXPERT_GATEUP_PAIR", "0", "routed experts on stored bytes (foreign_linear.ForeignSwitchGLU / foreign_moe_generic.GenericMoE, any stored gather-GEMV type, e.g. q3_k / q4_k), decode slots "
                               "(<= KYOJIN_FOREIGN_GATHER_MAX): gate and up are read in ONE gather pass (DevPack.gather_pair_raw; same stored type and shape, decided once per block, not per call) and "
                               "silu(gate) * up is folded into the down launch's input staging (2 launches per block instead of 3 + the stock activation ops; clamp / SiTU: one activation launch); "
                               "the router-weighted sum over the K slots of GenericMoE is one kernel (moe_reduce: f32 products added in slot order, written in the stream dtype) instead of 4 stock ops. "
                               "The activation is computed in f32 inside the kernel (not rounded to half first): not bit-identical, greedy ids are the gate. Read per call",
                               attrs=(("foreign_linear", "GATEUP_PAIR", _on), ("foreign_moe_generic", "GATEUP_PAIR", _on))),
}
_GLUE = ("glue_fuse", "glue_fuse_norm")


def _mod(name: str):
    return importlib.import_module(f"{__package__}.{name}")


def apply_defaults(environ=None) -> None:
    """Start value of every bool lever = its FLAGS default unless the environment sets it. The lever modules read their variable once at import with their
    own fallback literal, so without this a default flipped here only changed get_all(). Runs from the package __init__, before any lever module is imported.
    Int levers are left alone: some read the variable's presence (multiuser checks KYOJIN_HALF_ROWS)."""
    env = os.environ if environ is None else environ
    from . import opflags as OF
    OF.sync_env(env)                                                    # op-named spelling <-> model-named alias, before the start values below fill the gaps
    for f in FLAGS.values():
        if f.kind == "bool" and f.default in ("0", "1"):
            if f.env not in env and env is os.environ:
                OF.note_filled(f.env, f.default)                        # a start value, not a choice of the user (arch defaults may still replace it)
            env.setdefault(f.env, f.default)


def value(name: str) -> str:
    f = FLAGS[name]
    return os.environ.get(f.env, f.default)


def get_all() -> dict:
    """{name: {"env", "value", "default", "runtime", "doc"[, "reason"]}} with the value the code sees now."""
    out = {}
    for n, f in FLAGS.items():
        d = {"env": f.env, "value": value(n), "default": f.default, "runtime": f.runtime, "doc": f.doc}
        if f.reason:
            d["reason"] = f.reason
        out[n] = d
    return out


def _norm(f: Flag, v) -> str:
    if isinstance(v, bool):
        v = int(v)
    s = str(v).strip()
    if f.kind == "int":
        int(s)                                                   # ValueError on a non-integer
        return s
    if f.kind == "float":
        if f.default == "" and s.lower() in ("", "inf"):         # an optional bound: empty / inf = off
            return s
        x = float(s)
        if not 0.0 <= x < float("inf"):
            raise ValueError(f"{f.env}: expected a float >= 0, got {v!r}")
        return s
    if f.kind == "mode":
        if s not in f.choices:
            raise ValueError(f"{f.env}: expected one of {'/'.join(f.choices)}, got {v!r}")
        return s
    if f.kind == "str":
        from . import ablate
        return ablate.normalize(s)                               # the one str flag: validated class names in table order ("" = off)
    if s.lower() in ("1", "true", "on"):
        return "1"
    if s.lower() in ("0", "false", "off", ""):
        return "0"
    raise ValueError(f"{f.env}: expected 0/1, got {v!r}")


def _apply_glue() -> None:
    from . import fuse as F, glue_fuse as GN, gluefuse as GF
    GF.ENABLED = os.environ.get("KYOJIN_GLUE_FUSE", "1") == "1"
    GN.ENABLED = GF.ENABLED and os.environ.get("KYOJIN_GLUE_FUSE_NORM", "0") == "1"
    F.MIXPRE = os.environ.get("KYOJIN_FUSE_MIXPRE", "1") != "0" and not (GF.ENABLED and GF.MIXPREOFF)
    if GF.ENABLED:
        GF.install()                                             # wires the mix_vector cut once (it reads ENABLED per call after that)


def _opflags():
    from . import opflags
    return opflags


def set_many(values: dict) -> dict:
    """Validate every pair first (KeyError: unknown flag, PermissionError: not runtime, ValueError: bad value), then apply them all.
    Returns get_all()."""
    global GEN
    norm = {}
    for n, v in values.items():
        if n not in FLAGS:
            raise KeyError(n)
        f = FLAGS[n]
        if not f.runtime:
            raise PermissionError(f"{n}: not switchable at runtime ({f.reason})")
        norm[n] = _norm(f, v)
    for n, s in norm.items():
        f = FLAGS[n]
        for sp in _opflags().spellings(f.env):                          # op name and model-named aliases carry the same value
            os.environ[sp] = s
            _opflags()._FILLED.pop(sp, None)
        for mod, attr, fn in f.attrs:
            setattr(_mod(mod), attr, fn(s))
    if any(n in _GLUE for n in norm):
        _apply_glue()
    if norm:
        GEN += 1
    return get_all()


def set(name: str, v) -> dict:  # noqa: A001  (module API: flags.set)
    return set_many({name: v})


# ------------------------------------------------------------------------------------------------ per-request override
# A request body may carry {"kyojin_flags": {"spec_typical": 0, "spec_lookup": 0}}: these levers are set for the duration of that request and put back
# afterwards (POST /admin/flags stays the persistent switch). Only levers the engine reads per round and that do not touch the KV content are accepted,
# so the prefix cache is kept and no compiled region retraces (GEN is not bumped).
REQUEST_FLAGS = ("spec_typical", "spec_lookup")
_lease_lock = threading.Lock()
_lease_users = 0
_lease_saved: dict = {}


def check_request_flags(values) -> dict:
    """{name: normalised value} of a body's kyojin_flags; ValueError for anything but a dict of whitelisted levers (HTTP 400)."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("kyojin_flags must be an object")
    out = {}
    for n, v in values.items():
        if n not in REQUEST_FLAGS:
            raise ValueError(f"kyojin_flags: {n!r} is not a per-request flag (allowed: {', '.join(REQUEST_FLAGS)})")
        out[n] = _norm(FLAGS[n], v)
    return out


@contextlib.contextmanager
def request_override(values: dict):
    """Hold `values` (from check_request_flags) while the block runs; the first holder saves the baseline, the last one restores it."""
    global _lease_users, GEN
    if not values:
        yield
        return
    with _lease_lock:
        if _lease_users == 0:
            _lease_saved.clear()
            _lease_saved.update({n: value(n) for n in REQUEST_FLAGS})
        _lease_users += 1
        gen = GEN
        set_many(values)
        GEN = gen
    try:
        yield
    finally:
        with _lease_lock:
            _lease_users -= 1
            if _lease_users == 0:
                gen = GEN
                set_many(dict(_lease_saved))
                GEN = gen
