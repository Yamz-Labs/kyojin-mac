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
    "router_w16": Flag("KYOJIN_ROUTER_W16", "0", "decode router (rows.py): the fp32 gate is read through an exact bf16 / fp16 copy (lowp.exact_lowp: kept only when every value "
                       "round-trips), half the gate bytes per MoE layer, same logits bit for bit (rowdot upcasts each element to fp32 before the first fma); fp32 stays for a gate that is not exact",
                       attrs=(("rows", "ROUTER_W16", _on),)),
    "pf_rdadvise": Flag("KYOJIN_PF_RDADVISE", "0", "Engram prefill gather (engram_rows): F_RDADVISE each run of rows before the memory map take, one "
                       "advisory per run of rows within 16 of each other, skipped when a request scatters into more than 4096 runs; same bytes"),
    "gguf_fp16": Flag("KYOJIN_GGUF_FP16", "1", "foreign formats (foreign_linear.load_foreign, GGUF): 1-D F32 tensors (norm weights, biases; ggml always writes them F32) load in the model dtype, "
                      "so Q and K stay fp16: prefill attention 1.4x faster, decode reaches the fp16 flash kernel. gdn_norm's fp32 weight stays. 0 = kept F32, as before",
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
    "ablit_fused": Flag("KYOJIN_ABLIT_FUSED", "1", "decensor projection (ablit.project): clamp, the K dot products and the update of a row in ONE launch instead of ~7 "
                        "(fp16 clip pair, fp32 cast, two small fp32 matmuls, fused update); fp32 inside like the stock path, only the dot product's "
                        "summation order differs", attrs=(("ablit", "FUSED", _on),)),
    "attn_core_fuse": Flag("KYOJIN_ATTN_CORE_FUSE", "0", "DS4.1 plain decode attention core: the kv chain also writes the stacked window buffer the commit keeps, so the commit's slot scatter launch is gone "
                           "(bit-identical outputs and cache state; applies when the stack has spare slots or a non-identity slot map; 0 = the old path)",
                           attrs=(("dsv4_core", "ATTN_CORE_FUSE", _on),)),
    "ablate": Flag("KYOJIN_ABLATE", "", "serve probe (timing only, outputs become garbage): comma list of op classes of the DS4.1 decode step (1-row plain step and verify rows) "
                   "each replaced by the cheapest stand-in that keeps shapes and dtypes, to read its exposed cost as a round-time difference. Classes: moe_routed (routed experts -> zeros), "
                   "moe_shared (shared expert -> zeros), router (logits + top-k -> fixed ids / weights), moe_glue (routed GEMVs kept, rot / mid / final -> zero buffers), "
                   "mhc (mixes + sinkhorn -> constants, pre -> stream mean, expand -> add; block runs the stock chain), norm (RMSNorm -> identity; block runs the stock chain), "
                   "attn (attention core -> zeros, cache untouched), idx (indexer -> fixed first-k), ablit (decensor projection -> identity). \"\" = off, bit-identical",
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
    "idx_qblock": Flag("KYOJIN_IDX_QBLOCK", "512", "indexer query block rows in long prefill (0 = one block; read per call)", kind="int"),
}
_GLUE = ("glue_fuse", "glue_fuse_norm")


def _mod(name: str):
    return importlib.import_module(f"{__package__}.{name}")


def apply_defaults(environ=None) -> None:
    """Start value of every bool lever = its FLAGS default unless the environment sets it. The lever modules read their variable once at import with their
    own fallback literal, so without this a default flipped here only changed get_all(). Runs from the package __init__, before any lever module is imported.
    Int levers are left alone: some read the variable's presence (multiuser checks KYOJIN_HALF_ROWS)."""
    env = os.environ if environ is None else environ
    for f in FLAGS.values():
        if f.kind == "bool" and f.default in ("0", "1"):
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
        os.environ[f.env] = s
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
