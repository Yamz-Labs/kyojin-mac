# Bringing up a new architecture

For the Mac side of a release (fetch, calibrate, convert, qualify) read `docs/ADD_A_MODEL.md`; this page is the engine side: which fields describe the model, which checks must pass before any speed lever is looked at, and which levers come on by themselves. Nothing below was run on Metal where it was written; every command is the repo's own and every check named here exists in the tree (`tests/test_desc_fields.py`, `tests/test_packcheck.py`, `tests/test_conformance_synth.py`, `tools/cgeneric_mac.py`).

**Rule: a model is a description, not a branch.** No code path names a model. Class names, module paths, stop tokens, the CED midpoint, the draft-head prefix, the trunk attribute: all are fields of two tables, `modeldesc.DESCS` (what the model IS and which server pieces it selects) and `archs.ARCHS` (how its pack and its model code are patched). A literal such as `DeepseekV41Cache` outside those two files fails `tests/test_desc_fields.py::test_no_vendor_literal_in_executable_kernel_or_serve_code`.

## The unreleased models: open values

GLM 5.5, Qwen 4 and HY4 Flash are not released; their layouts are **not guessed here**. Every field below that depends on them is `open`: fill it from the released `config.json` and the model code, never from this page. DS4.1 values are the ones the lane runs today.

| field | DS4.1 (lane) | GLM 5.5 | Qwen 4 | HY4 Flash |
|---|---|---|---|---|
| `model_types`, `architectures` | `deepseek_v41`, `DeepseekV41ForCausalLM` | open | open | open |
| `block`, `attn`, `kv`, `moe`, `hc` | `csa_moe_hc`, `csa`, `native`, `yes`, `yes` | open | open | open |
| `backend`, `server`, `family` | `vlm`, `ds41`, `ds41` | open | open | open |
| `stop_tokens` | `<｜end▁of▁sentence｜>` | open | open | open |
| `mtp_auto` | `False` (draft head opt-in) | open | open | open |
| `chunk`, `mtp_depth` | 2048, 3 | open | open | open |
| `ArchSpec.vendor` (module + 5 class roles) | `mlx_vlm.models.deepseek_v41.language`, `DeepseekV41{Block,Attention,MoE,Cache}`, `BatchDeepseekV41Cache` | open | open | open |
| `CedSpec.mid`, `.tail` | `None`, `None` = layer 20 of 40, 128 rows (never lower: ledger) | open (is there a KV-source midpoint?) | open | open |
| `MtpSpec.prefix`, `.keep`, `.keys` | `mtp.`, 128, the `dspark_*` config keys | open | open | open |
| `ArchSpec.trunk` | `language_model` | open | open | open |
| `KERNEL` envelope match (`fast_paths`) | all covered | open: run `python -m kyojin_mlx.modeldesc config.json` | open | open |

## Steps, in order, each with its check

1. **Read the config against the engine (1 s, no model).** `python -m kyojin_mlx.modeldesc /path/config.json` prints the row found (`NONE` = add one), the geometry, and per fused path `covered` or `FALLBACK` with the reason. The FALLBACK list is the speed the model loses before anything is converted.
2. **Add the `ModelDesc` row** (`kyojin_mlx/modeldesc.py`). Fields (all are keyword fields with a default; the ones with a DS4.1 value above are the ones that moved out of code in this change):

   | field | what it decides |
   |---|---|
   | `model_types`, `architectures` | the config keys that find the row (`architectures` also says if there is a vision tower) |
   | `block`, `attn`, `kv`, `moe`, `hc` | layout words; `moe` / `hc` are promises the pack check enforces (`yes`, `no`, `config`) |
   | `backend` | `q35` (mlx-lm, `serve/q35mlx`) or `vlm` (mlx-vlm model patched by `patch_model`) |
   | `server`, `thinking_key`, `chat_prefixes`, `backend_prefixes` | chat family in `serve/chat.py`, the template kwarg that switches thinking, prefix fallbacks for unlisted model types |
   | `stop_tokens` | token strings besides the config / generation_config / tokenizer eos that end a turn |
   | `mtp_auto`, `mtp_prefix` | the draft head is switched on by itself when the pack carries tensors under `mtp_prefix` (otherwise `--mtp-depth`) |
   | `arch` | key of an `ArchSpec` (`None` = the model goes through its own loader, no `patch_model`) |
   | `chunk`, `mtp_depth`, `ref`, `own_tokenizer`, `builtin_template`, `one_buffer_gate` | `tools/qualify` and the server entry points |

3. **Add the `ArchSpec`** (`kyojin_mlx/archs.py`) when the model goes through `patch_model`: `name_rules` / `dense_rules` (pack tensor name -> module path), `moe` (`MoeSpec`), `rows`, `fuse`, `mtp` (`MtpSpec`: `prefix`, `keep`, `keys`), `ced` (`CedSpec`: `adapter`, `mid`, `tail`; `None` = n // 2 and the sliding window), `vendor` (`Vendor`: the module that defines the classes and the class name of each role: `block`, `attn`, `moe`, `cache`, `batch_cache`; a role left empty is not used and not checked), `trunk` (attribute that holds `.layers`, used by the decensor hook). The vendor classes are resolved through `archs.cls(role)` once and cached; `archs.activate(spec)` is called by `patch.build_model`, so one process serves one model.
4. **Check the pack against the row (no weight is read).** `python -m kyojin_mlx.packcheck PACK [--mtp] [--ced]`. Findings name the field that made the promise: `row`, `arch`, `config`, `layers`, `moe`, `hc`, `index`, `engram`, `mtp`, `ced`, `vendor`, `tokenizer`, `stop_tokens`, `template`. In the server the same check runs at load: `KYOJIN_DESC_STRICT=0` (default) logs each finding and loads on, `=1` raises `PackError` with all of them. Bring-up runs at `=1`.
5. **Describe a synthetic twin and run the CPU conformance suite.** Add a `Shape` to `kyojin_mlx/synth.py` (a tiny config of the same model code, hidden 64, 2 to 8 layers) and run `python -m pytest tests/test_conformance_synth.py`. It writes the shape as a pack, runs step 4 on it, loads it through the real loader, and compares with a slow reference (one forward over the whole sequence on a fresh cache): chunked prefill, every greedy decode step, KV fork (in memory, two forks, either order, bitwise), KvDisk save / lookup / continue (bitwise), and for a CED layout the CED prefill with the receptive-field tail. A new model must pass all of it on CPU before it goes to the Mac. A shape the stock model code cannot build (CED without MoE, today) is said in `synth.py`, not faked.
6. **Run the Metal-only paths on the Mac, bounded.** `python tools/cgeneric_mac.py --pack PACK --stage all`: `--pack` is read as config + index + tokenizer JSON only (never loaded); the synthetic packs go through the served `load_vlm` / `VlmBackend` (rows, fused-path installs, batch / step, fork, KvDisk, CED) against the CPU reference. Memory cap under 3.5 GiB, no timings printed.
7. **Convert, qualify, serve** as in `docs/ADD_A_MODEL.md` (`tools/qualify.sh --pack PACK`). The first A/B of a lever is a lane run; this page claims none.

## Which levers switch on by themselves, which need a kernel

**On by themselves (keyed on the config or the descriptor, no model code):**
- CED prefill: `ced.explain(config)` decides from the layer layout (even split or `CedSpec.mid`, ratio-1 decoder half, KV / index sources at the midpoint, no Engram at or after it). A layout it refuses is logged; with `desc_strict=1` and `--ced` the pack check fails instead of dropping CED silently.
- Rows router (`decrows`): any model with experts <= 1024 and 1 <= top-k <= 32.
- KV disk checkpoints and forks: the generic adapter takes any mlx-lm cache list (KVCache, ArraysCache, rotating, kvwin); the vendor cache class gets the `dsv4` adapter by name (`Vendor.cache`).
- Chat family and backend by prefix (`chat_prefixes`, `backend_prefixes`), stop tokens, thinking key.
- Draft head (`MtpSpec`) when the pack carries `mtp.*` tensors and the config carries the keys named in `MtpSpec.keys`.
- Shape-keyed tables (`tuning`, `sdpa_dec` split-K by head dim, `chip_defaults`): fall back to constants for a shape never measured.

**Need a kernel or code (the `FALLBACK` lines of `fast_paths`; the stock op runs, correct and slower):** the geometry the Metal sources are written for is `modeldesc.KERNEL`: 4 hyper-connection streams and 24 mixes (whole-block decode fusion, mix kernels), head dim 512 with 64 rope dims, 8 `wo_a` groups, indexer 32 heads x 128, `pf_glue.pre_norm_fq` at hidden exactly 5120. A model with another number is correct on the stock op; matching speed means a kernel written for that number and its gate widened **only after** it is written and run on the Mac. Also code, not config: a block adapter in the converter for a block `conv_mlx.py` does not know, a `Family` class in `serve/chat.py` for a chat format that is not Qwen / GLM / DS4.1, a cache adapter in `kvdisk.py` for a cache layout that is neither an mlx-lm cache nor the DS4.1 one, `RowsSpec` / `FuseSpec` entries for class-level hooks.

## What this page does not settle

Nothing here was timed. Whether a new model reaches the bars is a lane measurement after steps 1 to 6 pass, with the maintainers' measurement log as the do-not-redo list.
