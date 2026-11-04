# Adding a model: released safetensors -> pack -> served, on a Mac

What exists in the repo is described as it is. "Manual" marks a step that still needs a person or a code edit. Nothing below was run in the sandbox this was written in (no Metal); the commands are the repo's own.

## 0. Check the config against the engine (no GPU, 1 s)

    python -m kyojin_mlx.modeldesc /path/to/hf/config.json

Prints the table row found for the model (`NONE` = no row), the geometry read from the config (layers, experts, top-k, hc streams, head dims, indexer, window) and, per fused path, `covered` or `FALLBACK` with the reason (`hc_mult=2: kernels are written for 4`). A FALLBACK path runs the stock op: correct, slower. That list is the speed a new geometry loses before anything is converted.

## 1. Describe the model (manual, one table row)

`kyojin_mlx/modeldesc.py`, `DESCS`: one `ModelDesc` per architecture.

| field | picks |
|---|---|
| `model_types`, `architectures` | the config keys that find the row (`architectures` also says if there is a vision tower) |
| `block`, `attn`, `kv`, `moe`, `hc` | layout, attention kind, KV kind (`kvwin` quantized pool or `native` cache), MoE / hyper-connections (`yes` / `no` / `config`) |
| `backend` | `q35` (kyojin_mlx.serve.q35mlx) or `vlm` (patched mlx-vlm model, `patch_model`) |
| `server` | chat family in `serve/chat.py` (`qwen`, `glm`, `ds41`) and the `thinking_key` |
| `arch` | key of an `ArchSpec` in `kyojin_mlx/archs.py` (name mapping of pack tensors, rows, fuse, CED, MTP); `None` if not patched |
| `chunk`, `mtp_depth`, `ref`, `own_tokenizer`, `builtin_template`, `one_buffer_gate` | what `tools/qualify` and the server entry points use |

Unlisted `model_type`s keep the old prefix rules (`chat_prefixes`, `backend_prefixes`). A model that reuses an existing block needs only this row. A new block adds one module: an `ArchSpec` (`archs.py`) when it goes through `patch_model`, a `Family` class in `serve/chat.py` for a new chat format, a block adapter in the converter (step 3).

## 2. Fetch and calibrate

    python tools/calib_text.py tokens SRC_HF_DIR --rows 128 --cols 1024 --out cal.safetensors

Tokenizes the calibration text file (`--text`, not shipped) with the model's own tokenizer and chat template (no token ids stored, JSON fallback when there is no template). `tools/drill_mlx.py` does this itself.

## 3. Convert (EXL3 on Metal)

One command for a model the converter knows (Qwen3.5-style dense or MoE: GatedDeltaNet + attention + gated MLP):

    python tools/drill_mlx.py --hf ORG/MODEL --work WORK [--slice N]

fetch -> calibration -> `conv_mlx.py` -> bit-allocation ladder (`drill_rung.py`: +1 bit on the cells with the highest measured local error per parameter until the qualify gate passes, a size cap or a plateau) -> `tools/qualify.sh`. Resumable; heavy jobs hold `~/kyojin-mlx/.gpu.lock`. Allocation is the ladder (`--k-attn --k-mlp --k-expert --head-bits`, `--strategy k.json` for explicit per-linear K). Vendor FP8 / MXFP4 / NVFP4 sources are dequantized per linear (`kyojin_mlx/vendor_src.py`).

Experts one-tensor-per-expert -> the stacked layout the Mac kernels read:

    python tools/convert_stack.py SRC_PACK OUT_DIR --verify N

**Manual, still open:**
1. **A block the converter does not know.** `tools/conv_mlx.py` imports `mlx_lm.models.qwen3_5` (`DecoderLayer`, `TextModelArgs`, line ~585) and its calibration forward, tensor names (`model.language_model.`) and norm convention are Qwen3.5's. DS4.1 and GLM packs were converted with the CUDA / ROCm exllamav3 converter, not on the Mac. A new architecture needs a block adapter (forward of one block on the calibration state, tensor-name map, linear list) next to `conv_mlx.py`; this is the main remaining manual step and could not be made table-driven without Metal to check the result.
2. `quantization_config` must be emitted for the pack (`tools/conv_qconfig.py` does it for the Qwen path).
3. A tokenizer-less pack needs `own_tokenizer` / `--tokenizer`.

## 4. Qualify (conformance, autotune, parity, KV, max ctx, serve, smoke)

    tools/qualify.sh --pack PACK --dry-run      # the plan, the derived serve command and the estimates
    tools/qualify.sh --pack PACK                # resumable

Writes `qualify_out/<pack>/mlx/{REPORT.md,qualify.json}`. Stages read the table (`chunk`, `mtp_depth`, `kv`, `ref`). The "lane-kv-conformance" check is not a separate script in this repo (`lane-kv-conformance.py` does not exist): it is `conformance()` in `tools/qualify/load.py`, run by the `serve` stage. It renders the chat template with a system message and a tool and requires the tools before the system text (otherwise every new session re-pays the tool schemas). The `autotune` stage writes `<pack>/tuning/<device>.json` (per-shape kernel choices; no file = constants).

**Manual:** parity needs a BF16 reference adapter `tools/qualify/refs/<ref>.py` (only `qwen35` exists; `ref=None` skips parity and the report lists it as unmeasured). Bit allocation beyond the ladder (per-layer sensitivity for experts) is the ladder's third cell; nothing else is automatic.

## 5. Serve

    python -m kyojin_mlx.serve.server PACK

Backend, chat family and thinking key come from the row. `qualify.json` `serve.cmd` / `serve.env` hold the flags found by the run.

## 6. Tests (no Metal)

`python -m pytest tests/test_modeldesc.py tools/qualify/tests`. Copy the synthetic config pattern in `tests/test_modeldesc.py` (`UNSEEN`) for the new model: it checks the row lookup, the geometry and the expected `FALLBACK` list.
