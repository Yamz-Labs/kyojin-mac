# macair17 — decode step replay without per-step CPU encoding: feasibility

Machine: Air M5, macOS 27.0.1, MLX 0.32.4, Command Line Tools only (no `metal` tool; runtime Metal compile works). Branch `feat/replay` (from `feat/mtp`), worktree `~/kyojin-mlx/.worktrees/replay`.

## Decision

**Feasible, with one mechanism: Metal indirect command buffers (ICB), recorded from MLX's own encoder calls and replayed from a one-launch trigger op.** No MLX source change, no extension build: a 300-line Objective-C++ dylib (`replay/rp.mm`, clang from CLT) loaded by ctypes swizzles the AGX encoder, command-buffer, device and queue classes through the ObjC runtime. A recorded decode step of a DS4.1 slice replays bit-identical to the eager step; CPU per step drops from 0.5-1.7 ms to 20-40 us; GPU time per command buffer is unchanged within +-1.5 % (min).

What is proven: the **mechanism** (record, rebuild as ICB, replay, exact, cheap) on `ds2` and `ds2e` (2 layers, with and without Engram), for a chain of consecutive steps whose tokens are known when recording ("tape": state advances from step to step, per-step constants differ).

What is not built: the **live path** (tokens and positions that are not known at record time). The blocking facts are measured below (section 4); none is a wall, each has a known fix, about 3-5 days of engine work. Details and the Ultra number in sections 5 and 6.

## 1. What MLX exposes (evidence)

- MLX 0.32.4 has no capture or replay path, no ICB use, no public encoder access: `device.h` makes `get_command_encoder()` private; `mx.compile` removes Python and graph-build cost of a region but still encodes every kernel at every call (macair12's regions). `mx.fast.metal_kernel` gives custom kernels, not control of the encoder.
- The wheel ships `include/mlx` and `include/metal_cpp`, so a C++ extension could in principle link against MLX internals; not needed.
- MLX's encoder is a **concurrent-dispatch** compute encoder with manual `memoryBarrierWithScope` calls, command buffers from `commandBufferWithUnretainedReferences`, and a residency set attached to its queue.
- One warm ds2 decode step (probe `replay/rp_probe.mm` + `tools/rp_probe.py`, swizzling `AGXG17GDevice`, `...FamilyCommandQueue`, `...FamilyCommandBuffer`, `...FamilyComputeContext`): 1 command buffer, 1 compute encoder, 131 dispatches (128 `dispatchThreads`, 3 `dispatchThreadgroups`), 97 barriers, 536 `setBuffer`, 132 `setBytes`, 131 `setComputePipelineState`, 1 `updateFence`, plus a second command buffer that only signals an event. No blit encoder, no `waitForFence`, no `useResources`, no `setBuffers` range call. That is the whole vocabulary an ICB has to reproduce.
- MLX builds pipelines through `newComputePipelineStateWithFunction:error:`, which delegates to `...WithDescriptor:error:`. Hooking the descriptor variant and setting `supportIndirectCommandBuffers = YES` makes every MLX pipeline ICB-capable (the hook is installed before MLX creates any kernel; cost on the eager path: none measurable, eager min 416 us with the flag vs 442 us without).

## 2. ICB micro-benchmark (`replay/icb_micro.mm`, standalone, CLT clang, runtime Metal compile)

Dependent GEMV-like dispatches, 7 buffers + a constant buffer each, a barrier before every command, M5 Air:

| Case | direct encode, CPU | ICB execute, CPU | GPU direct vs ICB |
|---|---|---|---|
| 150 launches | 196 us | 25 us (95 us with `useResources`; 25 us with a residency set) | 2.26 vs 2.25 ms (GEMV), 311 vs 320 us (tiny kernels) |
| 2300 launches (a DS4.1 token) | 2242 us (min 1525) | 63 us (min 25) | 21.6 vs 22.1 ms (+1-2 %) |

- Bare Metal costs about 1 us per direct launch; MLX's 3.8-6 us per launch is mostly MLX and Python overhead, which an ICB removes entirely.
- ICB execute CPU is flat in the launch count (25-60 us).
- **The GPU-side dependent-kernel gap does not shrink** (about 2.1 us per dependent launch, same as direct). Replay cuts CPU only. It is not a GPU-side lever.

## 3. Prototype (`replay/rp.mm`, `tools/rp_chain.py`)

Design:
1. Record: `rp_rec_begin/end` swizzles the encoder and stores, per dispatch: pipeline, grid, threadgroup, `setBuffer` bindings, `setBytes` constants (into a 256-byte aligned arena), threadgroup memory, and where a barrier was.
2. Buffers: those inside a pinned range (the model's arrays, walked from the Python object graph), and big buffers that are unchanged by the recorded steps (hash before/after: weights), are bound as they are and kept alive by the chain. Every other buffer (born in a step, or a per-step external) is replaced by a private copy owned by the chain, initialised from the snapshot taken when it was first bound. Across steps the same recorded buffer maps to the same copy, so the state one step writes is the state the next one reads. Inputs the CPU rewrote between recorded steps (token, Engram indices) are detected by comparing the start-of-step contents with the end of the previous step, and refreshed before the replay (4 buffers on ds2, 45 on ds2e).
3. Replay: one ICB (`inheritPipelineState` and `inheritBuffers` NO, `setBarrier` where MLX had a barrier) and one constants arena per step; one residency set attached to MLX's queue and to every command buffer. The step is triggered by a one-launch `mx.fast.metal_kernel` (`kyojin_replay_trigger`): the hook on its dispatch runs barrier, `executeCommandsInBuffer`, barrier, then a tiny copy kernel moves the step's logits into the trigger op's own output buffer. In Python a replayed step is `call(token)` plus `mx.eval`.

Results (ds2 = 2 layers, ds2e = 2 layers + Engram, K = 6 consecutive teacher-forced steps recorded once, 20 passes each; the Air runs three other agents, so figures are min / p25):

| | dispatches per step | CPU per step, eager (build + async_eval) | CPU per step, replay | per layer (step / 2) eager -> replay | GPU per command buffer eager -> replay (min) |
|---|---|---|---|---|---|
| ds2 | 130 / 214 / 212 | min 0.52-0.83 ms (p25 1.1-3.2 ms) | min 21-68 us (p25 100-133 us) | 260-417 us -> 11-34 us | +0.3 % to +1.9 % (4046 -> 4124 us in the last run) |
| ds2e | 208 / 292 / 290 | min 1.14-1.72 ms (p25 3.7-4.9 ms) | min 26-40 us (p25 119-175 us) | 572-858 us -> 13-20 us | -0.6 % to +1.3 % (6545 -> 6610 us in the last run) |

(4 runs per slice, spread over the loaded Air: load average 2.2-4.3 from three other agents; the minimum is the honest figure, the p25 shows the host noise that replay also removes.)

The replay cost is flat in the launch count (the 2-layer slice and the 2300-launch microbench both sit at 20-60 us), so a full 43-layer DS4.1 step costs about the same: **eager 11.8 ms (macair12 median 13.2) -> replay about 0.05-0.15 ms**. "Per layer" is the slice's whole step divided by 2 and includes embedding, head and norm; the replay column is a constant per step, not per layer.

Exactness: **bit-identical to eager**. Gate (`tools/rp_chain.py`): K = 6 consecutive steps recorded once, replayed 20 passes with the private state rewound between passes, every replayed logits array compared with `np.array_equal` to the eager step: ds2 0 mismatches in 30 runs (3600 replayed steps) plus 4 timing runs, ds2e 0 mismatches in 6 runs plus 4 timing runs (120 steps per run); eager steps interleaved with replays on the live cache do not disturb either side (0 bad); the live cache is never touched by a replay (state lives in the private copies). One defect found and fixed on the way: a pass-through buffer (model array) that the chain did not retain could be freed and recycled by MLX under the ICB, which turned the output to all zeros from some pass on (about 1 run in 4 under load); two changes went in together and the run history does not separate them: retaining every pass-through buffer, and attaching the residency set to every command buffer with `useResidencySet:` (with the second change alone 1 run in 9 still failed); after both, 50 clean runs in a row, then 6 + 2 more.

Memory: RSS 1.1-1.5 GB. `mx.get_peak_memory()` is 3.29 GB (ds2) / 3.68 GB (ds2e) over a whole run, and 3.29 GB is already reached before any replay code runs (model build, 136-token prefill, warm-up and the K eager reference steps of the existing harness); recording K = 6 steps with everything retained peaks at 3.13 GB, the chain adds 17 MB of private copies (weights are bound in place, never copied). The 3 GB figure is therefore the harness's transient load and prefill, not the replay.

## 4. What a live decode needs on top (measured blockers)

Signature census (`tools/rp_diff.py`, `tools/rp_affine.py`, `tools/rp_ident.py`; ds2, 12-16 consecutive steps):

1. **Two step signatures alternate** (a group of pooled rows closes every `ratio` tokens): 213 commands on the closing step, 131 otherwise. Between two same-parity steps about 37 of 131 commands change a field.
2. **The pooled-row count n leaks into kernel constants and grids** (indexer sort, `carg_block_sort`, `attend`: `0x46 -> 0x47`, grid 2240 -> 2272, threadgroup 70 -> 71). Fields that vary between same-parity steps: 62-89 of about 3000, and 57-86 of them are exactly affine in the step index (second difference 0). The few that are not affine are boundary events: the sort kernel sequence changes at n >= 128 (the indexer top-k sort), and scratch buffer sizes change (irrelevant to the ICB: a scratch size is not a kernel parameter). A live replay therefore needs a patch list (about 90 field writes per step, microseconds) plus a re-record at each sort-sequence boundary, or shape-stable kernels (capacity-based, GPU-resident n).
3. **Buffer identity is not periodic**: between same-parity steps only 53-57 % of the bindings name the same buffer (allocator recycling). A live replay has to own its allocation (fixed private scratch, fixed state buffers with ping-pong or capacity-based caches); the pooled-KV buffers that grow by reallocation need capacity-based storage.
4. **Per-step CPU inputs**: token, positions, rope rows, and the Engram indices (`engram_mlx.py` reads the indices back with `np.asarray`, a GPU-to-CPU sync; `engram_rows.py` hashes on the host). They are 4-45 small buffers per step on the slices: written by the CPU into a ring of input buffers before each replay (the tape prototype does this with a refresh list).
5. **In flight**: the CPU must not write buffers an earlier replay still reads: a ring of 2-3 input arenas released from completion handlers. The tape test is synchronous (eval per step) and does not exercise it.
6. **Sampling**: the next token comes from the logits on the host (`.item()` in `spec.py`), or from a GPU argmax feeding the next step's token buffer (one tiny kernel appended to the ICB).

Not done: Metal API validation. With `MTL_DEBUG_LAYER=1` the layer rejects the ICB ("Setting a pipeline that does not have supportIndirectCommandBuffers = YES is invalid"): the validation layer wraps the device in its own classes, which the AGX-class swizzle does not reach, so the pipelines are built without the ICB flag. Running under validation needs the hook moved to the debug classes.

## 5. Rejected options

- **Per-layer persistent megakernel**: Metal has no grid-wide barrier with a forward-progress guarantee, and about 41 kernels per layer would have to be rewritten. Rejected.
- **MLX C++ extension against MLX internals**: not needed; the encoder swizzle gives the same control with no build of MLX.
- **`mx.compile` more regions**: cuts Python and graph build, not the encode (macair12 is already there).

## 6. Ultra solo estimate

**Headline: replay is what makes the Engram-model solo step GPU-bound; with it, solo is 62-76 tok/s (about 68 at a 14.5 ms GPU step), against about 36 today. 70 is reached only if the GPU side is 14.2 ms or less (dependent gap about 2.2 us or less): the same gate as ULTRA.md s.13.**

Evidence, in order:
- ULTRA.md s.13 models the step as max(CPU, GPU): CPU 11.8 (min) - 13.2 (median) ms, GPU 13.0-16.0 ms (9.1 ms at the roofs + 2302 launches x 1.7-3.0 us), so 63-77 tok/s. max() presumes the CPU encode of step N+1 overlaps the GPU of step N.
- **Measured on ds2e (`rp_chain.py`, eager path): one command buffer is committed inside the Python graph build, before `async_eval`** (2 command buffers per step, with `MLX_MAX_OPS_PER_BUFFER=100000`). ds2 without Engram commits none there. The cause is the Engram host read (`engram_mlx.py:25` `np.asarray(indices)`, `engram_rows.py:126` hashing token ids on the host): it waits for the previous step's GPU token. The DS4.1 mix has 2 Engram layers, and `spec.py` also reads the token with `.item()`. So the eager step cannot overlap its encode with the previous GPU step: it costs about GPU + CPU = 14.5 + 13.2 = 27.7 ms = **36 tok/s**, the 33-40 of the handoff. (The Python graph build before the first Engram layer still overlaps, so the real figure sits a little above 36.)
- Replay removes the encode from the critical path: the host computes the Engram indices from the token (microseconds, a refresh buffer) and triggers the ICB, 0.1 ms. Step = GPU + about 0.2 ms: 1000 / (16.1 + 0.2) to 1000 / (13.1 + 0.2) = **62-76 tok/s**. The GPU gap is not touched by replay (section 2), so this is the ceiling.
- MTP / verify rows are CPU-bound (verify costs 1.39 / 1.50 / 2.00 / 2.44 / 3.44 plain steps for n = 1..5, draft 4): with CPU removed they become GPU-bound, **62-78 -> 71-87 tok/s** (the s.11 GPU-bound multipliers); same for the multi-user rows. Robustness to a noisy host CPU is the other gain.

CPU per layer on the full model: 11.8 ms / 43 = about 275 us per layer now; replay is a constant per step, about 0.1 ms / 43 = 2-3 us per layer. (The slice figures in section 3 divide a whole step, head and embedding included, by 2.)

## 7. Files

`replay/icb_micro.mm` (ICB micro-benchmark), `replay/rp_probe.mm` + `tools/rp_probe.py` (what MLX encodes per step), `replay/rp.mm` (record, ICB build, replay hook), `tools/rp_chain.py` (the gate: exactness over passes, interleaving with eager steps, CPU and GPU timing), `tools/rp_diff.py`, `tools/rp_affine.py`, `tools/rp_ident.py` (per-step signature census). Build: `clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation replay/rp.mm -o replay/build/librp.dylib`. Run: `PYTHONPATH=$PWD:$HOME/kyojin-mlx/vlm_pkg ../../.venv/bin/python tools/rp_chain.py models/ds2 --k 6 --passes 20`.
