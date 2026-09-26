# Current progress and next steps

## STOP: work is paused, not complete

The user explicitly requested a pause on 2026-09-25, after approximately four
hours of implementation and measurement. Do not automatically resume agents,
builds, benchmarks, downloads, or optimization work. This file is a handoff for
the next agent after the user authorizes continuation.

The repository is **not yet optimal and has not met the requested performance
targets**. There is working, measured single-GPU inference, working two-GPU
inference, substantial improvements over the initial port, and useful numerical
and serving qualification. Several recent source changes and experimental
candidates still need integration or validation.

Implementation started from the following commit; this is the **pre-checkpoint
base**, not necessarily HEAD when this document is read:

```text
bace20dc perf(core): default to spin with configurable cuda synchronization
```

At the pause, the changes were a mixture of staged, unstaged, and untracked
files. The user subsequently requested that **all current source/documentation
changes be committed and pushed once the workers finish**. The resulting commit
is a paused checkpoint, not a claim of completed optimization. Build products,
model artifacts, raw profiles, and isolated candidate files remain ignored local
resources and are not included merely by committing tracked source.
Preserve any subsequent working-tree changes; do not reset, clean, discard, or
rewrite the checkpoint history.

**Read section10 first for the final source/binary and worker snapshots, then
section2 for the exact environment and section7 for continuation commands.**
This document exceeds common single-file tool output limits; read it in ranges
rather than assuming a truncated first read is complete. Most importantly, the
newest two-GPU output/head projection extension is not covered by the existing
paired performance numbers, and some source is newer than the executables.

## 1. Original requested deliverable

Optimize this from-scratch C++/CUDA engine for Qwen3.8-27B on:

- One Quadro RTX 8000, using physical GPUs 0/1 for single-card development.
- Two Quadro RTX 8000s over NVLink, using physical GPUs 2/3 for paired testing.
- One active request first, prioritizing decode without sacrificing prefill.
- Startup concurrency up to eight requests, with correct resource admission and
  useful batching.
- A 262,144-token logical context; benchmark 512/4096-token prefill, 128-token
  decode, and 32K prefill followed by 1K decode.
- Quality comparable to the reference Q4_K_M model. Custom conversion was
  permitted if necessary, but none has been needed.

The user selected the **best same-GPU-count historical baselines** as the targets.
Four-GPU results are stretch targets, not mandatory two-GPU targets.

| Configuration | pp512 target | pp4096 target | tg128 target |
|---|---:|---:|---:|
| One GPU | 719.07 tok/s | 681.79 tok/s | 28.58 tok/s |
| Two GPUs | 1245.92 tok/s | 1139.27 tok/s | 44.96 tok/s |

Reference authority, read-only:

```text
/home/algore/GIT/agorevski/TokenFurnace/targets/qwen3.8-27b/PERFORMANCE.md
```

The two-GPU maxima come from different physical pairs, not one jointly observed
configuration. Historical llama.cpp runs use different quantization, KV storage,
and token content from NInfer. Do not call these controlled runtime-only
comparisons. Do not modify TokenFurnace.

## 2. Machine, artifacts, and toolchain

Repository:

```text
/home/algore/GIT/agorevski/ninfer-rtx8000-48g-nvlink
```

Hardware is four 48-GiB Quadro RTX 8000s, compute capability 7.5. NVLink topology
is `0 <-> 1` and `2 <-> 3`, both reported as `NV2`. Driver is 580.173.02.
The machine has approximately 251 GiB RAM. No GPU clock, fan, power-limit, or
profiling-permission settings were changed. Long runs reached 85-89 C, so thermal
conditions matter.

### Use these explicit model paths

```text
NInfer v3:
/home/algore/models/qwen3.8-27b-ninfer/qwen3_8_27b.ninfer
Size:   20437521664 bytes
SHA256: 81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da

Independent reference:
/home/algore/models/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf
Size:   17106773984 bytes
SHA256: 7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b

Existing reference runtime:
/home/algore/llama.cpp-dspark-current
/home/algore/llama.cpp-dspark-current/build-peer2048
```

Hashes were checked. Neither model was downloaded, rewritten, or regenerated.
Do not accidentally select the older `qwen3.8-27b-ninfer-v2` artifact or an
NVFP4 artifact. This C++ reader requires v3.

The selected model uses `Qwen3_5ForCausalLM`: Qwen3.8 is an instance of the
existing mathematical architecture, not a new execution registration.
Important text shapes:

| Operand | Stored format | Shape `[output, input]` |
|---|---|---|
| Hidden / intermediate dimensions | - | 5120 / 17408 |
| Fused FFN gate/up parent | Q4 group64, FP16 scales | `[34816,5120]` |
| FFN down | Q5 group64, FP16 scales | `[5120,17408]` |
| Attention/GDN output projection | Q5 group64, FP16 scales | `[5120,6144]` |
| Main vocabulary head | Q8 group32, FP16 scales | `[248320,5120]` |

There are 64 text layers. Q6 occurs in vision, not the selected text backbone.
Selected text weights occupy approximately 15.9 GiB on the primary GPU.

### Preserve the working build directory

The working integration build is **`build-sm75/`**, not `build/` or
`build-rtx8000/`. It contains an assembled CUDA 12.9.86 toolkit from existing
local package-cache contents. Do not delete this directory to troubleshoot.

Verified CMake cache entries:

```text
CMAKE_CUDA_ARCHITECTURES=75
CMAKE_CUDA_COMPILER=<repo>/build-sm75/toolkit/bin/nvcc
CMAKE_CXX_COMPILER=/usr/bin/g++-13
CMAKE_MAKE_PROGRAM=/usr/bin/ninja
CUDA_cudart_LIBRARY=<repo>/build-sm75/toolkit/lib/libcudart.so
Python3_EXECUTABLE=/home/algore/miniconda3/envs/frigate-ai/bin/python3.11
```

The shell's default `nvcc` is CUDA 12.4, **not** this build's compiler. The
original AGENTS.md machine path under `/home/neroued/` is not this host's Python.
Use explicit Python 3.11 paths.

Standard environment for future build commands:

```bash
cd /home/algore/GIT/agorevski/ninfer-rtx8000-48g-nvlink
export TMPDIR="$PWD/build-sm75/work"
export PKG_CONFIG_PATH="$PWD/build-sm75/deps/lib/pkgconfig:/home/algore/miniconda3/envs/heretic/lib/pkgconfig"
PY=/home/algore/miniconda3/envs/frigate-ai/bin/python3.11
MODEL=/home/algore/models/qwen3.8-27b-ninfer/qwen3_8_27b.ninfer
```

FFmpeg 6.1 headers/libraries were reconciled with the existing installations;
CUDA header/runtime mixing was also corrected. Do not undo those paths by
reconfiguring from an unrelated Conda environment.

Portable `rtx8000` / `rtx8000-dev` presets were added, targeting SM75 in
`build-rtx8000/`. They are useful for a future clean installation but are **not**
a reason to abandon the already configured local integration build.

### Critical build-coordination lesson

Earlier concurrent/overlapping Ninja activity left `.ninja_deps` with
inconsistent path IDs. Every invocation then printed
`premature end of file; recovering` and rebuilt hundreds of unchanged CUDA
files. A large attention TU can take about 20 minutes to compile.

The build now uses existing **system Ninja 1.11.1**, not the Conda
`1.13.0.git.kitware.jobserver-pipe-1` binary. A complete build followed by the
identical build command was verified to return **`ninja: no work to do.`**
The exact original corruption cause was not conclusively established.

Only one owner may invoke Ninja/CMake against `build-sm75/` at a time. Avoid
even Ninja dry-runs or command-extraction invocations during an active build.
Other agents can use existing command logs for isolated compilations with
outputs outside the shared build tree. Do not launch competing "small target"
builds against it.

## 3. What is implemented

### SM75 platform and numerics

Build selection now accepts SM75 and SM120a in separate builds. The port adds
Turing-compatible staging, FP16 MMA, BF16 storage/rounding handling, PDL/copy
fallbacks, sampling support, and attention/materialization tiles that fit
Turing's shared-memory limits.

Main KV on SM75 is BF16 or INT8 group64; defaults select INT8. Unsupported
FP8/NVFP4 attention routes fail explicitly. Use the groupwise model above.

Important fixed failures, which should not be rediscovered as pending work:

- Q8 split-K/grouped variants exceeded the 48-KiB **static** shared-memory
  limit. Schedules and launch dimensions were corrected for SM75.
- Q8 pair column tails around 192/224/256 needed smaller valid tile launches.
- BF16 `[256,5120]` needed a single-stage shared-memory configuration.
- GDN cooperative prefill used SM120 occupancy assumptions. It now queries
  actual kernel occupancy and bounds/splits the grid; T512/1024/2048 and
  neighboring full/predicated cases were qualified.
- CUDA allocations, streams, events, and timers now preserve owning-device
  lifetime and caller binding.
- Two peer-context construction sites leaked the current CUDA device. They
  were fixed by guarding the entire peer construction, including member
  initialization. A guard created on an already-current device does not
  automatically undo arbitrary later `cudaSetDevice` calls.

Relevant areas: `src/core/`, `src/ops/common/`, `src/ops/softmax_attention/`,
`src/ops/gdn_gating_proj/`, and their tests.

### Turing linear execution

Primary files:

```text
src/ops/linear/turing.cu
src/ops/linear/turing.h
src/ops/linear/turing_tile.cuh
src/ops/linear/turing_gemv.cuh
src/ops/linear/turing_batched.cuh
src/ops/linear/a16_operand.cuh
src/ops/wrapper/linear_add.cpp
src/ops/wrapper/linear_swiglu.cpp
tests/ops/linear/test_turing_a16.cpp
tests/ops/linear/test_turing_batched.cpp
```

Q4/Q5/Q6/Q8 keep their stored row-split representation. Decode uses
bandwidth-oriented FP32 accumulation; prefill uses FP16 tensor operands while
preserving public BF16 boundaries and correcting values outside the usable
FP16 representation. No runtime weight repacking was introduced.

For Q4/Q5 at T >= 512, selected production code prepares activations once,
including exception masks and a per-token integer power-of-two exponent.
It scales **up** toward `2^14` when safe, uses exact FP16 round-trip
representability checks, and unscales before residual addition or SwiGLU.
Huge-input cases keep the existing fallback rather than being scaled toward
underflow. Do not discard tiny nonzero values or relax numerical tolerances.

Caller-owned workspace includes approximately `2*K*T + K*T/8 + 4*T` bytes,
with allocator alignment. Public Linear/Add/SwiGLU workspace queries and both
TP rank planners consume this requirement. It is not hidden allocation or
once-at-capture data: values, masks, and exponents must update on every replay.

Selected batched routes:

| Operation | Extents using new weight-reusing batched kernel |
|---|---|
| Q4 Linear `[34816,5120]` | T2-8 |
| Q4 fused SwiGLU | T2-8 |
| TP Q4 Linear `[17408,5120]` | T2-4 only |
| Q8 head `[248320,5120]` | T2-4 only |

T1 is unchanged. Retain existing Q5 routes, Q8 above T4, and TP Q4 above T4.
Measured rejected alternatives were slower, particularly TP Q4 T8
(approximately 0.65x existing performance).

### Two-GPU execution and the newest unfinished extension

Public option: `EngineOptions.tensor_parallel_device`; command-line option:
`--tensor-parallel-device N`. With `CUDA_VISIBLE_DEVICES=2,3`, select
`--device 0 --tensor-parallel-device 1`, **not physical ordinal 2**.

The validated initial design splits dense FFNs: primary computes gate, peer
computes up, both form the SiLU product, and each computes half of down's
output rows including its residual. Full primary weight parents remain
resident. Peer row planes are copied without requantization. KV, GDN state,
checkpoints, admission, and sampling remain primary-owned; VRAM is not pooled.
The paired path is dense SM75 text, without Vision or speculative decoding.

Primary files:

```text
src/artifact/materializer.{h,cpp}
src/models/qwen3_5/model.{h,cpp}
src/models/qwen3_5/execution/parameters.{h,cpp}
src/models/qwen3_5/execution/tensor_parallel.{h,cpp}
src/models/qwen3_5/execution/{text,ffn}.cpp
src/models/qwen3_5/program/{decode,prefill,graphs,program_impl}.cpp
src/models/qwen3_5/program/planning/startup.cpp
src/core/peer_copy.{h,cu}
tests/models/qwen3_5/test_tensor_parallel*.cpp
tests/test_peer_copy.cpp
```

Nsight proved captured peer DMA serialized gate/up work. Small decode peer
transfers were replaced with core-owned CUDA byte-copy kernels, restoring
overlap and improving measured pair decode. They have no allocation, device
switching, spins, or semaphores; existing streams/events own ordering.

**Source is ahead of the last measured pair configuration.** The source class
is now `TensorParallelProjections`, not just `TensorParallelFfn`.
The additional output-row extension is complete and syntax-checked in source,
but its final integration is not yet qualified:

- Q5 attention/GDN output: `[5120,6144]` -> two `[2560,6144]` halves.
- Q8 vocabulary head: `[248320,5120]` -> two `[124160,5120]` halves.
- Initial model selection is T=1 only; sampling/log-probability computation
  must still see the complete correctly ordered vocabulary on primary.

Model now has peer `output` rows per layer and `tensor_parallel_head_`.
New helper names include `TensorParallelProjections::ffn`, `project`,
`project_add`, `project_text_head`, and `output_projection_workspace_bytes`.
Every main head caller uses `project_text_head`, including exact-prefix reuse
and scalar causal scoring. Wider inputs retain the original full projections.
Startup queries cover the actual primary and peer operands; the peer workspace
is allocated before Engine publication.
The extra encoded peer weights total **1,336,033,280 bytes**, while the full
primary parents remain resident. This is calculated weight storage, not a newly
measured total VRAM footprint.
This extension is **not covered by the 37.40 tok/s measurement**.
Review its current source/build/qualification status before using it.

At the read-only pause inspection, the new Q8 head and Q5 output-shard Op
registrations explicitly reject T>1. Do not assume these new shapes support
batched head/prefill use. Review the supported-domain contract before broadening
them; a model's T1 optimization choice is not a reason to silently violate an
existing Op contract.

Another staged correction uses kernel peer copies for **all stream-captured
FFN transfers**, while retaining DMA for ordinary eager FFN prefill. New T1
output projections use kernel copies. The correction addresses a
`cudaGraphExecUpdateFailure` with update result5 at T128 in the larger synthetic
prefill test. Do not suppress that check: retest the full production-shape
capture/update cases, not just decode.

## 4. Measured results: do not confuse scopes

Detailed maintained record:
[`docs/performance/rtx8000-qwen3.8-27b.md`](docs/performance/rtx8000-qwen3.8-27b.md).
Raw reports are ignored local files under `profiles/`; they will not follow a
fresh clone automatically.

### Native single-GPU matrices

Full matrices use physical GPU0, the same artifact/corpus, INT8 KV, CUDA Graph
decode, context/capacity 8192, chunk **2048**, one warmup and five retained
repetitions. Means +/- sample standard deviations, tokens/second:

| State | pp512 | pp4096 | tg128 |
|---|---:|---:|---:|
| Initial corrected port | 199.3467 +/- 3.0069 | 175.1595 +/- 3.3392 | 29.0779 +/- 0.1459 |
| UP scaling + selected batching | 454.9166 +/- 4.6724 | 421.5799 +/- 9.1564 | 29.4489 +/- 0.0496 |
| Historical target | 719.07 | 681.79 | 28.58 |

Reports:

```text
profiles/bench/rtx8000-single-prepack-baseline/
profiles/bench/rtx8000-single-scaled/
```

Prefill improved 2.282x/2.407x versus this implementation's own baseline but
still misses the historical targets by 36.74%/38.17%.

An earlier cooler **decode-only** run measured 30.190640 +/- 0.190507 tok/s,
using chunk1024. Its file is
`profiles/bench/rtx8000-single-decode-initial.json`. Do not average it with the
full-matrix decode that follows long prefill/thermal loading.

### Native paired results

| State / scope | pp512 | pp4096 | tg128 |
|---|---:|---:|---:|
| Initial FFN TP, graph, five reps | 241.954 +/- 13.167 | 178.779 +/- 1.537 | 23.414 +/- 0.055 |
| Eager diagnostic, three reps | - | - | 34.346 +/- 0.533 |
| Kernel-peer-copy graph, five reps | - | - | 37.399 +/- 0.653 |
| Historical target | 1245.92 | 1139.27 | 44.96 |

Directory: `profiles/bench/rtx8000-pair-prepack-baseline/`.
The improved decode report is:

```text
json/native/tg128-kernel-peer-candidate.json
```

No complete paired matrix has established target completion for the latest
UP-scaling or additional output/head-sharding source. Do not apply old paired
prefill numbers to unmeasured new code.

### HTTP concurrency

Physical GPU1, one wave per startup concurrency, INT8, no speculation, greedy,
chunk1024, actual 262144-token shared KV pool. Each wave requested256 output
tokens per request. All15 requests completed256 tokens with `length`; complete
steady intervals had the actual batch size equal to C.

| C | Earlier aggregate decode tok/s | Selected batching aggregate tok/s |
|---|---:|---:|
| 1 | 30.62499 | 30.62475 |
| 2 | 18.07408 | 52.44444 |
| 4 | 33.10345 | 62.39999 |
| 8 | 47.60976 | 62.45161 |

```text
profiles/bench/rtx8000-single-concurrency-prepack/
profiles/bench/rtx8000-single-concurrency-scaled-batched/
```

The C2 regression is fixed. C8 improves31.17% and reaches about2.04x C1, but
C4->C8 throughput is essentially flat. This is useful measured improvement,
not linear scaling or an assertion that C8 is globally optimal.
These are HTTP steady-wave rates, not five-repeat native tg128 results.

### Quality screen already completed, but not final for new arithmetic

The independent reference and native baseline use the same exact IDs, text,
window resets and targets: four streams,32764 scored targets,12 windows,
context4096, stride2048,8192 input tokens per stream.

| Baseline candidate | Mean NLL | Delta from Q4_K_M | Aggregate <=0.02 |
|---|---:|---:|---|
| Native BF16 KV profile | 1.6004399212 | +0.0138283151 | Pass |
| Native INT8 deployment | 1.6004361601 | +0.0138245540 | Pass |

Reference NLL:1.5866116061. English-reference domain deltas0.02319/0.02281
are disclosed; no separate domain threshold was predeclared.

**Use this reference:**

```text
profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.json
```

Native `bf16` storage means **BF16 K / FP16 V**. The older
`reference-bf16-retry1.json` used BF16 for both and is diagnostic only.
The matched reference used a slow CPU attention fallback for its mixed KV
profile; its timing is not a performance result. Its scores are reusable.

Baseline candidates/comparisons:

```text
profiles/bench/rtx8000-quality/native-bf16-gpu1.json
profiles/bench/rtx8000-quality/native-bf16-gpu1.comparison.json
profiles/bench/rtx8000-quality/native-int8-gpu1.json
profiles/bench/rtx8000-quality/native-int8-gpu1.comparison.json
```

These passes predate activation prepacking/scaling and the newest projection
work. Run selected-build quality/equivalence checks before final claims.
This is a quality screen, not a full capability evaluation.

## 5. Profiling conclusions and candidate decisions

### Dominant prefill mechanism is known

The initial full pp512 trace attributed approximately:

- 57.6% to Q5 residual projections.
- 23.5% to Q4 SwiGLU.
- 12.4% to grouped attention/GDN input projections.
- Only about5.5% to GDN state/WY/output and0.1% to attention.

Do not optimize nonlinear attention/GDN mathematics as the primary prefill
fix without new evidence. Kernel names resembling legacy implementations do
not prove incorrect dispatch: SM75 branches reuse some existing template names.

```text
profiles/nsys/rtx8000-pp512-base.nsys-rep
profiles/nsys/rtx8000-pp512-base-stats_cuda_gpu_kern_sum.csv
profiles/nsys/rtx8000-pp512-prepack-gpu1.nsys-rep
profiles/nsys/rtx8000-pp512-prepack-gpu1-stats_cuda_gpu_kern_sum.csv
```

Controlled Q5 Add `[5120,17408]`, T512 reproduction:

| Activation distribution | Prepack operation time |
|---|---:|
| Normal, no exceptions | 3.914 ms |
| 50% exact zeros | 3.803 ms |
| 1% tiny nonzero values | 18.063 ms |
| 10% tiny nonzero values | 84.674 ms |
| Synthetic SwiGLU,7.789% exceptions | 70.650 ms |

The original mask marked every nonzero magnitude below `2^-14`.
The per-tile FP32 correction loop is very expensive on real tiny-valued inputs.
The selected UP-scaling/round-trip candidate reduced the synthetic SwiGLU
case to about19.11ms and exception density to0.557%, while preserving the
unmodified numerical criteria. The full-model improvement is measured above.

Candidate evidence:

```text
profiles/bench/sm75-prefill-candidate/
profiles/bench/sm75-scaled-candidate/
```

**Rejected candidate: using rounded shared FP16 weights for corrections.**
A valid Q5 cancellation case disproves this shortcut: code3 with FP16 scale
`0x3555` decodes to `1 - 2^-12`, but its shared FP16 representation is1.
With1024 active inputs of `2^-30` and residual `-2^-20`, the FP64 reference and
current exact correction produce `-2^-32`; rounded-weight correction produces
zero. A separate zero-weight column with activation `2^20` retains shift zero.
Relative L2 is1.0 against the unchanged1/256 limit. No production change or
GPU timing was made for this rejected rule.

```text
profiles/bench/sm75-shared-correction-candidate/cancellation_repro.py
profiles/bench/sm75-shared-correction-candidate/result.log
```

**Qualified but unselected alternative: cache exact packed weights in shared
memory.** This isolated candidate retains raw Q4/Q5 codes, high bits and stored
FP16 scales for the current K tile, then independently decodes FP32 correction
weights from those unchanged bytes. It does not substitute rounded shared
FP16 weights, allocate an FP32 tile, repack persistent weights, or change
global workspace capacity.

Same-GPU0 T512 paired medians against the built UPscale implementation:

| Operation and synthetic input | Current UPscale | Exact packed cache | Ratio |
|---|---:|---:|---:|
| Q5 Add `[5120,17408]`, normal | 3.975ms | 3.553ms | 1.119x |
| Q5 Add `[5120,17408]`, synthetic SwiGLU | 18.331ms | 14.164ms | 1.294x |
| Q4 SwiGLU `[34816,5120]`, normal | 5.844ms | 5.825ms | 1.003x |
| Q4 SwiGLU `[34816,5120]`, synthetic SwiGLU | 31.008ms | 25.031ms | 1.239x |

The candidate passed32 cases /160 eager and graph executions against unchanged
FP64 criteria, including tails, padded K, extreme inputs, overflowing weight
groups and cancellation. Its GPU check also preserves the exact `-2^-32`
counterexample above. Q5 static shared memory is28352bytes rather than25664,
with104 rather than108 registers; Q4 SwiGLU uses27840bytes and100 registers.
These footprints fit the same two-CTA resource budget.

```text
profiles/bench/sm75-exact-shared-candidate/
  scaled_tile.cuh
  scaled_kernels.cuh
  compare.cu
  compare_swiglu.cu
  qualify.cu
  cancellation.cu
  compare-results.log
  swiglu-results.log
  qualify-results.log
  cancellation-results.log
```

This candidate is **not integrated or selected**. It has no measured real-model
quality or end-to-end benefit. If selected after resumption, use a reviewed
surgical change to the prepared Q4/Q5 correction path, not a blind copy of the
experimental headers; then repeat public Op and selected-model qualification.

### Grouped projection tile experiment

An isolated WM32/WN32 alternative to the existing WM64/WN16 grouped
attention/GDN prefill configuration was investigated without changing public
math, workspace, or common headers.

```text
profiles/bench/sm75-grouped-candidate/
  comparison.csv
  baseline-qualification.log
  candidate-qualification.log
  q4_q5_attn_input_gemm_mma.cu
  q4_q5_gdn_input_gemm_mma.cu
  warp_tile.cuh
```

Do not copy these experimental files over production blindly. Read the final
pause notes and comparison first; selection and end-to-end benefit remain
separate from an isolated candidate result.

The completed candidate's more stable normal-input gains were approximately
10% for GDN at T512,7.3% at T1024,7.2% at T2048, and5.2-5.9% for attention.
Small-input cases improved approximately15%. A raw GDN-T512 ABBA ratio of1.204x
included first-baseline warmup/clock variability and is not a stable20% result.
This modest candidate passed the existing FP64 criteria but was **not selected
or integrated before the pause**.

### Two-GPU graph mechanism is known

The graph baseline had0/1024 gate/up pairs overlapping, while matching eager
execution had1024/1024 overlap. Captured peer-copy activity was absent from
the trace, which is missing attribution, not zero cost. Graph launch CPU
time was about3.29ms. The graph-native copy candidate restored1024/1024
overlap and improved measured tg128 to37.40tok/s.

```text
profiles/nsys/rtx8000-tp-graph-baseline.*
profiles/nsys/rtx8000-tp-eager-baseline.*
profiles/nsys/rtx8000-tp-graph-kernel-copy.*
profiles/bench/sm75-peer-copy/
```

Changing height-one copies from2D DMA to1D DMA was measured and rejected as
the primary fix: the difference was too small/inconsistent.
Do not introduce unbounded GPU spin waits, silently disable graphs, or
attribute all graph overhead to bandwidth without a trace.

## 6. Validation already performed

Evidence is spread across the retained logs and performance page:

- Whole SM75 product/test/benchmark build succeeded; incremental no-op behavior
  was subsequently verified with system Ninja.
- Core device/arena/lifetime and bidirectional peer-copy tests passed.
- Independent numerical attention coverage included262144 positions,
  batches1-8, masks and graph updates; this is not a full model262K input run.
- GDN cooperative-prefill regressions passed at T512/1024/2048 and adjacent
  boundaries, including eager and graph execution.
- Public Linear/Add/SwiGLU/Pair/TopK/input-projection suites and selected
  batched routes passed independent packed-weight FP64 qualification.
- UP-scaling production qualification included28 cases and156 eager/graph
  phases, smallest BF16 values, FP16 subnormals, cancellation, query/peak
  workspace agreement, and changing masks/scales.
- Initial TP FFN/seam, real-model placement, cancellation, B1-8 isolation and
  actual262144-token KV pool checks passed. Single/dual log-probability
  comparison reported RMS0.008768 and maximum0.193366; it is not the independent
  Q4_K_M quality screen.
- Six host options/admission/scoring protocol checks passed.
- Measurement Python regression suite last reported54 passing tests.

SM120a runtime was not tested: there is no RTX5090 here. Some affected SM120
sources were compile-checked with CUDA12.9, but do not claim complete Blackwell
qualification.

## 7. Exact continuation sequence after the user resumes work

### A. Reconcile source and build before executing performance tests

1. Read the final pause notes below, `git status --short`, and the latest source.
   New TP projection source may be ahead of binaries.
2. Check for other CUDA/Ninja jobs. Acquire explicit GPU ownership; do not
   infer that a brief idle sample gives permission to overlap another job.
3. Use the environment in section2 and one build owner:

```bash
cmake -S . -B build-sm75 -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja
cmake --build build-sm75 -j --target \
  apps/ninfer apps/ninfer-serve apps/ninfer-perplexity \
  ninfer_bench ninfer_score ninfer_peer_copy_test ninfer_device_target_test \
  ninfer_qwen3_5_tensor_parallel_test ninfer_qwen3_5_tensor_parallel_real_test \
  ninfer_linear_turing_a16_test ninfer_linear_turing_batched_test \
  ninfer_linear_add_q5_a16_test ninfer_linear_swiglu_q4_a16_test \
  ninfer_gdn_gating_proj_test
```

Use fresh failure output, not an old log. The earlier registration,
DeviceGuard, unsupported PTX and oversized Q8-static-shared blockers were
fixed. Do not reapply old proposed fixes. An isolated `nvcc` test does not
automatically update product executables.

### B. Run the affected checks, not every unrelated test

First check the new C1-only row-shard Op admission and gather boundaries. This
test source is newer than the last built central executable:

```bash
cmake --build build-sm75 -j --target ninfer_linear_turing_a16_test
CUDA_VISIBLE_DEVICES=0 build-sm75/tests/ninfer_linear_turing_a16_test --c1-row-shards
```

Use only a GPU explicitly assigned by the next coordinator. The bounded
selector covers Q5 LinearAdd `[2560,6144]` and Q8 Linear `[124160,5120]` at T1,
both encoded parent halves, independent FP64 results, residual/gather seams,
changed-input graphs, and rejection of T>1 intervals/execution before mutation.
Then run the broader affected suites:

```bash
ctest --test-dir build-sm75 -R '^ninfer_device_target_test$' --output-on-failure

CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-sm75 \
  -R '^ninfer_(linear_turing_a16|linear_turing_batched|linear_add_q5_a16|linear_swiglu_q4_a16)_test$' \
  --output-on-failure

CUDA_VISIBLE_DEVICES=0 build-sm75/tests/ninfer_gdn_gating_proj_test --prefill-only

CUDA_VISIBLE_DEVICES=2,3 build-sm75/tests/ninfer_peer_copy_test --peer
CUDA_VISIBLE_DEVICES=2,3 ctest --test-dir build-sm75 \
  -R '^ninfer_qwen3_5_tensor_parallel(_swiglu|_projection)?_test$' --output-on-failure

CUDA_VISIBLE_DEVICES=2,3 \
  build-sm75/tests/ninfer_qwen3_5_tensor_parallel_test --production-shape

CUDA_VISIBLE_DEVICES=2,3 \
NINFER_TEST_ARTIFACT="$MODEL" NINFER_TEST_TENSOR_PARALLEL=1 \
  build-sm75/tests/ninfer_qwen3_5_tensor_parallel_real_test
```

The projection CTest invokes `--projection-c1`: full-output FP64 qualification
of Q5 mixer outputs and the complete Q8 vocabulary, including large represented
inputs, residual cancellation, guards, changing-input graph replay, and update.
The real test additionally checks new peer-weight placement, scalar scoring,
and exact zero-suffix prefix reuse. These newest checks had not run against
fresh matching binaries at the pause.

Measurement-only checks, if those files change:

```bash
cmake --build build-sm75 -j --target ninfer_bench_support_test ninfer_score_protocol_test
ctest --test-dir build-sm75 \
  -R '^(ninfer_bench_support_test|ninfer_score_protocol_test)$' --output-on-failure

PYTHONPATH="$PWD/profiles/bench/python-deps:$PWD" \
  /home/algore/miniconda3/envs/vllm/bin/python -m pytest -q \
  tests/test_rtx8000_measurement.py tests/test_serve_corpus.py
```

### C. Measure the selected binary, using new output directories

Never use `--resume` to reuse reports from a different implementation.
The matrix preset uses chunk2048; report that fact. Compare chunk1024
separately if tuning it, rather than silently mixing settings.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" tools/bench/run_ninfer_bench_matrix.py \
  --preset rtx8000 --bench build-sm75/bench/ninfer_bench --build-dir build-sm75 \
  --weights "$MODEL" --device 0 --kv-dtype int8 --telemetry-gpus 0 \
  --output-dir profiles/bench/rtx8000-single-resumed-01 --no-build

CUDA_VISIBLE_DEVICES=2,3 "$PY" tools/bench/run_ninfer_bench_matrix.py \
  --preset rtx8000 --bench build-sm75/bench/ninfer_bench --build-dir build-sm75 \
  --weights "$MODEL" --device 0 --tensor-parallel-device 1 \
  --kv-dtype int8 --telemetry-gpus 2,3 \
  --output-dir profiles/bench/rtx8000-pair-resumed-01 --no-build
```

Choose unused explicit output paths on subsequent runs. Repeat the HTTP wave
after relevant batch/graph changes:

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" tools/bench/run_serve_concurrency.py \
  --serve build-sm75/apps/ninfer-serve --artifact "qwen3.8-27b=$MODEL" \
  --mode mtp0 --sampling greedy --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 256 --max-context 262144 --kv-capacity 262144 \
  --prefill-chunk 1024 --device 0 --port 18081 \
  --output profiles/bench/rtx8000-concurrency-resumed-01
```

The runner starts, health-checks and stops each server. Do not leave an extra
server consuming a GPU during another benchmark. Rates are aggregate steady
decode only over intervals with actual batch C.

### D. Final selected-build quality

Reuse the completed matched reference. Do not rerun its slow mixed-KV CPU
fallback unnecessarily. Select an unused candidate report path:

```bash
CUDA_VISIBLE_DEVICES=1 build-sm75/bench/ninfer_score \
  --weights "$MODEL" --corpus eval/corpora/perplexity-1m/manifest.json --quick \
  --context 4096 --stride 2048 --max-tokens 8192 --device 0 --kv-dtype int8 \
  --output profiles/bench/rtx8000-quality/native-resumed-int8.json

/home/algore/miniconda3/envs/vllm/bin/python -m tools.bench.compare_causal_scores \
  --candidate profiles/bench/rtx8000-quality/native-resumed-int8.json \
  --reference profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.json \
  --allow-kv-mismatch \
  --output profiles/bench/rtx8000-quality/native-resumed-int8.comparison.json
```

For a BF16-profile control, use `--kv-dtype bf16` and omit
`--allow-kv-mismatch`. For paired scoring, expose `2,3` and pass
`--device 0 --tensor-parallel-device 1` to the exporter. Keep all IDs/window
settings unchanged. The comparator's aggregate threshold defaults to0.02;
do not relax it after seeing results.

### E. Long-context work is still outstanding

An actual262144-token KV pool and short requests were tested. A full
262144-token input was **not** processed, and the requested32K/1K workload
was not completed as final evidence.

Start with a bounded public-Engine combined run after selecting performant code:

```bash
CUDA_VISIBLE_DEVICES=0 build-sm75/bench/ninfer_bench \
  --weights "$MODEL" -pg '32768,1024' --max-ctx 262144 \
  --prefill-chunk 1024 --kv-dtype int8 --device 0 \
  --warmup 0 -r 1 --output json \
  --output-file profiles/bench/rtx8000-long-resumed-01.json
```

Native `-pg P,G` requests G+1 completion tokens: one belongs to prefill and
exactly G to decode. Do not equate that blindly to an API's total output limit.
Then collect the repetitions needed for the claim and validate near-capacity
behavior. Existing TTFT tooling has a260096-token cold fixture; see the
performance page for its setup. Capacity allocation alone is not full-length
execution or quality validation.

## 8. Profiling tools that actually work here

Nsight Systems was absent and the PATH `ncu` was a broken wrapper. A verified
official Nsight Systems CLI package was extracted locally without changing
system packages:

```bash
NSYS="$PWD/.local/nsight-systems/opt/nvidia/nsight-systems-cli/2025.3.1/bin/nsys"
```

Example future single-case trace; use a free GPU and a new output path:

```bash
CUDA_VISIBLE_DEVICES=0 "$NSYS" profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=profiles/nsys/rtx8000-resumed-pp512 \
  build-sm75/bench/ninfer_bench --weights "$MODEL" \
  -p 512 -r 1 --warmup 1 --max-ctx 8192 --prefill-chunk 2048 \
  --kv-dtype int8 --device 0 --profile-measured \
  --output json --output-file profiles/nsys/rtx8000-resumed-pp512.json

"$NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  --output profiles/nsys/rtx8000-resumed-pp512-stats \
  profiles/nsys/rtx8000-resumed-pp512.nsys-rep
```

For paired graph analysis add `--cuda-graph-trace=node`; use matching graph
and eager tg16/tg32 captures to explain scheduling, then use unprofiled
five-repeat tg128 for the performance claim.

A real cached Nsight Compute binary exists at:

```text
/home/algore/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu
```

Its real-model hardware-counter attempt failed with `ERR_NVGPUCTRPERM`.
No permissions were changed. Use `--clock-control none` if authorized to
profile later; do not change clocks or driver settings implicitly.
CUDA activity tracing and controlled input-distribution reproductions were
sufficient for the current root-cause attribution.

## 9. Completion conditions and documentation cleanup

The original request is complete only after:

- Selected single- and two-GPU code is built, usable, and numerically qualified.
- Same-count native prefill/decode targets are met with sufficient evidence,
  or a genuine blocker is explicitly discussed with the user; do not silently
  redefine the targets.
- Final selected-build quality passes the declared screen, with scope/caveats.
- Long-context and concurrency behavior is validated at the claimed scope.
- Current CLI/build/architecture/serving documentation describes the retained
  implementation, including any new output/head sharding.

Some active documentation still describes FFN-only TP and may be behind the
newest source extension. Update the existing authorities, not just this file:
`README.md`, `docs/cli.md`, `docs/serving.md`,
`docs/maintainer/engine-architecture.md`,
`docs/maintainer/resource-scheduling-and-context-cache.md`,
`docs/maintainer/build-system.md`, `bench/README.md`, and the RTX8000 performance
record.

Remove or archive this temporary handoff when the work is completed or
abandoned; do not create parallel "final/v2/new-design" authorities.

## 10. Final pause notes

The main coordinator has sent a stop instruction to every worker and started
no further implementation, build, or benchmark after the pause request.
The last read-only process check found no CUDA compute process or active
Ninja/CUDA compiler. All workers have now reported idle, source-frozen, and
without GPU reservations; their final handoffs are recorded below. Only the
user-requested commit/push of this paused checkpoint is authorized in this
session. Do not treat an old agent message as fresher than the current files
and these final notes.

### Source/binary snapshot

At the pause inspection:

```text
apps/ninfer executable:                         2026-09-25 20:58:39 -0700
bench/ninfer_bench executable:                  2026-09-25 20:58:49 -0700
bench/ninfer_score executable:                  2026-09-25 20:58:51 -0700
src/ops/linear/turing.cu:                       2026-09-25 20:44:40 -0700
src/ops/linear/turing_tile.cuh:                  2026-09-25 20:44:40 -0700
src/models/qwen3_5/execution/tensor_parallel.cpp: 2026-09-25 21:15:43 -0700
src/models/qwen3_5/execution/text.cpp:           2026-09-25 21:15:44 -0700
tests/models/qwen3_5/test_tensor_parallel.cpp:   2026-09-25 21:15:44 -0700
tests/models/qwen3_5/test_tensor_parallel_real.cpp: 2026-09-25 21:06:18 -0700
src/runtime/contract/device_target.h:           2026-09-25 21:28:56 -0700
src/runtime/engine/engine.cpp:                  2026-09-25 21:30:01 -0700
```

Therefore **at least the latest TP implementation and startup source are newer
than the executables**. The active `tensor_parallel.cpp`, `text.cpp`, TP test,
and TP real-test objects were still timestamped approximately20:58, as confirmed
against `compile_commands.json`. The on-disk binaries contain an earlier
intermediate snapshot, not the final source or necessarily the validated
37.40 tok/s configuration. A fresh coordinated build is mandatory on resumption.
The main's latest successful build logs are
`build-sm75/scaled-batched-ops-build.log` and
`build-sm75/scaled-batched-product-build.log`.

The last worker edits added `src/runtime/contract/device_target.h` and
`tests/test_device_target.cpp`, checking the physical GPU against the compiled
SM75 or SM120a target. `Engine::initialize_device` now checks before stream
creation or artifact reading, and startup planning uses the same helper.
The pure host contract passed in both compile-definition configurations.
This supersedes the earlier
unconditional capability75-or120 acceptance. The change is newer than the
executables above: build and run the new host test on resumption and do not
mistake source inspection for integrated verification.

### Runtime/NVLink worker

Paused with **no running command or server and no GPU reservation**. Its last
process check found no compute process on GPUs2/3. The `nvlink-execution` todo is
blocked with `Paused by user`; no performance target was redefined or marked
complete.

Source edits are coherent, not half-written. New files are:

```text
src/models/qwen3_5/execution/tensor_parallel.{h,cpp}
src/runtime/contract/device_target.h
tests/models/qwen3_5/test_tensor_parallel.cpp
tests/models/qwen3_5/test_tensor_parallel_real.cpp
tests/test_device_target.cpp
```

Supporting changes cover model materialization/parameters/text execution,
Program planning and execution, Engine initialization, public options,
CLI/perplexity/serving option wiring, and matching CMake/host tests.
The final T1 extension requires the source registrations in
`src/ops/linear/q8/q8_dispatch.cpp` and `src/ops/wrapper/linear_add.cpp`; both were
confirmed present, admitting only T1 for the new half-row profiles.

The next runtime action is the fresh build and exact checks in section7,
not another design rewrite. In particular, finish the pending T128
capture/update correction, qualify the full-vocabulary T1 projection,
singleton scoring and zero-suffix reuse, then measure C1 before/after.
Do not quote the old FFN-only pair rate as the extension's performance.

An isolated public C1 Op test was reported ready after the pause:
`profiles/bench/sm75-c1-row-shards/public-test --c1-row-shards`.
The runtime worker did **not** execute it after the stop instruction.
Readiness of that ignored local binary is not a qualification result and does
not replace rebuilding the current product.

### Measurement worker

Stopped with no processes or GPU reservations. Measurement-owned source is
coherent and predates the latest executable links. Documentation has newer
staged/unstaged results; preserve its working-tree content.
Baseline artifact preparation and baseline quality are complete, not final
qualification of all subsequent arithmetic.

### Platform/batched worker

Stopped with no running command, server, or GPU reservation. Platform production
changes are coherent and included in the latest linked products; there is no
partially applied platform production edit. Its completed status remains done.

The WM32/WN32 grouped-projection experiment described above is isolated only.
Temporary binaries, objects and archive copies were removed, while its source
shims, logs and comparison remain. If selected after resumption, change only
the SM75 WM/WN arguments in:

```text
src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
```

Then build `ninfer_gdn_input_proj_test`, `ninfer_attn_input_proj_test` and
`ninfer_bench`, run the two public projection tests on an exclusively assigned
GPU, and measure the whole Engine. Do not replace common helpers or claim
that its isolated gain establishes the native prefill target.

### Linear worker

Paused with **no running command, server, or GPU reservation**. GPU0 and GPU1
were explicitly returned to main. No partially applied production edit remains.
The `sm75-linear` todo is blocked with **Paused by user**; do not automatically
resume optimization or claim the requested performance targets are achieved.

The combined UPscale/exact-representability prepared path and selected batched
hooks are in the20:57-20:58 central binaries. The worker qualified actual
production entry points through isolated RDC/device-linked drivers, not only
proposal kernels:28 cases /156 eager and graph executions passed FP64,
workspace-peak, guard, range, cancellation and input-preservation checks.
Exact production tests also covered FP16 subnormal products and the smallest
BF16 input using integer exponent147 with `ldexpf`. Platform separately passed
actual public Linear/SwiGLU tests for the selected batched profiles and TP T2-4.

```text
profiles/bench/sm75-scaled-candidate/production_qualify-results.log
profiles/bench/sm75-scaled-candidate/production_subnormals-results.log
profiles/bench/sm75-batched-candidate/production/public-results.log
profiles/bench/sm75-batched-candidate/production/extended-public-results.log
```

The following later, **host-only C1 admission changes are not in those central
binaries**:

| Path | Last modification,2026-09-25 -0700 | Change |
|---|---|---|
| `src/ops/linear/q8/q8_dispatch.cpp` | 21:05:07 | Q8 Linear `[124160,5120]`, exactly T1 |
| `src/ops/wrapper/linear_add.cpp` | 21:05:08 | Q5 LinearAdd `[2560,6144]`, exactly T1 |
| `tests/ops/linear/test_turing_a16.cpp` | 21:10:59 | `--c1-row-shards` numerical/admission tests |

Matching contracts were updated in `include/ninfer/ops/linear.h` and
`include/ninfer/ops/linear_add.h`. Workspace queries return zero only for
`[min_tokens,max_tokens]=[1,1]` on these new profiles and reject any interval
containing T>1. Execution likewise rejects T>1 before output mutation.
Original full-row profiles retain their existing positive-T domain. No new
CUDA arithmetic was introduced for these C1 extensions.

Comparison artifact timestamps:

```text
build-sm75/src/ops/libninfer_ops.a:                20:57:41
build-sm75/tests/ninfer_linear_turing_a16_test:    20:57:43
build-sm75/bench/ninfer_bench:                    20:58:49
profiles/bench/sm75-c1-row-shards/public-test:     21:12:32
```

The isolated C1 executable links the updated public host dispatch with the
existing production CUDA kernels. It and both SM75/non-SM75 host compilations
succeeded. **No GPU execution result for the new C1 test was received before
the pause.** Do not mistake the older TP FFN tests or standalone compilation
for qualification of the new head/output shards.

The minimal next action is section7B's bounded C1 build/test, followed by the
runtime owner's corresponding C1 projection and real-Engine checks after one
coordinated rebuild. The independent exact packed shared-cache experiment in
section5 requires an explicit selection and separate integration; leave it
isolated while validating the already-landed C1 changes.

Completed exploratory binaries/objects were cleaned from
`profiles/bench/sm75-linear/`; its reports and reproduction sources remain.
Some later candidate and C1 isolated executables/objects are still present
under their named local directories. They are not running, not production
build outputs, and not evidence of a central rebuild.
