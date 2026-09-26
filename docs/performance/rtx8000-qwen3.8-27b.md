# Qwen3.8-27B on Quadro RTX 8000

## Scope and status

Target: one Quadro RTX 8000 48 GiB (`sm_75`), or one NVLink pair (physical GPUs
0–1 or 2–3), using the public `.ninfer` Engine. Four-GPU reference results are
not two-GPU targets. MTP/API rates are not native non-speculative decode rates.

The coordinated `build-sm75` configuration selects CUDA **12.9.86** from
`build-sm75/toolkit`, host `/usr/bin/g++-13`, and `NINFER_SM75=1`.
This is not the CUDA 13.1 toolchain described for the original Blackwell host.
Platform configuration uses Python 3.11.15 at
`/home/algore/miniconda3/envs/frigate-ai/bin/python3.11`; the independent
measurement tooling below uses the separately verified vLLM-environment
Python 3.11.15. These are configuration facts, not a completed GPU qualification.

The current best complete single-GPU matrix measures **454.92 pp512,
421.58 pp4096 and 29.45 tg128 tok/s** after scaled activation preparation
and batch changes. Prefill improved **2.28×/2.41×** over this implementation's
retained prepack baseline, but remains **36.74%/38.17% below** the historical
same-count targets. This is not all-target completion or a controlled
runtime-only comparison with llama.cpp.

The earlier prepack-baseline native BF16-profile and INT8 deployment scores
both pass the predeclared **0.02 nats/token aggregate** quality screen.
That result must not be presented as final qualification of later arithmetic
changes: a selected-build quality/equivalence check remains pending.
No full 262,144-token input pass is claimed.
The initial NVLink pair matrix misses **all three** historical dual-card
targets, and its default multi-device graph decode is slower than single-card
decode. Functional two-device execution is not yet a demonstrated accelerated
default configuration.
The updated HTTP C1/2/4/8 wave completes all 15 requests with a real
262,144-token shared KV pool. It removes the earlier C2 regression and reaches
about **2.04× C1** at both C4 and C8; C4→C8 is essentially flat, not linear
scaling. Both before/after waves remain documented separately.
The independent Q4_K_M **BF16-K/FP16-V matched-storage reference** has completed
on GPU3; native comparisons ran subsequently on GPU1. NInfer's `bf16` profile
stores BF16 K / FP16 V, which is the
adapter's `bf16-fp16` profile. An earlier both-BF16 reference is retained only
as a separate diagnostic.

## Explicit local artifacts

The existing groupwise v3 artifact is reusable without conversion or runtime
weight repacking:

```text
/home/algore/models/qwen3.8-27b-ninfer/qwen3_8_27b.ninfer
bytes: 20437521664
sha256: 81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da
architecture: Qwen3_5ForCausalLM
artifact ID: ba82761a4e9040968fe462707a8fdd19
components: text, vision, mtp, dflash2
```

The current v3 reader accepts its framing and directory. Inventory: 183
`q4_g64_fp16`, 246 `q5_g64_fp16`, one `q6_g64_fp16`, 30 `q8_g32_fp16`,
627 BF16, 96 FP32, and one INT32 tensor. There are 460
`row_split_k128_v1` tensors and 724 contiguous tensors, plus six resources.
No stored NVFP4 or FP8 weight requires native Blackwell instructions. BF16
storage still requires the Turing execution adaptations; directory acceptance
alone does not qualify a CUDA route. One artifact serves both single-GPU and
tensor-parallel placement.

Independent quantized reference:

```text
/home/algore/models/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf
bytes: 17106773984
sha256: 7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b
llama.cpp source: /home/algore/llama.cpp-dspark-current
llama.cpp build: /home/algore/llama.cpp-dspark-current/build-peer2048
```

Both hashes were independently recomputed locally. No checkpoint was downloaded,
modified, or regenerated; 678 GiB was free when checked. TokenFurnace was read
only. Inspection JSON and hash records are under ignored
`profiles/bench/rtx8000-artifact/`.

The historical `/home/algore/models/qwen3.8-27b-ninfer-v2/qwen3_8_27b.ninfer`
is **not** the selected product artifact. Its 18,210,531,328-byte v2 framing is
not accepted by this product's v3 reader.

## Native reference targets

Authority: the user's local
`/home/algore/GIT/agorevski/TokenFurnace/targets/qwen3.8-27b/PERFORMANCE.md`,
“Matched four-layout topology sweep (2026-09-23 UTC)”. Values below are
**previously measured reference results**, not measurements of this checkout.
They used llama.cpp `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3` (build 359),
the GGUF above, FP16 KV, flash attention, batch 8192, ubatch 2048,
`GGML_CUDA_P2P=1`, stock launch queues, and five retained repetitions.

| Physical placement | pp512 mean ± SD | pp4096 mean ± SD | tg128 mean ± SD |
|---|---:|---:|---:|
| GPU 0, layer | 719.07 ± 4.02 | 681.79 ± 21.11 | 28.58 ± 0.07 |
| NVLink 0–1, tensor | 1182.40 ± 16.67 | 1139.27 ± 7.27 | 44.96 ± 0.22 |
| NVLink 2–3, tensor | 1245.92 ± 6.36 | 1122.99 ± 39.89 | 40.60 ± 1.94 |

Units: tokens/second. The conservative best same-count targets are single
**719.07 / 681.79 / 28.58**, dual **1245.92 / 1139.27 / 44.96**.
The dual maxima come from different physical pairs; they are not one
jointly observed configuration. Reference temperatures reached 85–87 °C,
so sequential thermal differences limit pair attribution.

The historical v2 NInfer MTP-0 result was 84.47 / 79.29 / 18.34 tok/s.
MTP-3 had only 26.5% acceptance on its native corpus and slower native
decode, despite better short-prompt HTTP throughput. Neither API throughput
nor favorable speculative acceptance substitutes for the targets above.

## Current best complete single-GPU matrix: scaled activation preparation

The later UP-scale activation-preparation and batch implementation was measured
on physical GPU0 with the **same** official artifact, fixed token-ID corpus,
INT8 KV, context/capacity8192, chunk2048, CUDA Graph decode, no speculation,
one warmup and **five retained repetitions** as the earlier complete single-GPU
baseline. Means and sample standard deviations reproduce from the retained
public-Engine phase timings:

| Workload | Scaled/batch mean ± SD (tok/s) | Earlier full baseline | Ratio to own baseline | Historical target | Difference from historical |
|---|---:|---:|---:|---:|---:|
| pp512 | **454.916639 ± 4.672412** | 199.346653 | **2.282×** | 719.07 | **−36.74%** |
| pp4096 | **421.579940 ± 9.156379** | 175.159507 | **2.407×** | 681.79 | **−38.17%** |
| tg128 | **29.448874 ± 0.049608** | 29.077902 | 1.013× | 28.58 | +3.04% |

This is a substantial measured improvement to this implementation's whole
prefill route, **not completion of the requested prefill targets**. The two
historical comparisons still differ in weight representation, KV format and
token content. The own-baseline comparison holds those workload choices fixed,
but combines activation preparation and batch changes; it does not isolate
one kernel's contribution.

The 90 sampled GPU states show 34 °C initially, 88 °C maximum and 85 °C on
completion; peak memory 17,229 MiB and power 260.61 W. Final clocks were
1770 MHz SM/7000 MHz memory. No competing GPU0 compute process was detected.
As in the earlier full matrix, decode follows both prefill shapes; do not
average its 29.45 tok/s with the earlier cooler decode-only 30.19 tok/s result.

Retained reports: `profiles/bench/rtx8000-single-scaled/summary.csv`,
`summary.json`, `manifest.json`, `commands.sh`,
`json/native/pp512_pp4096_tg128.json` and
`logs/native.pp512_pp4096_tg128.stdout.telemetry.json`.
The manifest records the unchanged corpus hash and exact binary timestamp.
All earlier baselines below remain available for before/after interpretation.
Final numerical qualification of the selected arithmetic implementation
remains pending; the earlier prepack-baseline quality pass is not silently
transferred to these changes.

## Initial measured single-GPU decode

Measured on **2026-09-26 UTC**, physical GPU0, through the public Engine using
the official groupwise v3 artifact above. The binary used the coordinated
CUDA12.9/GCC13 SM75 build; the native report records CUDA runtime12.9 and
driver API13.0. Decode used INT8 group-64 KV, context/capacity8192,
prefill chunk1024, CUDA Graphs, no MTP, one discarded warmup and **five retained
repetitions**. Ordinary public generation primed the decode graph beforehand.

| Workload | NInfer mean ± sample SD | Historical single-card target | Rate difference |
|---|---:|---:|---:|
| tg128 | **30.190640 ± 0.190507 tok/s** | 28.58 ± 0.07 tok/s | **+5.64%** |

The difference is `(30.19063976 / 28.58 - 1) * 100`. It is a same-count
historical comparison, **not isolation of the runtime**: NInfer groupwise
weights differ from Q4_K_M, INT8 KV differs from FP16, and the committed
NInfer token-ID corpus differs from llama-bench's internal token sequence.
The separately measured quality screen below passes, but does not make this
historical timing comparison a controlled runtime-only experiment.

Exact retained command:

```bash
build-sm75/bench/ninfer_bench \
  --weights /home/algore/models/qwen3.8-27b-ninfer/qwen3_8_27b.ninfer \
  -n 128 -r 5 --warmup 1 --max-ctx 8192 --prefill-chunk 1024 \
  --kv-dtype int8 --device 0 --output json \
  --output-file profiles/bench/rtx8000-single-decode-initial.json
```

Each repetition generated129 output tokens: the first belongs to the untimed
seed/prefill phase and exactly128 tokens to the reported decode phase. Prefix
reuse and model-default stops were disabled. No competing GPU0 compute process
was present before or after the run. Before/after samples show temperature
35→78 °C, SM clock300→1860 MHz, memory clock405→7000 MHz, and power22.54→151.65 W.
These are endpoint samples, not measured in-run minima/maxima; both memory
samples were7MiB outside the process lifetime. Native allocation records report
17,093,490,688 bytes of selected weights,443,263,232 bytes of sequence storage,
and159,981,568 bytes of workspace.

Raw per-repetition timings, full configuration and allocation records are in
`profiles/bench/rtx8000-single-decode-initial.json`; endpoint telemetry is in
`rtx8000-single-decode-before.csv` and `rtx8000-single-decode-after.csv` in the
same directory.

This initial binary predates the latest GDN-prefill and tensor-parallel binding
fixes; the measured single-token decode route is unchanged. `pp512` was blocked
by the GDN cooperative-prefill issue, so no prefill result accompanies this
decode-only record. The subsequently relinked full baseline is recorded next.
Cold generation smoke timings are not throughput evidence, and this one passed
decode target does not establish completion of prefill, dual-GPU, long-context,
or concurrency goals. Model quality has separate measured evidence below.

## Full single-GPU prepack baseline

The corrected, relinked public-Engine binary completed all three native shapes
on physical GPU0 with the official groupwise artifact, INT8 KV, context/capacity
8192, CUDA Graph decode, no MTP, one warmup and **five retained repetitions per
shape**. Unlike the initial decode-only run, the `rtx8000` matrix preset used
**prefill chunk 2048**, then ran pp512, pp4096 and tg128 in that order.

| Workload | NInfer mean ± sample SD (tok/s) | Historical single-card target | Difference | Outcome |
|---|---:|---:|---:|---|
| pp512 | **199.346653 ± 3.006947** | 719.07 ± 4.02 | **−72.28%** | Far below target |
| pp4096 | **175.159507 ± 3.339213** | 681.79 ± 21.11 | **−74.31%** | Far below target |
| tg128 | **29.077902 ± 0.145896** | 28.58 ± 0.07 | **+1.74%** | Above historical rate |

Prefill reaches only **27.72%** and **25.69%** of the selected historical
single-GPU targets. This is a valid baseline, not a successful completion of
the performance goal. Whole-prefill profiling and the selected activation
preparation optimization remain integration work; isolated kernel timings do
not explain or replace these public-Engine results.

The preset's chunk 2048 differs from the initial native run's chunk 1024 and
the historical v2 NInfer reference's chunk 1024. The faster historical
llama.cpp target used batch 8192/ubatch 2048, which is not an equivalent
NInfer scheduling contract. Weight formats, KV storage, token content and
execution boundaries also differ as disclosed above; this is not a
controlled runtime-only comparison.

The later tg128 rate is **29.08**, not the earlier **30.19**. Do not average
them: the latter full-matrix decode followed six pp512 and six pp4096
generations including warmups. Its 169 GPU samples show 34 °C initially,
89 °C maximum and 85 °C at completion, versus 35→78 °C endpoint samples for
the decode-only run. Thermal history plausibly contributes to the difference,
but timing order, chunk configuration and binary revision are also different;
temperature alone has not been isolated. Full-run peak sampled memory was
17,229 MiB and power 261.02 W. Final clocks were 1770 MHz SM/7000 MHz memory.
The strict selected-GPU contention guard completed without detecting another
GPU0 compute workload.

Raw machine-readable evidence is under
`profiles/bench/rtx8000-single-prepack-baseline/`: `summary.csv`, `summary.json`,
the exact `manifest.json`/`commands.sh`, native
`json/native/pp512_pp4096_tg128.json`, topology/NVLink records, and
`logs/native.pp512_pp4096_tg128.stdout.telemetry.json`.
The full manifest records CUDA visibility 0, chunk 2048 and the corpus SHA256;
the reported means/SDs reproduce from all five raw phase timings.

This full baseline precedes the upcoming private activation-prepack change.
The separate native quality pass applies to that prepack-baseline build;
any later performance improvement requires new whole-inference measurements
and appropriate numerical-equivalence evidence.

## Initial NVLink pair baseline and graph regression

Measured on physical GPUs **2–3**, `CUDA_VISIBLE_DEVICES=2,3`,
`--device 0 --tensor-parallel-device 1`, using the same artifact and corpus.
The full pair matrix uses INT8 KV, context/capacity8192, prefill chunk2048,
CUDA Graph decode, no speculation, one warmup and **five retained repetitions**
for each shape. The primary retains the full selected weights and owns
attention/GDN/KV/state; peer FFN work therefore is not equivalent to the
historical llama.cpp tensor decomposition.

| Workload | Pair mean ± sample SD (tok/s) | Best historical same-count dual target | Difference |
|---|---:|---:|---:|
| pp512 | **241.954232 ± 13.166829** | 1245.92 | **−80.58%** |
| pp4096 | **178.778675 ± 1.537431** | 1139.27 | **−84.31%** |
| tg128, graph | **23.414042 ± 0.054773** | 44.96 | **−47.92%** |

**All targets are missed.** Pair prefill is only modestly above the single-card
prepack baseline, while default graph decode is below both the initial 30.19
single-card decode and the later 29.08 full-matrix decode. These results must not
be described as an accelerated default configuration.

Bounded follow-up decode diagnostics kept artifact, corpus, INT8 KV,
context8192, chunk2048 and tg128 fixed:

| Placement / decode mode | Retained repetitions | Mean ± sample SD (tok/s) | Mean phase ms/token |
|---|---:|---:|---:|
| Pair2–3, graph (matrix above) | 5 | 23.414042 ± 0.054773 | 42.7096 |
| Pair2–3, eager | 3 | 34.346340 ± 0.532933 | 29.1198 |
| GPU2 only, graph | 3 | 27.817099 ± 1.942443 | 36.0628 |

Each diagnostic used one warmup. Mean phase milliseconds/token is
`mean(decode_seconds) / 128 * 1000`, not the reciprocal of the mean token rate.
The observed pair graph phase costs **13.5898 ms/token more than pair eager**.
Eager pair execution is faster than the sampled same-primary single-card
graph run, but still misses 44.96 tok/s and is a three-repeat diagnostic rather
than a promoted five-repeat configuration. Runs were sequential, not a
thermal-matched randomized comparison. The evidence motivates graph/peer
transfer profiling; it does not establish that all 13.5898 ms is communication,
nor justify silently changing the product's graph policy.

The full pair matrix's sampled temperatures were GPU2: 58→84 °C, max 87 °C;
GPU3: 60→74 °C, max 86 °C. Peak sampled device memory was 17,247 MiB primary
and 5167 MiB peer, with peak power 261.04 W and 262.86 W respectively. The
strict selected-GPU contention guard completed without a competing compute
workload. Native memory reports separate 17,093,490,688 bytes of primary
weights from 4,902,092,800 bytes of peer weights and 245,366,784 bytes of peer
workspace; peer storage must not be omitted from total-residency discussion.

All raw results remain under
`profiles/bench/rtx8000-pair-prepack-baseline/`, including
`json/native/pp512_pp4096_tg128.json`,
`json/native/tg128-eager-diagnostic.json`, and
`json/native/tg128-single-gpu2-diagnostic.json`. The full matrix manifest,
commands, topology and per-GPU telemetry preserve placement and conditions.
The runtime's successful functional/numerical two-device tests are separate
evidence; they do not convert these failed performance targets into a speedup.
Final quality and performance evidence for activation-prepack/range fixes
remains pending.

## Current HTTP concurrency: batch improvement, C4→C8 plateau

The later UP-scale plus selected-batch implementation repeated the existing
HTTP decode-saturation wave on physical GPU1. Workload settings were unchanged:
official groupwise artifact, INT8 KV, CUDA Graphs, no speculation/prefix reuse,
greedy sampling, chunk 1024, logical context 262144 and a real shared
262,144-token KV pool. All **15 requests** again completed 256 tokens with
finish reason `length`; complete steady intervals observed every configured
batch column, including all eight at C8.

| Startup C | Updated aggregate steady tok/s | Earlier wave tok/s | Change vs earlier | Ratio to updated C1 | Full-batch steady seconds |
|---|---:|---:|---:|---:|---:|
| 1 | **30.624749** | 30.624986 | effectively unchanged | 1.000× | 8 |
| 2 | **52.444442** | 18.074078 | **+190.16%** | 1.712× | 9 |
| 4 | **62.399990** | 33.103452 | **+88.50%** | 2.038× | 15 |
| 8 | **62.451614** | 47.609761 | **+31.17%** | 2.039× | 31 |

The C2 regression is removed in this observed wave. However, doubling
concurrency from C4 to C8 increases aggregate steady throughput by only
**0.083%**: effectively a plateau, not evidence of optimal or linear C8
scaling. HTTP wave makespans were 10.149/13.467/24.014/48.076 s for C1/2/4/8;
the larger wave contains more requests, so makespan alone is not a
per-request latency comparison.

As before, these are **one-wave HTTP steady-interval measurements**, not
five-repeat native tg128 rates. No repeatability standard deviation or
isolated attribution to one kernel is claimed. They demonstrate the current
configuration's functional completion and observed batch-rate improvement,
not a full 262K input pass or final selected-build quality qualification.

Exact requests, counters, capacity/environment records and commands are
retained under `profiles/bench/rtx8000-single-concurrency-scaled-batched/`,
particularly `summary.json`, `summary.md`, `points/` and `server/`.
The earlier measurements below remain the explicit before-change record.

## Earlier HTTP concurrency: functional pass, uneven scaling

The existing `run_serve_concurrency.py` runner completed one decode-saturation
wave at each startup concurrency **1, 2, 4, 8** on physical GPU1. It used the
current activation-prepack binary before the subsequent range-shift fix,
official groupwise weights, INT8 KV, CUDA Graph decode, no MTP, greedy
sampling, no prefix reuse and prefill chunk 1024. The report confirms both
logical context 262144 and an **actually allocated shared KV capacity of
262,144 tokens** at every point.

All **15 HTTP requests** completed exactly 256 output tokens with finish
reason `length`, using 335-token prompts. Every retained steady interval had
a full decode batch equal to the configured concurrency:

| Startup concurrency | Requests | Complete steady intervals / seconds | Aggregate steady decode tok/s | Ratio to C1 | HTTP wave makespan |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 8 / 8.00 | **30.624986** | 1.000× | 10.213 s |
| 2 | 2 | 27 / 27.00 | **18.074078** | **0.590×** | 31.984 s |
| 4 | 4 | 29 / 29.00 | **33.103452** | 1.081× | 38.943 s |
| 8 | 8 | 41 / 41.00 | **47.609761** | **1.555×** | 59.226 s |

The source JSON—not an earlier approximate handoff—defines these exact rates.
They count committed decode tokens only during complete full-batch intervals
of the HTTP wave. The first output token belongs to prefill, leaving 255 decode
tokens per completed request; the selected intervals need not include every
decode token. These are **aggregate** rates, not per-request throughput.
Wave makespan spans concurrent client release through the last full response.

**C2 is slower in aggregate than C1**, C4 gains only 8.1%, and C8 gains 55.5%
rather than approaching linear scaling. Successful admission and completion
do not demonstrate efficient batching. There is one wave per configuration,
not five retained native tg128 repetitions; no repeatability standard
deviation or historical native speedup is claimed. Startup max concurrency
was changed by restarting the service for each point, and only the selected
C1/2/4/8 capacities are measured here.

The 262K allocation plus short requests establishes pool allocation and
request handling, **not a 262K input pass** or long-context quality. This
HTTP wave does not replace the required native 32K-prefill/1K-decode workload.
It also does not retroactively qualify later arithmetic changes.

Raw request outcomes, complete server commands, environment/memory records,
steady-interval counters and individual HTTP timing are retained in
`profiles/bench/rtx8000-single-concurrency-prepack/summary.json`,
`summary.md`, `points/` and `server/`. The structured report identifies
CUDA12.9/runtime12.9/driver13.0 and the physical GPU1 UUID. These rates are
kept separate from native single/dual 128-token benchmarks and from the
independent quality evaluations.

## Reproducible throughput apparatus

Select Python 3.11 explicitly on this host:

```bash
PY=/home/algore/miniconda3/envs/vllm/bin/python
MODEL=/home/algore/models/qwen3.8-27b-ninfer/qwen3_8_27b.ninfer
```

The existing matrix runner now has three non-speculative presets:

| Preset | Work | Warmups / retained repetitions |
|---|---|---|
| `rtx8000` | pp512, pp4096, tg128, context 8192 | 1 / 5 |
| `rtx8000-long` | pp32768+tg1024, context allocation 262144 | 1 / 5 |
| `rtx8000-capacity` | pp16+tg8, context allocation 262144 | 0 / 1 |

All use prefill chunks of 2048. The capacity case establishes only allocation
and short generation, never a full-length input pass.

After the selected GPUs are released by other workers:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m tools.bench.run_ninfer_bench_matrix \
  --preset rtx8000 --bench build-sm75/bench/ninfer_bench \
  --build-dir build-sm75 --no-build --weights "$MODEL" --kv-dtype int8 \
  --device 0 --telemetry-gpus 0 \
  --output-dir profiles/bench/rtx8000-single-native

CUDA_VISIBLE_DEVICES=2,3 "$PY" -m tools.bench.run_ninfer_bench_matrix \
  --preset rtx8000 --bench build-sm75/bench/ninfer_bench \
  --build-dir build-sm75 --no-build --weights "$MODEL" --kv-dtype int8 \
  --device 0 --tensor-parallel-device 1 --telemetry-gpus 2,3 \
  --output-dir profiles/bench/rtx8000-pair23-native
```

Select `rtx8000-long` or `rtx8000-capacity` with a distinct output directory
for the corresponding run. `--dry-run` records commands without building,
querying the GPU, or executing a model.

Schema-v16 native JSON retains each repetition, mean/standard deviation, and
single/dual CUDA device ordinals. Peer weight/workspace arenas and each test's
peer workspace peak are reported separately; they must not be mistaken for
primary-device usage or omitted when describing total residency. The
runner records exact commands, corpus hash, artifact path/size, CUDA visibility,
topology/NVLink status, and one-second samples of clocks, temperature, power,
memory, utilization and compute processes. `--telemetry-gpus` takes **physical**
IDs; Engine flags take CUDA-visible ordinals. Competing compute on a selected
GPU causes a visible failure. Other GPUs may be busy but can still affect host
power/thermal conditions. Failed reruns cannot retain stale native reports.

**Comparison limits:** NInfer uses the committed fixed token-ID corpus,
not llama-bench's internally generated tokens. Counts and phase shapes match,
not token content. Native tg128 starts after a one-token seed and NInfer's
prefill/begin token; pp requests one output token. Graph priming is outside
retained repetitions. Prefix reuse and model stops are disabled. NInfer INT8
group-64 KV differs from the historical FP16 reference. NInfer chunk2048 is
not llama batch8192, although it matches ubatch2048. Preserve these differences
in any reported speed comparison; rerun both backends on exact shared IDs
before calling a comparison token-matched.

For startup-fixed concurrency acceptance, reuse the existing serving harness
rather than interpreting single-request native rates as concurrent throughput:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m tools.bench.run_serve_concurrency \
  --serve build-sm75/apps/ninfer-serve --artifact qwen38="$MODEL" \
  --mode mtp0 --sampling greedy --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 256 --max-context 8192 --kv-capacity 65536 \
  --output profiles/bench/rtx8000-concurrency
```

For a pair, make both GPUs visible and add `--tensor-parallel-device 1`.
The harness preserves the existing 1–8 validation and uses INT8 KV; passing a
dry run does not prove GPU scheduling acceptance. Reserve the selected GPUs
and retain external telemetry during serving campaigns as well.
Add `--concurrency 3 --concurrency 5 --concurrency 6 --concurrency 7` when
validating every supported startup capacity rather than the C1/2/4/8 performance
sample.

For the exact native long workload, reuse the matrix:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m tools.bench.run_ninfer_bench_matrix \
  --preset rtx8000-long --bench build-sm75/bench/ninfer_bench \
  --build-dir build-sm75 --no-build --weights "$MODEL" --kv-dtype int8 \
  --device 0 --telemetry-gpus 0 \
  --output-dir profiles/bench/rtx8000-single-long
```

For near-capacity **HTTP** evidence, reuse the existing low-level TTFT runner
against a separately started foreground server. Do **not** use
`run_serve_ttft_campaign.py`'s standard artifact/profile: that controller fixes
NVFP4 weights and FP8 KV. Record the actual groupwise/INT8 SM75 server command,
since the low-level runner's profile label verifies the request graph, not the
server's weight or KV precision:

```bash
# First terminal; wait for the endpoint to become healthy before the client.
CUDA_VISIBLE_DEVICES=0 build-sm75/apps/ninfer-serve "$MODEL" \
  --host 127.0.0.1 --port 18080 --kv-dtype int8 --device 0 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 1 \
  --no-prefix-reuse --no-thinking --greedy

# Second terminal, after readiness:
"$PY" -m tools.bench.run_serve_ttft \
  --base-url http://127.0.0.1:18080 --case cold-long-256k \
  --profile-label text-cold-256k --timeout-seconds 7200 \
  --output profiles/bench/rtx8000-http-long/ttft.json
```

This existing fixture has 260,096 prompt tokens and a 32-token output limit
under its frozen tokenizer/template facts. Require its reported actual token
counts and constructed-case checks to pass for the selected artifact; a
template mismatch is a visible failure, not a silently relabeled input.
It is a near-capacity input test, not the native 32K/1K workload or an exact
262,144-token prompt pass. `cold-long-64k` with `text-cold-64k` is the analogous
smaller HTTP case; configure the server context/capacity to 65,536 for it.

## Independent quality comparison

`ninfer_score` is a benchmark-only export of public Engine `CausalScoring`,
not a Python inference route or a replacement runtime. It reuses the product
perplexity corpus loader and canonical window planner. It exports exact token
IDs, original UTF-8 streams, window bounds and every target log probability.
No template, BOS, cross-stream history or generated-answer heuristic is used.

`llama_score` is an **external** reference adapter, not linked into NInfer.
It opens the Q4_K_M GGUF with the already-built llama.cpp, independently
tokenizes each stream, rejects differing token IDs/counts, resets all KV and
recurrent state for every window, and scores the exact same targets.
Log-softmax uses FP64 accumulation over the complete FP32 logits vector.
Reference execution uses one GPU, BF16 K / FP16 V by default, flash attention,
and up to 1024 context-only tokens per batch. This matches the physical storage
of NInfer's `bf16` profile, whose BF16 V is converted once to FP16 on append
([KV contract](../maintainer/paged-kv-cache.md)). Nominal dtype labels alone
must not be treated as evidence of matching physical K and V representations.
Scored batches are bounded to a **128 MiB FP32 logits tile**: 128 rows
(121.25 MiB) for this 248,320-token vocabulary. The native score tile remains
1024 rows; this execution-shape difference is recorded, not concealed.
The limit is for each logits tile, not total model/workspace/host memory.
`--kv-dtype bf16-fp16` explicitly selects that matched profile;
`--kv-dtype bf16` means both reference K and V are BF16, while
`--kv-dtype fp16` selects both FP16 (the historical performance profile).

Builds (CPU compilation; the reference build tree is not modified):

```bash
cmake --build build-sm75 -j --target ninfer_score
"$PY" -m tools.bench.build_llama_score \
  --source-dir /home/algore/llama.cpp-dspark-current \
  --build-dir /home/algore/llama.cpp-dspark-current/build-peer2048 \
  --output profiles/bench/rtx8000-artifact/llama_score
```

The reference helper requires that existing Makefiles build's
`llama-bench.dir/link.txt`; it reuses the recorded compiler, libraries and
runtime search paths rather than silently building another llama revision.
An existing output is rejected. The helper's CPU compile and `--help` have
passed on this host.

The independent reference can run first if main releases a GPU before native
Engine qualification completes. A CPU-only plan reads the artifact's embedded
tokenizer, disables implicit padding/truncation/special-token insertion, and
preserves original text bytes. It performs **no model inference**:

```bash
"$PY" -m tools.bench.prepare_score_plan \
  --weights "$MODEL" --corpus eval/corpora/perplexity-1m/manifest.json --quick \
  --context 4096 --stride 2048 --max-tokens 8192 \
  --output profiles/bench/rtx8000-quality/cpu-plan.json
profiles/bench/rtx8000-artifact/llama_score --check-plan \
  profiles/bench/rtx8000-quality/cpu-plan.json
```

The plan above has been generated locally on CPU (four streams, 32,764
targets). Once main explicitly releases a GPU, pass `cpu-plan.json` in place
of `ninfer.json` to the reference command below. The reference independently
verifies its GGUF tokenization against that plan, and the later native export
must match the resulting reference IDs, text, counts and windows exactly.

Run the following **sequentially on a released GPU**, retaining stderr logs:

```bash
mkdir -p profiles/bench/rtx8000-quality
CUDA_VISIBLE_DEVICES=0 build-sm75/bench/ninfer_score \
  --weights "$MODEL" --corpus eval/corpora/perplexity-1m/manifest.json --quick \
  --context 4096 --stride 2048 --max-tokens 8192 --kv-dtype bf16 \
  --output profiles/bench/rtx8000-quality/ninfer.json \
  2>profiles/bench/rtx8000-quality/ninfer.stderr.log

CUDA_VISIBLE_DEVICES=0 profiles/bench/rtx8000-artifact/llama_score \
  /home/algore/models/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf \
  profiles/bench/rtx8000-quality/ninfer.json \
  profiles/bench/rtx8000-quality/llama.json --kv-dtype bf16-fp16 \
  2>profiles/bench/rtx8000-quality/llama.stderr.log

"$PY" -m tools.bench.compare_causal_scores \
  --candidate profiles/bench/rtx8000-quality/ninfer.json \
  --reference profiles/bench/rtx8000-quality/llama.json \
  --output profiles/bench/rtx8000-quality/comparison.json
```

Create the log directory before shell redirection. The default pilot scores
8191 targets across three windows in each of four fixed domains: English
reference, long-form English, Chinese reference and C++/CUDA source
(32,764 targets, twelve paired windows total).
The corpus is `ninfer-ppl-1m-v1`, not a claim to have evaluated a million
tokens. `--max-tokens 0` removes the per-stream prefix limit; omit `--quick`
to include all 16 streams. Larger coverage and overlapping windows should be
run after the pilot, before broad quality claims.

Predeclared screening tolerance: candidate minus reference token-weighted
mean NLL ≤ **0.02 nats/token** overall (PPL ratio ≤ exp(0.02) ≈ 1.0202),
and at least **16,000** targets. Every domain is reported; a separate
`--max-domain-nll-delta` guard is optional and must be selected before evaluation.
Exact IDs, text and windows must match; missing/nonfinite scores fail.
Boolean/null “scores,” noninteger bounds/counts, and a truncated prefix that
does not match its declared limit also fail. The reference validates these
invariants before GPU initialization; the native exporter validates before
publishing its report.
Per-token absolute-error median/p95/p99/max are diagnostics, not equality
thresholds between different quantizers. Schema-v2 comparisons also retain each
paired window's candidate/reference mean NLL and delta, weighted window
dispersion, leave-one-window-out range, and a deterministic 2,000-resample
within-domain paired-window bootstrap interval (seed zero). If any domain has
fewer than two windows, its bootstrap interval is unavailable rather than a
fabricated zero-width confidence interval. Overlapping context and common
streams correlate windows: these are descriptive sensitivity summaries, not
independent-sample statistical guarantees. The criterion is an engineering
non-inferiority screen, not proof of equal general quality or an independent
mathematical oracle for individual CUDA Ops.

The default comparison matches BF16 K / FP16 V storage, but the engines and weight
quantizers do not have identical internal rounding boundaries; this is model
quality comparison, not kernel equality. If the selected reference build cannot
execute that profile, retain the failure before separately selecting another.
Schema-v2 score exports declare physical `kv_key_dtype` and `kv_value_dtype`;
the comparator validates them against each backend's actual profile instead
of equating the two backends' ambiguous `bf16` strings.
Comparing differing physical profiles requires
`--allow-kv-mismatch`, which is explicitly recorded. INT8 deployment quality
requires a separate candidate export with `--kv-dtype int8` against the same
reference and the same acknowledgement. Do not present a BF16 score as
validation of INT8 KV, or a quantized-model comparison as isolation of kernel
rounding error.

### Measured matched-storage reference

Completed on **2026-09-26 UTC**, physical GPU3
(`CUDA_VISIBLE_DEVICES=3`, reference logical device0), with the same fixed
streams, tokenizer IDs, context4096/stride2048 and 8192-token stream prefixes.
The runtime log confirms **K BF16 / V FP16**, matching native `--kv-dtype bf16`
physical storage. All **32,764 target log probabilities** are finite and all
**12 exact windows** pass the protocol validator.

| Domain | Scored targets | Mean NLL (nats/token) | Perplexity |
|---|---:|---:|---:|
| English reference | 8,191 | 1.864506 | 6.452750 |
| English long-form | 8,191 | 1.836116 | 6.272132 |
| Chinese reference | 8,191 | 2.097598 | 8.146577 |
| C++/CUDA source | 8,191 | 0.548226 | 1.730181 |
| **Token-weighted overall** | **32,764** | **1.586612** | **4.887161** |

Use this persisted reference for subsequent native BF16 and INT8 comparisons:

```text
profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.json
profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.summary.json
profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.run.json
profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.telemetry.jsonl
profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.stderr.log
```

The 1686.197 s wall time is **not performance-qualified**. This existing llama
build has `GGML_CUDA_FA_ALL_QUANTS=OFF`; heterogeneous K/V flash attention falls
back to CPU, with 34 graph partitions rather than the two observed for
both-BF16. That mixed CPU/GPU execution is acceptable for the independent
numerical reference, not for a GPU throughput comparison. No competing compute
was observed. GPU samples reached 74 °C, 264.56 W and 16,259 MiB. The process
exited successfully, and GPU3 was returned to its owner before CPU report
summarization.

Storage matching does not make the engines' full internal cast/quantization
boundaries identical. In particular, native V is BF16 before the FP16 storage
cast; the reference has its own upstream arithmetic. The full quality
comparison measures the resulting model distributions, not bitwise kernel
equivalence.

### Measured native quality: BF16 profile and INT8 deployment

Both public-Engine `CausalScoring` exports completed successfully on physical
GPU1 on **2026-09-26 UTC**, using the freshly linked all-fixes binary confirmed
at 19:50 local time. This is the **pre-activation-prepack baseline**. They use
exactly the reference's four streams, 32,764 targets, 12 windows,
context4096/stride2048 and 8192-token stream prefixes. The comparator verified
identical UTF-8 text, token IDs, full token counts, target bounds and score
counts; all log probabilities are finite.

| Candidate | Physical K / V | Mean NLL | Delta vs Q4_K_M | Perplexity | PPL ratio | Aggregate screen |
|---|---|---:|---:|---:|---:|---|
| Native `bf16` | BF16 / FP16 | 1.600440 | **+0.013828** | 4.955212 | 1.013924 | **Pass** |
| Native `int8` | INT8-G64 / INT8-G64 | 1.600436 | **+0.013825** | 4.955193 | 1.013921 | **Pass** |
| Independent Q4_K_M | BF16 / FP16 | 1.586612 | reference | 4.887161 | 1.000000 | reference |

Both aggregate regressions are below the predeclared 0.02 nats/token bound
(approximately **1.392%** higher PPL, versus the allowed 2.020%). The BF16
profile matches reference K/V storage formats. The INT8 comparison deliberately
changes both key and value representation; `--allow-kv-mismatch` is explicitly
recorded, and this result assesses the deployment representation rather than
isolating weight quantization.

Every domain is reported; the aggregate pass must not hide the English-reference
domain's larger regression:

| Domain | Reference NLL | BF16 NLL | BF16 delta | INT8 NLL | INT8 delta |
|---|---:|---:|---:|---:|---:|
| English reference | 1.864506 | 1.887699 | **+0.023193** | 1.887321 | **+0.022815** |
| English long-form | 1.836116 | 1.848534 | +0.012418 | 1.848062 | +0.011946 |
| Chinese reference | 2.097598 | 2.109026 | +0.011428 | 2.108978 | +0.011380 |
| C++/CUDA source | 0.548226 | 0.556500 | +0.008274 | 0.557383 | +0.009157 |

No separate domain threshold was predeclared. Each domain has 8191 targets and
three paired windows. The deterministic within-domain paired-window bootstrap
95% intervals for the aggregate NLL delta are **[0.010488,0.016600]** (BF16)
and **[0.010379,0.016556]** (INT8); leave-one-window-out aggregate ranges are
[0.012222,0.014865] and [0.012215,0.014837], respectively. These descriptive
intervals do not assume away shared-stream/overlapping-context correlation and
are not a guarantee of performance on unseen domains. All individual window
deltas and per-domain sensitivity intervals are retained in the JSON reports.
Per-token absolute NLL-delta p95 is 0.598367 (BF16) and 0.597307 (INT8);
maximum is 5.284263 and 5.307039, respectively. Different quantizers need not
agree tokenwise even when aggregate quality passes.

Durable evidence:

```text
profiles/bench/rtx8000-quality/native-bf16-gpu1.json
profiles/bench/rtx8000-quality/native-bf16-gpu1.comparison.json
profiles/bench/rtx8000-quality/native-int8-gpu1.json
profiles/bench/rtx8000-quality/native-int8-gpu1.comparison.json
profiles/bench/rtx8000-quality/native-gpu1-quality-results.json
```

Each candidate also has a `.run.json` exact-command/binary-mtime record,
`.stderr.log`, and `.stdout.telemetry.json` GPU samples. No competing GPU1
compute was observed. BF16 reached 84 °C/17,393 MiB/274.8 W; INT8 reached
85 °C/17,269 MiB/269.66 W. Load/scoring/CPU comparison wall times were 258.005 s
and 253.456 s, **not throughput-qualified measurements**. GPU1 was released
immediately after both runs; no benchmark rate is derived from those times.

Reproduce the retained comparisons without GPU work:

```bash
"$PY" -m tools.bench.compare_causal_scores \
  --candidate profiles/bench/rtx8000-quality/native-bf16-gpu1.json \
  --reference profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.json \
  --output profiles/bench/rtx8000-quality/native-bf16-gpu1.comparison.json
"$PY" -m tools.bench.compare_causal_scores \
  --candidate profiles/bench/rtx8000-quality/native-int8-gpu1.json \
  --reference profiles/bench/rtx8000-quality/reference-bf16-fp16-gpu3.json \
  --allow-kv-mismatch \
  --output profiles/bench/rtx8000-quality/native-int8-gpu1.comparison.json
```

This validates the measured single-GPU candidate build and deployment profile
at the stated corpus/window scope, not full 262K quality, task accuracy, or
all future binary changes. Representation-preserving activation-preparation
changes can use bounded same-input native scoring equivalence against this
baseline; the expensive independent reference need not be regenerated.

### Earlier both-BF16 diagnostic

Completed on **2026-09-26 UTC**, physical GPU0, using the explicitly identified
Q4_K_M GGUF and llama.cpp revision above. The reference independently verified
all four source streams' tokenizer IDs/counts against the CPU plan. All
**32,764 target log probabilities** are finite and the **12 windows** pass the
strict protocol validator: context4096, stride2048, 8192-token prefix per
stream, **BF16 K / BF16 V**, no added special tokens, FP64 log-softmax accumulation.
This valid reference-only result is **not storage-matched** to NInfer's
BF16-K/FP16-V profile; use the completed matched reference above for acceptance.

| Domain | Scored targets | Mean NLL (nats/token) | Perplexity |
|---|---:|---:|---:|
| English reference | 8,191 | 1.865247 | 6.457532 |
| English long-form | 8,191 | 1.836123 | 6.272172 |
| Chinese reference | 8,191 | 2.097717 | 8.147545 |
| C++/CUDA source | 8,191 | 0.547928 | 1.729665 |
| **Token-weighted overall** | **32,764** | **1.586754** | **4.887855** |

Reusable full-precision records:

```text
profiles/bench/rtx8000-quality/cpu-plan.json
profiles/bench/rtx8000-quality/reference-bf16-retry1.json
profiles/bench/rtx8000-quality/reference-bf16-retry1.physical-kv.json
profiles/bench/rtx8000-quality/reference-bf16-retry1.summary.json
profiles/bench/rtx8000-quality/reference-bf16-retry1.run.json
profiles/bench/rtx8000-quality/reference-bf16-retry1.telemetry.jsonl
profiles/bench/rtx8000-quality/reference-bf16-retry1.stderr.log
```

The original raw report is unchanged. The `physical-kv.json` derivative adds
schema-v2 physical-storage metadata proven by the retained llama KV allocation
log; all numerical scores are identical, and the source hash is recorded.
It remains usable as a diagnostic with explicit precision-mismatch acknowledgement,
not as an automatically matched native-quality baseline.

This is **numerical-quality evidence, not a throughput benchmark**. The 144.727 s
wall time includes load, tokenizer verification, GPU execution and CPU FP64
reductions; it has no five-repeat/warmup performance qualification. The
quality-only monitor permitted known small in-tree numerical tests below a
1 GiB aggregate overlap budget, but observed **zero overlap samples** in the
successful run. GPU samples reached 88 °C, 282.8 W, and 16,251 MiB device memory.
Do not derive a native prefill/decode rate from that wall time.

An earlier `reference-bf16` attempt was canceled by the strict no-contention
guard when `ninfer_arena_test` appeared on GPU0; it produced no final quality
report. Its logs are retained, and it is not a numerical failure or a retained
performance sample. The retry did not weaken the throughput runner's guard.
After success the reference process exited and GPU0 returned to 7 MiB/0%
utilization; scheduling ownership was returned immediately to main/platform.

## Checked without GPU execution

- Python 3.11.15: targeted measurement tests pass (protocol/ID/window rejection,
  nonfinite handling, domain/overall tolerances, exact presets, telemetry
  contention refusal and stale-report invalidation).
- Python `py_compile` passes for the new/changed measurement modules.
- Native benchmark-support CLI/report tests pass in a standalone host-only
  C++20 build, including invalid pair rejection and schema-v16 placement.
- The public Engine scorer and shared corpus/window sources pass C++20 syntax
  compilation; final CUDA-linked executable validation belongs to the
  coordinated platform build.
- The CPU-only score-protocol test rejects malformed score/count/window input
  and verifies the reference logits-memory bound.
- Existing official v3 artifact inspection and both full SHA-256 checks pass.
- Native and 32K/1K matrix dry runs pass.
- External reference adapter compiles against the explicit build above;
  its `--help` succeeds without model/GPU execution.

Targeted CPU checks:

```bash
PYTHONPATH="$PWD/profiles/bench/python-deps:$PWD" \
  "$PY" -m pytest -q tests/test_rtx8000_measurement.py tests/test_serve_corpus.py \
  --basetemp=profiles/bench/rtx8000-test-work
"$PY" -m py_compile tools/bench/run_ninfer_bench_matrix.py \
  tools/bench/run_serve_concurrency.py tools/bench/build_llama_score.py \
  tools/bench/compare_causal_scores.py tools/bench/prepare_score_plan.py
```

On this host pytest was initially missing from the selected interpreter; it was
installed locally under ignored `profiles/bench/python-deps` and the tests ran
with that directory on `PYTHONPATH` (54 passed). The shared vLLM environment was
not modified.

The measured native-versus-reference quality screen and initial decode rate
above are separate from the CPU checks. Full prefill/dual-GPU throughput,
long-prefill/decode execution and 1–8 concurrency execution still require their
own integration evidence; the quality pass does not establish those outcomes.
