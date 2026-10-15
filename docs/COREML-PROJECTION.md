# Fixed-shape Core ML projection diagnostic

`CoreMLProjectionExecutor` is an explicit, adapter-owned execution primitive.
It accepts a reviewed `.mlmodelc` tree, verifies its SHA-256 identity, copies at
most 64 MiB of regular files into private storage, verifies that retained copy,
and loads it with `CPU_AND_NE`. Input and output must match the declared finite
float16 shapes. Core ML may expose an fp16 model feature to Python as float32;
the executor accepts only native float16 or float32, validates the exact shape
and finite range, and returns a private float16 copy. Prediction and close share one lock; close retains the model
and files until an in-flight call returns. No serving adapter selects it yet.

The receipt distinguishes the following evidence:

- `selected=true`: this executor was explicitly selected by its caller.
- `prediction_completed=true`: the native prediction call returned. A returned
  prediction receipt also means its output feature, shape, dtype and finite
  values passed validation. A compute-plan receipt counts native completions;
  it does not replace the separate numerical parity receipt.
- `compute_plan`: anticipated placement reported by `MLComputePlan`.
- `observed_used=false` and `execution_device_observed=false`: no post-execution
  device trace was collected. An anticipated Neural Engine placement or a
  successful `CPU_AND_NE` prediction does not change these fields.
- `qualified=false`: this primitive and its tiny diagnostic do not establish
  model-serving qualification or a speed benefit.

Apple documents `CPU_AND_NE` as allowing CPU and Neural Engine while excluding
GPU. The [Core ML Tools model API](https://apple.github.io/coremltools/source/coremltools.models.html)
also documents compiled-model prediction and compute-plan inspection. The
declared tensor byte count is a lower bound, not Core ML's resident or peak
memory use. This diagnostic does not integrate asynchronous model-head results,
sampling, KV state, or scheduler admission.

## Running the bounded M3 probe

Use a macOS arm64 Python environment with compatible `coremltools` and `numpy`.
Prediction with this float16 model and `CPU_AND_NE` requires macOS 13 or later;
the optional compute-plan API requires macOS 14.4 or later. The probe checks
that the chip is an Apple M3. Core ML Tools is absent from the development
environment used for the CPU contract tests, so those tests substitute a fake
runtime and never execute Core ML. A separate owned run on an Apple M3 Pro with
Core ML Tools 9.0 completed all six predictions with exact output parity. That
runtime widened the native output feature to float32 before the executor
validated and narrowed it. Its Python package did not expose
`coremltools.models.MLComputePlan`, so the run supplies no placement evidence;
`observed_used` and `execution_device_observed` remain false.

First obtain the normal CPG/radio lease and matching paired GPU locks, after
checking waiter order, foreign work, services, swap and thermals. The probe
does not acquire or release ownership. Both its supervisor and its worker
validate the supplied live lease and ancestor-bound lock receipts. Run from
the source tree inside that already-owned window:

```sh
PYTHONPATH="$PWD/src" /path/to/m3/coreml-python \
  scripts/probe_coreml_projection.py \
  --output-dir /path/to/fresh/external-evidence \
  --cpg-task TASK_ID \
  --cpg-owner-lock /path/to/cpg-owner-lock-directory \
  --cpg-generation GENERATION \
  --timeout 120 --repetitions 2
```

The output directory must not exist. The probe builds a deterministic synthetic
float16 `[2,8] -> [2,4]` linear model with no bias, compiles it, and checks every
output against NumPy float32 multiplication for three input cases per repetition
(absolute and relative tolerance 0.002). It writes the generated package and
compiled artifact, worker log, atomic prediction/parity/plan receipt and atomic
supervisor receipt outside the repository. These are diagnostic artifacts,
not trained weights, a performance result, or observed ANE execution evidence.

The supervisor applies one deadline to conversion, compilation, prediction and
plan inspection. On timeout or SIGINT/SIGTERM it terminates its worker process
group and reaps the worker before returning. A rare uninterruptible OS call can
delay reaping after SIGKILL; in that case the supervisor remains alive to retain
ownership rather than claim safe completion. Filesystem operations and ownership
validation are not a universal hard wall-clock bound. Process exit also does
not supply a per-device Core ML daemon drain trace: inspect the ownership and
device state before handing the window back. No deadline or prediction result
should be presented as native Core ML cancellation or observed ANE placement.
