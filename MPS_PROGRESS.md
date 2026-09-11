# MPS implementation log

Target: M4 Pro, 24 GiB, Python 3.12.11, PyTorch 2.11.0.
Branch: `feature/mps-mamba1-backend`. Use `.venv/bin/python` exclusively.
Dependency changes require permission. Validate and commit each step before starting the next.

## Step 1 — reference baseline

Packages are pinned in `requirements-dev.txt`. The user installed these packages.
Real GPU execution succeeded outside the sandbox. CPU fallback is disabled for GPU checks.

Command: `MAMBAPY_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/test_chunk_step_equivalence.py tests/test_checkpointing.py tests/test_mps_baseline.py -q`

Result: 12 passed, 1 explicitly skipped CUDA-only peak-counter test. CPU and MPS chunk-gradient tests passed. Retained-activation storage has a separate test; it is not presented as peak memory.

Benchmark: `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m benchmarks.mps --suite blocks --repeats 5 --output benchmark-results/baseline-blocks.json`.
Single residual block, FP32, state size 16. Three warmups; five individually synchronized samples. Compilation is excluded from medians.

| Batch | Length | Width | Forward ms | Forward/backward ms | AdamW step including forward/backward ms |
|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 128 | 2.085 | 3.345 | 4.838 |
| 2 | 512 | 128 | 6.980 | 17.594 | 17.917 |
| 2 | 2048 | 128 | 25.564 | 66.064 | 66.235 |
| 2 | 512 | 384 | 19.514 | 50.295 | 50.493 |
| 1 | 2048 | 384 | 34.670 | 87.138 | 87.560 |

Raw samples and allocation snapshots are in ignored `benchmark-results/` files. Snapshot counters are not peak measurements. These timings use different batching from the prior PR audit and must not be treated as a direct controlled comparison with that report.

## Step 2 — backend contract

Added explicit backend configuration and a compact reference scan using the existing `pscan`. Both output and final state are differentiable. Checkpoint keys remain compatible. Metal dispatch deliberately stays disabled until its backward is validated.

Validation: 17 passed, 1 CUDA-only skip across scan-contract, baseline, chunk-equivalence and checkpointing tests. Includes double-precision gradcheck and CPU/MPS model gradients.

## Step 3 — FP32 Metal forward

Three kernels calculate chunk summaries, boundary states and outputs. No full-sequence state tensor is allocated. State capacities are bounded power-of-two specializations; explicit native support is 1..128 states, with validated rejection outside that range.

Validation: 35 GPU cases passed for chunk sizes 64/128/256, lengths through 8192, states through 128, views and storage offsets. A long-sequence failure exposed loss of accuracy for near-unit decay factors. Reusing PyTorch's existing Metal `expm1f` helper and accumulating chunk log-decays fixed it without relaxing tolerances (rtol=1e-4, atol=1e-5), checked against a CPU double-precision reference.

## Step 4 — Metal autograd

Added reverse chunk summaries, boundary adjoints, local state recomputation and gradients for every scan input. Compensated accumulation fixes long-sequence derivative cancellation without changing test tolerances. SIMD state reductions and compact atomic accumulation avoid full-sequence state/gradient intermediates. MPS model dispatch is now enabled for supported FP32 inputs.

Validation: 62 passed, 1 CUDA-only skip. Includes all input gradients against double precision through length 2048, non-power-of-two state dimensions, a 100000-scaled analytical derivative, final-state-only loss, frozen inputs, repeated backward/gradient accumulation, and three complete-model AdamW steps matching the reference. Schema and FakeTensor opchecks passed. PyTorch 2.11's autograd-registration opcheck excludes MPS explicitly; direct numerical gradient tests cover that requirement instead.
