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
