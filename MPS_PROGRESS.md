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

## Step 5 — convolution

Added convolution/SiLU forward and backward with optional bias, filter widths 1/4/9, differentiable history and views. Validation: 65 scan/convolution/model cases passed.

`python -m benchmarks.conv` compared identical complete MPS blocks with only convolution changed (B=2,D=128). At L=64, forward improved 1.881 -> 1.766 ms and forward/backward 2.445 -> 1.763 ms. At L=512, results were essentially tied; at L=2048 native backward regressed 9.179 -> 9.947 ms. Consequently automatic fused-convolution selection is limited to lengths <=128; longer sequences retain PyTorch convolution on MPS. This is a measured dispatch decision, not an unsupported-operator CPU fallback.

## Step 6 — stateful execution

Chunk execution requests only final state, reusing `ssm_chunk_step` with an optional `return_states=False`; the default still returns every state for compatibility. Native step execution shares the same stateful path. The existing `checkpointed_sequence` helper is reused. Fixed empty convolution history handling for width one and added positive segment-size validation.

Validation covers nonzero initial states, losses on final states, all parameter/input/cache gradients, uneven chunks, checkpointed chunks and individual token steps. The existing cache list is intentionally updated by the model API; tests retain separate references to original states when checking gradients.

## Step 7 — cached generation

`LM.generate` now prefills Mamba-1 once and decodes one token at a time using existing core cache APIs. The output token tensor is preallocated and only the last prompt position is projected to vocabulary logits. Sampling options and the input-device return convention remain intact. Zero-token requests return an empty result. Each module's prior train/eval flag is restored even on an exception.

Validation: 13 generation/stateful tests passed on CPU and MPS, including greedy parity against full-history execution, one-token prompts, batch generation, sampling, zero output tokens, invalid arguments, and assertions of one prefill plus exactly N-1 single-token decode calls.

## Step 8 — precision coverage

Added actual half/bfloat Metal buffer variants, with FP32 recurrent states, delta, reductions and gradient accumulation; outputs and input gradients retain their intended dtype. Reference scan and normalization handle mixed precision consistently. Convolution cache inputs follow activation dtype. The existing checkpoint helper now captures and restores autocast settings during backward recomputation, verified by hooks observing projection output dtypes.

Validation: 98 passed, 1 CUDA-only skip. FP16 tolerances are rtol=1e-2/atol=1e-3; BF16 rtol=5e-2/atol=5e-3; FP32 stays rtol=1e-4/atol=1e-5. Tests cover explicit conversion, autocast, optional bias/inner normalization, all model gradients, optimizer steps, cached generation and checkpoint backward outside the autocast context. Autocast AdamW uses FP32 parameters; explicitly low-precision parameters were validated with SGD. The CPU fallback environment flag is unset (PyTorch's default is disabled).

## Step 9 — measured tuning

Reused `F.linear` for the MPS delta projection, eliminating its transposed batched multiplication and subsequent layout copy. Scan output uses SIMD state reductions and 256-thread groups. Short scans skip summary/boundary launches; wide inference uses a recurrent specialization and narrow prompts use temporal tiles. Training uses bounded 64/128-token choices, selected after measuring 64/128/256 in all three precisions. Long inference now uses native convolution; long training retains the faster PyTorch convolution.

An added 8192-token derivative test exposed cancellation in the A gradient. Compensated reductions and a long-sequence forward-sensitivity specialization using pairs of FP32 values resolve it without changing tolerances or clipping. This specialization starts at length 4096. At length 8192, a B=2, width=128 block takes 76.89 ms for forward/backward versus 258.10 ms for the PyTorch path; inference takes 6.43 versus 99.27 ms. Fixed fake output strides for dense transposed inputs, including backward operators.

Validation: **117 passed, 1 CUDA-only skip**, with real MPS required and CPU fallback disabled. Coverage includes long FP16/BF16 gradients and CPU validation of the comparison Transformer's real KV cache.

FP32 small-model results at lengths 512/2048: **7.24x geometric-mean training speedup** and **12.09x prompt speedup** over PyTorch Mamba. Generating 256 tokens after those prompts improves from 18.49/68.01 seconds to 0.273/0.263 seconds. Most of the generation improvement comes from caching: cached PyTorch generation takes 0.345/0.473 seconds. All five PR forward cases meet the 10% gate in two controlled runs. PR comparisons rotate timing order after warming every model to distribute clock/thermal bias.

The 129,135,360-parameter Mamba benchmark ties embedding/head weights only in the benchmark; the library's default remains untied. The matched Transformer has 128,457,216 parameters. At length 2048, FP32 Mamba training/prompt processing takes 669.31/149.50 ms versus 954.47/306.44 ms for the Transformer. The PyTorch Mamba reference exceeds the 16 GiB allocator budget during training at this length; its missing training result is reported as out of budget. It does complete cached generation.

Two 30-second small-model training runs remain stable: first/last-quarter medians are 21.407/21.398 ms (length 512) and 65.732/65.741 ms (length 2048). Sustained-run results are saved separately from the fresh-model speed measurements. Memory sampling is a lower bound, not an exact peak; GPU driver counters and process RSS overlap in unified memory and must not be added as a measured peak.

Raw samples, precision comparisons, memory readings, compilation/pipeline preparation, and dispatch diagnostics are consolidated by `python -m benchmarks.report`. The final benchmark report and reproduction guide accompany packaging in step 10.

## Step 10 — packaging and documentation

Packaged both Metal sources in the wheel and source distribution, using the fork-local version `1.2.0+mps`. Metadata records Python >=3.10, PyTorch >=2.11.0, and einops >=0.8.1. CPU forward and chunk execution now avoid importing the Metal convolution implementation. No dependencies were installed or upgraded.

Added usage, precision, checkpointing, generation, installation, benchmark reproduction, and limitation guides. The measured tables distinguish native kernels from caching improvements, record the 130M reference training memory limit, and explicitly identify unmeasured full-history 130M generation and full-model FP16 performance. Mamba-2, Jamba, Vision Mamba, accelerated HF integration, higher-order derivatives, and full compile optimization remain follow-on work.

Final validation: **119 passed, 1 CUDA-only skip**, with real MPS required and CPU fallback disabled. Distribution tests build and inspect both archives, then run CPU and MPS training/generation from an unpacked wheel outside the checkout. The CPU test rejects any Metal compilation and verifies that backend modules stay unloaded. All three Python examples in the MPS usage guide executed successfully. The final diff passed whitespace checks.

## Follow-up — real-text training comparison

Added `benchmarks.real_text`, using pinned Tiny Shakespeare data and unchanged state-spaces `Mamba`/`selective_scan_ref` definitions through an explicit MPS reference adapter. Original CUDA performance is unavailable on this Mac and is not claimed. Reused the existing LM wrapper, exact state-dict loading, AdamW, and PR loader. Every implementation sees the same initial weights, batches, and held-out split. No packages were installed.

The default 73,920-parameter, two-layer run completed 100 updates. Upstream reference/ours/PR median update times are 43.752/4.043/2.922 ms; final validation losses are 2.26471/2.26471/2.40998. Upstream and ours pass initial and learned-weight audits. PR has 14 missing-gradient parameter tensors, all unchanged after training; its timing is explicitly invalid for correct-training comparison. The script returns exit code 2 after saving the complete report when any implementation fails correctness.

Audits compare logits and every gradient with the upstream CPU reference. Optimizer correctness uses identical supplied gradients on CPU/MPS; cross-device updates with independently calculated gradients remain visible as diagnostics because AdamW amplifies near-zero derivative roundoff. Four regressions cover dataset split/window integrity, deterministic batches, source checksums, and detection of absent/incorrect/nonfinite gradients even with matching logits.

Validation: **123 passed, 1 CUDA-only skip**, requiring real MPS with CPU fallback disabled. The short execution check and complete default real-text run both completed; PR correctness failures were captured as expected. Saved measurements are in `benchmarks/results/real-text-m4-pro.json` and usage instructions in `docs/REAL_TEXT_BENCHMARK.md`.

## Follow-up — ECG200 classification from scratch

Added `benchmarks.ecg200` using only this fork. The user's LayerNorm/Mamba/Dropout/residual block wraps the existing bare mixer; two blocks with width 256, state size 64, convolution width 4, expansion 2 and dropout 0.1 feed mean pooling and an MLP classifier. There are 1,057,666 trainable parameters and no pretrained weights.

Downloaded and checksum-pinned the real ECG200 archive. A fixed stratified split reserves 20 of the official 100 training examples for validation; normalization is fitted on the remaining 80 only. The 100 official test examples do not select epochs or settings. Three predefined seeds train with AdamW, with early stopping on validation loss and no gradient clipping.

Seeds 2026/2027/2028 select epochs 53/27/31 and score 86%/80%/83% test accuracy. Mean accuracy is 83% with 3 percentage-point sample standard deviation; mean balanced accuracy is 82.47%. The training-majority test baseline is 64%. Median full training-update times are 16.142/16.065/16.118 ms for batch 16. All initial/final numerical audits against this fork's CPU path pass, every update has finite gradients for every parameter, all parameter tensors change, and saved checkpoints reload with matching logits. Validation loss becomes unstable in later epochs on this small dataset; no settings were adjusted based on the test outcomes.

Validation: **128 passed, 1 CUDA-only skip**, with real MPS required and CPU fallback disabled. No packages were installed. Instructions are in `docs/ECG200_CLASSIFICATION.md`, measured results in `benchmarks/results/ecg200-m4-pro.json`, and the three local checkpoints in ignored `benchmark-results/ecg200-checkpoints/`.
