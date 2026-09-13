# M4 Pro benchmark results

Measured with PyTorch 2.11.0, Python 3.12.11, macOS 26.6.2, and an M4 Pro with 24 GiB. The implementation, raw samples, and acceptance checks were committed together in `0f37c64`. [Machine-readable results](../benchmarks/results/m4-pro.json) include input settings, source hashes, first-call times, medians, tail samples, and memory readings.

These are synthetic compute benchmarks, not measurements of trained model quality. All runs use initialized weights and random token tensors, without pretrained downloads or tokenization. GPU benchmarks run sequentially, with CPU fallback disabled.

For actual text training and gradient checks against state-spaces and PR #78, see the separate [Tiny Shakespeare benchmark](REAL_TEXT_BENCHMARK.md).

For from-scratch sequence classification using only this fork, see the [ECG200 Mamba classifier](ECG200_CLASSIFICATION.md).

## Models and timing

| Model class | Layers | Width | State size | Vocabulary | Mamba parameters | Transformer parameters |
|---|---:|---:|---:|---:|---:|---:|
| Small | 6 | 384 | 16 | 8192 | 12,077,184 | 12,214,272 |
| Approximately 130M | 24 | 768 | 16 | 50,280 | 129,135,360 | 128,457,216 |

Model benchmarks use batch size one. The 130M benchmark ties input embedding and output-head weights in both architectures; the library's default remains untied. The causal Transformer uses PyTorch SDPA, matching width/depth, a feedforward width chosen to approximately match parameters, and a real preallocated KV cache. Its cache implementation is checked against full-history greedy generation.

Speed runs use three warmups and five synchronized samples. Forward/prompt samples average three executions; training samples contain one complete update. Generation uses one warmup and three synchronized 256-token samples. First calls are separate from these medians. Reported p95 values are empirical tail observations; with three or five samples they are the maximum sample, not a well-estimated population percentile.

## FP32 training and prompt processing

Training includes forward, cross-entropy loss, backward, and AdamW. Times are milliseconds per update.

| Size | Length | Native Mamba | PyTorch Mamba | SDPA Transformer |
|---|---:|---:|---:|---:|
| Small | 512 | 21.39 | 139.51 | 17.56 |
| Small | 2048 | 65.82 | 528.34 | 102.41 |
| 130M | 512 | 206.64 | 1133.98 | 178.81 |
| 130M | 2048 | 669.31 | Out of budget | 954.47 |

Prompt processing builds caches and projects only the last prompt position to vocabulary logits. Times are milliseconds.

| Size | Prompt | Native Mamba | PyTorch Mamba | SDPA Transformer |
|---|---:|---:|---:|---:|
| Small | 512 | 4.37 | 52.96 | 3.87 |
| Small | 2048 | 16.91 | 203.68 | 35.11 |
| 130M | 512 | 38.46 | 421.46 | 36.74 |
| 130M | 2048 | 149.50 | 1684.64 | 306.44 |

Small-model geometric-mean improvements over PyTorch Mamba are **7.24x for training** and **12.09x for prompt processing**, exceeding the 2x targets. The Transformer is faster at length 512; Mamba is faster at 2048. Full-logit forward and forward/backward timings are also retained in the raw results.

## FP32 generation

Milliseconds to generate **256 new tokens**, including prompt processing:

| Size | Prompt | Native cached Mamba | Cached PyTorch Mamba | Cached Transformer | Full-history PyTorch Mamba |
|---|---:|---:|---:|---:|---:|
| Small | 128 | 263.02 | 339.00 | 233.38 | 7242.93 |
| Small | 512 | 272.60 | 345.40 | 233.63 | 18485.00 |
| Small | 2048 | 263.04 | 473.24 | 321.97 | 68007.61 |
| 130M | 128 | 974.09 | 1330.03 | 926.44 | Not measured |
| 130M | 512 | 979.30 | 1689.29 | 1195.27 | Not measured |
| 130M | 2048 | 1005.72 | 2924.67 | 2618.02 | Not measured |

The small-model full-history comparison exceeds the 2x target by **67.81x at prompt 512** and **258.54x at prompt 2048**. These large ratios mainly measure the benefit of caching. Comparing the cached implementations separates that change from backend optimization. Full-history 130M generation was not measured; no speedup against that baseline is claimed.

Decode-only runs exclude prefill and include first-token selection plus 255 recurrent updates. At prompt 2048, native/PyTorch/Transformer decode medians are **265.55/341.86/292.72 ms** for the small model and **998.15/1323.77/2379.25 ms** for 130M. Stage timings use separate synchronization and samples, so they need not add to end-to-end latency; asynchronous execution can overlap CPU dispatch with GPU work.

## Precision comparisons

BF16 autocast with FP32 optimizer parameters, in milliseconds:

| Size | Length | Mamba training | Transformer training | Mamba prompt | Transformer prompt |
|---|---:|---:|---:|---:|---:|
| Small | 512 | 21.67 | 19.11 | 4.42 | 4.02 |
| Small | 2048 | 63.88 | 109.40 | 15.42 | 37.36 |
| 130M | 512 | 204.65 | 187.71 | 39.59 | 39.24 |
| 130M | 2048 | 630.54 | 984.55 | 137.60 | 324.50 |

Converted BF16 weights reduce storage but do not improve Mamba decoding on this setup. End-to-end generation of 256 tokens, milliseconds:

| Size | Prompt | Mamba FP32 | Mamba BF16 weights | Transformer FP32 | Transformer BF16 weights |
|---|---:|---:|---:|---:|---:|
| Small | 128 | 263.02 | 393.78 | 233.38 | 235.50 |
| Small | 512 | 272.60 | 402.28 | 233.63 | 237.23 |
| Small | 2048 | 263.04 | 406.42 | 321.97 | 285.76 |
| 130M | 128 | 974.09 | 1505.20 | 926.44 | 882.68 |
| 130M | 512 | 979.30 | 1535.33 | 1195.27 | 900.61 |
| 130M | 2048 | 1005.72 | 1649.71 | 2618.02 | 2211.26 |

Autocast generation is slower still for Mamba: at prompt 2048 it takes 542.26 ms (small) and 2155.18 ms (130M). FP32 is therefore the fastest measured Mamba generation choice among these modes. FP16 has correctness and block-tuning coverage; complete-model FP16 performance was not benchmarked. Further low-precision decode optimization remains useful.

## PR #78 and kernel tuning

The comparison uses the audited PR head `94c9dc285e104f2fceab0119d6d3198c091b6e01`, with its unchanged extension compiled for the same Python/PyTorch environment. Model weights and inputs match across implementations. Only the PR's valid FP32 forward behavior is compared; its disconnected/incorrect training gradients do not provide a valid backward baseline.

Every model receives 25 warmups. Fifteen samples of five forwards rotate model order to distribute clock and thermal effects. Both complete runs pass the requirement that native forward be no more than 10% slower than the PR. The repeated run gives these milliseconds:

| Batch | Length | Width | Native | PyTorch | PR #78 | Native / PR |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 128 | 0.267 | 1.077 | 0.545 | 0.490 |
| 2 | 512 | 128 | 0.515 | 6.575 | 0.703 | 0.733 |
| 2 | 2048 | 128 | 1.602 | 25.047 | 1.819 | 0.881 |
| 2 | 512 | 384 | 1.312 | 19.032 | 1.272 | 1.032 |
| 1 | 2048 | 384 | 2.762 | 34.067 | 2.766 | 0.998 |

These block timings are not interchangeable with full-LM timings. Dispatch uses bounded state/dtype/chunk variants, not one shader per sequence length. Measurements cover chunk sizes 64/128/256 and output threadgroups 64/128/256/512/1024. Native convolution helps inference and short training; PyTorch convolution remains faster for long training. Reusing PyTorch `F.linear` for delta projection avoids a costly transpose/copy.

The accuracy specialization for lengths >=4096 adds work to the A gradient. At length 8192 (B=2, width=128, one block), native forward/backward still takes **76.89 ms versus 258.10 ms** for PyTorch, and inference takes **6.43 versus 99.27 ms**. This measured result includes that extra accuracy work.

## Sustained runs, startup, and memory

Two 30-second training runs completed 1403 and 457 updates at lengths 512 and 2048. First/last-quarter medians were **21.407/21.398 ms** and **65.732/65.741 ms**, respectively. These runs are stored separately from fresh-model speed measurements. The PR gate was repeated after the longer workloads.

First-call costs are excluded from steady-state medians. For example, the small FP32 Mamba forward first call took 440.04 ms, versus a 4.85 ms median at length 512. Separate shader-library and pipeline preparation measurements range from about 0.16 to 2.57 ms per recorded variant with existing driver caches. Driver caches were not purged, so these preparation numbers do not describe a fully cold machine.

Memory is measured in a separate instrumented execution, polling live-tensor and driver allocation counters every 5 ms. Their maxima are **sampled lower bounds**, not exact peaks. Process maximum RSS is recorded separately; unified-memory counters overlap and must not be summed as an observed peak. The MPS allocator is capped at 16 GiB.

At length 2048, sampled FP32 driver maxima are **1.14 GiB** for small Mamba versus **2.31 GiB** for the Transformer; at 130M they are **7.95 versus 12.89 GiB**. The 130M PyTorch Mamba training baseline hit the 16 GiB cap. It is recorded as out of budget, not as a timed success or skipped validation.

## Reproduction

Run from this fork with the existing `.venv` and pinned development dependencies. These commands do not install anything:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=0
MAMBAPY_REQUIRE_MPS=1 .venv/bin/python -m pytest tests -q

for size in small 130m; do
  for backend in mps torch transformer; do
    .venv/bin/python -m benchmarks.acceptance --size "$size" --backend "$backend" \
      --mode speed --output "benchmark-results/$size-$backend-speed.json"
    .venv/bin/python -m benchmarks.acceptance --size "$size" --backend "$backend" \
      --mode generation --lengths 128 512 2048 --repeats 3 \
      --output "benchmark-results/$size-$backend-generation.json"
  done
done

.venv/bin/python -m benchmarks.acceptance --size small --backend torch \
  --mode generation --uncached --lengths 128 512 2048 --repeats 3 \
  --output benchmark-results/small-torch-uncached.json
.venv/bin/python -m benchmarks.acceptance --size small --backend mps \
  --mode speed --sustain-seconds 30 --output benchmark-results/small-mps-sustained.json
```

For BF16 training/prompt runs, add `--dtype bfloat16`. For converted-weight generation add both `--dtype bfloat16 --precision-mode weights`; the default precision mode uses autocast with FP32 parameters. Use separate output filenames to retain each result.

```sh
.venv/bin/python -m benchmarks.tune > benchmark-results/tuning.jsonl
.venv/bin/python -m benchmarks.groups > benchmark-results/groups.jsonl
.venv/bin/python -m benchmarks.conv > benchmark-results/convolution.jsonl
.venv/bin/python -m benchmarks.long_sequence > benchmark-results/long-sequence.jsonl
.venv/bin/python -m benchmarks.compile > benchmark-results/compilation.jsonl
.venv/bin/python -m benchmarks.blocks --output benchmark-results/blocks-without-pr.json
```

The optional PR comparison needs an independently built audited checkout. Pass `--pr-root /path/to/audit`, containing `pr78-review/mambapy/mamba.py` and the importable extension under `pr78-build/metal_pscan`, to `benchmarks.blocks`. Save the two runs as `controlled-blocks.json` and `controlled-blocks-repeat.json`. This harness neither installs nor merges PR code.

After collecting those files, consolidate and check the numerical speed gates:

```sh
.venv/bin/python -m benchmarks.report --output benchmarks/results/m4-pro.json
```

The command fails if a required gate fails. Hardware, power state, software versions, and background load affect results; rerun the full workloads when changing dispatch choices.
