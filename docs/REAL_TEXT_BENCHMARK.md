# Train three Mamba implementations on real text

[benchmarks/real_text.py](../benchmarks/real_text.py) trains a small character-level language model on **Tiny Shakespeare**, using the same starting weights, training windows, validation windows, and AdamW configuration for all three implementations. It needs only the existing development environment and the already built PR #78 extension. It installs nothing.

From this fork on the configured Mac:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m benchmarks.real_text \
  --download \
  --pr-root /Users/michalszczygiel/Documents/ChatGPT/mamba-mps \
  --output benchmark-results/real-text.json
```

`--download` fetches the corpus and upstream source once into `benchmark-results/real-text-assets/`. Subsequent runs can omit it and work offline. Every asset is checked against a pinned SHA-256, including cached files. Downloading is outside timing. No pretrained weights are used.

`--pr-root` points to the existing audit directory containing `pr78-review/` at commit `94c9dc285e104f2fceab0119d6d3198c091b6e01` and its Python 3.12/PyTorch 2.11 extension in `pr78-build/metal_pscan/`. The script refuses an unavailable extension, a different PR revision, or modified tracked PR implementation files. It does not install or rebuild that extension.

## What “original state-spaces” means on this Mac

The original optimized state-spaces Mamba backend requires CUDA. **It cannot be benchmarked natively on MPS.** This script loads the original `Mamba` class and the original `selective_scan_ref` function from pinned upstream source. It leaves their definition bodies unchanged, bypasses CUDA module imports, disables the fused CUDA path, and supplies the upstream reference scan through an explicit adapter.

The resulting row is named **`state_spaces_reference_mps`**. Its timing measures the upstream PyTorch reference on MPS, **not the performance of original CUDA Mamba**. A CUDA performance comparison requires a separate NVIDIA machine and software environment.

All three models share this fork's LM embedding, residual connections, RMSNorm, and output head; only each Mamba mixer is replaced. Strict state-dict loading verifies identical parameter names and values. The other two rows use our `backend="mps"` mixer and PR #78's original fused Metal dispatch, without repairing its autograd.

Source: [upstream Mamba class](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py), [upstream reference scan](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py). The downloaded upstream Apache-2.0 license is retained alongside its source.

## Dataset and defaults

The real text is [Tiny Shakespeare from char-rnn](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt): 1,115,394 characters from Shakespeare's works. The first 90% is training text and the final 10% is validation text. No input/target window crosses that boundary. Character zero is reserved for padding; every real character has a trainable embedding.

Default workload:

- Two layers, width 64, state size 16, convolution width 4, FP32.
- Batch 4, sequence length 128, 100 training updates, learning rate 0.002.
- AdamW with betas `(0.9, 0.95)`, weight decay 0.01, no gradient clipping.
- Fixed seeds and identical batches, prepared before timing.
- Validation at step 0 and every 20 updates using four fixed held-out batches.

This is a short learning and implementation check, not training to convergence or a model-quality benchmark. Increase `--steps 500` for a longer run. `--width`, `--layers`, `--batch`, `--length`, `--eval-every`, and `--eval-batches` are configurable. State size is restricted to <=32 because of the audited PR's unchecked fixed-size arrays. The comparison intentionally uses FP32 because PR #78 does not provide valid mixed-precision training.

## Measured default run

On the M4 Pro, PyTorch 2.11.0, using 73,920 parameters and 100 updates (51,200 training characters), every implementation starts at validation cross-entropy 4.20963. [Saved measurements and audits](../benchmarks/results/real-text-m4-pro.json):

| Implementation | Median update, ms | p95, ms | Final validation loss | Perplexity | Gradient/optimizer correctness |
|---|---:|---:|---:|---:|---|
| State-spaces reference on MPS | 43.752 | 44.385 | 2.26471 | 9.628 | Pass |
| Our Metal backend | 4.043 | 4.521 | 2.26471 | 9.628 | Pass |
| PR #78 Metal | 2.922 | 5.195 | 2.40998 | 11.134 | **Fail: 14 parameter tensors have no gradients** |

PR #78's lower time reflects an incomplete backward pass. Those same 14 tensors remain unchanged after training. All parameter tensors update in the other two implementations. Our and the upstream model's held-out next-character accuracy is 34.91%, versus 31.49% for the PR. These are results for this short run, not claims of convergence or general language-model quality.

## Timing and correctness

Each timed update includes zeroing gradients, forward, cross-entropy loss, backward, and AdamW. MPS synchronization surrounds the update. Data transfers, evaluation, numerical audits, and CPU logging are excluded. Backend order rotates each step. First-call time is recorded separately; warmup updates are discarded and both weights and optimizer state start fresh for training. Reports contain median, empirical p95, every timing sample, and training tokens per second. Missing MPS access or enabled CPU fallback is an error.

The correctness audit runs before training and again at our model's learned weights. Each implementation receives the same weights and real-text batch. It checks:

- Logits against the original upstream implementation executed independently on CPU.
- Every parameter gradient, plus the gradient into the mixer stack, including missing and nonfinite gradients.
- An AdamW update against CPU AdamW supplied with **the exact same gradients**.

Gradient tolerances are `rtol=1e-3, atol=1e-7`; logits use `1e-4, 1e-5`; same-gradient optimizer updates use `1e-3, 1e-6`. CPU/MPS updates made from independently computed gradients are also recorded as diagnostics. AdamW's normalization can amplify tiny derivative differences near zero; this diagnostic does not replace the direct gradient check or decide correctness.

Training loss reduction alone does not establish correct gradients. PR #78 can learn through its remaining connected paths while its SSM parameters receive no gradients. The script still executes that training loop to show the behavior, but labels its timings **invalid for a correct-training comparison** if the audits fail. It reports unchanged parameters after training and all missing/mismatched gradients by name.

The final JSON includes held-out cross-entropy, perplexity, next-character accuracy, full training-loss curves, numerical differences, corpus window indices, source revisions/hashes, environment details, and timings. The exit code is **0** when every implementation passes, **2** when the completed comparison finds correctness failures. Exit code 2 is expected for the audited PR and does not mean the report is missing.

For a quick execution check:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m benchmarks.real_text \
  --pr-root /Users/michalszczygiel/Documents/ChatGPT/mamba-mps \
  --steps 2 --length 16 --batch 2 --width 16 --layers 1 \
  --warmup 1 --eval-batches 1 --eval-every 1 \
  --output benchmark-results/real-text-smoke.json
```
