# ECG200 sequence classification with Mamba

[benchmarks/ecg200.py](../benchmarks/ecg200.py) trains **only this fork's Mamba-1 implementation**, from random initialization, on the real [ECG200 archive](https://timeseriesclassification.com/description.php?Dataset=ECG200). No pretrained model, state-spaces dependency, PR #78 extension, or extra package is needed.

## Run

From the fork's root, with its existing virtual environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m benchmarks.ecg200 --download
```

The first run downloads the 299 KB ECG200 archive and verifies its SHA-256 and both text-file checksums. It caches the archive under `benchmark-results/ecg200-data/`; subsequent runs work offline and can omit `--download`.

Default output is `benchmark-results/ecg200.json`, plus one validation-selected checkpoint per seed under `benchmark-results/ecg200-checkpoints/`. To make a separate longer run, for example:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m benchmarks.ecg200 \
  --epochs 200 --patience 30 --output benchmark-results/ecg200-longer.json
```

Changes to settings should be selected using training/validation data, not the reported test accuracy.

## Architecture

The model uses the requested residual block exactly:

```text
input [B,96,1]
  -> Linear(1,256)
  -> 2 x [LayerNorm -> Mamba mixer -> Dropout(0.1) -> add residual]
  -> mean over time
  -> Linear(256,128) -> GELU -> Dropout(0.1) -> Linear(128,2)
```

Every mixer has state size 64, convolution width 4, expansion 2, and `backend="mps"`. The stack is `nn.Sequential(*[MambaBlock(...) for _ in range(num_layers)])`. It has **1,057,666 trainable parameters**. There is no attention, positional embedding, or pretrained feature extractor.

In this fork, `mambapy.mamba.MambaBlock` is the bare mixer equivalent to the `Mamba(...)` inside the supplied residual wrapper. The benchmark imports it as `MambaMixer` and adds the user's LayerNorm, dropout, and residual. Using the fork's top-level `Mamba` there would unintentionally add another normalization and residual connection.

The classes can also be used directly:

```python
import torch
from benchmarks.ecg200 import ECGClassifier

model = ECGClassifier(hidden_dim=256, d_state=64, d_conv=4, expand=2,
                      dropout=0.1, num_layers=2).to("mps")
# signals: float32 [batch,96,1], labels: int64 [batch] containing 0 or 1
# logits = model(signals)
# torch.nn.functional.cross_entropy(logits, labels).backward()
```

## Data and training protocol

ECG200 contains 100 training and 100 test sequences, each with 96 scalar observations and one of two labels. Labels map from `-1/+1` to `0/1`. The benchmark reserves 20 examples from the official training set for stratified validation, leaving **80 training, 20 validation, 100 test**. The split is fixed with seed 42, shared across all runs. Normalization uses only the 80 training examples' scalar mean and standard deviation.

Training uses FP32, batch size 16, AdamW at learning rate 0.001 and weight decay 0.01, ordinary cross-entropy, and no gradient clipping. It runs up to 100 epochs and stops after 20 epochs without improved validation loss. The lowest validation-loss checkpoint is restored before test evaluation. Three initialization/shuffle seeds—2026, 2027, and 2028—are fixed before running. The test set does not select epochs, hyperparameters, normalization, or seeds.

The benchmark records per-epoch training and validation loss, test loss, accuracy, balanced accuracy, macro F1, confusion matrices, and per-example test predictions. Repeated runs can differ slightly due to GPU floating-point reductions and nondeterministic operations.

## Validation and timing

Before training and at the selected checkpoint, dropout is disabled temporarily and the same classifier/weights are compared against **this fork's CPU PyTorch reference path**. This is a numerical check within our implementation, not training another architecture. It compares logits, input gradients, and every parameter gradient. Tolerances are `rtol=1e-4, atol=1e-5` for logits and `rtol=1e-3, atol=1e-7` for gradients.

Every training update checks that all parameters receive finite gradients. The selected checkpoint must change every parameter tensor from initialization. The saved checkpoint is loaded with strict state-dict matching and its logits are checked against the pre-save logits. Missing MPS access, CPU fallback, or failed correctness checks do not count as successful tests.

Reported milliseconds measure a synchronized full forward/loss/backward/AdamW update. Data indexing, numerical checks, evaluation, logging, and checkpoint work are excluded. First-call time is separate. Warmup weights, optimizer state, and dropout RNG are reset before training. Epoch training time is the sum of timed updates, not total experiment wall time.

## Measured results on M4 Pro

The default configuration above produced these [saved results](../benchmarks/results/ecg200-m4-pro.json):

| Seed | Selected epoch | Test accuracy | Balanced accuracy | Macro F1 | Median update | p95 update |
|---|---:|---:|---:|---:|---:|---:|
| 2026 | 53 | 86% | 85.42% | 0.8498 | 16.142 ms | 16.369 ms |
| 2027 | 27 | 80% | 80.12% | 0.7898 | 16.065 ms | 16.351 ms |
| 2028 | 31 | 83% | 81.86% | 0.8166 | 16.118 ms | 16.338 ms |

Mean test accuracy is **83%, with sample standard deviation 3 percentage points across seeds**; mean balanced accuracy is **82.47%**. Predicting the training majority class for every test example gives 64% accuracy. All initial/final numerical audits, gradient checks, parameter-update checks, and checkpoint reload checks passed.

Training accuracy at the selected checkpoints is 93.75–96.25%. Later validation loss is unstable in some runs, so validation selection matters. With only 100 test examples and 80 optimization examples, this demonstrates working learning and generalization on a small real dataset; it does not establish a state-of-the-art classification result.

## Restore a checkpoint

```python
from pathlib import Path
import torch
from benchmarks.ecg200 import ECGClassifier

saved = torch.load(Path("benchmark-results/ecg200-checkpoints/ecg200-seed-2026.pt"),
                   map_location="cpu", weights_only=True)
model = ECGClassifier(**saved["model_config"]).to("mps").eval()
model.load_state_dict(saved["state_dict"], strict=True)
mean = torch.tensor(saved["normalization"]["mean"], device="mps")
std = torch.tensor(saved["normalization"]["std"], device="mps")
# raw_signals: [batch,96,1], on MPS
# with torch.no_grad():
#     predictions = model((raw_signals - mean) / std).argmax(-1)
```
