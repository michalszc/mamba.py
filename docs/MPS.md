# Mamba-1 on Apple Silicon

This fork provides first-order training, prompt processing, stateful chunks, checkpointing, and cached language-model generation on MPS. It preserves Mamba-1 parameter names, checkpoint weights, cache layouts, and the public `pscan(A, X)` function. Measured performance and limitations are in [MPS_BENCHMARKS.md](MPS_BENCHMARKS.md).

## Installation

The validated environment is Python **3.12.11**, PyTorch **2.11.0**, macOS **26.6.2**, and an **M4 Pro with 24 GiB**. Package metadata requires Python >=3.10 and PyTorch >=2.11.0. Other Apple chips and OS versions have not been benchmarked here.

Use the existing environment. Development versions are pinned in [requirements-dev.txt](../requirements-dev.txt): torch 2.11.0, pytest 8.3.5, numpy 2.2.6, einops 0.8.1, and setuptools 80.9.0. No separate CUDA library, Objective-C++ extension, or Metal package is required. Shader source is distributed with the package and compiled lazily by `torch.mps.compile_shader`.

After the required dependencies are present, install this fork from its root:

```sh
.venv/bin/python -m pip install --no-build-isolation --no-deps .
```

This is an installation command for the reader; repository validation builds and unpacks distributions without installing or modifying dependencies. The fork uses the local version `1.2.0+mps`, distinct from the upstream release.

## Backend selection

| Configuration | Behavior |
|---|---|
| `backend="auto"` (default) | Uses validated native operations for supported MPS inputs; otherwise uses the existing PyTorch path on the tensor's device. |
| `backend="torch"` | Forces the PyTorch reference operations, including when tensors are on MPS. |
| `backend="mps"` | Requires MPS inputs and native scan support; unsupported inputs raise an error. |

Native scan supports state sizes **1 through 128**, including non-powers of two. Larger states use PyTorch on MPS under `auto`; explicit `mps` rejects them. Native convolution supports optional bias and arbitrary positive filter widths; widths 1, 4, and 9 are covered by tests. Long training sequences intentionally use PyTorch convolution because it is faster there.

`pscan=False` keeps the existing reference behavior under `auto`/`torch`; combining it with explicit `mps` is an error. Explicit backend settings cannot be combined with `use_cuda=True`. Existing CUDA selection remains available under `auto`. CPU execution does not import the Metal implementation or compile shaders. The `torch` path on MPS is distinct from PyTorch's optional unsupported-operator CPU fallback, which is disabled in validation.

## Training and precision

Keep parameters in FP32 for AdamW. Autocast can use FP16 or BF16 activations while the scan retains FP32 recurrent states and sensitive reductions:

```python
import torch
import torch.nn.functional as F
from mambapy.lm import LM
from mambapy.mamba import MambaConfig

assert torch.backends.mps.is_available()
model = LM(MambaConfig(128, 2, backend="mps"), vocab_size=8192).to("mps")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
tokens = torch.randint(0, 8192, (2, 512), device="mps")
targets = tokens.roll(-1, dims=1)

optimizer.zero_grad(set_to_none=True)
with torch.autocast("mps", dtype=torch.bfloat16):
    logits = model(tokens)
    loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
loss.backward()
optimizer.step()
```

Remove the autocast context for FP32. Explicit `model.to(torch.float16)` and `model.to(torch.bfloat16)` are also supported; training with explicitly converted parameters is tested with SGD. FP16 AdamW parameters are not the validated training recipe. Outputs preserve activation dtype, while recurrent caches remain FP32. Projection/convolution bias options and inner normalization are covered.

FP32 is the fastest measured cached-generation precision on this machine. Explicit BF16 conversion reduces parameter and convolution-history storage; recurrent states remain FP32. Mamba decoding is slower in both BF16 autocast and converted-weight measurements. Choose precision using complete-model measurements rather than assuming a smaller dtype is faster.

## Stateful chunks and checkpointing

Reuse the model's existing cache APIs:

```python
from mambapy.mamba import Mamba, MambaConfig

config = MambaConfig(128, 2, backend="mps")
core = Mamba(config).to("mps")
x = torch.randn(2, 513, config.d_model, device="mps", requires_grad=True)
caches = [(None, x.new_zeros(2, config.d_inner, config.d_conv - 1))
          for _ in core.layers]
outputs, caches = core.checkpointed_chunk_step(x, caches, segment_size=128)
loss = outputs.square().mean() + sum(h.square().mean() for h, _ in caches)
loss.backward()
```

`chunk_step(x, caches)` processes an arbitrary chunk; `step(x, caches)` accepts one token shaped `(batch, width)`. Keep states attached to preserve gradients across chunk boundaries. `checkpointed_chunk_step` reuses `checkpointed_sequence`, including the original autocast settings during backward recomputation. Uneven chunks, nonzero initial states, and losses on final caches are supported.

As in the existing API, model execution updates the cache **list** in place. Retain a separate list when comparing alternative executions. The internal compact scan returns outputs plus final state. `ssm_chunk_step` still returns every state by default for callers that request that behavior.

## Generation

`LM.generate` accepts token tensors and returns only the newly generated tokens:

```python
prompt = torch.randint(0, 8192, (2, 128), device="mps")
new_tokens = model.generate(prompt, num_tokens=256, sample=False)
sampled = model.generate(prompt, num_tokens=32, sample=True,
                         top_k=40, temperature=0.8)
```

Mamba-1 prefills once, projects only the last prompt position to logits, and then processes one token per decoding update. Recurrent caches stay fixed in size. Batched and one-token prompts are supported; requesting zero tokens returns shape `(batch, 0)`. The model's original per-module train/eval flags are restored. The output returns to the prompt tensor's original device; using MPS prompts avoids that final device transfer.

`eval()` alone does not disable gradients. For direct prompt/block inference, use `torch.no_grad()` or `torch.inference_mode()`; `generate()` already does this.

## Numerical and implementation boundaries

The native path uses compact chunk boundaries instead of full-sequence state tensors. It recomputes local states during backward and differentiates all scan inputs, including initial/final state. Slowly decaying long sequences use compensated FP32 arithmetic; the A gradient at lengths >=4096 additionally uses forward sensitivities with two FP32 components. There is no hidden gradient clipping or recurrence clamp.

Regression tolerances were fixed before tuning: FP32 `rtol=1e-4, atol=1e-5`; FP16 `1e-2, 1e-3`; BF16 `5e-2, 5e-3`. Floating-point reductions, including atomic gradient accumulation, need not be bitwise identical across runs. Fake implementations match output shape, dtype, device, and strides, including dense transposed inputs. PyTorch 2.11's autograd-registration opcheck excludes MPS; independent numerical gradient tests cover actual backward execution.

This release targets **first-order Mamba-1** execution. Higher-order derivatives and full `torch.compile` optimization are follow-on work. Mamba-2, end-to-end Jamba/Vision Mamba validation, accelerated HF Transformers integration, quantization, and distributed training need separate work. In particular, HF callers of the unchanged public `pscan` do not automatically reach the compact Metal operator.

## Validation and distributions

```sh
MAMBAPY_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests -q
.venv/bin/python setup.py sdist bdist_wheel
```

The packaging tests build both formats using existing setuptools, inspect both Metal files, and run CPU and actual MPS smoke tests from the unpacked wheel outside the source checkout. A missing MPS device fails the release check rather than being counted as a successful skipped GPU run. No package installation occurs in those tests.
