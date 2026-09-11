"""Compact selective-scan contract shared by model backends.

The public pscan(A, X) primitive still returns every state. Model backends use
this contract to avoid materializing those states when only the last is needed.
"""
import torch
from mambapy.pscan import pscan


def validate_config(backend, pscan_enabled=True, use_cuda=False):
    if backend not in ('auto', 'torch', 'mps'):
        raise ValueError('backend must be auto, torch, or mps')
    if backend != 'auto' and use_cuda:
        raise ValueError('An explicit backend cannot be combined with use_cuda=True')
    if backend == 'mps' and not pscan_enabled:
        raise ValueError('backend=mps cannot be combined with pscan=False')


def reference_scan(x, delta, A, B, C, D, initial_state=None):
    """x/delta: BLD; A: DN; B/C: BLN; D: D; initial_state: BDN.

    Returns (y: BLD, final_state: BDN). Both outputs are differentiable.
    Uses the existing parallel scan and preserves its mathematical definition.
    """
    if x.size(1) == 0:
        h = initial_state if initial_state is not None else x.new_zeros(x.size(0), x.size(2), A.size(1))
        return x.clone(), h.clone()
    a = torch.exp(delta.unsqueeze(-1) * A)
    bx = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    if initial_state is not None:
        bx = torch.cat((bx[:, :1] + a[:, :1] * initial_state.unsqueeze(1), bx[:, 1:]), dim=1)
    states = pscan(a, bx)
    y = (states @ C.unsqueeze(-1)).squeeze(-1) + D * x
    return y, states[:, -1]


def selective_scan(x, delta, A, B, C, D, initial_state=None, *, backend='auto'):
    validate_config(backend)
    if backend == 'mps':
        if x.device.type != 'mps':
            raise ValueError('backend=mps requires MPS inputs')
    native = (x.device.type == 'mps' and x.dtype == torch.float32 and
              x.size(1) > 0 and 1 <= A.size(1) <= 128 and hasattr(torch.mps,'compile_shader'))
    if backend == 'mps' or (backend == 'auto' and native):
        from mambapy.mps.scan import scan
        return scan(x,delta,A,B,C,D,initial_state)
    return reference_scan(x, delta, A, B, C, D, initial_state)
