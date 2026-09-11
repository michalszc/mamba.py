import copy
import os

import pytest
import torch

from mambapy.mamba import Mamba, MambaConfig


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        if os.environ.get('MAMBAPY_REQUIRE_MPS') == '1':
            pytest.fail('MPS required but unavailable: run with GPU access')
        pytest.skip('MPS unavailable')
    return request.param


def caches(model, batch, device):
    return [(None, torch.zeros(batch, model.config.d_inner, model.config.d_conv - 1,
                               device=device)) for _ in model.layers]


@pytest.mark.parametrize('length,state,width', [(1, 1, 1), (3, 16, 4), (65, 32, 9), (257, 64, 4)])
def test_chunk_gradients(device, length, state, width):
    torch.manual_seed(19)
    reference = Mamba(MambaConfig(16, 2, d_state=state, d_conv=width)).to(device)
    chunked = copy.deepcopy(reference)
    x = torch.randn(1, length, 16, device=device, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y = reference(x)
    state_cache = caches(chunked, 1, device)
    pieces = []
    for segment in x2.split(17, dim=1):
        piece, state_cache = chunked.chunk_step(segment, state_cache)
        pieces.append(piece)
    y2 = torch.cat(pieces, dim=1)
    torch.testing.assert_close(y2, y, rtol=1e-4, atol=1e-5)
    weight = torch.randn_like(y)
    (y * weight).mean().backward()
    (y2 * weight).mean().backward()
    torch.testing.assert_close(x2.grad, x.grad, rtol=1e-4, atol=1e-5)
    for a, b in zip(reference.parameters(), chunked.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-5)


def test_checkpointing_retains_less_storage():
    # This measures tensors retained by autograd, not process/GPU peak memory.
    model = Mamba(MambaConfig(32, 2))
    x = torch.randn(1, 257, 32)
    def retained(checkpoint):
        storages = {}
        def pack(t):
            s = t.untyped_storage()
            storages[s.data_ptr()] = s.nbytes()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            if checkpoint:
                y, _ = model.checkpointed_chunk_step(x, caches(model, 1, 'cpu'), 32)
            else:
                y = model(x)
        assert y.shape == x.shape
        return sum(storages.values())
    assert retained(True) < retained(False)
