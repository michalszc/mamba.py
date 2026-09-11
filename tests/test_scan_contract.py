import pytest
import torch
from mambapy.scan import reference_scan, selective_scan
from mambapy.mamba import Mamba, MambaConfig


def sequential(x, dt, A, B, C, D, h):
    ys = []
    for t in range(x.size(1)):
        h = torch.exp(dt[:,t,:,None] * A) * h + dt[:,t,:,None]*B[:,t,None,:]*x[:,t,:,None]
        ys.append((h*C[:,t,None,:]).sum(-1) + D*x[:,t])
    return torch.stack(ys,1), h


def inputs(device='cpu', dtype=torch.float64, length=7, state=3):
    torch.manual_seed(7)
    shapes = [(2,length,4),(2,length,4),(4,state),(2,length,state),(2,length,state),(4,),(2,4,state)]
    data = [torch.randn(s,device=device,dtype=dtype) for s in shapes]
    data[1] = data[1].sigmoid()*.1
    data[2] = -data[2].abs()
    return [t.requires_grad_() for t in data]


def test_reference_gradcheck():
    args = inputs(length=3,state=2)
    assert torch.autograd.gradcheck(reference_scan,tuple(args))
    got = reference_scan(*args)
    expected = sequential(*args)
    for a,b in zip(got,expected):
        torch.testing.assert_close(a,b)


@pytest.mark.parametrize('kwargs', [dict(backend='bad'),dict(backend='mps',pscan=False),dict(backend='torch',use_cuda=True)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MambaConfig(16,1,**kwargs)


def test_checkpoint_keys_and_sequential_opt_out():
    a = Mamba(MambaConfig(16,1,backend='torch'))
    b = Mamba(MambaConfig(16,1,pscan=False))
    b.load_state_dict(a.state_dict(),strict=True)
    x = torch.randn(2,13,16)
    torch.testing.assert_close(a(x),b(x))
    with pytest.raises(ValueError,match='MPS inputs'):
        selective_scan(*inputs(),backend='mps')
