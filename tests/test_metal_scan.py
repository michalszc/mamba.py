import pytest
import torch
from mambapy.mps.scan import forward_raw
from mambapy.scan import reference_scan

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Real MPS required')


def data(length=7, state=16, width=8):
    torch.manual_seed(25)
    shapes = [(2,length,width),(2,length,width),(width,state),(2,length,state),(2,length,state),(width,),(2,width,state)]
    tensors = [torch.randn(s) for s in shapes]
    tensors[1] = tensors[1].sigmoid()*.1
    tensors[2] = -tensors[2].abs()
    return tensors


@pytest.mark.parametrize('length,state',[(1,1),(2,16),(3,32),(63,64),(64,128),(65,16),(255,32),(256,16),(257,64),(2048,16),(8192,16)])
@pytest.mark.parametrize('chunk',[64,128,256])
def test_forward(length,state,chunk):
    args = data(length,state)
    expected = [t.float() for t in reference_scan(*[a.double() for a in args])]
    got = forward_raw(*[a.to('mps') for a in args],chunk_size=chunk)
    for a,b in zip(got,expected):
        torch.testing.assert_close(a.cpu(),b,rtol=1e-4,atol=1e-5)


def test_views_and_offsets():
    args = data(65)
    viewed = []
    for t in args:
        storage = torch.zeros(*t.shape,2,device='mps')
        v = storage[...,1]
        v.copy_(t.to('mps'))
        viewed.append(v)
    got = forward_raw(*viewed)
    for a,b in zip(got,reference_scan(*args)):
        torch.testing.assert_close(a.cpu(),b,rtol=1e-4,atol=1e-5)


def test_invalid_state_rejected():
    with pytest.raises(ValueError,match='state size'):
        forward_raw(*[t.to('mps') for t in data(2,129)])
