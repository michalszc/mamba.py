import pytest
import torch
import torch.nn.functional as F
from mambapy.mps.conv import conv_silu,_conv

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Real MPS required')


@pytest.mark.parametrize('length,window',[(1,1),(1,4),(3,9),(65,4),(257,9)])
@pytest.mark.parametrize('bias_enabled',[False,True])
def test_conv_gradients(length,window,bias_enabled):
    torch.manual_seed(3)
    args=[torch.randn(2,length,8),torch.randn(8,1,window),torch.randn(8),torch.randn(2,8,window-1)]
    cpu=[t.double().requires_grad_() for t in args]
    gpu=[t.to('mps').requires_grad_() for t in args]
    full=torch.cat((cpu[3],cpu[0].transpose(1,2)),dim=2)
    expected=F.silu(F.conv1d(full,cpu[1],cpu[2] if bias_enabled else None,groups=8)).transpose(1,2)
    got=conv_silu(gpu[0],gpu[1],gpu[2] if bias_enabled else None,gpu[3])
    torch.testing.assert_close(got.cpu(),expected.float(),rtol=1e-4,atol=1e-5)
    weight=torch.randn_like(args[0])
    (expected*weight.double()).sum().backward()
    (got*weight.to('mps')).sum().backward()
    for a,b in zip(gpu,cpu):
        if b.grad is not None:
            torch.testing.assert_close(a.grad.cpu(),b.grad.float(),rtol=1e-4,atol=1e-5)


def test_conv_views_and_metadata():
    x=torch.randn(2,13,16,device='mps')[...,::2].requires_grad_()
    w=torch.randn(8,1,9,device='mps',requires_grad=True)
    bias=torch.randn(8,device='mps',requires_grad=True)
    torch.library.opcheck(_conv,(x,w,bias),test_utils=('test_schema','test_faketensor'))
