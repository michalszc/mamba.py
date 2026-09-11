import pytest
import torch
from mambapy.mps.scan import forward_raw
from mambapy.mps.scan import scan, _scan_op
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


@pytest.mark.parametrize('length,state',[(1,1),(3,3),(3,16),(65,32),(257,64),(2048,16)])
def test_backward(length,state):
    args = data(length,state)
    ref = [t.double().requires_grad_() for t in args]
    gpu = [t.to('mps').requires_grad_() for t in args]
    y,h = scan(*gpu)
    yr,hr = reference_scan(*ref)
    torch.manual_seed(6)
    gy,gh = torch.randn_like(args[0]),torch.randn_like(args[-1])
    expected = torch.autograd.grad((yr,hr),ref,(gy.double(),gh.double()))
    actual = torch.autograd.grad((y,h),gpu,(gy.to('mps'),gh.to('mps')))
    for name,a,b in zip(('x','delta','A','B','C','D','initial'),actual,expected):
        torch.testing.assert_close(a.cpu(),b.float(),rtol=1e-4,atol=1e-5,msg=lambda s: name+': '+s)


def test_analytical_and_scaled_gradients():
    x = torch.ones(1,3,1,device='mps',requires_grad=True)
    dt = torch.ones_like(x)
    A = torch.full((1,1),-0.6931471805599453,device='mps',requires_grad=True)
    bc = torch.ones(1,3,1,device='mps')
    skip = torch.zeros(1,device='mps')
    y,_ = scan(x,dt,A,bc,bc,skip)
    gx,ga = torch.autograd.grad(y.sum()*100000,(x,A))
    torch.testing.assert_close(gx.cpu(),torch.tensor([[[175000.],[150000.],[100000.]]]))
    torch.testing.assert_close(ga.cpu(),torch.tensor([[150000.]]))


def test_registration():
    args = [t.to('mps').requires_grad_() for t in data(3,2)]
    # PyTorch 2.11's autograd_registration_check explicitly excludes MPS.
    # Gradients are checked numerically above, including every input.
    torch.library.opcheck(_scan_op,tuple(args)+(128,),
                          test_utils=('test_schema','test_faketensor'))


def test_full_model_training():
    import copy
    from mambapy.mamba import Mamba,MambaConfig
    torch.manual_seed(17)
    native = Mamba(MambaConfig(32,2,backend='mps')).to('mps')
    ref = copy.deepcopy(native)
    ref.config.backend = 'torch'
    # Shared configs in copied layers retain the copied backend configuration.
    opt = torch.optim.AdamW(native.parameters(),lr=1e-4)
    opt_ref = torch.optim.AdamW(ref.parameters(),lr=1e-4)
    for _ in range(3):
        x = torch.randn(2,65,32,device='mps')
        opt.zero_grad(set_to_none=True)
        opt_ref.zero_grad(set_to_none=True)
        native(x).square().mean().backward()
        ref(x).square().mean().backward()
        for a,b in zip(native.parameters(),ref.parameters()):
            assert a.grad is not None and torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
        opt.step(); opt_ref.step()
    for a,b in zip(native.parameters(),ref.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)


def test_final_state_loss_and_accumulation():
    args = [t.to('mps').requires_grad_() for t in data(65,7)]
    args[2].requires_grad_(False)
    _,h = scan(*args)
    h.square().sum().backward(retain_graph=True)
    first = args[0].grad.clone()
    h.square().sum().backward()
    torch.testing.assert_close(args[0].grad,2*first,rtol=1e-4,atol=1e-5)
    assert args[2].grad is None
