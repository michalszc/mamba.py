import copy
import pytest
import torch
from mambapy.mamba import Mamba,MambaConfig
from mambapy.lm import LM
from mambapy.mps.scan import scan,_scan_op
from mambapy.mps.conv import _conv
from mambapy.scan import reference_scan

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Real MPS required')
TOL={torch.float16:dict(rtol=1e-2,atol=1e-3),torch.bfloat16:dict(rtol=5e-2,atol=5e-3)}


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_scan_dtype_and_gradients(dtype):
    torch.manual_seed(41)
    args=[torch.randn(2,65,8,dtype=dtype),torch.rand(2,65,8)*.1,
          -torch.rand(8,7),torch.randn(2,65,7,dtype=dtype),torch.randn(2,65,7,dtype=dtype),
          torch.randn(8),torch.randn(2,8,7)]
    gpu=[t.to('mps').requires_grad_() for t in args]
    ref=[t.double().requires_grad_() for t in args]
    y,h=scan(*gpu)
    yr,hr=reference_scan(*ref)
    assert y.dtype==dtype and h.dtype==torch.float32
    torch.testing.assert_close(y.cpu().float(),yr.float(),**TOL[dtype])
    torch.testing.assert_close(h.cpu(),hr.float(),rtol=1e-4,atol=1e-5)
    gy,gh=torch.randn_like(args[0]),torch.randn_like(args[-1])
    actual=torch.autograd.grad((y,h),gpu,(gy.to('mps'),gh.to('mps')))
    expected=torch.autograd.grad((yr,hr),ref,(gy.double(),gh.double()))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a.cpu().float(),b.float(),**TOL[dtype])
    torch.library.opcheck(_scan_op,tuple(gpu)+(128,),test_utils=('test_schema','test_faketensor'))
    w=torch.randn(8,1,4,device='mps',dtype=dtype,requires_grad=True)
    bias=torch.randn(8,device='mps',dtype=dtype,requires_grad=True)
    torch.library.opcheck(_conv,(gpu[0],w,bias),test_utils=('test_schema','test_faketensor'))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('autocast',[False,True])
@pytest.mark.parametrize('inner_norm,bias',[(False,True),(True,False)])
def test_model_precision(dtype,autocast,inner_norm,bias):
    torch.manual_seed(52)
    cfg=MambaConfig(32,2,backend='mps',bias=bias,conv_bias=bias,inner_layernorms=inner_norm)
    native=LM(cfg,128).to('mps')
    if not autocast:
        native=native.to(dtype)
    ref=copy.deepcopy(native)
    ref.config.backend='torch'
    tokens=torch.randint(0,128,(2,65),device='mps')
    with torch.autocast('mps',dtype=dtype,enabled=autocast):
        y=native(tokens)
        expected=ref(tokens)
        loss=y.float().square().mean()
        ref_loss=expected.float().square().mean()
    torch.testing.assert_close(y,expected,**TOL[dtype])
    loss.backward(); ref_loss.backward()
    for a,b in zip(native.parameters(),ref.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad,b.grad,**TOL[dtype])
    # Explicit low-precision parameters need a numerically suitable optimizer;
    # autocast training uses normal FP32 master parameters with AdamW.
    optimizer=torch.optim.AdamW(native.parameters(),lr=1e-4) if autocast else torch.optim.SGD(native.parameters(),lr=1e-3)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in native.parameters())
    with torch.autocast('mps',dtype=dtype,enabled=autocast):
        out=native.generate(tokens[:,:3],3,sample=False)
        caches=[(None,native.embedding.weight.new_zeros(2,64,3)) for _ in native.mamba.layers]
        hidden=native.embedding(tokens)
        yc,states=native.mamba.checkpointed_chunk_step(hidden,caches,17)
        checkpoint_loss=yc.float().square().mean()
    checkpoint_loss.backward()
    assert out.shape==(2,3)
    assert all(h.dtype==torch.float32 for h,_ in states)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_checkpoint_recomputation_preserves_autocast(dtype):
    model=Mamba(MambaConfig(16,1,backend='mps')).to('mps')
    seen=[]
    hook=model.layers[0].mixer.in_proj.register_forward_hook(lambda module,args,out: seen.append(out.dtype))
    try:
        x=torch.randn(1,33,16,device='mps',requires_grad=True)
        caches=[(None,torch.zeros(1,32,3,device='mps'))]
        with torch.autocast('mps',dtype=dtype):
            y,_=model.checkpointed_chunk_step(x,caches,17)
            loss=y.square().mean()
        loss.backward()
        assert len(seen)>=4
        assert all(d==dtype for d in seen),seen
    finally:
        hook.remove()
