import copy
import pytest
import torch
from mambapy.mamba import Mamba,MambaConfig

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Real MPS required')


def initial(model,batch):
    return [(torch.randn(batch,model.config.d_inner,model.config.d_state,device='mps',requires_grad=True),
             torch.randn(batch,model.config.d_inner,model.config.d_conv-1,device='mps',requires_grad=True)) for _ in model.layers]


@pytest.mark.parametrize('window',[1,4,9])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_stateful_outputs_and_gradients(window,checkpoint):
    torch.manual_seed(33)
    a=Mamba(MambaConfig(16,2,d_state=7,d_conv=window,backend='mps')).to('mps')
    b=copy.deepcopy(a)
    b.config.backend='torch'
    ca=initial(a,2)
    cb=[tuple(t.detach().clone().requires_grad_() for t in pair) for pair in ca]
    initial_a=[t for pair in ca for t in pair]
    initial_b=[t for pair in cb for t in pair]
    xa=torch.randn(2,65,16,device='mps',requires_grad=True)
    xb=xa.detach().clone().requires_grad_()
    if checkpoint:
        ya,ha=a.checkpointed_chunk_step(xa,ca,17)
    else:
        pieces=[]
        ha=list(ca)
        for piece in xa.split(17,dim=1):
            y,ha=a.chunk_step(piece,ha)
            pieces.append(y)
        ya=torch.cat(pieces,dim=1)
    yb,hb=b.chunk_step(xb,list(cb))
    def loss(y,states):
        return y.square().mean()+sum(t.square().sum()*.01 for pair in states for t in pair)
    loss(ya,ha).backward()
    loss(yb,hb).backward()
    torch.testing.assert_close(ya,yb,rtol=1e-4,atol=1e-5)
    for pa,pb in zip(ha,hb):
        for ta,tb in zip(pa,pb):
            torch.testing.assert_close(ta,tb,rtol=1e-4,atol=1e-5)
    aa=[xa]+list(a.parameters())+initial_a
    bb=[xb]+list(b.parameters())+initial_b
    for ta,tb in zip(aa,bb):
        assert ta.grad is not None and tb.grad is not None
        torch.testing.assert_close(ta.grad,tb.grad,rtol=1e-4,atol=1e-5)


def test_steps_match_chunk():
    model=Mamba(MambaConfig(16,2,backend='mps')).to('mps')
    x=torch.randn(2,17,16,device='mps')
    state=initial(model,2)
    with torch.no_grad():
        expected,end=model.chunk_step(x,list(state))
        ys=[]
        for token in x.unbind(1):
            y,state=model.step(token,state)
            ys.append(y)
    torch.testing.assert_close(torch.stack(ys,1),expected,rtol=1e-4,atol=1e-5)
    for a,b in zip(state,end):
        for aa,bb in zip(a,b):
            torch.testing.assert_close(aa,bb,rtol=1e-4,atol=1e-5)
