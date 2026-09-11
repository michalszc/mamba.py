from unittest.mock import patch
import pytest
import torch
from mambapy.lm import LM
from mambapy.mamba import MambaConfig


@pytest.fixture(params=['cpu','mps'])
def model(request):
    if request.param=='mps' and not torch.backends.mps.is_available():
        pytest.skip('Real MPS required')
    torch.manual_seed(4)
    return LM(MambaConfig(32,2),64).to(request.param)


@pytest.mark.parametrize('batch,length',[(1,1),(2,17)])
def test_greedy_and_cache_execution(model,batch,length):
    prompt=torch.randint(0,64,(batch,length),device=model.embedding.weight.device)
    with torch.inference_mode():
        history=prompt
        for _ in range(5):
            token=model(history)[:,-1].argmax(-1,keepdim=True)
            history=torch.cat((history,token),1)
    model.eval()
    with patch.object(model.mamba,'chunk_step',wraps=model.mamba.chunk_step) as prefill, patch.object(model.mamba,'step',wraps=model.mamba.step) as step:
        got=model.generate(prompt,5,sample=False)
        assert prefill.call_count==1
        assert step.call_count==4
        assert all(call.args[0].shape==(batch,32) for call in step.call_args_list)
    assert not model.training
    torch.testing.assert_close(got,history[:,-5:])


def test_sampling_zero_and_mode_restore(model):
    prompt=torch.ones(2,1,dtype=torch.long,device=model.embedding.weight.device)
    model.train()
    model.norm_f.eval()
    modes=[m.training for m in model.modules()]
    assert model.generate(prompt,0).shape==(2,0)
    generated=model.generate(prompt,3,sample=True,top_k=100,temperature=.7)
    assert generated.shape==(2,3)
    assert torch.all((generated>=0)&(generated<64))
    assert [m.training for m in model.modules()]==modes
    with pytest.raises(ValueError):
        model.generate(prompt,-1)
    with pytest.raises(ValueError):
        model.generate(prompt,1,top_k=0)
    with pytest.raises(ValueError):
        model.generate(prompt[:,:0],1)
