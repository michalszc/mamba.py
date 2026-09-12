import torch
import pytest
from benchmarks.acceptance import prefill, decode
from benchmarks.transformer import TransformerLM
from mambapy.lm import LM
from mambapy.mamba import MambaConfig


def test_transformer_kv_cache_matches_full_history():
    torch.manual_seed(44)
    model=TransformerLM(64,2,128,64).eval()
    prompt=torch.randint(0,128,(2,7))
    with torch.inference_mode():
        expected=prompt
        for _ in range(4):
            expected=torch.cat((expected,model(expected)[:,-1].argmax(-1,keepdim=True)),1)
        actual=model.generate(prompt,4)
    torch.testing.assert_close(actual,expected[:,-4:])


@pytest.mark.parametrize('kind',['mamba','transformer'])
def test_decode_benchmark_matches_generation(kind):
    torch.manual_seed(12)
    model=(LM(MambaConfig(64,2,backend='torch'),128) if kind=='mamba'
           else TransformerLM(64,2,128,64)).eval()
    tokens=torch.randint(0,128,(2,7))
    with torch.inference_mode():
        logits,caches=prefill(model,tokens,11)
        actual=decode(model,logits,caches,7,4)
        torch.testing.assert_close(actual,model.generate(tokens,4,sample=False))
