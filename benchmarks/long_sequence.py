"""Measure the long-sequence accuracy specialization on a complete block."""
import json
import torch
from benchmarks.mps import measure
from mambapy.mamba import Mamba, MambaConfig


def main():
    assert torch.backends.mps.is_available()
    torch.mps.set_per_process_memory_fraction(min(1.0,16*2**30/torch.mps.recommended_max_memory()))
    torch.manual_seed(72)
    model=Mamba(MambaConfig(128,1,d_state=16,backend='mps')).to('mps')
    x=torch.randn(2,8192,128,device='mps')
    for backend in ('mps','torch'):
        model.config.backend=backend
        row=dict(batch=2,length=8192,width=128,backend=backend)
        with torch.inference_mode():
            row['forward']=measure(lambda:model(x),repeats=5)
        def backward():
            model.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
        row['forward_backward']=measure(backward,repeats=5)
        print(json.dumps(row),flush=True)
        model.zero_grad(set_to_none=True)
        torch.mps.empty_cache()


if __name__=='__main__':main()
