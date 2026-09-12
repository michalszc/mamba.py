import json
from unittest.mock import patch
import torch
from benchmarks.mps import measure
from mambapy.mamba import Mamba,MambaConfig


def main():
    assert torch.backends.mps.is_available()
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        for batch,length,width in ((2,64,128),(2,512,128),(2,2048,128),(2,512,384),(1,2048,384)):
            torch.manual_seed(72)
            model=Mamba(MambaConfig(width,1,backend='mps')).to(device='mps',dtype=dtype)
            x=torch.randn(batch,length,width,device='mps',dtype=dtype)
            for chunk in (64,128,256):
                with patch('mambapy.mps.scan.select_chunk',return_value=chunk):
                    # Training forward must retain grad mode: no-grad inference
                    # intentionally uses the recurrent specialization, not tiles.
                    with torch.enable_grad():
                        f=measure(lambda:model(x),repeats=7)['median_ms']
                    def backward():
                        model.zero_grad(set_to_none=True)
                        model(x).float().square().mean().backward()
                    b=measure(backward,repeats=7)['median_ms']
                print(json.dumps(dict(dtype=str(dtype),batch=batch,length=length,width=width,chunk=chunk,training_forward_ms=f,backward_ms=b)),flush=True)


if __name__=='__main__':
    main()
