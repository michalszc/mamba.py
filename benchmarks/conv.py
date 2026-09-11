"""Compare native and PyTorch convolution inside the same MPS Mamba block."""
import json
from unittest.mock import patch
import torch
from benchmarks.mps import measure
from mambapy.mamba import Mamba,MambaConfig


def main():
    assert torch.backends.mps.is_available()
    for length in (64,512,2048):
        torch.manual_seed(8)
        model=Mamba(MambaConfig(128,1,backend='mps')).to('mps')
        x=torch.randn(2,length,128,device='mps')
        row={'length':length}
        for native in (False,True):
            with patch('mambapy.mps.conv.eligible',return_value=native):
                with torch.inference_mode():
                    row[str(native)+'_forward']=measure(lambda:model(x))['median_ms']
                def backward():
                    model.zero_grad(set_to_none=True)
                    model(x).square().mean().backward()
                row[str(native)+'_training']=measure(backward)['median_ms']
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
