import json
from unittest.mock import patch
import torch
from benchmarks.mps import measure
from mambapy.mamba import Mamba,MambaConfig
from mambapy.mps.scan import forward_raw


def main():
    assert torch.backends.mps.is_available()
    for batch,length,width in ((2,64,128),(2,512,128),(2,2048,128),(2,512,384),(1,2048,384)):
        model=Mamba(MambaConfig(width,1,backend='mps')).to('mps').eval()
        x=torch.randn(batch,length,width,device='mps')
        row=dict(batch=batch,length=length,width=width)
        with torch.inference_mode():
            row['serial']=measure(lambda:model(x),warmup=10,iterations=5)['median_ms']
            for chunk in (64,128,256):
                with patch('mambapy.mps.scan._inference_op',side_effect=lambda *a:forward_raw(*a,chunk_size=chunk)[:2]):
                    row[str(chunk)]=measure(lambda:model(x),warmup=10,iterations=5)['median_ms']
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
