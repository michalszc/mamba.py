"""Compare bounded Metal threadgroup sizes on complete MPS blocks."""
import json
import importlib
import torch
from benchmarks.blocks import compare
from mambapy.mamba import Mamba, MambaConfig


def main():
    assert torch.backends.mps.is_available()
    scan_module=importlib.import_module('mambapy.mps.scan')
    for batch,length,width in ((2,64,128),(2,512,128),(2,2048,128),(2,512,384),(1,2048,384)):
        torch.manual_seed(72)
        model=Mamba(MambaConfig(width,1,backend='mps')).to('mps').eval()
        x=torch.randn(batch,length,width,device='mps')
        row=dict(batch=batch,length=length,width=width)
        with torch.inference_mode():
            functions={}
            for group in (64,128,256,512,1024):
                def run(inputs,group=group):
                    scan_module.OUTPUT_GROUP_SIZE=group
                    return model(inputs)
                functions[str(group)]=run
            previous=scan_module.OUTPUT_GROUP_SIZE
            try:
                row.update(compare(functions,x))
            finally:
                scan_module.OUTPUT_GROUP_SIZE=previous
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
