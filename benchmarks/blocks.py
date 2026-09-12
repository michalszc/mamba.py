"""Controlled FP32 forward comparison, optionally including audited PR #78."""
import argparse
import copy
import importlib.util
import json
import statistics
import time
from pathlib import Path
import sys
import torch
from mambapy.mamba import Mamba,MambaConfig


def compare(models,x,warmup=25,repeats=15,iterations=5):
    """Warm every model, then rotate order to distribute clock/thermal bias."""
    first={}
    for name,model in models.items():
        torch.mps.synchronize()
        start=time.perf_counter()
        model(x)
        torch.mps.synchronize()
        first[name]=(time.perf_counter()-start)*1000
    for _ in range(warmup):
        for model in models.values():
            model(x)
    torch.mps.synchronize()
    samples={name:[] for name in models}
    names=list(models)
    for repeat in range(repeats):
        for name in names[repeat%len(names):]+names[:repeat%len(names)]:
            start=time.perf_counter()
            for _ in range(iterations):
                models[name](x)
            torch.mps.synchronize()
            samples[name].append((time.perf_counter()-start)*1000/iterations)
    return {name:dict(first_ms=first[name],median_ms=statistics.median(values),
                      p95_ms=sorted(values)[int(.95*len(values))],samples_ms=values)
            for name,values in samples.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pr-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert torch.backends.mps.is_available()
    pr=None
    if args.pr_root:
        sys.path.insert(0,str(args.pr_root/'pr78-build'))
        spec=importlib.util.spec_from_file_location('audited_pr78',args.pr_root/'pr78-review/mambapy/mamba.py')
        pr=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=pr
        spec.loader.exec_module(pr)
        assert pr._USE_METAL_SSM and pr._USE_METAL_CONV
    rows=[]
    for batch,length,width in ((2,64,128),(2,512,128),(2,2048,128),(2,512,384),(1,2048,384)):
        torch.manual_seed(72)
        native=Mamba(MambaConfig(width,1,backend='mps')).to('mps').eval()
        ref=copy.deepcopy(native)
        ref.config.backend='torch'
        models={'mps':native,'torch':ref}
        if pr:
            old=pr.Mamba(pr.MambaConfig(width,1)).to('mps').eval()
            old.load_state_dict(native.state_dict(),strict=True)
            models['pr78']=old
        x=torch.randn(batch,length,width,device='mps')
        row=dict(batch=batch,length=length,width=width)
        with torch.inference_mode():
            expected=ref(x)
            for name,model in models.items():
                torch.testing.assert_close(model(x),expected,rtol=1e-4,atol=1e-5)
            row.update(compare(models,x))
        if pr:
            row['pr_ratio']=row['mps']['median_ms']/row['pr78']['median_ms']
            row['pr_gate_pass']=row['pr_ratio']<=1.1
        rows.append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(torch=torch.__version__,
            protocol='25 warmups per model, 15 rotating-order samples of 5 forwards, synchronized',rows=rows),indent=2)+'\n')
        print(json.dumps({k:v if not isinstance(v,dict) else v['median_ms'] for k,v in row.items()}),flush=True)


if __name__=='__main__':
    main()
