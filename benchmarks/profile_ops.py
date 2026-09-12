import argparse
import json
import sys
from pathlib import Path
import torch
from benchmarks.mps import measure
from mambapy.mamba import Mamba,MambaConfig
from mambapy.mps.conv import conv_silu
from mambapy.mps.scan import scan


def main():
    p=argparse.ArgumentParser();p.add_argument('--pr-root',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.pr_root/'pr78-build'))
    import metal_pscan.torch_ops
    for batch,length,width in ((2,512,128),(2,512,384),(1,2048,384)):
        m=Mamba(MambaConfig(width,1,backend='mps')).to('mps').layers[0].mixer
        x=torch.randn(batch,length,width,device='mps')
        with torch.inference_mode():
            raw,z=m.in_proj(x).chunk(2,-1)
            activated=conv_silu(raw,m.conv1d.weight,m.conv1d.bias)
            delta,B,C=m.x_proj(activated).split((m.config.dt_rank,16,16),-1)
            delta=torch.nn.functional.softplus(m.dt_proj(delta))
            A=-m.A_log.exp();D=m.D
            ops={
                'native_conv':lambda:conv_silu(raw,m.conv1d.weight,m.conv1d.bias),
                'pr_conv':lambda:torch.ops.metal_pscan.conv1d_silu(raw.transpose(1,2),m.conv1d.weight,m.conv1d.bias),
                'native_scan':lambda:scan(activated,delta,A,B,C,D),
                'pr_scan':lambda:torch.ops.metal_pscan.ssm_output_fused(delta,A,B,activated,C,D),
            }
            row=dict(batch=batch,length=length,width=width)
            for name,fn in ops.items():
                row[name]=measure(fn,warmup=10,iterations=5)['median_ms']
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
