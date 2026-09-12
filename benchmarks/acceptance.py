"""Model-level acceptance benchmarks, with synthetic data and no downloads."""
import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.mps import measure
from benchmarks.transformer import TransformerLM
from mambapy.lm import LM
from mambapy.mamba import MambaConfig


def prefill(model,tokens,capacity=None):
    if isinstance(model,TransformerLM):
        return model.prefill(tokens,capacity)
    states=[(None,model.embedding.weight.new_zeros(tokens.size(0),model.config.d_inner,model.config.d_conv-1)) for _ in model.mamba.layers]
    x,states=model.mamba.chunk_step(model.embedding(tokens),states)
    return model._project_logits(x[:,-1]),states


@torch.inference_mode()
def uncached(model,tokens,count):
    history=tokens
    for _ in range(count):
        token=model(history)[:,-1].argmax(-1,keepdim=True)
        history=torch.cat((history,token),dim=1)
    return history[:,-count:]


def decode(model,logits,caches,start,count):
    """Greedy decoding from an existing prefill, using the model's cache APIs."""
    out=torch.empty(logits.size(0),count,device=logits.device,dtype=torch.long)
    for i in range(count):
        token=logits.argmax(-1,keepdim=True)
        out[:,i:i+1]=token
        if i+1<count:
            if isinstance(model,TransformerLM):
                x,caches=model.hidden(token,caches,start+i)
                logits=model.lm_head(model.norm(x[:,0]))
            else:
                x,caches=model.mamba.step(model.embedding(token[:,0]),caches)
                logits=model._project_logits(x)
    return out


def measure_decode(model,tokens,count,repeats):
    samples=[]
    for i in range(repeats+1):
        logits,caches=prefill(model,tokens,tokens.size(1)+count)
        torch.mps.synchronize()
        start=time.perf_counter()
        decode(model,logits,caches,tokens.size(1),count)
        torch.mps.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        if i:
            samples.append(elapsed)
    return dict(median_ms=statistics.median(samples),p95_ms=sorted(samples)[int(.95*len(samples))],
                samples_ms=samples,generated_tokens=count,recurrent_updates=count-1,
                description='Prefill excluded; includes selection of the first token and subsequent cached updates')


def sampled_memory(fn):
    """Separate instrumented run: sampled lower bound, never an exact peak."""
    stop=threading.Event()
    values=[]
    def sample():
        while not stop.is_set():
            values.append((torch.mps.current_allocated_memory(),torch.mps.driver_allocated_memory()))
            stop.wait(.005)
    worker=threading.Thread(target=sample,daemon=True)
    worker.start()
    try:
        fn();torch.mps.synchronize()
    finally:
        stop.set();worker.join()
    return dict(sampled_live_max_bytes=max((v[0] for v in values),default=0),
                sampled_driver_max_bytes=max((v[1] for v in values),default=0),
                process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                sampling_interval_ms=5,method='sampled lower bound; allocator limited to 16 GiB; RSS overlaps unified GPU memory')


def sustained(fn,seconds):
    samples=[]
    deadline=time.perf_counter()+seconds
    while time.perf_counter()<deadline:
        start=time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    window=max(1,len(samples)//4)
    return dict(duration_seconds=sum(samples)/1000,iterations=len(samples),
                median_ms=statistics.median(samples),p95_ms=sorted(samples)[int(.95*len(samples))],
                first_quarter_median_ms=statistics.median(samples[:window]),
                last_quarter_median_ms=statistics.median(samples[-window:]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',choices=['small','130m'],default='small')
    p.add_argument('--backend',choices=['mps','torch','transformer'],required=True)
    p.add_argument('--mode',choices=['speed','generation'],default='speed')
    p.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='float32')
    p.add_argument('--precision-mode',choices=['autocast','weights'],default='autocast')
    p.add_argument('--lengths',type=int,nargs='+',default=[512,2048])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--uncached',action='store_true')
    p.add_argument('--sustain-seconds',type=float,default=0)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.repeats<1 or args.tokens<1 or any(length<1 for length in args.lengths):
        p.error('repeats, generated tokens, and sequence lengths must be positive')
    if args.precision_mode=='weights' and args.mode!='generation':
        p.error('Converted-weight benchmarks are for inference; training uses FP32 optimizer parameters')
    assert torch.backends.mps.is_available(),'Real MPS required'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='1','Disable CPU fallback'
    torch.mps.set_per_process_memory_fraction(min(1.0,16*2**30/torch.mps.recommended_max_memory()))
    dtype=getattr(torch,args.dtype)
    width,layers,vocab=(384,6,8192) if args.size=='small' else (768,24,50280)
    torch.manual_seed(93)
    config=MambaConfig(width,layers,backend='torch' if args.backend=='transformer' else args.backend)
    model=LM(config,vocab)
    # This repository's default untied 24x768 LM is ~165M. Tie only this
    # benchmark's embedding/head to represent the ~130M parameter class.
    tied=args.size=='130m'
    if tied:
        model.lm_head.weight=model.embedding.weight
    mamba_parameters=sum(p.numel() for p in model.parameters())
    block_parameters=sum(p.numel() for p in model.mamba.layers[0].parameters())
    if args.backend=='transformer':
        ff_width=max(64,round((block_parameters-4*width*width-9*width)/(2*width+1)/64)*64)
        del model
        model=TransformerLM(width,layers,vocab,ff_width,tied)
    model=model.to(device='mps',dtype=dtype if args.precision_mode=='weights' else torch.float32)
    autocast_enabled=dtype!=torch.float32 and args.precision_mode=='autocast'
    report=dict(args={**vars(args),'output':str(args.output)},torch=torch.__version__,
                platform=platform.platform(),parameters=sum(p.numel() for p in model.parameters()),
                reference_mamba_parameters=mamba_parameters,tied_embeddings=tied,
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),rows=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    for length in args.lengths:
        row={'length':length}
        report['rows'].append(row)
        try:
            tokens=torch.randint(0,vocab,(1,length),device='mps')
            model.eval()
            if args.mode=='generation':
                with torch.inference_mode(),torch.autocast('mps',dtype=dtype,enabled=autocast_enabled):
                    fn=(lambda:uncached(model,tokens,args.tokens)) if args.uncached else (lambda:model.generate(tokens,args.tokens,sample=False))
                    row['generation']=measure(fn,warmup=1,repeats=args.repeats)
                    row['memory']=sampled_memory(fn)
                    if not args.uncached:
                        row['decode']=measure_decode(model,tokens,args.tokens,args.repeats)
            else:
                with torch.inference_mode(),torch.autocast('mps',dtype=dtype,enabled=autocast_enabled):
                    row['forward']=measure(lambda:model(tokens),repeats=args.repeats,iterations=3)
                    row['prefill']=measure(lambda:prefill(model,tokens),repeats=args.repeats,iterations=3)
                model.train()
                optimizer=(model.configure_optimizers(.01,1e-4,(.9,.95),'mps') if isinstance(model,LM)
                           else torch.optim.AdamW(model.parameters(),lr=1e-4))
                targets=tokens.roll(-1,1)
                def backward():
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('mps',dtype=dtype,enabled=autocast_enabled):
                        logits=model(tokens)
                        loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
                    loss.backward()
                row['forward_backward']=measure(backward,repeats=args.repeats)
                def training():
                    backward();optimizer.step()
                row['training']=measure(training,repeats=args.repeats)
                row['memory']=sampled_memory(training)
                if args.sustain_seconds>0:
                    row['sustained_training']=sustained(training,args.sustain_seconds)
                assert all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite parameters'
                optimizer.zero_grad(set_to_none=True)
                del optimizer
            print(json.dumps(row),flush=True)
        except RuntimeError as exc:
            if not isinstance(exc,torch.OutOfMemoryError) and 'out of memory' not in str(exc).lower():
                raise
            row['out_of_budget']=str(exc)
            print(json.dumps(row),flush=True)
        finally:
            save();gc.collect();torch.mps.empty_cache()


if __name__=='__main__':main()
