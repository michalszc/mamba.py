"""Reproducible standalone benchmark: python -m benchmarks.mps --suite blocks.

MPS allocation readings are snapshots, not peak-memory measurements. No CPU
fallback or model downloads. Compilation/first call is reported separately.
"""
import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from mambapy.mamba import Mamba, MambaConfig
from mambapy.lm import LM


def sync():
    torch.mps.synchronize()


def measure(fn, warmup=3, repeats=10, iterations=1):
    sync()
    start = time.perf_counter()
    fn()
    sync()
    first = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    ordered = sorted(samples)
    return dict(first_ms=first, median_ms=statistics.median(samples),
                p95_ms=ordered[min(len(ordered)-1, int(.95*len(ordered)))],
                samples_ms=samples, live_tensor_bytes=torch.mps.current_allocated_memory(),
                driver_bytes_snapshot=torch.mps.driver_allocated_memory())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['blocks', 'small', '130m', 'smoke'], default='blocks')
    parser.add_argument('--backend', default='torch')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--lengths', type=int, nargs='+', default=[512, 2048])
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert torch.backends.mps.is_available(), 'Real MPS access required'
    # Keep allocator within the agreed working budget; no high-watermark disabling.
    torch.mps.set_per_process_memory_fraction(min(1.0, 16*2**30 / torch.mps.recommended_max_memory()))
    dtype = getattr(torch, args.dtype)
    report = dict(torch=torch.__version__, platform=platform.platform(),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  args=vars(args), results=[])
    cases = [(2,64,128),(2,512,128),(2,2048,128),(2,512,384),(1,2048,384)]
    if args.suite != 'blocks':
        cases = [(1,L,32 if args.suite == 'smoke' else 384 if args.suite == 'small' else 768) for L in args.lengths]
    for batch,length,width in cases:
        torch.manual_seed(2026)
        layers = 1 if args.suite == 'blocks' else 2 if args.suite == 'smoke' else 6 if args.suite == 'small' else 24
        cfg = MambaConfig(width, layers)
        if hasattr(cfg, 'backend'):
            cfg.backend = args.backend
        elif args.backend != 'torch':
            raise ValueError('This revision has only the torch backend')
        model = (Mamba(cfg) if args.suite == 'blocks' else LM(cfg, 256 if args.suite == 'smoke' else 8192 if args.suite == 'small' else 50280)).to(device='mps',dtype=dtype)
        x = (torch.randn(batch,length,width,device='mps',dtype=dtype) if args.suite == 'blocks' else torch.randint(0,model.embedding.num_embeddings,(batch,length),device='mps'))
        row = dict(batch=batch,length=length,width=width,layers=layers,parameters=sum(p.numel() for p in model.parameters()))
        model.eval()
        with torch.inference_mode():
            row['forward'] = measure(lambda: model(x), repeats=args.repeats)
            if args.generate and args.suite != 'blocks':
                row['generation'] = measure(lambda: model.generate(x,args.generate,sample=False),warmup=1,repeats=args.repeats)
        model.train()
        def backward():
            model.zero_grad(set_to_none=True)
            model(x).float().square().mean().backward()
        row['forward_backward'] = measure(backward,repeats=args.repeats)
        optimizer = model.configure_optimizers(.01,1e-4,(.9,.95),'mps') if isinstance(model,LM) else torch.optim.AdamW(model.parameters(),lr=1e-4)
        def train():
            backward()
            optimizer.step()
        row['training'] = measure(train,repeats=args.repeats)
        report['results'].append(row)
        print(json.dumps(row), flush=True)
        del optimizer,model,x
        gc.collect()
        torch.mps.empty_cache()
        target = Path(args.output)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
