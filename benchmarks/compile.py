"""Measure shader creation separately from execution (driver caches may persist)."""
import json
import time
import torch
from mambapy.mps.scan import shaders as scan_shaders
from mambapy.mps.conv import shaders as conv_shaders


def main():
    assert torch.backends.mps.is_available(), 'Real MPS required'
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for chunk in (64, 128, 256):
            start = time.perf_counter()
            lib = scan_shaders(16, dtype, chunk)
            library_ms = (time.perf_counter()-start)*1000
            # Resolve every pipeline too: library creation alone can be lazy.
            for name in ('summaries','boundaries','output_scan','reverse_summaries',
                         'reverse_boundaries','scan_gradients','reduce_A','long_A_gradient'):
                assert getattr(lib,name).max_threads_per_threadgroup > 0
            print(json.dumps(dict(operator='scan', state_capacity=16, chunk=chunk,
                                  dtype=str(dtype), library_ms=library_ms,
                                  compile_ms=(time.perf_counter()-start)*1000)), flush=True)
        start = time.perf_counter()
        lib = conv_shaders(dtype)
        library_ms = (time.perf_counter()-start)*1000
        for name in ('conv_forward','conv_activation_grad','conv_input_grad','conv_weight_grad'):
            assert getattr(lib,name).max_threads_per_threadgroup > 0
        print(json.dumps(dict(operator='convolution', dtype=str(dtype),
                              library_ms=library_ms,
                              compile_ms=(time.perf_counter()-start)*1000)), flush=True)


if __name__ == '__main__':
    main()
