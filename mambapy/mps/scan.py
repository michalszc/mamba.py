from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=8)
def shaders(capacity):
    source = Path(__file__).with_suffix('.metal').read_text()
    return torch.mps.compile_shader(source.replace('STATE_CAPACITY', str(capacity)))


def validate(x, delta, A, B, C, D, initial_state):
    if x.ndim != 3 or A.ndim != 2:
        raise ValueError('Expected x: BLD and A: DN')
    batch, length, width = x.shape
    state = A.size(1)
    expected = [(batch,length,width),(width,state),(batch,length,state),(batch,length,state),(width,),(batch,width,state)]
    for tensor, shape in zip((delta,A,B,C,D,initial_state),expected):
        if tuple(tensor.shape) != shape:
            raise ValueError(f'Expected shape {shape}, got {tuple(tensor.shape)}')
    for tensor in (x,delta,A,B,C,D,initial_state):
        if tensor.device != x.device or tensor.device.type != 'mps':
            raise ValueError('All scan tensors must be on the same MPS device')
        if tensor.dtype != torch.float32:
            raise TypeError('The FP32 scan requires float32 tensors')
    if not (batch > 0 and length > 0 and width > 0 and 1 <= state <= 128):
        raise ValueError('Positive dimensions and state size 1..128 required')


def forward_raw(x, delta, A, B, C, D, initial_state, chunk_size=128):
    """Internal forward only; use the registered operation for autograd."""
    validate(x,delta,A,B,C,D,initial_state)
    if chunk_size not in (64,128,256):
        raise ValueError('chunk_size must be 64, 128, or 256')
    x,delta,A,B,C,D,initial_state = [t.contiguous() for t in (x,delta,A,B,C,D,initial_state)]
    batch,length,width = x.shape
    state = A.size(1)
    chunks = (length+chunk_size-1)//chunk_size
    lib = shaders(1 << (state-1).bit_length())
    alpha = x.new_empty(batch,chunks,width,state)
    beta = torch.empty_like(alpha)
    states = x.new_empty(batch,chunks+1,width,state)
    y = torch.empty_like(x)
    lib.summaries(x,delta,A,B,alpha,beta,length,width,state,chunks,chunk_size,
                  threads=alpha.numel())
    lib.boundaries(alpha,beta,initial_state,states,width,state,chunks,
                   threads=batch*width*state)
    lib.output_scan(x,delta,A,B,C,D,states,y,length,width,state,chunks,chunk_size,
                    threads=batch*chunks*width)
    return y, states[:,-1].contiguous(), states
