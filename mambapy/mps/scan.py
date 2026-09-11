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
    return y, states[:,-1].clone(), states


def backward_raw(x, delta, A, B, C, D, states, gy, gh, chunk_size):
    x,delta,A,B,C,D,gy,gh = [t.contiguous() for t in (x,delta,A,B,C,D,gy,gh)]
    batch,length,width = x.shape
    state = A.size(1)
    chunks = states.size(1)-1
    lib = shaders(1 << (state-1).bit_length())
    alpha = x.new_empty(batch,chunks,width,state)
    beta = torch.empty_like(alpha)
    adjoints = torch.empty_like(states)
    gx = gy*D
    gdt = torch.zeros_like(delta)
    gA = torch.empty_like(alpha)
    gB, gC = torch.zeros_like(B), torch.zeros_like(C)
    lib.reverse_summaries(delta,A,C,gy,alpha,beta,length,width,state,chunks,chunk_size,
                          threads=alpha.numel())
    lib.reverse_boundaries(alpha,beta,gh,adjoints,width,state,chunks,
                           threads=batch*width*state)
    lib.scan_gradients(x,delta,A,B,C,states,adjoints,gy,gx,gdt,gA,gB,gC,
                       length,width,state,chunks,chunk_size,
                       threads=batch*chunks*width*(1 << (state-1).bit_length()))
    return gx,gdt,gA.sum((0,1)),gB,gC,(gy*x).sum((0,1)),adjoints[:,0].clone()


@torch.library.custom_op('mambapy_mps::scan_backward', mutates_args=())
def _backward_op(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                 B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                 states: torch.Tensor, gy: torch.Tensor, gh: torch.Tensor,
                 chunk_size: int) -> list[torch.Tensor]:
    return list(backward_raw(x,delta,A,B,C,D,states,gy,gh,chunk_size))


@_backward_op.register_fake
def _backward_fake(x,delta,A,B,C,D,states,gy,gh,chunk_size):
    return [torch.empty_like(t) for t in (x,delta,A,B,C,D,gh)]


@torch.library.custom_op('mambapy_mps::scan', mutates_args=())
def _scan_op(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
             B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
             initial_state: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    return forward_raw(x,delta,A,B,C,D,initial_state,chunk_size)


@_scan_op.register_fake
def _scan_fake(x,delta,A,B,C,D,initial_state,chunk_size):
    chunks = (x.size(1)+chunk_size-1)//chunk_size
    return (torch.empty_like(x),torch.empty_like(initial_state),
            x.new_empty(x.size(0),chunks+1,x.size(2),A.size(1)))


def _setup_context(ctx,inputs,output):
    x,delta,A,B,C,D,h,chunk = inputs
    ctx.save_for_backward(x,delta,A,B,C,D,output[2])
    ctx.chunk = chunk
    ctx.mark_non_differentiable(output[2])


def _backward(ctx,gy,gh,_):
    x,delta,A,B,C,D,states = ctx.saved_tensors
    if gy is None:
        gy = torch.zeros_like(x)
    if gh is None:
        gh = torch.zeros_like(states[:,0])
    return (*_backward_op(x,delta,A,B,C,D,states,gy,gh,ctx.chunk),None)


_scan_op.register_autograd(_backward,setup_context=_setup_context)


def scan(x,delta,A,B,C,D,initial_state=None,chunk_size=128):
    if initial_state is None:
        initial_state = x.new_zeros(x.size(0),x.size(2),A.size(1))
    y,h,_ = _scan_op(x,delta,A,B,C,D,initial_state,chunk_size)
    return y,h
