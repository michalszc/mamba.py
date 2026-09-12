from functools import lru_cache
from pathlib import Path

import torch

OUTPUT_GROUP_SIZE = 256


@lru_cache(maxsize=72)
def shaders(capacity,dtype=torch.float32,chunk_size=128):
    source = Path(__file__).with_suffix('.metal').read_text()
    scalar={torch.float32:'float',torch.float16:'half',torch.bfloat16:'bfloat'}[dtype]
    return torch.mps.compile_shader(source.replace('STATE_CAPACITY', str(capacity)).replace('SCALAR_TYPE',scalar).replace('CHUNK_CAPACITY',str(chunk_size)))


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
    if x.dtype not in (torch.float32,torch.float16,torch.bfloat16) or B.dtype!=x.dtype or C.dtype!=x.dtype:
        raise TypeError('Scan activations and B/C must share a supported floating dtype')
    if any(t.dtype!=torch.float32 for t in (delta,A,D,initial_state)):
        raise TypeError('Scan delta, A, D and recurrent state require float32')
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
    capacity=1 << (state-1).bit_length()
    lib = shaders(capacity,x.dtype,chunk_size)
    states = A.new_empty(batch,chunks+1,width,state)
    y = torch.empty_like(x)
    last=torch.empty_like(initial_state)
    if chunks>1:
        alpha = A.new_empty(batch,chunks,width,state)
        beta = torch.empty_like(alpha)
        lib.summaries(x,delta,A,B,alpha,beta,length,width,state,chunks,chunk_size,
                      threads=alpha.numel())
        lib.boundaries(alpha,beta,initial_state,states,width,state,chunks,
                       threads=batch*width*state)
    lib.output_scan(x,delta,A,B,C,D,states,y,initial_state,last,length,width,state,chunks,chunk_size,
                    threads=batch*chunks*width*min(capacity,32),group_size=OUTPUT_GROUP_SIZE)
    return y, last, states


def backward_raw(x, delta, A, B, C, D, states, gy, gh, chunk_size):
    x,delta,A,B,C,D,gy,gh = [t.contiguous() for t in (x,delta,A,B,C,D,gy,gh)]
    batch,length,width = x.shape
    state = A.size(1)
    chunks = states.size(1)-1
    lib = shaders(1 << (state-1).bit_length(),x.dtype,chunk_size)
    alpha = A.new_empty(batch,chunks,width,state)
    beta = torch.empty_like(alpha)
    adjoints = torch.empty_like(states)
    gx = gy*D
    gdt = torch.zeros_like(delta)
    gA = torch.empty_like(alpha)
    gB, gC = torch.zeros_like(B,dtype=torch.float32), torch.zeros_like(C,dtype=torch.float32)
    lib.reverse_summaries(delta,A,C,gy,alpha,beta,length,width,state,chunks,chunk_size,
                          threads=alpha.numel())
    lib.reverse_boundaries(alpha,beta,gh,adjoints,width,state,chunks,
                           threads=batch*width*state)
    lib.scan_gradients(x,delta,A,B,C,states,adjoints,gy,gx,gdt,gA,gB,gC,
                       length,width,state,chunks,chunk_size,
                       threads=batch*chunks*width*(1 << (state-1).bit_length()))
    ga=torch.empty_like(A)
    if length>=4096:
        lib.long_A_gradient(x,delta,A,B,C,states,gy,gh,ga,length,width,state,chunks,batch,threads=ga.numel())
    else:
        lib.reduce_A(gA,ga,width*state,batch*chunks,threads=ga.numel())
    return gx.to(x.dtype),gdt,ga,gB.to(B.dtype),gC.to(C.dtype),(gy*x.float()).sum((0,1)),adjoints[:,0].clone()


@torch.library.custom_op('mambapy_mps::scan_backward', mutates_args=())
def _backward_op(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                 B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                 states: torch.Tensor, gy: torch.Tensor, gh: torch.Tensor,
                 chunk_size: int) -> list[torch.Tensor]:
    return list(backward_raw(x,delta,A,B,C,D,states,gy,gh,chunk_size))


@_backward_op.register_fake
def _backward_fake(x,delta,A,B,C,D,states,gy,gh,chunk_size):
    return [torch.empty_like(t,memory_format=torch.contiguous_format) for t in (x,delta,A,B,C,D,gh)]


@torch.library.custom_op('mambapy_mps::scan', mutates_args=())
def _scan_op(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
             B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
             initial_state: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    return forward_raw(x,delta,A,B,C,D,initial_state,chunk_size)


@_scan_op.register_fake
def _scan_fake(x,delta,A,B,C,D,initial_state,chunk_size):
    chunks = (x.size(1)+chunk_size-1)//chunk_size
    return (torch.empty_like(x,memory_format=torch.contiguous_format),
            torch.empty_like(initial_state,memory_format=torch.contiguous_format),
            A.new_empty(x.size(0),chunks+1,x.size(2),A.size(1)))


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
    return (*_backward_op(x,delta,A,B,C,D,states,gy.float(),gh.float(),ctx.chunk),None)


_scan_op.register_autograd(_backward,setup_context=_setup_context)


def select_chunk(length,width,dtype):
    # Bounded M4 Pro tuning: short sequences use one launch; 64-token tiles
    # lower backward register pressure for wide or long sequences.
    if length<=128:
        return 128
    if length>=4096:
        return 128
    return 64 if length>=1024 or width>=512 else 128


@torch.library.custom_op('mambapy_mps::scan_inference',mutates_args=())
def _inference_op(x: torch.Tensor,delta: torch.Tensor,A: torch.Tensor,B: torch.Tensor,
                  C: torch.Tensor,D: torch.Tensor,initial_state: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    # Narrow prompt workloads gain temporal parallelism from tiling. Wide
    # workloads already fill the GPU and benefit from the single-launch scan.
    if x.size(1)>128 and x.size(2)<512:
        return forward_raw(x,delta,A,B,C,D,initial_state,64 if x.size(1)>=1024 else 128)[:2]
    validate(x,delta,A,B,C,D,initial_state)
    x,delta,A,B,C,D,initial_state=[t.contiguous() for t in (x,delta,A,B,C,D,initial_state)]
    batch,length,width=x.shape
    state=A.size(1)
    capacity=1 << (state-1).bit_length()
    states=A.new_empty(batch,2,width,state)
    y=torch.empty_like(x)
    last=torch.empty_like(initial_state)
    shaders(capacity,x.dtype,128).output_scan(x,delta,A,B,C,D,states,y,initial_state,last,
                                            length,width,state,1,length,
                                            threads=batch*width*min(capacity,32),group_size=OUTPUT_GROUP_SIZE)
    return y,last


@_inference_op.register_fake
def _inference_fake(x,delta,A,B,C,D,initial_state):
    return (torch.empty_like(x,memory_format=torch.contiguous_format),
            torch.empty_like(initial_state,memory_format=torch.contiguous_format))


def scan(x,delta,A,B,C,D,initial_state=None,chunk_size=None):
    if chunk_size is None:
        chunk_size=select_chunk(x.size(1),x.size(2),x.dtype)
    if initial_state is None:
        initial_state = x.new_zeros(x.size(0),x.size(2),A.size(1),dtype=torch.float32)
    if not torch.is_grad_enabled():
        return _inference_op(x,delta.float(),A.float(),B.to(x.dtype),C.to(x.dtype),D.float(),initial_state.float())
    y,h,_ = _scan_op(x,delta.float(),A.float(),B.to(x.dtype),C.to(x.dtype),D.float(),initial_state.float(),chunk_size)
    return y,h
