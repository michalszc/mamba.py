from functools import lru_cache
from pathlib import Path
import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def shaders():
    return torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())


@torch.library.custom_op('mambapy_mps::conv_silu',mutates_args=())
def _conv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.device.type != 'mps' or any(t.device != x.device or t.dtype != torch.float32 for t in (x,w,bias)):
        raise ValueError('Native convolution requires FP32 MPS tensors')
    if x.ndim != 3 or w.ndim != 3 or w.size(1) != 1 or w.size(0) != x.size(2) or bias.shape != (x.size(2),):
        raise ValueError('Invalid depthwise convolution shapes')
    batch,size,width = x.shape
    window=w.size(2)
    length=size-window+1
    if window<1 or length<1:
        raise ValueError('Convolution requires a positive output length and filter width')
    x,w,bias = [t.contiguous() for t in (x,w,bias)]
    y=x.new_empty(batch,length,width)
    shaders().conv_forward(x,w,bias,y,length,width,window,threads=y.numel())
    return y


@_conv.register_fake
def _conv_fake(x,w,bias):
    return x.new_empty(x.size(0),x.size(1)-w.size(2)+1,x.size(2))


@torch.library.custom_op('mambapy_mps::conv_silu_backward',mutates_args=())
def _conv_backward(x: torch.Tensor,w: torch.Tensor,bias: torch.Tensor,gy: torch.Tensor) -> list[torch.Tensor]:
    x,w,bias,gy=[t.contiguous() for t in (x,w,bias,gy)]
    batch,length,width=gy.shape
    window=w.size(2)
    gv=torch.empty_like(gy)
    gx,gw=torch.empty_like(x),torch.empty_like(w)
    lib=shaders()
    lib.conv_activation_grad(x,w,bias,gy,gv,length,width,window,threads=gy.numel())
    lib.conv_input_grad(gv,w,gx,length,width,window,threads=x.numel())
    lib.conv_weight_grad(x,gv,gw,batch,length,width,window,threads=w.numel())
    return [gx,gw,gv.sum((0,1))]


@_conv_backward.register_fake
def _conv_backward_fake(x,w,bias,gy):
    return [torch.empty_like(t) for t in (x,w,bias)]


def _setup(ctx,inputs,output):
    ctx.save_for_backward(*inputs)


def _backward(ctx,gy):
    return tuple(_conv_backward(*ctx.saved_tensors,gy))


_conv.register_autograd(_backward,setup_context=_setup)


def conv_silu(x,weight,bias=None,history=None):
    """BLD input; optional BD(W-1) history, differentiable through both."""
    window=weight.size(2)
    if history is None:
        history=x.new_zeros(x.size(0),x.size(2),window-1)
    if history.shape != (x.size(0),x.size(2),window-1):
        raise ValueError('Invalid convolution history shape')
    full=torch.cat((history.transpose(1,2),x),dim=1)
    bias=x.new_zeros(x.size(2)) if bias is None else bias
    return _conv(full,weight,bias)


def eligible(x,config):
    return (x.device.type=='mps' and x.dtype==torch.float32 and config.backend!='torch'
            and config.pscan and not config.use_cuda and 0<x.size(1)<=128)
