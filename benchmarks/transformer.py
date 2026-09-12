"""Causal SDPA comparison model with a real, preallocated generation KV cache."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,width,ff_width):
        super().__init__()
        self.heads=width//64
        self.norm1=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width)
        self.out=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,ff_width),nn.GELU(),nn.Linear(ff_width,width))

    def forward(self,x,cache=None):
        batch,length,width=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(batch,length,3,self.heads,64).permute(2,0,3,1,4).unbind(0)
        causal=True
        if cache is not None:
            keys,values,used=cache
            keys[:,:,used:used+length].copy_(k)
            values[:,:,used:used+length].copy_(v)
            k,v=keys[:,:,:used+length],values[:,:,:used+length]
            causal=used==0
            cache=(keys,values,used+length)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=causal)
        x=x+self.out(y.transpose(1,2).reshape(batch,length,width))
        return x+self.ff(self.norm2(x)),cache


class TransformerLM(nn.Module):
    def __init__(self,width,layers,vocab,ff_width,tied=False):
        super().__init__()
        self.width=width
        self.embedding=nn.Embedding(vocab,width,padding_idx=0)
        self.layers=nn.ModuleList([Block(width,ff_width) for _ in range(layers)])
        self.norm=nn.LayerNorm(width)
        self.lm_head=nn.Linear(width,vocab,bias=False)
        if tied:
            self.lm_head.weight=self.embedding.weight
        pos=torch.arange(16384)[:,None]
        frequency=torch.exp(torch.arange(0,width,2)*(-math.log(10000.0)/width))
        encoding=torch.empty(16384,width)
        encoding[:,0::2]=torch.sin(pos*frequency)
        encoding[:,1::2]=torch.cos(pos*frequency)
        self.register_buffer('positions',encoding,persistent=False)

    def hidden(self,tokens,caches=None,start=0):
        x=self.embedding(tokens)+self.positions[start:start+tokens.size(1)].to(self.embedding.weight.dtype)
        for i,layer in enumerate(self.layers):
            x,state=layer(x,None if caches is None else caches[i])
            if caches is not None:
                caches[i]=state
        return x,caches

    def forward(self,tokens):
        x,_=self.hidden(tokens)
        return self.lm_head(self.norm(x))

    def prefill(self,tokens,capacity=None):
        capacity=tokens.size(1) if capacity is None else capacity
        dtype=torch.get_autocast_dtype('mps') if torch.is_autocast_enabled('mps') else self.embedding.weight.dtype
        shape=(tokens.size(0),self.width//64,capacity,64)
        caches=[(torch.empty(shape,device=tokens.device,dtype=dtype),
                 torch.empty(shape,device=tokens.device,dtype=dtype),0) for _ in self.layers]
        x,caches=self.hidden(tokens,caches)
        return self.lm_head(self.norm(x[:,-1])),caches

    @torch.inference_mode()
    def generate(self,tokens,num_tokens,sample=False):
        logits,caches=self.prefill(tokens,tokens.size(1)+num_tokens)
        out=tokens.new_empty(tokens.size(0),num_tokens)
        for i in range(num_tokens):
            token=logits.argmax(-1,keepdim=True)
            out[:,i:i+1]=token
            if i+1<num_tokens:
                x,caches=self.hidden(token,caches,tokens.size(1)+i)
                logits=self.lm_head(self.norm(x[:,0]))
        return out
