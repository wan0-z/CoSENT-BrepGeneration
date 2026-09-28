"""Small causal prefix-memory Transformer for a CPU-scale sequence experiment.

All condition tokens remain visible to target queries. Prefix queries use local
causal attention, avoiding quadratic 8k-token condition self-attention. This is
an explicit sparse-attention demo, not full dense attention over 156k tokens.
"""
import math
import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self,dim,heads,dropout,prefix_window):
        super().__init__()
        self.heads=heads; self.head_dim=dim//heads; self.prefix_window=prefix_window
        self.norm1=nn.LayerNorm(dim); self.qkv=nn.Linear(dim,3*dim); self.out=nn.Linear(dim,dim)
        self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.Dropout(dropout))
        self.drop=nn.Dropout(dropout)

    def attend(self,q,k,v,mask):
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores=scores.masked_fill(mask.unsqueeze(0),float("-inf"))
        return (self.drop(scores.softmax(dim=-1))@v).transpose(0,1).contiguous().flatten(1)

    def forward(self,x,nprefix):
        q,k,v=self.qkv(self.norm1(x)).chunk(3,dim=-1)
        q,k,v=[t.reshape(len(t),self.heads,self.head_dim).transpose(0,1) for t in (q,k,v)]
        result=[]; window=self.prefix_window
        for start in range(0,nprefix,window):
            end=min(nprefix,start+window); left=max(0,start-window)
            mask=torch.arange(left,end,device=x.device)[None,:]>torch.arange(start,end,device=x.device)[:,None]
            result.append(self.attend(q[:,start:end],k[:,left:end],v[:,left:end],mask))
        if nprefix<len(x):
            # Every body query can see every condition key and only past body keys.
            mask=torch.arange(len(x),device=x.device)[None,:]>torch.arange(nprefix,len(x),device=x.device)[:,None]
            result.append(self.attend(q[:,nprefix:],k,v,mask))
        x=x+self.drop(self.out(torch.cat(result,dim=0)))
        return x+self.ff(self.norm2(x))


class PrefixTransformer(nn.Module):
    def __init__(self,vocab_size,dim=64,layers=2,heads=4,dropout=0.1,prefix_window=128):
        super().__init__()
        if dim%heads: raise ValueError("dim must be divisible by heads")
        self.dim=dim; self.embedding=nn.Embedding(vocab_size,dim)
        self.blocks=nn.ModuleList([CausalBlock(dim,heads,dropout,prefix_window) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim); self.lm_head=nn.Linear(dim,vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=0.02)
            if getattr(module,"bias",None) is not None: nn.init.zeros_(module.bias)

    def positions(self,positions):
        freq=torch.exp(torch.arange(0,self.dim,2,device=positions.device)*(-math.log(10000.)/self.dim))
        phase=positions.float()[:,None]*freq[None,:]
        pe=torch.zeros(len(positions),self.dim,device=positions.device)
        pe[:,0::2]=phase.sin(); pe[:,1::2]=phase.cos()
        return pe

    def forward(self,prefix,body,positions):
        ids=torch.cat([prefix,body]); pos=torch.cat([torch.arange(len(prefix),device=ids.device),positions])
        x=self.embedding(ids)*math.sqrt(self.dim)+self.positions(pos)
        for block in self.blocks: x=block(x,len(prefix))
        return self.lm_head(self.norm(x[len(prefix):]))
