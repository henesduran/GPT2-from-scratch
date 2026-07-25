from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024 # sequence length (context window)
    vocab_size: int = 50257 # vocabulary(token) size : 50,000 (Byte pair encoding) merges + 256 bytes tokens(base tokens) + 1 special (<|endoftext|>) token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # must be evenly divided
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # key, query, value

        # projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_FLAG = 1 #scaling to prevent std expansion

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #casual mask(or bias) for attention ,to mask out future tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimension
        
        qkv = self.c_attn(x) # produce query,key,value as a batch,more efficient this way
        q, k, v = qkv.split(self.n_embd, dim=2) # split

        #moving number of heads dimension to front,to be the batch dimension (B,n_head),to compute in parallel
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True) #casual attention computation
        #attention computation can also be done doing the following:

        """ 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) 
        """

        y = y.transpose(1, 2).contiguous().view(B, T, C) # concatenate all head outputs back
        y = self.c_proj(y) # output projection
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # expand the layer to produce more information
        self.gelu    = nn.GELU(approximate='tanh') #non-linearity
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # project the information back into embedding size
        self.c_proj.SCALE_FLAG = 1 #scaling to prevent std expansion

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # pre-layernorm + residual connection
        x = x + self.mlp(self.ln_2(x)) # pre-layernorm + residual connection
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding table
            wpe = nn.Embedding(config.block_size, config.n_embd), #positional embedding table
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), #final layernorm before classifier
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classifier
        self.transformer.wte.weight = self.lm_head.weight # weight sharing (input ~ final classifier)

        self.apply(self._init_weights)

    def _init_weights(self,module):
        std = 0.02
        if isinstance(module,nn.Linear):
            if hasattr(module,'SCALE_FLAG'):
                std *= (2*self.config.n_layer)**-0.5 #normalize std if flag exists
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)

    def forward(self, idx,targets=None):
        B, T = idx.size() # input size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is limited to {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings

        x = tok_emb + pos_emb #add up position + token embedding

        # forward the blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x) # layernorm before lm_head
        logits = self.lm_head(x)
        loss  = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss


    def configure_optimizers(self, weight_decay, learning_rate, device):

        #all parameters requiring grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        #only params with dimension bigger than 2 will be weight-decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Create AdamW optimizer(and fused if possible)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

