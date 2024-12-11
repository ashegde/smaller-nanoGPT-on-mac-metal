"""
This module contains the basic NanoGPT Transformer model. 
The code is adapted from https://github.com/ashegde/build-nanoGPT/,
and the core architecture is based on Andrej Karpathy's GPT2 tutorial.

Note that I have made some modifications below based on recent works 
describing performant initialization strategies. This is still exploratory.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # ensure divisibility for multiple heads 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #3 for the query, key, and value matrices
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # this is more of a mask than a bias, but following Karpathy's lecture, which follows the 
        # OpenAI/HuggingFace naming convention, we will use "bias".
        # EDIT 6-27-2024, this is handled in scaled_dot_product_attention
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x is (B, T, C), 
        # [B]atches
        # [T]okens := block_size = number of tokens in the sequence
        # [C]hannels := n_embd = n_heads * h_size. Recall, h_size = n_embd // n_heads = C // n_heads

        B, T, C = x.size()
        qkv = self.c_attn(x) #(B, T, 3 * n_embd) 

        # split dim 2 into chunks of size n_embd --> in this case, we will have 3 chunks for query, key, and value outputs
        q, k, v = qkv.split(self.n_embd, dim=2) # q, k, v are each (B,T,n_embd)

        # distribute the projections across different heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, n_head, T, h_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, n_head, T, h_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, n_head, T, h_size)

        # attn = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1))) # (B, n_head, T, T) inner product of token embeddings for each head
        # attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # lower triangular masking (pre-softmax), for causality
        # attn = F.softmax(attn, dim=-1) #(B, n_head, T, T)
        # y = attn @ v # (B, n_head, T, T) @ (B, n_head, T, h_size) = (B, n_head, T, h_size)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention

        y = y.transpose(1,2).contiguous().view(B,T,C) #(B,T, C = n_heads * h_size) 

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #approximate GELU here is an artifact to replicate GPT2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        # note the residual stream remains intact
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                 # weights of the token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),                # weights of the position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),    # attention block
            ln_f = nn.LayerNorm(config.n_embd),                                   # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)    # language model head, final classifier
       
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module,'NANOGPT_SCALE_INIT'):
            std *= (2*self.config.n_layer)**(-0.5)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
  
    def forward(self, idx, targets=None):
        # idx is (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T} since the block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # position indices in the sequence
        pos_embd = self.transformer.wpe(pos) # (T, n_embd) position embeddings for each token in the sequence
        tok_embd = self.transformer.wte(idx) # (B, T, n_embd) token embedding for each sequence element
        x = tok_embd + pos_embd # (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)
        x = self.transformer.ln_f(x) #(B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # for a given batch b and token t, logits[b,t,:] gives the predictive distribution (in logits, pre-softmax)
        # for the next token t+1 given the previous :t tokens.

        return logits

    def generate(self, past_tokens:torch.Tensor, max_new_tokens:int) -> torch.Tensor:
    # past_tokens is of dim (B,T) whose (b,t)th entry corresponds to the vocabulary index in batch b at time t
        for _ in range(max_new_tokens):
            #ensure we stay within scope (context never exceeds block_size, i.e., the context = the most recent upt-to-block_size tokens) 
            context = past_tokens[:,-self.config.block_size:] 
            logits = self(context) # logits is (B,T,C), loss is (B*T)
            logits = logits[:, -1, :] # (B,C)
            next_token_probs = F.softmax(logits, dim=-1) #(B,C)
            next_token = torch.multinomial(next_token_probs, num_samples=1) #(B,1)
            past_tokens = torch.cat((past_tokens, next_token), dim=1) #(B,T+1)
        return past_tokens