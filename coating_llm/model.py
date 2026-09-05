# -*- coding: utf-8 -*-
"""CoatingLLM：GQA + SwiGLU + RMSNorm + RoPE 的纯 PyTorch 小语言模型。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def precompute_rope(head_dim, max_seq, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv)
    return torch.cat([freqs.cos(), freqs.cos()], dim=-1), torch.cat([freqs.sin(), freqs.sin()], dim=-1)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    cos, sin = cos[..., :half], sin[..., :half]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, max_seq, dropout):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.head_dim = hidden // heads
        self.n_rep = heads // kv_heads
        self.q = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.k = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)
        cos, sin = precompute_rope(self.head_dim, max_seq)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, mask):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim)
        k = self.k(x).view(B, T, self.kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.kv_heads, self.head_dim)
        cos = self.rope_cos[:T].view(1, T, 1, -1)
        sin = self.rope_sin[:T].view(1, T, 1, -1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
        y = self.drop(att) @ v
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, hidden, intermediate, dropout):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg["hidden"])
        self.att = Attention(cfg["hidden"], cfg["heads"], cfg["kv_heads"], cfg["max_seq"], cfg["dropout"])
        self.n2 = RMSNorm(cfg["hidden"])
        self.mlp = SwiGLU(cfg["hidden"], cfg["intermediate"], cfg["dropout"])

    def forward(self, x, mask):
        x = x + self.att(self.n1(x), mask)
        return x + self.mlp(self.n2(x))


class CoatingLLM(nn.Module):
    def __init__(self, vocab_size=10000, hidden=256, layers=4, heads=4, kv_heads=2,
                 intermediate=640, max_seq=1024, dropout=0.1):
        super().__init__()
        cfg = dict(hidden=hidden, heads=heads, kv_heads=kv_heads,
                   intermediate=intermediate, max_seq=max_seq, dropout=dropout)
        self.cfg = cfg
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(layers))
        self.norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.apply(self._init)
        print(f"CoatingLLM 初始化完成: {self.num_params / 1e6:.2f}M 参数")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, mask)
        x = self.norm(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            targets[:, 1:].contiguous().view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.6, top_p=0.9, top_k=30,
                 repetition_penalty=1.1, eos_token_id=2):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq else idx[:, -self.max_seq:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                for i in range(idx.shape[0]):
                    seen = torch.unique(idx[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty,
                                                  score * repetition_penalty)
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < kth] = float("-inf")
            if top_p < 1.0:
                sl, ii = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cum - F.softmax(sl, dim=-1) > top_p] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, ii, sl)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_token_id is not None and (nxt == eos_token_id).all():
                break
        return idx
