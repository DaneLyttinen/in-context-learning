import torch
from torch import nn
from torch import nn, einsum
from einops import rearrange

class Attention(nn.Module):
    def __init__(
        self,
        model_dim,
        heads=2,
        dim_head=16,
        dropout=0.,
        causal=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(model_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if self.causal:
            # apply causal mask
            mask = torch.ones(size=sim.shape[-2:], device=sim.device).triu_(1).bool()
            sim.masked_fill_(mask, float("-inf"))

        attn = sim.softmax(dim=-1) # (batch, heads, query, key)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads) # merge heads
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        token_dim=784 + 10,
        inner_dim=None,
        dropout=0.,
        causal=False,
    ):
        super().__init__()
        self.embed_proj = nn.Linear(token_dim, dim)
        self.layers = nn.ModuleList([])
        inner_dim = inner_dim or 4 * dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, inner_dim),
                    nn.GELU(),
                    nn.Linear(inner_dim, dim),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x):
        x = self.embed_proj(x)
        for attn, ln1, mlp, ln2 in self.layers:
            x = x + attn(x)
            x = x + mlp(ln1(x))
        return x

class InContextLearner(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        heads=4,
        dim_head=16,
        inner_dim=None,
        dropout=0.1,
        whole_seq_prediction=False,
    ):
        super().__init__()
        inner_dim = inner_dim or 4 * dim
        self.whole_seq_prediction = whole_seq_prediction
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            inner_dim=inner_dim,
            dropout=dropout,
            causal=whole_seq_prediction,
        )
        self.final_classifier = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.transformer(x) # (batch, seq_len, dim)
        if self.whole_seq_prediction:
            return self.final_classifier(x)
        else:
            return self.final_classifier(x[:,-1,:])