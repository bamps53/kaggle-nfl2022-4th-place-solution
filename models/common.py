import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    # from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.dist_linear = nn.Sequential(
            nn.Linear(3, inner_dim, bias=True),
            nn.ReLU(),
            nn.Linear(inner_dim, 1, bias=True))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, distance, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dist = self.dist_linear(distance).squeeze(3)[:, None]
        dots += dist

        if attn_mask is not None:
            # Here attn_mask's meaning is different from pytorch original.
            # True means valid example and False means invalid(empty) example.
            # Therefore, -inf should be added to mask == False position.
            attn_mask = attn_mask[:, None]  # add multihead dim
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float32)
            new_attn_mask.masked_fill_(attn_mask == False, -1e3)  # to deal with overflow when mixed precision
            dots += new_attn_mask

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, distance, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, distance=distance, attn_mask=attn_mask) + x
            x = ff(x) + x
        return x
