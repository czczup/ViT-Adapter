import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, Mlp


class GroundingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, r):
        B, N, C = x.shape
        B_, N_, C_ = r.shape

        kv = self.kv(r).reshape(B_, N_, 2, self.num_heads, C_ //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N_)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroundingCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., with_cp=False,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 layer_scale=False):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(dim)

        self.attn = GroundingAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x, r):

        def _inner_forward(x, r):
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), self.norm1(r)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(r)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, r)
        else:
            x = _inner_forward(x, r)

        return x
