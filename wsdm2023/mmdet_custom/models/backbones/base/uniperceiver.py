import logging
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from timm.models.layers import DropPath
from torch import nn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def batch_mean(x, batch_size):
    """
    Args:
        x (torch.Tensor): shape=(B,N,C)
        batch_size (int): target batch size
    Returns:
        s: shape=(batch_size,N,C)
    """
    B, N, C = x.shape
    assert B % batch_size == 0
    Nw = B // batch_size
    x = x.reshape(Nw, batch_size, N, C)
    x = torch.mean(x, dim=0)
    return x.reshape(batch_size, N, C)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 masked_value=1e-12, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.masked_value = masked_value

        self.in_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q, q_mask, H, W):
        # mask.shape = [Bq, Nq]
        B, N, C = x.shape
        Bq, Nq, Cq = q.shape
        assert B == Bq
        x = torch.cat([x, q], dim=1)

        qkv = self.in_proj(x).reshape(
            B, N+Nq, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N+Nq, C // num_heads]

        attn = (q @ k.transpose(-2, -1)) * \
            self.scale  # [B, num_heads, N+Nq, N+Nq]

        if self.use_mask:
            mask = torch.cat(
                [torch.ones([B, N], dtype=q_mask.dtype).cuda(), q_mask], dim=1)
            mask = mask.reshape(B, 1, 1, N+Nq)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N+Nq, C)

        x = self.out_proj(x)
        x = self.proj_drop(x)
        q = x[:, N:, :]
        x = x[:, :N, :]

        return x, q
    

class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.in_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x, q, q_mask, H, W):
        B, N, C = x.shape
        Bq, Nq, Cq = q.shape
        assert B == Bq

        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, [0, 0, 0, W_ - W, 0, H_ - H])

        # nW*B, window_size, window_size, C
        x = window_partition(x, window_size=self.window_size)
        x = x.view(-1, N_, C)

        B_ = x.shape[0]
        q = q.unsqueeze(0).expand(B_//Bq, Bq, Nq, Cq).reshape(B_, Nq, Cq)
        x = torch.cat([x, q], dim=1)

        qkv = self.in_proj(x).view(-1, N_+Nq, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * \
            self.scale  # [B, L, num_head, N_, N_]

        # mask
        if self.use_mask:
            mask = torch.cat(
                [torch.ones([B, N_], dtype=q_mask.dtype).cuda(), q_mask], dim=1)
            mask = mask.reshape(B, 1, 1, N_+Nq)
            mask = mask.unsqueeze(0).expand(
                B_//B, B, 1, 1, N_+Nq).reshape(B_, 1, 1, N_+Nq)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        x = (attn @ v).transpose(1, 2).reshape(-1, N_+Nq, C)

        i = x[:, :N_, :].reshape(-1, self.window_size, self.window_size, C)
        t = batch_mean(x[:, N_:, :], Bq)

        x = window_reverse(i, self.window_size, H_, W_)
        x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
        x = self.out_proj(x)
        x = self.proj_drop(x)
        q = self.out_proj(t)
        q = self.proj_drop(q)

        return x, q


class MultiModelBertLayer(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                 drop_path_ratio=0.1, windowed=False, window_size=14, with_cp=False, use_mask=False):

        super(MultiModelBertLayer, self).__init__()
        self.with_cp = with_cp
        if windowed:
            self.self_attn = WindowedAttention(hidden_size, num_attention_heads, qkv_bias=True, attn_drop=0.,
                                               proj_drop=0., window_size=window_size, use_mask=use_mask)
        else:
            self.self_attn = Attention(hidden_size, num_attention_heads, qkv_bias=True,
                                       attn_drop=0., proj_drop=0., use_mask=use_mask)
        # self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = nn.GELU()

        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.gamma_1 = nn.Parameter(torch.zeros((hidden_size)), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.zeros((hidden_size)), requires_grad=True)

    def ffn_forward(self, x, q):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        q = self.linear1(q)
        q = self.act_fn(q)
        q = self.linear2(q)
        return x, q

    def forward(self, x, q, q_mask, H, W):

        def _inner_forward(x, q, q_mask):
            x_, q_ = self.self_attn(self.norm1(x), self.norm1(q), q_mask, H, W)
            x = x + self.gamma_1 * self.drop_path(x_)
            q = q + self.gamma_1 * self.drop_path(q_)
            x_, q_ = self.ffn_forward(self.norm2(x), self.norm2(q))
            x = x + self.gamma_2 * self.drop_path(x_)
            q = q + self.gamma_2 * self.drop_path(q_)
            return x, q

        if self.with_cp and x.requires_grad:
            x, q = cp.checkpoint(_inner_forward, x, q, q_mask)
        else:
            x, q = _inner_forward(x, q, q_mask)

        return x, q


class VisualPatchEmbedding(nn.Module):
    def __init__(self, in_dim=3, out_dim=768, patch_size=16, image_size=224, dropout=0.):
        super(VisualPatchEmbedding, self).__init__()
        self.embeddings_act = None
        self.embeddings_norm = nn.LayerNorm(out_dim)
        # self.embeddings_type = nn.Embedding(1, 768)
        self.embeddings_dropout = nn.Dropout(dropout)
        
        self.patch_embed = PatchEmbed(
            img_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
            in_chans=in_dim, embed_dim=out_dim,
        )
    
    def forward(self, x):
        embeddings, H, W = self.patch_embed(x)
        # data_type = torch.zeros(1).long().cuda()
        # embeddings_type = self.embeddings_type(data_type).unsqueeze(1)
        # embeddings = embeddings + embeddings_type
        # embeddings = embeddings + self.embeddings_type.weight[0].unsqueeze(0).unsqueeze(1).to(embeddings.dtype)
        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)
        
        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)
        
        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)
        
        return embeddings, H, W


class PatchEmbed(torch.nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pretrain_size = img_size
        
        self.spatial_pos_embed = nn.Embedding(num_patches, embed_dim)
        self.temporal_pos_embed = nn.Embedding(8, embed_dim)
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        temp_len = 1
        pos_embed = self._get_pos_embed(self.spatial_pos_embed.weight.unsqueeze(0), H // 16, W // 16)
        temporal_pos_ids = torch.arange(temp_len, dtype=torch.long, device=x.device)
        temporal_pos_embed = self.temporal_pos_embed(temporal_pos_ids).unsqueeze(0)
        x = x + pos_embed + temporal_pos_embed
        return x, H // 16, W // 16


class NNEmbeddingEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(NNEmbeddingEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)
    
    def forward(self, x, start_time=0):
        if isinstance(x, int):
            position_embeddings = self.position_embeddings(torch.tensor([x], dtype=torch.long).cuda())
        elif isinstance(x, torch.Tensor) and x.dim() == 1:
            position_embeddings = self.position_embeddings(x)
        else:
            x_size = x.size(1)
            position_ids = torch.arange(x_size, dtype=torch.long, device=x.device) + start_time
            position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class TokenBaseEmbedding(nn.Module):
    def __init__(self, dim, vocab_size):
        super(TokenBaseEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.embeddings_norm = nn.LayerNorm(dim)
        self.embeddings_pos = NNEmbeddingEncoding(dim, 512)
        self.embeddings_token_type = nn.Embedding(2, dim)
    
    def forward(self, input_ids, time_step=None, start_time=0):
        embeddings = self.embeddings(input_ids)
        pos_inputs = input_ids if time_step is None else time_step
        position_embeddings = self.embeddings_pos(pos_inputs, start_time=start_time)
        embeddings = embeddings + position_embeddings.to(embeddings.dtype)
        
        embeddings_token_type = self.embeddings_token_type.weight[0].unsqueeze(0).unsqueeze(1)
        embeddings = embeddings + embeddings_token_type.to(embeddings.dtype)
        embeddings = self.embeddings_norm(embeddings)
        
        return embeddings


class UnifiedBertEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 window_attn=False, window_size=14, with_cp=False, use_mask=False,
                 pretrained=None):

        super(UnifiedBertEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        logging.info('window attention:', window_attn)
        logging.info('window size:', window_size)
        logging.info('use mask:', use_mask)
        layers = []
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            layers.append(
                MultiModelBertLayer(hidden_size=embed_dim, intermediate_size=int(embed_dim * mlp_ratio),
                                    num_attention_heads=num_heads, drop_path_ratio=dpr[i],
                                    windowed=window_attn[i], window_size=window_size[i],
                                    with_cp=with_cp, use_mask=use_mask)
            )

        self.layers = nn.ModuleList(layers)
        self.visual_embed = VisualPatchEmbedding(in_dim=in_chans, out_dim=embed_dim,
                                                 patch_size=patch_size, image_size=img_size)
        self.token_embed = TokenBaseEmbedding(dim=embed_dim, vocab_size=49411)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu',
                            strict=False, logger=logger)

    def forward(self, img, question):
        x, H, W = self.visual_embed(img)
        q = self.token_embed(question)

        for layer in self.layers:
            x, q = layer(x, q, H, W)
        return x, q
