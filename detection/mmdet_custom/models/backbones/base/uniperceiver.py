# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import logging
import math
from functools import partial

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


class BertSelfWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, attention_probs_dropout_prob=0.,
                 proj_drop=0., window_size=14):
        super().__init__()
        self.num_heads = num_attention_heads
        head_dim = hidden_size // num_attention_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_drop = nn.Dropout(attention_probs_dropout_prob)
        self.window_size = window_size

    def forward(self, x, H, W):
        B, N, C = x.shape

        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, [0, 0, 0, W_ - W, 0, H_ - H])

        x = window_partition(x, window_size=self.window_size)  # nW*B, window_size, window_size, C
        x = x.view(-1, N_, C)

        q = self.query(x).view(-1, N_, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x).view(-1, N_, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x).view(-1, N_, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)

        x = window_reverse(x, self.window_size, H_, W_)
        x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
        return x


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number '
                'of attention heads (%d)' % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)
        # return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, H, W):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=3072, layer_norm_eps=1e-6, hidden_dropout_prob=0., drop_path_ratio=0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, drop_path_ratio=0.1,
                 windowed=False, window_size=14):
        super(BertAttention, self).__init__()
        if not windowed:
            self.self = BertSelfAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads)
        else:
            self.self = BertSelfWindowAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                window_size=window_size)
        self.output = BertSelfOutput(hidden_size=hidden_size, drop_path_ratio=drop_path_ratio)

    def forward(self, input_tensor, H, W):
        self_output = self.self(input_tensor, H, W)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, intermediate_drop=0.):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dropout = nn.Dropout(intermediate_drop)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, layer_norm_eps=1e-6,
                 ffn_dropout_prob=0., drop_path_ratio=0.1):

        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(ffn_dropout_prob)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                 drop_path_ratio=0.1, windowed=False, window_size=14, with_cp=False):

        super(BertLayer, self).__init__()
        self.with_cp = with_cp
        self.attention = BertAttention(hidden_size, num_attention_heads,
                                       drop_path_ratio, windowed, window_size)

        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size=hidden_size,
                                 intermediate_size=intermediate_size,
                                 drop_path_ratio=drop_path_ratio)

    def forward(self, hidden_states, H, W):
        
        def _inner_forward(hidden_states):
            attention_output = self.attention(hidden_states, H, W)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output

        if self.with_cp and hidden_states.requires_grad:
            x = cp.checkpoint(_inner_forward, hidden_states)
        else:
            x = _inner_forward(hidden_states)

        return x


class VisualPatchEmbedding(nn.Module):
    def __init__(self, in_dim=3, out_dim=768, patch_size=16, image_size=224, dropout=0.):
        super(VisualPatchEmbedding, self).__init__()
        self.embeddings_act = None
        self.embeddings_norm = nn.LayerNorm(out_dim, eps=1e-12)
        self.embeddings_type = nn.Embedding(1, 768)
        self.embeddings_dropout = nn.Dropout(dropout)

        self.patch_embed = PatchEmbed(
            img_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
            in_chans=in_dim, embed_dim=out_dim,
        )

    def forward(self, x):
        embeddings, H, W = self.patch_embed(x)
        data_type = torch.zeros(1).long().cuda()
        embeddings_type = self.embeddings_type(data_type).unsqueeze(1)
        embeddings = embeddings + embeddings_type

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

        self.pos_embed = nn.Embedding(num_patches, embed_dim)
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        pos_embed = self._get_pos_embed(self.pos_embed.weight.unsqueeze(0), H // 16, W // 16)
        x = x + pos_embed
        return x, H // 16, W // 16


class UnifiedBertEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 embed_layer=VisualPatchEmbedding, window_attn=False, window_size=14,
                 with_cp=False, pretrained=None):

        super(UnifiedBertEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        logging.info('window attention:', window_attn)
        logging.info('window size:', window_size)

        layers = []
        for i in range(depth):
            layers.append(
                BertLayer(hidden_size=embed_dim, intermediate_size=int(embed_dim * mlp_ratio),
                          num_attention_heads=num_heads, drop_path_ratio=drop_path_rate,
                          windowed=window_attn[i], window_size=window_size[i], with_cp=with_cp)
            )

        self.layers = nn.ModuleList(layers)
        self.visual_embed = embed_layer(in_dim=in_chans, out_dim=embed_dim,
                                        patch_size=patch_size, image_size=img_size)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward(self, x):  # 'v', 't', 'vt'
        x, H, W = self.visual_embed(x)
        for layer in self.layers:
            x = layer(x, H, W)
        return x
