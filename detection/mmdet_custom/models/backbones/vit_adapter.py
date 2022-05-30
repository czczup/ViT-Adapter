# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .base.vit import TIMMVisionTransformer

_logger = logging.getLogger(__name__)


class ConvFFN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2)
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W)
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2)
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 deform_ratio=1.0,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim,
                                 n_levels=n_levels,
                                 n_heads=num_heads,
                                 n_points=n_points,
                                 ratio=deform_ratio)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim,
                               hidden_features=int(dim * cffn_ratio),
                               drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, H, W):
        attn = self.attn(self.query_norm(query), reference_points,
                         self.feat_norm(feat), spatial_shapes,
                         level_start_index, None)
        query = query + attn

        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
        return query


class Injector(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim,
                                 n_levels=n_levels,
                                 n_heads=num_heads,
                                 n_points=n_points,
                                 ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)),
                                  requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index):
        attn = self.attn(self.query_norm(query), reference_points,
                         self.feat_norm(feat), spatial_shapes,
                         level_start_index, None)
        return query + self.gamma * attn


class InteractionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 init_values=0.,
                 deform_ratio=1.0,
                 extra_extractor=False):
        super().__init__()

        self.injector = Injector(dim=dim,
                                 n_levels=3,
                                 num_heads=num_heads,
                                 init_values=init_values,
                                 n_points=n_points,
                                 norm_layer=norm_layer,
                                 deform_ratio=deform_ratio)
        self.extractor = Extractor(dim=dim,
                                   n_levels=1,
                                   num_heads=num_heads,
                                   n_points=n_points,
                                   norm_layer=norm_layer,
                                   deform_ratio=deform_ratio,
                                   with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio,
                                   drop=drop,
                                   drop_path=drop_path)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim,
                          num_heads=num_heads,
                          n_points=n_points,
                          norm_layer=norm_layer,
                          with_cffn=with_cffn,
                          cffn_ratio=cffn_ratio,
                          deform_ratio=deform_ratio) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x,
                          reference_points=deform_inputs1[0],
                          feat=c,
                          spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c,
                           reference_points=deform_inputs2[0],
                           feat=x,
                           spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],
                           H=H,
                           W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c,
                              reference_points=deform_inputs2[0],
                              feat=x,
                              spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],
                              H=H,
                              W=W)
        return x, c


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(
                3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes,
                      2 * inplanes,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes,
                      4 * inplanes,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes,
                      4 * inplanes,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes,
                             embed_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes,
                             embed_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes,
                             embed_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes,
                             embed_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


@BACKBONES.register_module()
class ViTAdapter(TIMMVisionTransformer):
    def __init__(self,
                 pretrain_size=224,
                 num_heads=12,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 *args,
                 **kwargs):

        super().__init__(num_heads=num_heads, *args, **kwargs)

        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [
            i for i in range(-1, self.num_block, self.num_block // 4)
        ][1:]
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim,
                             num_heads=deform_num_heads,
                             n_points=n_points,
                             init_values=init_values,
                             drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer,
                             with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio,
                             deform_ratio=deform_ratio,
                             extra_extractor=((True if i ==
                             len(interaction_indexes) - 1
                             else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size[0] // 16,
                                      self.pretrain_size[1] // 16,
                                      -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5,
                               H_ - 0.5,
                               H_,
                               dtype=torch.float32,
                               device=device),
                torch.linspace(0.5,
                               W_ - 0.5,
                               W_,
                               dtype=torch.float32,
                               device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def _deform_inputs(self, x):
        bs, c, h, w = x.shape
        spatial_shapes = torch.as_tensor([(h // 8, w // 8), (h // 16, w // 16),
                                          (h // 32, w // 32)],
                                         dtype=torch.long,
                                         device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self._get_reference_points([(h // 16, w // 16)],
                                                      x.device)
        deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

        spatial_shapes = torch.as_tensor([(h // 16, w // 16)],
                                         dtype=torch.long,
                                         device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self._get_reference_points([(h // 8, w // 8),
                                                       (h // 16, w // 16),
                                                       (h // 32, w // 32)],
                                                      x.device)
        deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

        return deform_inputs1, deform_inputs2

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = self._deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3,
                               scale_factor=4,
                               mode='bilinear',
                               align_corners=False)
            x2 = F.interpolate(x3,
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=False)
            x4 = F.interpolate(x3,
                               scale_factor=0.5,
                               mode='bilinear',
                               align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
