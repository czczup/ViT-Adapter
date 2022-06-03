import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath
from torch.nn.init import normal_
from .base.vit import TIMMVisionTransformer
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn

_logger = logging.getLogger(__name__)


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        super(DWConv, self).__init__()
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


class ExtractLayer(nn.Module):
    
    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=ratio)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat),
                         spatial_shapes, level_start_index, None)
        query = query + attn
        return query


class InsertLayer(nn.Module):
    
    def __init__(self,
                 dim,
                 num_heads=6,
                 n_points=4,
                 n_levels=1,
                 ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat),
                         spatial_shapes, level_start_index, None)
        return query + self.gamma * attn


class InteractBlock(nn.Module):
    
    def __init__(self, dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_ffn=True,
                 ffn_ratio=0.25,
                 init_values=0.,
                 extract_deform_ratio=1.0,
                 insert_deform_ratio=1.0):
        super().__init__()
        self.extract = ExtractLayer(dim=dim, n_levels=1, num_heads=num_heads,
                                    n_points=n_points, norm_layer=norm_layer, ratio=extract_deform_ratio)
        self.insert = InsertLayer(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                  n_points=n_points, norm_layer=norm_layer, ratio=insert_deform_ratio)
        
        self.with_ffn = with_ffn
        if with_ffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * ffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, c, blocks, deform_pkg1, deform_pkg2, H, W):
        x = self.insert(x, deform_pkg1[0], c, deform_pkg1[1], deform_pkg1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extract(c, deform_pkg2[0], x, deform_pkg2[1], deform_pkg2[2])
        if self.with_ffn:
            c = c + self.drop_path(self.ffn(self.ffn_norm(c), H, W))
        return x, c


class ExtractBlock(nn.Module):
    
    def __init__(self, dim,
                 num_heads=6,
                 n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0.,
                 drop_path=0.,
                 with_ffn=True,
                 ffn_ratio=0.25,
                 deform_ratio=1.0):
        super().__init__()
        self.extract = ExtractLayer(dim=dim, n_levels=1, num_heads=num_heads,
                                    n_points=n_points, norm_layer=norm_layer, ratio=deform_ratio)
        
        self.with_ffn = with_ffn
        if with_ffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * ffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, c, x, deform_pkg, H, W):
        
        c = self.extract(c, deform_pkg[0], x, deform_pkg[1], deform_pkg[2])
        if self.with_ffn:
            c = c + self.drop_path(self.ffn(self.ffn_norm(c), H, W))
        return c


class ConvBranch(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super(ConvBranch, self).__init__()
        
        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
    
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
                 interact_with_ffn=False,
                 cffn_ratio=0.25,
                 num_extract_block=3,
                 extract_with_ffn=True,
                 extract_ffn_ratio=0.25,
                 add_vit_feature=True,
                 extract_deform_ratio=1.0,
                 deform_ratio=1.0,
                 pretrained=None,
                 *args,
                 **kwargs):
        
        super().__init__(num_heads=num_heads, pretrained=pretrained,
                         *args, **kwargs)
        
        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [i for i in range(-1, self.num_block, self.num_block // 4)][1:]
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.conv_branch = ConvBranch(inplanes=conv_inplane, embed_dim=embed_dim)
        self.interact_blocks = nn.Sequential(*[
            InteractBlock(dim=embed_dim,
                          num_heads=deform_num_heads,
                          n_points=n_points,
                          init_values=init_values,
                          drop_path=self.drop_path_rate,
                          norm_layer=self.norm_layer,
                          with_ffn=interact_with_ffn,
                          ffn_ratio=cffn_ratio,
                          extract_deform_ratio=deform_ratio,
                          insert_deform_ratio=deform_ratio
                          ) for _ in range(len(interaction_indexes))])
        self.extract_blocks = nn.Sequential(*[
            ExtractBlock(dim=embed_dim,
                         num_heads=deform_num_heads,
                         n_points=n_points,
                         norm_layer=self.norm_layer,
                         with_ffn=extract_with_ffn,
                         ffn_ratio=extract_ffn_ratio,
                         deform_ratio=extract_deform_ratio
                         ) for _ in range(num_extract_block)
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)
        
        self.up.apply(self._init_weights)
        self.conv_branch.apply(self._init_weights)
        self.interact_blocks.apply(self._init_weights)
        self.extract_blocks.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        
        self.init_weights(pretrained)

    
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
                                      self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode="bicubic",
                                  align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed
    
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    
    def _get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    
    def forward_deform_pkgs(self, x):
        bs, c, h, w = x.shape
        spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                          (h // 16, w // 16),
                                          (h // 32, w // 32)], dtype=torch.long).cuda()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self._get_reference_points([(h // 16, w // 16)], "cuda").cuda()
        deform_pkg1 = [reference_points, spatial_shapes, level_start_index]
        
        spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long).cuda()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self._get_reference_points([(h // 8, w // 8),
                                                       (h // 16, w // 16),
                                                       (h // 32, w // 32)], "cuda").cuda()
        deform_pkg2 = [reference_points, spatial_shapes, level_start_index]
        
        return deform_pkg1, deform_pkg2
    
    def add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def forward(self, x):
        deform_pkg1, deform_pkg2 = self.forward_deform_pkgs(x)
        
        c1, c2, c3, c4 = self.conv_branch(x)
        c2, c3, c4 = self.add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        
        outs = []
        for i, layer in enumerate(self.interact_blocks):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_pkg1, deform_pkg2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        for extract_block in self.extract_blocks:
            c = extract_block(c, x, deform_pkg2, H, W)
            
        c2 = c[:, 0: c2.size(1), :]
        c3 = c[:, c2.size(1): c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]
        
        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1
        
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
        
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
