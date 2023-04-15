# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
pretrained = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        layer_scale=False,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,
                     14, 14, None, 14, 14, None],
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0002, weight_decay=0.01,
    paramwise_cfg=dict(
    custom_keys={
        'level_embed': dict(decay_mult=0.),
        'pos_embed': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'bias': dict(decay_mult=0.)
    }))
optimizer_config = dict(grad_clip=None)
evaluation = dict(save_best='auto')
# fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)