# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_augreg.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# pretrained = 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz'
# pretrained = 'https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth'
pretrained = 'pretrained/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        img_size=384,
        pretrain_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.4,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        window_attn=[True, True, True, True, True, False,
                     True, True, True, True, True, False,
                     True, True, True, True, True, False,
                     True, True, True, True, True, False],
        window_size=[14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None],
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        num_outs=5)
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.80))
optimizer_config = dict(grad_clip=None)
evaluation = dict(save_best='auto')
# fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
)
