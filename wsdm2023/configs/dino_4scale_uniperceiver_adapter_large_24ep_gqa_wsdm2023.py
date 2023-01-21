_base_ = [
    '_base_/datasets/wsdm2023.py',
    '_base_/default_runtime.py'
]
# https://github.com/czczup/ViT-Adapter/releases/download/wsdm2023/dino_4scale_uniperceiver_adapter_large_6ep_gqa.pth
load_from = 'pretrained/dino_4scale_uniperceiver_adapter_large_6ep_gqa.pth'
model = dict(
    type='GroundingDINO',
    with_aux_loss=True,
    backbone=dict(
        type='UniPerceiverAdapter',
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        num_classes=1,
        with_cp=True,
        out_indices=(1, 2, 3),
        num_cross_attn=0,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        window_attn=[False] * 24,
        window_size=[None] * 24,
        pretrained=None),
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024, 1024, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DINOHead',
        num_query=100,
        num_classes=1,
        in_channels=2048,  # TODO
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='DinoTransformer',
            two_stage_num_proposals=100,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        dropout=0.0),  # 0.1 for DeformDETR
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.,
                        use_checkpoint=True,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),  # 0.1 for DeformDETR
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),  # 0.1 for DeformDETR
                    ],
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.,
                        use_checkpoint=True,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=1))  # TODO: Originally 100
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadRefer', tag='question'),
    dict(type='RandomParaPhrase',
         phrase_cache='data/wsdm2023/annotations/paraphrase_train.json'),
    dict(type='RandomFlipWithRefer', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='TokenizeRefer', max_sent_len=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'refer', 'r_mask', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRefer', tag='question'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 600), (1333, 800), (1333, 1000)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlipWithRefer'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='TokenizeRefer', max_sent_len=64),
            dict(type='Collect', keys=['img', 'refer', 'r_mask'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(filter_empty_gt=True, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
evaluation = dict(save_best='auto')
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]