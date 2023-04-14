# Upgraded Mask R-CNN

> [Upgraded Mask R-CNN](https://arxiv.org/abs/2111.11429)

<!-- [ALGORITHM] -->

## Introduction

Relative to the original Mask R-CNN, Li et al. modernized several of its modules. Concisely, the modifications include: 

1. following the convolutions in FPN with batch normalization (BN).

2. using two convolutional layers in the region proposal network (RPN) instead of one.

3. using four convolutional layers with BN followed by one linear layer for the region-of-interest (RoI) classification and box regression head instead of a two-layer MLP without normalization.

4. following the convolutions in the standard mask head with BN.

## Results and Models

| Backbone      | Pretrain                                                                    | Lr schd | box AP | mask AP | #Param | Config                                                      | Download                                                                                                                     |
|:-------------:|:----------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:-----------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------:|
| ViT-B         | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | 25ep+LSJ | 48.1   | 43.2    | 116M   | -                                                           | -                                                                                                                            |
| ViT-B         | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | 50ep+LSJ | 50.1   | 44.6    | 116M   | -                                                           | -                                                                                                                            |
| ViT-B         | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | 100ep+LSJ | 50.3   | 44.9    | 116M   | -                                                           | -                                                                                                                            |
| ViT-Adapter-B | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | 25ep+LSJ | 50.3   | 44.7    | 122M   | [config](./mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.4/mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.pth.tar) |
| ViT-Adapter-B | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | 50ep+LSJ | 50.8   | 45.1    | 122M   | [config](./mask_rcnn_mae_adapter_base_lsj_fpn_50ep_coco.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.4/mask_rcnn_mae_adapter_base_lsj_fpn_50ep_coco.pth.tar) |
