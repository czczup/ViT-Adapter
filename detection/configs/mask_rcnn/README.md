# Mask R-CNN

> [Mask R-CNN](https://arxiv.org/abs/1703.06870)

<!-- [ALGORITHM] -->

## Introduction

Mask R-CNN is a conceptually simple, flexible, and general framework for object instance segmentation. It efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. And it extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. 

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967081-c2552bed-9af2-46c4-ae44-5b3b74e5679f.png"/>
</div>

## Results and Models

| Backbone      | Pre-train                                                                                                                                                        | Lr schd | box AP | mask AP | #Param | Config                                                         | Download                                                                                                                        |
|:-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:--------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-T | [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                                                 | 3x      | 46.0   | 41.0    | 28M    | [config](./mask_rcnn_deit_adapter_tiny_fpn_3x_coco.py)         | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.pth.tar)         |
| ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                                                | 3x      | 48.2   | 42.8    | 48M    | [config](./mask_rcnn_deit_adapter_small_fpn_3x_coco.py)        | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar)        |
| ViT-Adapter-B | [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                                                 | 3x      | 49.6   | 43.6    | 120M   | [config](./mask_rcnn_deit_adapter_base_fpn_3x_coco.py)         | [model]()                                                                                                                       |
| ViT-Adapter-B | [Uni-Perceiver](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.1/uniperceiver_pretrain.pth)                                                        | 3x      | 50.7   | 44.9    | 120M   | [config](./mask_rcnn_uniperceiver_adapter_base_fpn_3x_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.1/mask_rcnn_uniperceiver_adapter_base_fpn_3x_coco.pth.tar) |
| ViT-Adapter-L | [AugReg](https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz) | 3x      | 50.9   | 44.8    | 348M   | [config](./mask_rcnn_augreg_adapter_large_fpn_3x_coco.py)      | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_augreg_adapter_large_fpn_3x_coco.pth.tar)      |