# Sparse R-CNN

> [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450)

<!-- [ALGORITHM] -->

## Abstract

Sparse R-CNN is a purely sparse method for object detection in images. Existing works on object detection heavily rely on dense object candidates, such as k anchor boxes pre-defined on all grids of image feature map of size H×W. In Sparse R-CNN, however, a fixed sparse set of learned object proposals, total length of N, are provided to object recognition head to perform classification and location. By eliminating HWk (up to hundreds of thousands) hand-designed object candidates to N (e.g. 100) learnable proposals, Sparse R-CNN completely avoids all efforts related to object candidates design and many-to-one label assignment. More importantly, final predictions are directly output without non-maximum suppression post-procedure. Sparse R-CNN demonstrates accuracy, run-time and training convergence performance on par with the well-established detector baselines on the challenging COCO dataset, e.g., achieving 45.0 AP in standard 3× training schedule and running at 22 fps using ResNet-50 FPN model. 

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143998489-8a5a687d-ceec-4590-8347-708e427e7dfe.png" height="300"/>
</div>

## Results and Models

| Backbone      | Pretrain                                                                         | Lr schd | box AP | #Param | Config                                                    | Download                                                                                                                   |
|:-------------:|:---------------------------------------------------------------------------------:|:-------:|:------:|:------:|:---------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS   | 48.1   | 110M   | [config](./sparse_rcnn_deit_adapter_small_fpn_3x_coco.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/sparse_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar) |
