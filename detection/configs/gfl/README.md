# GFL

> [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)

<!-- [ALGORITHM] -->

## Introduction

One-stage detector basically formulates object detection as dense classification and localization. The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an individual prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the representations of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including (1) the inconsistent usage of the quality estimation and classification between training and inference and (2) the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty in complex scenes. To address the problems, we design new representations for these elements. Specifically, we merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain continuous labels, which is beyond the scope of Focal Loss. We then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the continuous version for successful optimization. On COCO test-dev, GFL achieves 45.0% AP using ResNet-101 backbone, surpassing state-of-the-art SAPD (43.5%) and ATSS (43.6%) with higher or comparable inference speed, under the same backbone and training settings. Notably, our best model can achieve a single-model single-scale AP of 48.2%, at 10 FPS on a single 2080Ti GPU.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143887865-44dc384d-ba0d-44e8-b3d7-d5fa837838cf.png"/>
</div>

## Results and Models

| Backbone      | Pretrain                                                                         | Lr schd | box AP | #Param | Config                                             | Download                                                                                                           |
|:-------------:|:---------------------------------------------------------------------------------:|:-------:|:------:|:------:|:--------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS      | 50.0   | 36M    | [config](./gfl_deit_adapter_small_fpn_3x_coco.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/gfl_deit_adapter_small_fpn_3x_coco.pth.tar) |
