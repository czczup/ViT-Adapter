# ATSS

> [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424)

<!-- [ALGORITHM] -->

## Introduction

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have become popular due to the proposal of FPN and Focal Loss. In this paper, Zhang et al. first point out that the essential difference between anchor-based and anchor-free detection is actually how to define positive and negative training samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative samples during training, there is no obvious difference in the final performance, no matter regressing from a box or a point. This shows that how to select positive and negative training samples is important for current object detectors. Then, they propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples according to statistical characteristics of object. It significantly improves the performance of anchor-based and anchor-free detectors and bridges the gap between them. Finally, they discuss the necessity of tiling multiple anchors per location on the image to detect objects. Extensive experiments conducted on MS COCO support their aforementioned analysis and conclusions. 

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143870776-c81168f5-e8b2-44ee-978b-509e4372c5c9.png"/>
</div>

## Results and Models

| Backbone      | Pretrain                                                                         | Lr schd | box AP | #Param | Config                                             | Download                                                                                                            |
|:-------------:|:---------------------------------------------------------------------------------:|:-------:|:------:|:------:|:--------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS      | 49.6   | 36M    | [config](./atss_deit_adapter_small_fpn_3x_coco.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.5/atss_deit_adapter_small_fpn_3x_coco.pth.tar) |
