# Cascade R-CNN

> [Cascade R-CNN: High Quality Object Detection and Instance Segmentation](https://arxiv.org/abs/1906.09756)

<!-- [ALGORITHM] -->

## Introduction

In object detection, the intersection over union (IoU) threshold is frequently used to define positives/negatives. The threshold used to train a detector defines its quality. While the commonly used threshold of 0.5 leads to noisy (low-quality) detections, detection performance frequently degrades for larger thresholds. This paradox of high-quality detection has two causes: 1) overfitting, due to vanishing positive samples for large thresholds, and 2) inference-time quality mismatch between detector and test hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, composed of a sequence of detectors trained with increasing IoU thresholds, is proposed to address these problems. The detectors are trained sequentially, using the output of a detector as training set for the next. This resampling progressively improves hypotheses quality, guaranteeing a positive training set of equivalent size for all detectors and minimizing overfitting. The same cascade is applied at inference, to eliminate quality mismatches between hypotheses and detectors. An implementation of the Cascade R-CNN without bells or whistles achieves state-of-the-art performance on the COCO dataset, and significantly improves high-quality detection on generic and specific object detection datasets, including VOC, KITTI, CityPerson, and WiderFace. Finally, the Cascade R-CNN is generalized to instance segmentation, with nontrivial improvements over the Mask R-CNN.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143872197-d99b90e4-4f05-4329-80a4-327ac862a051.png"/>
</div>

## Results and Models

| Backbone      | Pretrain                                                                         | Lr schd | box AP | mask AP | #Param | Config                                                          | Download                                                                                                                                                                                                                      |
|:-------------:|:---------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:---------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS   | 51.5   | 44.3    | 86M    | [config](./cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.3/cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.log) |
| ViT-Adapter-B | [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)  | 3x+MS   | 52.1   | 44.8    | 158M   | [config](./cascade_mask_rcnn_deit_adapter_base_fpn_3x_coco.py)  | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/cascade_mask_rcnn_deit_adapter_base_fpn_3x_coco.pth) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/cascade_mask_rcnn_deit_adapter_base_fpn_3x_coco.log)                                                                                                                                                                                                                          |
