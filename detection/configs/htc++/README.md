# HTC++

> [Improved Hybrid Task Cascade by Swin Paper](https://arxiv.org/abs/2103.14030)

<!-- [ALGORITHM] -->

## Abstract

For system-level comparison, Swin adopts an improved HTC (denoted as HTC++) with instaboost, stronger multi-scale training (resizing the input such that the shorter side is between 400 and 1400 while the longer side is at most 1600), 6x schedule (72 epochs with the learning rate decayed at epochs 63 and 69 by a factor of 0.1), softNMS, and an extra global self-attention layer appended at the output of last stage and ImageNet-22K pre-trained model as initialization.

## Introduction

HTC++ requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
detection
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

The results on **COCO 2017val** are shown in the below table.

| Backbone           | Pre-train                                                                                                             | Lr schd | box AP | mask AP | #Param | Config                                              | Download                                                                                                             |
|:------------------:|:---------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:---------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-L      | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x      | 57.9   | 50.2    | 401M   | [config](./htc++_beit_adapter_large_fpn_3x_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.0/htc++_beit_adapter_large_fpn_3x_coco.pth.tar) |
| ViT-Adapter-L (MS) | -                                                                                                                     | -       | 59.8   | 51.7    | 401M   | TODO                                                | -                                                                                                                    |

- MS denotes multi-scale testing. Note that the ms config is only for testing.
- We use 16 A100 GPUs with 1 image/GPU for ViT-Adapter-L models.

The results on **COCO 2017test-dev** are shown in the below table.

| Backbone           | Pre-train                                                                                                             | Lr schd | box AP | mask AP | #Param | Config                                                 | Download                                                                                                             |
|:------------------:|:---------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-L      | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x      | 58.5   | 50.8    | 401M   | [config](./htc++_beit_adapter_large_fpn_3x_coco.py)    | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.0/htc++_beit_adapter_large_fpn_3x_coco.pth.tar) |
| ViT-Adapter-L (MS) | -                                                                                                                     | -       | 60.1   | 52.1    | 401M   | [config](./htc++_beit_adapter_large_fpn_3x_coco_ms.py) | -                                                                                                                    |
