# ADE20K

<!-- [ALGORITHM] -->

## Introduction

The ADE20K semantic segmentation dataset contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.

## Results and Models

| Method      | Backbone      | Pre-train                                                                                                             | Batch Size | Lr schd | Crop Size | mIoU (SS) | mIoU (MS) | #Param | Config                                                           | Download                                                                                                                                                                                                               |
|:-----------:|:-------------:|:---------------------------------------------------------------------------------------------------------------------:|:----------:|:-------:|:---------:|:---------:|:---------:|:------:|:----------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| UperNet     | ViT-Adapter-L | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 8x2        | 160k    | 640       | 58.0      | 58.4      | 451M   | [config](./upernet_beit_adapter_large_640_160k_ade20k_ss.py)     | [model]() \| [log]()                                                                                                                                                                                                   |
| Mask2Former | ViT-Adapter-L | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 8x2        | 160k    | 640       | 58.3      | 59.0      | 568M   | [config](./mask2former_beit_adapter_large_640_160k_ade20k_ss.py) | [model]() \| [log]()                                                                                                                                                                                                   |
| Mask2Former | ViT-Adapter-L | COCO-Stuff-164K                                                                                                       | 16x1       | 80k     | 896       | 59.4      | 60.5      | 571M   | [config](./mask2former_beit_adapter_large_896_80k_ade20k_ss.py)  | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.0/mask2former_beit_adapter_large_896_80k_ade20k.zip) \| [log](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.0/20220430_154104.log) |
