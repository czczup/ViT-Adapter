# ViT-Adapter
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=vision-transformer-adapter-for-dense)


The official implementation of the paper "[Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534)".

## News

(2022/05/17) ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev.\
(2022/05/12) ViT-Adapter-L reaches 85.2 mIoU on Cityscapes test set without coarse data.\
(2022/05/05) ViT-Adapter-L achieves the SOTA on ADE20K val set with 60.5 mIoU!

## Abstract

This work investigates a simple yet powerful adapter for Vision Transformer (ViT). Unlike recent visual transformers that introduce vision-specific inductive biases into their architectures, ViT achieves inferior performance on dense prediction tasks due to lacking prior information of images. To solve this issue, we propose a Vision Transformer Adapter (ViT-Adapter), which can remedy the defects of ViT and achieve comparable performance to vision-specific models by introducing inductive biases via an additional architecture. Specifically, the backbone in our framework is a vanilla transformer that can be pre-trained with multi-modal data. When fine-tuning on downstream tasks, a modality-specific adapter is used to introduce the data and tasks' prior information into the model, making it suitable for these tasks. We verify the effectiveness of our ViT-Adapter on multiple downstream tasks, including object detection, instance segmentation, and semantic segmentation. Notably, when using HTC++, our ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev, surpassing Swin-L by 1.4 box AP and 1.0 mask AP. For semantic segmentation, our ViT-Adapter-L establishes a new state-of-the-art of 60.5 mIoU on ADE20K val. We hope that the proposed ViT-Adapter could serve as an alternative for vision-specific transformers and facilitate future research.

## Method

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/168661265-494ecb50-353f-471c-a1d1-c3f98bd82b74.png">

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/168972788-60e08b3b-5f4a-43d6-b731-3ea52878534f.png">


## SOTA Model Zoo

### COCO test-dev

| Method | Framework | Pre-train | Lr schd | box AP | mask AP | #Param |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-Adapter-L | HTC++ | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x  | [58.5](https://drive.google.com/file/d/11zpPSvmuAn7aP5brxzHE8naObnOfFxby/view?usp=sharing) | [50.8](https://drive.google.com/file/d/1wIbtzfHfPqkvZaSivzcsh4HWu1oSiun6/view?usp=sharing) | 401M |
| ViT-Adapter-L (MS) | HTC++ | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x  | [60.1](https://drive.google.com/file/d/1i-qjgUK4CMwZcmu5pkndldwfVbdkw5sU/view?usp=sharing) | [52.1](https://drive.google.com/file/d/16mlEOPY7K-Xpx_CL650A-LWbVDm2vl4X/view?usp=sharing) | 401M |

### ADE20K val
| Method | Framework | Pre-train | Iters | Crop Size | mIoU | +MS | #Param |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-Adapter-L | UperNet | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 160k | 640 | [58.0](https://drive.google.com/file/d/1KsV4QPfoRi5cj2hjCzy8VfWih8xCTrE3/view?usp=sharing) | [58.4](https://drive.google.com/file/d/1haeTUvQhKCM7hunVdK60yxULbRH7YYBK/view?usp=sharing) | 451M |
| ViT-Adapter-L | Mask2Former | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 160k | 640 | [58.3](https://drive.google.com/file/d/1jj56lSbc2s4ZNc-Hi-w6o-OSS99oi-_g/view?usp=sharing) | [59.0](https://drive.google.com/file/d/1hgpZB5gsyd7LTS7Aay2CbHmlY10nafCw/view?usp=sharing) | 568M |
| ViT-Adapter-L | Mask2Former | [COCO-Stuff-164k]() | 80k | 896 | [59.4](https://drive.google.com/file/d/1B_1XSwdnLhjJeUmn1g_nxfvGJpYmYWHa/view?usp=sharing) | [60.5](https://drive.google.com/file/d/1UtjmgcYKR-2h116oQXklUYOVcTw15woM/view?usp=sharing) | 571M |

### Cityscapes val/test

| Method        | Framework   | Pre-train     | Iters | Crop Size | val mIoU                                                                                   | val/test +MS                                                                                                                                                                                                                 | #Param |
|:-------------:|:-----------:|:-------------:|:-----:|:---------:|:------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|
| ViT-Adapter-L | Mask2Former | [Mapillary]() | 80k   | 896       | [84.9](https://drive.google.com/file/d/1LKy0zz-brCBbKGmUWquadILaBHdDLR6s/view?usp=sharing) | [85.8](https://drive.google.com/file/d/1LSJvK1BPSbzm9eWpKL8Xo7RmYBrd2xux/view?usp=sharing)/[85.2](https://www.cityscapes-dataset.com/anonymous-results/?id=0ca6821dc3183ff970bd5266f812df2eaa4519ecb1973ca1308d65a3b546bf27) | 571M   |

### COCO-Stuff-10k

| Method        | Framework   | Pre-train                                                                                                             | Iters | Crop Size | mIoU                                                                                       | +MS                                                                                        | #Param |
|:-------------:|:-----------:|:---------------------------------------------------------------------------------------------------------------------:|:-----:|:---------:|:------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:------:|
| ViT-Adapter-L | UperNet     | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 80k   | 512       | [51.0](https://drive.google.com/file/d/1xZodiAvOLGaLtMGx_btYVZIMC2VKrDhI/view?usp=sharing) | [51.4](https://drive.google.com/file/d/1bmFG9GA4bRqOEJfqXcO7nWYPwG3wSk2J/view?usp=sharing) | 451M   |
| ViT-Adapter-L | Mask2Former | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 40k   | 512       | [53.2](https://drive.google.com/file/d/1Buewc1n7GBAcBDXeia-QarujrDZqc_Sx/view?usp=sharing) | [54.2](https://drive.google.com/file/d/1kQgJUHDeQoO3pPY6QoXRKwyF7heT7wCJ/view?usp=sharing) | 568M   |

### Pascal Context

| Method        | Framework   | Pre-train                                                                                                             | Iters | Crop Size | mIoU                                                                                       | +MS                                                                                        | #Param |
|:-------------:|:-----------:|:---------------------------------------------------------------------------------------------------------------------:|:-----:|:---------:|:------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:------:|
| ViT-Adapter-L | UperNet     | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 80k   | 480       | [67.0](https://drive.google.com/file/d/1BxnwkxGZzncpd_G4cDmHPB6Pq681YogD/view?usp=sharing) | [67.5](https://drive.google.com/file/d/1Ff-6CGyNs5_ORrlLnyYkV9spE59UjwiT/view?usp=sharing) | 451M   |
| ViT-Adapter-L | Mask2Former | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 40k   | 480       | [67.8](https://drive.google.com/file/d/1AbC7DZeTjZVIqNTRWFCkc16FHEmxvDNK/view?usp=sharing) | [68.2](https://drive.google.com/file/d/1gl_gPF7pxjWKvUubK7g1CB5wtihuAWgA/view?usp=sharing) | 568M   |

## Regular Model Zoo

### COCO mini-val

#### Baseline Detectors

| Method | Framework | Pre-train | Lr schd | Aug | box AP | mask AP | #Param |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-Adapter-T | Mask R-CNN | DeiT | 3x  | Yes | 46.0 | 41.0 | 28M |
| ViT-Adapter-S | Mask R-CNN | DeiT | 3x  | Yes | 48.2 | 42.8 | 48M |
| ViT-Adapter-B | Mask R-CNN | DeiT | 3x  | Yes | 49.6 | 43.6 | 120M |
| ViT-Adapter-L | Mask R-CNN | DeiT | 3x  | Yes | 50.9 | 44.8 | 348M |

#### Advanced Detectors

| Method        | Framework          | Pre-train | Lr schd | Aug | box AP | mask AP | #Param |
|:-------------:|:------------------:|:---------:|:-------:|:---:|:------:|:-------:|:------:|
| ViT-Adapter-S | Cascade Mask R-CNN | DeiT      | 3x      | Yes | 51.5   | 44.5    | 86M    |
| ViT-Adapter-S | ATSS               | DeiT      | 3x      | Yes | 49.6   | -       | 36M    |
| ViT-Adapter-S | GFL                | DeiT      | 3x      | Yes | 50.0   | -       | 36M    |
| ViT-Adapter-S | Sparse R-CNN       | DeiT      | 3x      | Yes | 48.1   | -       | 110M   |
| ViT-Adapter-B | Upgraded Mask R-CNN | MAE      | 25ep    | LSJ | 50.3   | 44.7    | 122M    |
| ViT-Adapter-B | Upgraded Mask R-CNN | MAE      | 50ep    | LSJ | 50.8   | 45.1    | 122M    |

### ADE20K val

| Method        | Framework | Pre-train | Iters | Crop Size | mIoU | +MS  | #Param |
|:-------------:|:---------:|:---------:|:-----:|:---------:|:----:|:----:|:------:|
| ViT-Adapter-T | UperNet   | DeiT      | 160k  | 512       | 42.6 | 43.6 | 36M  |
| ViT-Adapter-S | UperNet   | DeiT      | 160k  | 512       | 46.6 | 47.4 | 58M  |
| ViT-Adapter-B | UperNet   | DeiT      | 160k  | 512       | 48.1 | 49.2 | 134M |
| ViT-Adapter-B | UperNet   | AugReg    | 160k  | 512       | 51.9 | 52.5 | 134M |
| ViT-Adapter-L | UperNet   | AugReg    | 160k  | 512       | 53.4 | 54.4 | 364M |


## Catalog

- [ ] Detection checkpoints
- [ ] Segmentation checkpoints
- [ ] Model code
- [ ] Detection logs
- [ ] Segmentation logs
- [x] Initialization

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{chen2021vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE.md) file.
