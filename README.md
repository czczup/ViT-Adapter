# ViT-Adapter


The official implementation of the paper "[Vision Transformer Adapter for Dense Predictions]()".

## News

[2022/05/17]: ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev.\
[2022/05/05]: ViT-Adapter-L achieves the [SOTA](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k) on ADE20K val set with 60.5 mIoU!

## Abstract

This work investigates a simple yet powerful adapter for Vision Transformer (ViT). Unlike recent visual transformers that introduce vision-specific inductive biases into their architectures, ViT achieves inferior performance on dense prediction tasks due to lacking prior information of images. To solve this issue, we propose a Vision Transformer Adapter (ViT-Adapter), which can remedy the defects of ViT and achieve comparable performance to vision-specific models by introducing inductive biases via an additional architecture. Specifically, the backbone in our framework is a vanilla transformer that can be pre-trained with multi-modal data. When fine-tuning on downstream tasks, a modality-specific adapter is used to introduce the data and tasks' prior information into the model, making it suitable for these tasks. We verify the effectiveness of our ViT-Adapter on multiple downstream tasks, including object detection, instance segmentation, and semantic segmentation. Notably, when using HTC++, our ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev, surpassing Swin-L by 1.4 box AP and 1.0 mask AP. For semantic segmentation, our ViT-Adapter-L establishes a new state-of-the-art of 60.5 mIoU on ADE20K val, 0.6 points higher than SwinV2-G. We hope that the proposed ViT-Adapter could serve as an alternative for vision-specific transformers and facilitate future research.

## Method

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/168661265-494ecb50-353f-471c-a1d1-c3f98bd82b74.png">

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
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE.md) file.
