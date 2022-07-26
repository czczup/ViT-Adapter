# ViT-Adapter

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=vision-transformer-adapter-for-dense)

The official implementation of the paper "[Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534)".

## News
(2022/06/09) ViT-Adapter-L yields 60.4 box AP and 52.5 mask AP on COCO test-dev.\
(2022/06/04) Code and models are released.\
(2022/05/17) ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev. \
(2022/05/12) ViT-Adapter-L reaches 85.2 mIoU on Cityscapes test set without coarse data.\
(2022/05/05) ViT-Adapter-L achieves the SOTA on ADE20K val set with 60.5 mIoU!

## Abstract

This work investigates a simple yet powerful adapter for Vision Transformer (ViT). Unlike recent visual transformers that introduce vision-specific inductive biases into their architectures, ViT achieves inferior performance on dense prediction tasks due to lacking prior information of images. To solve this issue, we propose a Vision Transformer Adapter (ViT-Adapter), which can remedy the defects of ViT and achieve comparable performance to vision-specific models by introducing inductive biases via an additional architecture. Specifically, the backbone in our framework is a vanilla transformer that can be pre-trained with multi-modal data. When fine-tuning on downstream tasks, a modality-specific adapter is used to introduce the data and tasks' prior information into the model, making it suitable for these tasks. We verify the effectiveness of our ViT-Adapter on multiple downstream tasks, including object detection, instance segmentation, and semantic segmentation. Notably, when using HTC++, our ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev, surpassing Swin-L by 1.4 box AP and 1.0 mask AP. For semantic segmentation, our ViT-Adapter-L establishes a new state-of-the-art of 60.5 mIoU on ADE20K val. We hope that the proposed ViT-Adapter could serve as an alternative for vision-specific transformers and facilitate future research.

## Method

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/168661265-494ecb50-353f-471c-a1d1-c3f98bd82b74.png">

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/168972788-60e08b3b-5f4a-43d6-b731-3ea52878534f.png">

## Catalog

- [x] Segmentation checkpoints
- [x] Segmentation code
- [x] Detection checkpoints
- [x] Detection code
- [x] Initialization

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{chen2022vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE.md) file.
