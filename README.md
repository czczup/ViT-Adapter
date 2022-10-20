# ViT-Adapter

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=vision-transformer-adapter-for-dense)

The official implementation of the paper "[Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534)".

## News

(2022/10/20) ViT-Adapter is adopted by Zhang et al. and ranked 1st in the [UVO Challenge 2022](https://arxiv.org/pdf/2210.09629.pdf).\
(2022/08/22) ViT-Adapter is adopted by [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) and created new SOTA of 62.8 mIoU on ADE20K.\
(2022/06/09) ViT-Adapter-L yields 60.4 box AP and 52.5 mask AP on COCO test-dev.\
(2022/06/04) Code and models are released.\
(2022/05/17) ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev. \
(2022/05/12) ViT-Adapter-L reaches 85.2 mIoU on Cityscapes test set without coarse data.\
(2022/05/05) ViT-Adapter-L achieves the SOTA on ADE20K val set with 60.5 mIoU!

## Abstract

This work investigates a simple yet powerful dense prediction task adapter for Vision Transformer (ViT). Unlike recently advanced variants that incorporate vision-specific inductive biases into their architectures, the plain ViT suffers inferior performance on dense predictions due to weak prior assumptions. To address this
issue, we propose the ViT-Adapter, which allows plain ViT to achieve comparable performance to vision-specific transformers. Specifically, the backbone in our
framework is a plain ViT that can learn powerful representations from large-scale
multi-modal data. When transferring to downstream tasks, a pre-training-free
adapter is used to introduce the image-related inductive biases into the model,
making it suitable for these tasks. We verify ViT-Adapter on multiple dense prediction tasks, including object detection, instance segmentation, and semantic segmentation. Notably, without using extra detection data, our ViT-Adapter-L yields
state-of-the-art 60.9 box AP and 53.0 mask AP on COCO test-dev. We hope that
the ViT-Adapter could serve as an alternative for vision-specific transformers and
facilitate future research. The code and models will be released.

## Method

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/194904519-30d2a5d1-b203-419d-a597-608ee90bb3bb.png">

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/194904786-ea9c40a3-f6ac-4fe1-90ad-976e7b9e8f03.png">

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
