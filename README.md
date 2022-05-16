# ViT-Adapter

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/168661265-494ecb50-353f-471c-a1d1-c3f98bd82b74.png">


The official implementation of the paper "[Vision Transformer Adapter for Dense Predictions]()".

## News

[2022/05/17]: ViT-Adapter-L yields 60.1 box AP and 52.1 mask AP on COCO test-dev.\
[2022/05/05]: ViT-Adapter-L achieves the [SOTA](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k) on ADE20K val set with 60.5 mIoU!

## Abstract

This work investigates a simple yet powerful adapter for Vision Transformer (ViT). Unlike recent visual transformers that introduce vision-specific inductive biases into their architectures, ViT achieves inferior performance on dense prediction tasks due to lacking prior information of images. To solve this issue, we propose a Vision Transformer Adapter (ViT-Adapter), which can remedy the defects of ViT and achieve comparable performance to vision-specific models by introducing inductive biases via an additional architecture. Specifically, the backbone in our framework is a vanilla transformer that can be pre-trained with multi-modal data. When fine-tuning on downstream tasks, a modality-specific adapter is used to introduce the data and tasks' prior information into the model, making it suitable for these tasks. We verify the effectiveness of our ViT-Adapter on multiple downstream tasks, including object detection, instance segmentation, and semantic segmentation. Notably, with comparable parameters, ViT-Adapter-B yields 49.6 box AP on the COCO dataset, surpassing Swin-B (48.6 box AP) by 1.0 points. With multi-modal pre-training, the performance of our model can be further improved to 50.7 box AP. We hope that the proposed ViT-Adapter could serve as an alternative for vision-specific transformers and facilitate future research.

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
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
