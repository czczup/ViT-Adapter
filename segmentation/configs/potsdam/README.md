# ISPRS Potsdam

<!-- [ALGORITHM] -->

## Introduction

The Potsdam dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/). The `2_Ortho_RGB.zip` and `5_Labels_all_noBoundary.zip` are required.

For Potsdam dataset, please run the [script](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/potsdam.py) provided by mmseg official to download and re-organize the dataset.

```python
python /path/to/convertor/potsdam.py /path/to/potsdam
```

In the default setting, it will generate 3456 images for training and 2016 images for validation.

## Results and Models

| Method      | Backbone      | Pretrain | Batch Size | Lr schd | Crop Size | mIoU (SS) | #Param | Config                                                           | Download                                               |
|:-----------:|:-------------:|:--------:|:----------:|:-------:|:---------:|:---------:|:------:|:----------------------------------------------------------------:|:------------------------------------------------------:|
| Mask2Former | ViT-Adapter-L | BEiT-L   | 8x1        | 80k     | 512       | 80.0      | 352M   | [config](./mask2former_beit_adapter_large_512_80k_potsdam_ss.py) | [log](https://github.com/czczup/ViT-Adapter/issues/38) |
