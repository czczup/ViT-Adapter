# Panoptic Segmentation with Mask2Former

> [Masked-attention Mask Transformer for Universal Image Segmentation](http://arxiv.org/abs/2112.01527)

<!-- [ALGORITHM] -->

## Abstract

Image segmentation is about grouping pixels with different semantics, e.g., category or instance membership, where each choice of semantics defines a task. While only the semantics of each task differ, current research focuses on designing specialized architectures for each task. We present Masked-attention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

<div align=center>
<img src="https://camo.githubusercontent.com/455d3116845b1d580b1f8a8542334b9752fdf39364deee2951cdd231524c7725/68747470733a2f2f626f77656e63303232312e6769746875622e696f2f696d616765732f6d61736b666f726d657276325f7465617365722e706e67" height="300"/>
</div>

## Introduction

Mask2Former requires COCO and [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) dataset for training and evaluation. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
|   |   |   ├── instances_train2017.json
|   |   |   ├── instances_val2017.json
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Note

**Environment:** For panoptic segmentation, please upgrade `mmdetection` to `v2.25.0`; otherwise, some errors may occur.

**Test:** If you meet this error during testing:

```
TypeError: Mask2Former: __init__() got an unexpected keyword argument 'pretrained'
    raise type(e)(f'{obj_cls.__name__}: {e}')
```

Please comment out `line 132` in the `test.py`

```
# cfg.model.pretrained = None
```

**Evaluation Server:** See [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/tutorials/test_results_submission.md).

## Results and Models

The results on COCO mini-val are shown in the below table.

| Backbone      | Pretrain                                                                                                                  | Lr schd | PQ   | PQst | PQth | box AP | mask AP | Config                                                                | Download                                                                                                                                                                                                                                                                   |
|:-------------:|:--------------------------------------------------------------------------------------------------------------------------:|:-------:|:----:|:----:|:----:|:------:|:-------:|:---------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-L | [BEiTv2-L](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth) | 3x+MS   | 58.4 | 65.0 | 48.4 | 52.9   | 48.9    | [config](./mask2former_beitv2_adapter_large_16x1_3x_coco-panoptic.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/panoptic/mask2former_beitv2_adapter_large_16x1_3x_coco-panoptic.pth) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beitv2_adapter_large_16x1_3x_coco-panoptic.txt) |

To evaluate ViT-Adapter-L + Mask2Former on COCO val2017 on a single node with 8 gpus:

```shell
sh dist_test.sh <config> <checkpoint> 8 --eval PQ bbox segm
```

This should give:

```
Panoptic Evaluation Results:
+--------+--------+--------+--------+------------+
|        | PQ     | SQ     | RQ     | categories |
+--------+--------+--------+--------+------------+
| All    | 58.387 | 83.694 | 69.004 | 133        |
| Things | 64.998 | 84.892 | 76.147 | 80         |
| Stuff  | 48.409 | 81.886 | 58.223 | 53         |
+--------+--------+--------+--------+------------+

{'PQ': 58.38731797774417, 'SQ': 83.69399287047781, 'RQ': 69.00400946467386, 'PQ_th': 64.99819608860089, 'SQ_th': 84.89159636576206, 'RQ_th': 76.14657876127653, 'PQ_st': 48.40863403682838, 'SQ_st': 81.88628948136942, 'RQ_st': 58.222772790556576, 'PQ_copypaste': '58.387 83.694 69.004 64.998 84.892 76.147 48.409 81.886 58.223', 'bbox_mAP': 0.529, 'bbox_mAP_50': 0.737, 'bbox_mAP_75': 0.571, 'bbox_mAP_s': 0.36, 'bbox_mAP_m': 0.565, 'bbox_mAP_l': 0.709, 'bbox_mAP_copypaste': '0.529 0.737 0.571 0.360 0.565 0.709', 'segm_mAP': 0.489, 'segm_mAP_50': 0.738, 'segm_mAP_75': 0.529, 'segm_mAP_s': 0.303, 'segm_mAP_m': 0.535, 'segm_mAP_l': 0.708, 'segm_mAP_copypaste': '0.489 0.738 0.529 0.303 0.535 0.708'}
```
