# Applying ViT-Adapter to Object Detection

Our detection code is developed on top of [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

For details see [Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534).

If you use this code for a paper please cite:

```
@article{chen2022vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```

## Usage

Install [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0
pip install instaboostfast # for htc++
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

## Pre-training Sources

| Name          | Type       | Year | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | ---------- | ---- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeiT          | Supervised | 2021 | ImageNet-1K  | [repo](https://github.com/facebookresearch/deit/blob/main/README_deit.md)                               | [paper](https://arxiv.org/abs/2012.12877)                                                                                                                                       |
| AugReg        | Supervised | 2021 | ImageNet-22K | [repo](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) | [paper](https://arxiv.org/abs/2106.10270)                                                                                                                                       |
| BEiT          | MIM        | 2021 | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit)                                             | [paper](https://arxiv.org/abs/2106.08254)                                                                                                                                       |
| MAE           | MIM        | 2021 | ImageNet-1K  | [repo](https://github.com/facebookresearch/mae)                                                         | [paper](https://arxiv.org/abs/2111.06377)                                                                                                                                       |
| Uni-Perceiver | Supervised | 2022 | Multi-Modal  | -                                                                                                       | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf) |

## Results and Models

**HTC++**

<table>
   <tr  align=center>
      <td rowspan="2" align=center><b>Backbone</b></td>
      <td rowspan="2" align=center><b>Pre-train</b></td>
      <td rowspan="2" align=center><b>Lr schd</b></td>
      <td colspan="2" align=center><b>mini-val</b></td>
      <td colspan="2" align=center><b>test-dev</b></td>
      <td rowspan="2" align=center><b>#Param</b></td>
      <td rowspan="2" align=center><b>Config</b></td>
      <td rowspan="2" align=center><b>Download</b></td>
   </tr>
   <tr>
      <td>box AP</td>
      <td>mask AP</td>
      <td>box AP</td>
      <td>mask AP</td>
   </tr>
   <tr align=center>
      <td>ViT-Adapter-L</td>
      <td><a href="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth">BEiT-L</a></td>
      <td>3x+MS</td>
      <td>58.4</td>
      <td>50.8</td>
      <td><a href="https://drive.google.com/file/d/1lXQxf5PJ0g0bQNkMMrhG63jal0NsmYjb/view?usp=sharing">58.9</a></td>
      <td><a href="https://drive.google.com/file/d/1nyuONJcHHXki0Cn8dCgbPZ9D_MURh47t/view?usp=sharing">51.3</a></td>
      <td>401M</td>
      <td><a href="./configs/htc++/htc++_beit_adapter_large_fpn_3x_coco.py">config</a> </td>
      <td><a href="https://github.com/czczup/ViT-Adapter/releases/download/0.3.0/htc++_beit_adapter_large_fpn_3x_coco.pth.tar">model</a></td>
   </tr>
   </tr>
   <tr align=center>
      <td>ViT-Adapter-L (TTA)</td>
      <td><a href="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth">BEiT-L</a></td>
      <td>3x+MS</td>
      <td>60.2</td>
      <td>52.2</td>
      <td><a href="https://drive.google.com/file/d/15t2Oc3FiNeLr6RnKOJ-0IbI7b2LalxbX/view?usp=sharing">60.4</a></td>
      <td><a href="https://drive.google.com/file/d/1TIPOJC6ieZS_ZRNCbo_AW4UqYAkQIjyN/view?usp=sharing">52.5</a></td>
      <td>401M</td>
      <td>-</td>
      <td>-</td>
   </tr>
</table>

**Mask R-CNN**

| Method     | Backbone      | Pre-train                                                                                                                                                                        | Lr schd | box AP | mask AP | #Param | Config                                                                           | Download                                                                                                                        |
|:----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|
| Mask R-CNN | ViT-Adapter-T | [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                                                                 | 3x+MS   | 46.0   | 41.0    | 28M    | [config](./configs/mask_rcnn/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.py)         | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.pth.tar)         |
| Mask R-CNN | ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                                                                | 3x+MS   | 48.2   | 42.8    | 48M    | [config](./configs/mask_rcnn/mask_rcnn_deit_adapter_small_fpn_3x_coco.py)        | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar)        |
| Mask R-CNN | ViT-Adapter-B | [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                                                                 | 3x+MS   | 49.6   | 43.6    | 120M   | [config](./configs/mask_rcnn/mask_rcnn_deit_adapter_base_fpn_3x_coco.py)         | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/mask_rcnn_deit_adapter_base_fpn_3x_coco.pth.tar)         |
| Mask R-CNN | ViT-Adapter-B | [Uni-Perceiver](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.1/uniperceiver_pretrain.pth)                                                                        | 3x+MS   | 50.7   | 44.9    | 120M   | [config](./configs/mask_rcnn/mask_rcnn_uniperceiver_adapter_base_fpn_3x_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.1/mask_rcnn_uniperceiver_adapter_base_fpn_3x_coco.pth.tar) |
| Mask R-CNN | ViT-Adapter-L | [AugReg-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth) | 3x+MS   | 50.9   | 44.8    | 348M   | [config](./configs/mask_rcnn/mask_rcnn_augreg_adapter_large_fpn_3x_coco.py)      | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_augreg_adapter_large_fpn_3x_coco.pth.tar)      |

**Advanced Detectors**

| Method        | Framework           | Pre-train                                                                         | Lr schd  | box AP | mask AP | #Param | Config                                                                                 | Download                                                                                                                         |
|:-------------:|:-------------------:|:---------------------------------------------------------------------------------:|:--------:|:------:|:-------:|:------:|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | Cascade Mask R-CNN  | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS    | 51.5   | 44.3    | 86M    | [config](./configs/cascade_rcnn/cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.py)   | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.3/cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar) |
| ViT-Adapter-S | ATSS                | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS    | 49.6   | -       | 36M    | [config](./configs/atss/atss_deit_adapter_small_fpn_3x_coco.py)                        | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.5/atss_deit_adapter_small_fpn_3x_coco.pth.tar)              |
| ViT-Adapter-S | GFL                 | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS    | 50.0   | -       | 36M    | [config](./configs/gfl/gfl_deit_adapter_small_fpn_3x_coco.py)                          | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/gfl_deit_adapter_small_fpn_3x_coco.pth.tar)               |
| ViT-Adapter-S | Sparse R-CNN        | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x+MS    | 48.1   | -       | 110M   | [config](./configs/sparse_rcnn/sparse_rcnn_deit_adapter_small_fpn_3x_coco.py)          | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/sparse_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar)       |
| ViT-Adapter-B | Upgraded Mask R-CNN | [MAE-B](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)    | 25ep+LSJ | 50.3   | 44.7    | 122M   | [config](./configs/upgraded_mask_rcnn/mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.4/mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.pth.tar)     |
| ViT-Adapter-B | Upgraded Mask R-CNN | [MAE-B](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)    | 50ep+LSJ | 50.8   | 45.1    | 122M   | [config](./configs/upgraded_mask_rcnn/mask_rcnn_mae_adapter_base_lsj_fpn_50ep_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.4/mask_rcnn_mae_adapter_base_lsj_fpn_50ep_coco.pth.tar)     |

## Evaluation

To evaluate ViT-Adapter-L + HTC++ on COCO val2017 on a single node with 8 gpus run:

```shell
sh dist_test.sh configs/htc++/htc++_beit_adapter_large_fpn_3x_coco.py /path/to/checkpoint_file 8 --eval bbox segm
```

This should give

```
Evaluate annotation type *bbox*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.771
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.642
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.622
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.742
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.742
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.742
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.864

Evaluate annotation type *segm*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.508
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.750
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.556
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.542
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.780
```

## Training

To train ViT-Adapter-T + Mask R-CNN on COCO train2017 on a single node with 8 gpus for 36 epochs run:

```shell
sh dist_train.sh configs/mask_rcnn/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.py 8
```

## Image Demo & Video Demo 
Please see [issue#23](https://github.com/czczup/ViT-Adapter/issues/23).
