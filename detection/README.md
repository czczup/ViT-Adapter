# Applying ViT-Adapter to Object Detection

Our detection code is developed on top of [MMDetection v2.23.0](https://github.com/open-mmlab/mmdetection/tree/v2.23.0).

For details see [Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534).

If you use this code for a paper please cite:

```
@article{chen2021vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```

## Usage

Install [MMDetection v2.23.0](https://github.com/open-mmlab/mmdetection/tree/v2.23.0).

```
cd ops & sh make.sh # compile deformable attention
pip install timm==0.4.12
pip install mmdet==2.23.0
# recommended environment: torch1.9 + cuda11.1
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install instaboostfast # for htc++
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.23.0](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

## Results and models

#### ViT-Adapter on COCO test-dev

HTC++

| Method | Backbone           | Pre-train                                                                                                             | Lr schd | box AP                                                                                     | mask AP                                                                                    | #Param | Config                                                            | Download                                                                                                             |
|:------:|:------------------:|:---------------------------------------------------------------------------------------------------------------------:|:-------:|:------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:------:|:-----------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
| HTC++  | ViT-Adapter-L      | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x      | [58.5](https://drive.google.com/file/d/11zpPSvmuAn7aP5brxzHE8naObnOfFxby/view?usp=sharing) | [50.8](https://drive.google.com/file/d/1wIbtzfHfPqkvZaSivzcsh4HWu1oSiun6/view?usp=sharing) | 401M   | [config](./configs/htc++/htc++_beit_adapter_large_fpn_3x_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.0/htc++_beit_adapter_large_fpn_3x_coco.pth.tar) |
| HTC++  | ViT-Adapter-L (MS) | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x      | [60.1](https://drive.google.com/file/d/1i-qjgUK4CMwZcmu5pkndldwfVbdkw5sU/view?usp=sharing) | [52.1](https://drive.google.com/file/d/16mlEOPY7K-Xpx_CL650A-LWbVDm2vl4X/view?usp=sharing) | 401M   | TODO                                                              | -                                                                                                                    |

#### ViT-Adapter on COCO minival

HTC++

| Method | Backbone           | Pre-train                                                                                                             | Lr schd | box AP | mask AP | #Param | Config                                                            | Download                                                                                                             |
|:------:|:------------------:|:---------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:-----------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
| HTC++  | ViT-Adapter-L      | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x      | 57.9   | 50.2    | 401M   | [config](./configs/htc++/htc++_beit_adapter_large_fpn_3x_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.0/htc++_beit_adapter_large_fpn_3x_coco.pth.tar) |
| HTC++  | ViT-Adapter-L (MS) | [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth) | 3x      | 59.8   | 51.7    | 401M   | TODO                                                              | -                                                                                                                    |

Baseline Detectors

| Method     | Backbone      | Pre-train                                                                                                                                                                      | Lr schd | Aug | box AP | mask AP | #Param | Config                                                                           | Download                                                                                                                        |
|:----------:|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:---:|:------:|:-------:|:------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|
| Mask R-CNN | ViT-Adapter-T | [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                                                               | 3x      | Yes | 46.0   | 41.0    | 28M    | [config](./configs/mask_rcnn/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.py)         | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.pth.tar)         |
| Mask R-CNN | ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                                                              | 3x      | Yes | 48.2   | 42.8    | 48M    | [config](./configs/mask_rcnn/mask_rcnn_deit_adapter_small_fpn_3x_coco.py)        | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar)        |
| Mask R-CNN | ViT-Adapter-B | [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                                                               | 3x      | Yes | 49.6   | 43.6    | 120M   | [config](./configs/mask_rcnn/mask_rcnn_deit_adapter_base_fpn_3x_coco.py)         | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/mask_rcnn_deit_adapter_base_fpn_3x_coco.pth.tar)         |
| Mask R-CNN | ViT-Adapter-B | [Uni-Perceiver](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.1/uniperceiver_pretrain.pth)                                                                      | 3x      | Yes | 50.7   | 44.9    | 120M   | [config](./configs/mask_rcnn/mask_rcnn_uniperceiver_adapter_base_fpn_3x_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.1/mask_rcnn_uniperceiver_adapter_base_fpn_3x_coco.pth.tar) |
| Mask R-CNN | ViT-Adapter-L | [AugReg](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth) | 3x      | Yes | 50.9   | 44.8    | 348M   | [config](./configs/mask_rcnn/mask_rcnn_augreg_adapter_large_fpn_3x_coco.py)      | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_augreg_adapter_large_fpn_3x_coco.pth.tar)      |

Advanced Detectors

| Method        | Framework           | Pre-train                                                                         | Lr schd | Aug | box AP | mask AP | #Param | Config                                                                                 | Download                                                                                                                         |
|:-------------:|:-------------------:|:---------------------------------------------------------------------------------:|:-------:|:---:|:------:|:-------:|:------:|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | Cascade Mask R-CNN  | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x      | Yes | 51.5   | 44.5    | 86M    | [config](./configs/cascade_rcnn/cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.py)   | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.3/cascade_mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar) |
| ViT-Adapter-S | ATSS                | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x      | Yes | 49.6   | -       | 36M    | [config](./configs/atss/atss_deit_adapter_small_fpn_3x_coco.py)                        | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.5/atss_deit_adapter_small_fpn_3x_coco.pth.tar)              |
| ViT-Adapter-S | GFL                 | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x      | Yes | 50.0   | -       | 36M    | [config](./configs/gfl/gfl_deit_adapter_small_fpn_3x_coco.py)                          | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/gfl_deit_adapter_small_fpn_3x_coco.pth.tar)               |
| ViT-Adapter-S | Sparse R-CNN        | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | 3x      | Yes | 48.1   | -       | 110M   | [config](./configs/sparse_rcnn/sparse_rcnn_deit_adapter_small_fpn_3x_coco.py)          | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/sparse_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar)       |
| ViT-Adapter-B | Upgraded Mask R-CNN | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)      | 25ep    | LSJ | 50.3   | 44.7    | 122M   | [config](./configs/upgraded_mask_rcnn/mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.4/mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.pth.tar)     |
| ViT-Adapter-B | Upgraded Mask R-CNN | [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)      | 50ep    | LSJ | 50.8   | 45.1    | 122M   | [config](./configs/upgraded_mask_rcnn/mask_rcnn_mae_adapter_base_lsj_fpn_50ep_coco.py) | [model](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.4/mask_rcnn_mae_adapter_base_lsj_fpn_50ep_coco.pth.tar)     |

## Evaluation

To evaluate ViT-Adapter-L + HTC++ on COCO val2017 on a single node with 8 gpus run:

```
sh dist_test.sh configs/htc++/htc++_beit_adapter_large_fpn_3x_coco.py /path/to/checkpoint_file 8 --eval bbox segm
```

This should give

```
Evaluate annotation type *bbox*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.766
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.635
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.436
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.616
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.726
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.736
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.736
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.736
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.768
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.863

Evaluate annotation type *segm*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.744
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.549
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.328
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.533
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.683
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.776
```

## Training

To train ViT-Adapter-T + Mask R-CNN on COCO train2017 on a single node with 8 gpus for 36 epochs run:

```
sh dist_train.sh configs/mask_rcnn/mask_rcnn_deit_adapter_tiny_fpn_3x_coco.py 8
```
