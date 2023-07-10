# ViT-Adapter with DINOv2

## Preparation

Please download the DINOv2 pretrained weights into the `pretrained/` folder:

<table style="margin: auto">
  <tr>
    <th>model</th>
    <th># of<br />params</th>
    <th>ImageNet<br />k-NN</th>
    <th>ImageNet<br />linear</th>
    <th>download</th>
  </tr>
  <tr>
    <td>ViT-S/14 distilled</td>
    <td align="right">21 M</td>
    <td align="right">79.0%</td>
    <td align="right">81.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth">backbone only</a></td>
  </tr>
  <tr>
    <td>ViT-B/14 distilled</td>
    <td align="right">86 M</td>
    <td align="right">82.1%</td>
    <td align="right">84.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth">backbone only</a></td>
  </tr>
  <tr>
    <td>ViT-L/14 distilled</td>
    <td align="right">300 M</td>
    <td align="right">83.5%</td>
    <td align="right">86.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">backbone only</a></td>
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="right">1,100 M</td>
    <td align="right">83.5%</td>
    <td align="right">86.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth">backbone only</a></td>
  </tr>
</table>

Then convert these models to have patch size 16:

```shell
python convert_14to16.py pretrained/dinov2_vits14_pretrain.pth
python convert_14to16.py pretrained/dinov2_vitb14_pretrain.pth
python convert_14to16.py pretrained/dinov2_vitl14_pretrain.pth
python convert_14to16.py pretrained/dinov2_vitg14_pretrain.pth
```

After that, the directory structure is:

```shell
detection
├── pretrained
│   └── dinov2_vits14_pretrain.pth
│   └── dinov2_vitb14_pretrain.pth
│   └── dinov2_vitl14_pretrain.pth
│   └── dinov2_vitg14_pretrain.pth
│   └── dinov2_vits14_pretrain_14to16.pth
│   └── dinov2_vitb14_pretrain_14to16.pth
│   └── dinov2_vitl14_pretrain_14to16.pth
│   └── dinov2_vitg14_pretrain_14to16.pth
└── convert_14to16.py
```

## Results and Models

| Backbone      | Pretrain                                                                                   | Lr schd | box AP | mask AP | #Param | Config                                                    | Download                                                                                                                                                                                                                      |
|:-------------:|:------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:|:---------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ViT-Adapter-S | [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                                                     | 3x+MS   | 48.2   | 42.8    | 48M    | [config](../mask_rcnn_deit_adapter_small_fpn_3x_coco.py)  | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.2/mask_rcnn_deit_adapter_small_fpn_3x_coco.pth.tar)                                                                                                       |
| ViT-Adapter-S | [DINOv2-S](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | 3x+MS   | 51.5 (+3.3)   | 45.6 (+2.8)   | 48M    | [config](./mask_rcnn_dinov2_adapter_small_fpn_3x_coco.py) | [ckpt](https://huggingface.co/czczup/ViT-Adapter/resolve/main/mask_rcnn_dinov2_adapter_small_fpn_3x_coco.pth) \| [log](https://huggingface.co/czczup/ViT-Adapter/resolve/main/mask_rcnn_dinov2_adapter_small_fpn_3x_coco.log) |
| ViT-Adapter-B | [DINOv2-B](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | 3x+MS   | -      | -       | 120M   | [config](./mask_rcnn_dinov2_adapter_base_fpn_3x_coco.py)  | -                                                                                                                                                                                                                             |
| ViT-Adapter-L | [DINOv2-L](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | 3x+MS   | -      | -       | 348M   | [config](./mask_rcnn_dinov2_adapter_large_fpn_3x_coco.py) | -                                                                                                                                                                                                                             |

Note that, the hyper-parameter `layer_decay_rate`  has a significant impact on performance of DINOv2. For example, for the `ViT-Adapter-S` with `DINOv2-S`, the box AP of different `layer_decay_rate` are:

| Backbone      | Pretrain                                                                                   | 0.70 | 0.75 | 0.80 | 0.90 | 0.95 |
|:-------------:|:------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| ViT-Adapter-S | [DINOv2-S](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | 51.5 | 51.0 | 50.8 | 49.4 | 48.8 |

Perhaps further reducing `layer_decay_rate` will continue to improve performance.


