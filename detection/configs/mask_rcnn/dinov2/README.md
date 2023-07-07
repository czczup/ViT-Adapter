

<!-- [ALGORITHM] -->

## Preparation

Please download the DINOv2 pretrained weights into `pretrained/` folder:

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

| Backbone      | Pretrain                                                                                   | Lr schd | box AP | mask AP | #Param | Config       | Download            |
|:-------------:|:------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:------:| ------------ |:-------------------:|
| ViT-Adapter-S | [DINOv2-S](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | 3x+MS   | TODO   | TODO    | 48M    | [config](./) | [ckpt]() \| [log]() |
| ViT-Adapter-B | [DINOv2-B](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | 3x+MS   | TODO   | TODO    | 120M   | [config](./) | [ckpt]() \| [log]() |
| ViT-Adapter-L | [DINOv2-L](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | 3x+MS   | TODO   | TODO    | 348M   | [config](./) | [ckpt]() \| [log]() |
| ViT-Adapter-g | [DINOv2-g](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth) | 3x+MS   | TODO   | TODO    | 348M   | [config](./) | [ckpt]() \| [log]() |
