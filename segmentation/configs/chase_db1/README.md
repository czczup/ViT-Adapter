# CHASE DB1

<!-- [ALGORITHM] -->

## Introduction

The training and validation set of CHASE DB1 could be download from [here](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip).

To convert CHASE DB1 dataset to MMSegmentation format, you should run the [script](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/chase_db1.py) provided by mmseg official:

```shell
python /path/to/convertor/chase_db1.py /path/to/CHASEDB1.zip
```

The script will make directory structure automatically.

## Results and Models

| Method      | Backbone      | Pretrain | Batch Size | Lr schd | Crop Size | mDice |  #Param | Config                                                           | Download                                               |
|:-----------:|:-------------:|:---------:|:----------:|:-------:|:---------:|:---------:|:------:|:----------------------------------------------------------------:|:------------------------------------------------------:|
| Mask2Former | ViT-Adapter-L | BEiT-L    | 4x4        | 40k     | 128       | 89.4      |  350M   | [config](./mask2former_beit_adapter_large_128_40k_chase_db1_ss.py) | [log](https://github.com/czczup/ViT-Adapter/issues/11) |
