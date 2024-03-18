# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class PizzeDataset(CustomDataset):
    CLASSES = ('bg','Anchovy', 'Olives', 'Salami', 'Red_Pepper',
               'Yellow_Pepper')

    PALETTE = [ [255, 255, 255],[0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(PizzeDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)