# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class APASDatasetAll(CocoDataset):

    CLASSES = ('Right_hand', 'Left_hand', 'Needle_driver', 'Forceps', 'Forceps_not_used', 'Scissors')

