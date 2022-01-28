# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class APASDataset(CocoDataset):

    CLASSES = ('Hand', )
