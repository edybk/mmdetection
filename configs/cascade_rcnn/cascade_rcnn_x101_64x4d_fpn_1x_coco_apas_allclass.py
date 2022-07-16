# _base_ = './cascade_mask_rcnn_r50_fpn_1x_coco_apas_allclass.py'

_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_allclass.py',
    '../_base_/datasets/apas_detection_allclass.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
gpu_ids = range(0, 1)


model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
load_from = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth' 
runner = dict(type='EpochBasedRunner', max_epochs=100)

workflow = [('train', 1), ('val', 1)]