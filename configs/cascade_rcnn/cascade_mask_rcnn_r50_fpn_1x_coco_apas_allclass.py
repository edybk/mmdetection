_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_allclass.py',
    '../_base_/datasets/apas_detection_allclass.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
gpu_ids = range(0, 1)