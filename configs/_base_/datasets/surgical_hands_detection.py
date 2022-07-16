# dataset settings
dataset_type = 'SurgicalHands1'
data_root = 'data/surgical_hands_release/'

img_norm_cfg = dict(
    mean=[144.7125, 132.8805, 124.7715], std=[65.1270, 69.1305, 70.737], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/coco/surgical_hands_coco_train.json',
        img_prefix=f'{data_root}/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/coco/surgical_hands_coco_val.json',
        img_prefix=f'{data_root}/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/coco/surgical_hands_coco_test.json',
        img_prefix=f'{data_root}/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
