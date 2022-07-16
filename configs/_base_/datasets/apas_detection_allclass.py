# dataset settings
dataset_type = 'APASDatasetAll'
data_root = 'data/apas_allclass/'

img_norm_cfg = dict(
        mean=[144.01279543590812, 131.91472349201618, 123.31423661654405], std=[64.7040393831716, 68.7886688576991, 70.7488325943194], to_rgb=True)

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
        ann_file=f'{data_root}/coco/apas_train.json',
        img_prefix=f'{data_root}/coco/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/coco/apas_val.json',
        img_prefix=f'{data_root}/coco/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/coco/apas_val.json',
        img_prefix=f'{data_root}/coco/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
