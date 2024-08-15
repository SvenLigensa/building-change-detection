base_channels = 16
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='DualInputSegDataPreProcessor')
data_root = 'data/S2Looking'
dataset_type = 'S2Looking_Dataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=4166, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        img_shape=(
            1024,
            1024,
            3,
        ),
        interval=1,
        type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'none'
load_from = 'fc_siam_conc/iter_41660.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(base_channel=16, in_channels=3, type='FC_Siam_conc'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='DualInputSegDataPreProcessor'),
    decode_head=dict(
        channels=16,
        concat_input=False,
        in_channels=16,
        in_index=-1,
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        num_classes=2,
        num_convs=0,
        type='mmseg.FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='DIEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=0.001, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=41667,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test/Image1',
            img_path_to='test/Image2',
            seg_map_path='test/label'),
        data_root='data/S2Looking',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='S2Looking_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='MultiImgResize'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_iters=41667, type='IterBasedTrainLoop', val_interval=4166)
train_dataloader = dict(
    batch_size=24,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/Image1',
            img_path_to='train/Image2',
            seg_map_path='train/label'),
        data_root='data/S2Looking',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    512,
                    512,
                ),
                type='MultiImgRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(
                brightness_delta=10,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='MultiImgPhotoMetricDistortion'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='S2Looking_Dataset'),
    num_workers=24,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
    dict(
        cat_max_ratio=0.75, crop_size=(
            512,
            512,
        ), type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(
        brightness_delta=10,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='MultiImgPhotoMetricDistortion'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True, scale_factor=0.75, type='MultiImgResize'),
                dict(keep_ratio=True, scale_factor=1.0, type='MultiImgResize'),
                dict(
                    keep_ratio=True, scale_factor=1.25, type='MultiImgResize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip'),
            ],
            [
                dict(type='MultiImgLoadAnnotations'),
            ],
            [
                dict(type='MultiImgPackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='val/Image1',
            img_path_to='val/Image2',
            seg_map_path='val/label'),
        data_root='data/S2Looking',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='S2Looking_Dataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    save_dir='./fc_siam_conc/s2looking',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = './fc_siam_conc'
