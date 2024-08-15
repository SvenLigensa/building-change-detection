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
data_root = 'data/WHU'
dataset_type = 'WHU_CD_Dataset'
default_hooks = dict(
    logger=dict(interval=50, log_metric_by_epoch=True, type='LoggerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        img_shape=(
            512,
            512,
            3,
        ),
        interval=1,
        type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'changeformer/iter_41660.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='mmseg.MixVisionTransformer'),
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
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        type='mmseg.SegformerHead'),
    neck=dict(policy='concat', type='FeatureFusionNeck'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='SiamEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='all/A',
            img_path_to='all/B',
            seg_map_path='all/label'),
        data_root='data/WHU',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='WHU_CD_Dataset'),
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
        512,
        512,
    ), type='MultiImgResize'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    save_dir='./changeformer/whu',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = './changeformer'
