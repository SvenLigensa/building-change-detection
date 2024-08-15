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
fpn_norm_cfg = dict(requires_grad=True, type='mmpretrain.LN2d')
launcher = 'none'
load_from = 'ttp/iter_325000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        encoder_cfg=dict(
            arch='large',
            img_size=(
                512,
                512,
            ),
            init_cfg=dict(
                checkpoint=
                'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth',
                prefix='backbone.',
                type='Pretrained'),
            layer_cfgs=dict(type='TimeFusionTransformerEncoderLayer'),
            out_channels=256,
            patch_size=16,
            type='ViTSAM_Custom',
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14),
        peft_cfg=dict(
            bias='lora_only',
            lora_dropout=0.01,
            r=16,
            target_modules=[
                'qkv',
            ]),
        type='VisionTransformerTurner'),
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
        dropout_ratio=0,
        in_channels=[
            256,
            256,
            256,
            256,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
            4,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        out_size=(
            128,
            128,
        ),
        type='MLPSegHead'),
    neck=dict(
        necks=[
            dict(out_indices=(0, ), policy='concat', type='FeatureFusionNeck'),
            dict(
                backbone_channel=512,
                in_channels=[
                    128,
                    256,
                    512,
                    512,
                ],
                norm_cfg=dict(requires_grad=True, type='mmpretrain.LN2d'),
                num_outs=5,
                out_channels=256,
                type='SimpleFPN'),
        ],
        type='SequentialNeck'),
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        256,
        256,
    )),
    train_cfg=dict(),
    type='TimeTravellingPixels')
norm_cfg = dict(requires_grad=True, type='SyncBN')
resume = False
sam_pretrain_ckpt_path = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth'
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
    save_dir='./ttp/whu',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = './ttp'
