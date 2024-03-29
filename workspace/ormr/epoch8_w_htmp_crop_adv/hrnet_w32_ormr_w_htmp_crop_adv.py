checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'workspace/ormr/epoch7_w_htmp_crop_adv/epoch_1.pth'
workflow = [('train', 1)]
use_adversarial_train = True
img_res = 224
optimizer = dict(
    backbone=dict(type='Adam', lr=0.0001),
    head=dict(type='Adam', lr=0.0001),
    disc=dict(type='Adam', lr=5e-05))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=200)
hrnet_extra = dict(
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4, ),
        num_channels=(64, )),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(32, 64)),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(32, 64, 128)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(32, 64, 128, 256)),
    return_list=False,
    single_task=False,
    multi_tasks=True,
    downsample=False,
    use_conv=False,
    final_conv_kernel=1,
    pretrained_layers=[
        'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2',
        'transition2', 'stage3', 'transition3', 'stage4'
    ])
find_unused_parameters = True
model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='PoseHighResolutionNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)),
            return_list=False,
            single_task=False,
            multi_tasks=True,
            downsample=False,
            use_conv=False,
            final_conv_kernel=1,
            pretrained_layers=[
                'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
                'stage2', 'transition2', 'stage3', 'transition3', 'stage4'
            ]),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/pretrained_models/hrnet_pretrain.pth')),
    head=dict(
        type='HMRHead',
        feat_dim=2048,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_54',
        model_path='data/body_models/smpl',
        keypoint_approximate=True,
        extra_joints_regressor='data/body_models/J_regressor_extra.npy'),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    convention='smpl_54',
    loss_keypoints3d=dict(type='SmoothL1Loss', loss_weight=100),
    loss_keypoints2d=dict(type='SmoothL1Loss', loss_weight=10),
    loss_vertex=dict(type='L1Loss', loss_weight=2),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=3),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.02),
    loss_adv=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1),
    disc=dict(type='SMPLDiscriminator'),
    loss_heatmap2d=dict(type='HeatmapMSELoss', loss_weight=300))
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_prob=0.5),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_54'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='ToTensor',
        keys=[
            'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
            'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
        ]),
    dict(
        type='Collect',
        keys=[
            'img', 'has_smpl', 'smpl_body_pose', 'smpl_global_orient',
            'smpl_betas', 'smpl_transl', 'keypoints2d', 'keypoints3d',
            'sample_idx'
        ],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
adv_data_keys = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_adv_pipeline = [
    dict(
        type='Collect',
        keys=[
            'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
        ],
        meta_keys=[])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='ToTensor',
        keys=[
            'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
            'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
        ]),
    dict(
        type='Collect',
        keys=[
            'img', 'has_smpl', 'smpl_body_pose', 'smpl_global_orient',
            'smpl_betas', 'smpl_transl', 'keypoints2d', 'keypoints3d',
            'sample_idx'
        ],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
inference_pipeline = [
    dict(type='MeshAffine', img_res=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='AdversarialDataset',
        train_dataset=dict(
            type='MixedDataset',
            configs=[
                dict(
                    type='HumanImageDataset',
                    dataset_name='h36m',
                    data_prefix='data',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='RandomCrop', crop_prob=0.5),
                        dict(type='RandomChannelNoise', noise_factor=0.4),
                        dict(
                            type='RandomHorizontalFlip',
                            flip_prob=0.5,
                            convention='smpl_54'),
                        dict(
                            type='GetRandomScaleRotation',
                            rot_factor=30,
                            scale_factor=0.25),
                        dict(type='MeshAffine', img_res=224),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='ToTensor',
                            keys=[
                                'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ],
                            meta_keys=[
                                'image_path', 'center', 'scale', 'rotation'
                            ])
                    ],
                    convention='smpl_54',
                    ann_file='h36m_train.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='mpi_inf_3dhp',
                    data_prefix='data',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='RandomCrop', crop_prob=0.5),
                        dict(type='RandomChannelNoise', noise_factor=0.4),
                        dict(
                            type='RandomHorizontalFlip',
                            flip_prob=0.5,
                            convention='smpl_54'),
                        dict(
                            type='GetRandomScaleRotation',
                            rot_factor=30,
                            scale_factor=0.25),
                        dict(type='MeshAffine', img_res=224),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='ToTensor',
                            keys=[
                                'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ],
                            meta_keys=[
                                'image_path', 'center', 'scale', 'rotation'
                            ])
                    ],
                    convention='smpl_54',
                    ann_file='mpi_inf_3dhp_train.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='lsp',
                    data_prefix='data',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='RandomCrop', crop_prob=0.5),
                        dict(type='RandomChannelNoise', noise_factor=0.4),
                        dict(
                            type='RandomHorizontalFlip',
                            flip_prob=0.5,
                            convention='smpl_54'),
                        dict(
                            type='GetRandomScaleRotation',
                            rot_factor=30,
                            scale_factor=0.25),
                        dict(type='MeshAffine', img_res=224),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='ToTensor',
                            keys=[
                                'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ],
                            meta_keys=[
                                'image_path', 'center', 'scale', 'rotation'
                            ])
                    ],
                    convention='smpl_54',
                    ann_file='lsp_train.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='lspet',
                    data_prefix='data',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='RandomCrop', crop_prob=0.5),
                        dict(type='RandomChannelNoise', noise_factor=0.4),
                        dict(
                            type='RandomHorizontalFlip',
                            flip_prob=0.5,
                            convention='smpl_54'),
                        dict(
                            type='GetRandomScaleRotation',
                            rot_factor=30,
                            scale_factor=0.25),
                        dict(type='MeshAffine', img_res=224),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='ToTensor',
                            keys=[
                                'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ],
                            meta_keys=[
                                'image_path', 'center', 'scale', 'rotation'
                            ])
                    ],
                    convention='smpl_54',
                    ann_file='lspet_train.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='mpii',
                    data_prefix='data',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='RandomCrop', crop_prob=0.5),
                        dict(type='RandomChannelNoise', noise_factor=0.4),
                        dict(
                            type='RandomHorizontalFlip',
                            flip_prob=0.5,
                            convention='smpl_54'),
                        dict(
                            type='GetRandomScaleRotation',
                            rot_factor=30,
                            scale_factor=0.25),
                        dict(type='MeshAffine', img_res=224),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='ToTensor',
                            keys=[
                                'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ],
                            meta_keys=[
                                'image_path', 'center', 'scale', 'rotation'
                            ])
                    ],
                    convention='smpl_54',
                    ann_file='mpii_train.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='coco',
                    data_prefix='data',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='RandomCrop', crop_prob=0.5),
                        dict(type='RandomChannelNoise', noise_factor=0.4),
                        dict(
                            type='RandomHorizontalFlip',
                            flip_prob=0.5,
                            convention='smpl_54'),
                        dict(
                            type='GetRandomScaleRotation',
                            rot_factor=30,
                            scale_factor=0.25),
                        dict(type='MeshAffine', img_res=224),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='ToTensor',
                            keys=[
                                'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ]),
                        dict(
                            type='Collect',
                            keys=[
                                'img', 'has_smpl', 'smpl_body_pose',
                                'smpl_global_orient', 'smpl_betas',
                                'smpl_transl', 'keypoints2d', 'keypoints3d',
                                'sample_idx'
                            ],
                            meta_keys=[
                                'image_path', 'center', 'scale', 'rotation'
                            ])
                    ],
                    convention='smpl_54',
                    ann_file='coco_2014_train.npz')
            ],
            partition=[0.35, 0.15, 0.1, 0.1, 0.1, 0.2]),
        adv_dataset=dict(
            type='MeshDataset',
            dataset_name='cmu_mosh',
            data_prefix='data',
            pipeline=[
                dict(
                    type='Collect',
                    keys=[
                        'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
                        'smpl_transl'
                    ],
                    meta_keys=[])
            ],
            ann_file='cmu_mosh.npz')),
    test=dict(
        type='HumanImageDataset',
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
            dict(type='MeshAffine', img_res=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='ToTensor',
                keys=[
                    'has_smpl', 'smpl_body_pose', 'smpl_global_orient',
                    'smpl_betas', 'smpl_transl', 'keypoints2d', 'keypoints3d',
                    'sample_idx'
                ]),
            dict(
                type='Collect',
                keys=[
                    'img', 'has_smpl', 'smpl_body_pose', 'smpl_global_orient',
                    'smpl_betas', 'smpl_transl', 'keypoints2d', 'keypoints3d',
                    'sample_idx'
                ],
                meta_keys=['image_path', 'center', 'scale', 'rotation'])
        ],
        ann_file='pw3d_test.npz'))
work_dir = 'workspace/ormr/epoch8_w_htmp_crop_adv'
gpu_ids = range(0, 4)
