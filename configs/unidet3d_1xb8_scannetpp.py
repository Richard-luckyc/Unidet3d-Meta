# configs/unidet3d_scannetpp.py

_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'unidet3d',  # 已有的导入项
        'custom_hooks.rename_sp_mask_hook'
    ],
    allow_failed_imports=False  # 保持此设置不变
)

# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #
classes_scannetpp = [
    'table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair',
    'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 'window', 'box',
    'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can',
    'book', 'plant', 'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket',
    'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 'backpack', 'crate',
    'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave',
    'shoes', 'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack',
    'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 'cup',
    'coat hanger', 'light switch', 'speaker', 'table lamp', 'kettle',
    'smoke detector', 'container', 'power strip', 'slippers', 'paper bag',
    'mouse', 'cutting board', 'toilet paper', 'paper towel', 'pot', 'clock',
    'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box',
    'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones',
    'stapler', 'marker'
]

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
num_channels = 32
voxel_size = 0.02

model = dict(
    type='UniDet3D_Meta',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=[False],
    target_by_distance=[True],
    use_superpoints=[False],
    fast_nms=[None],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True
    ),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_scannetpp],
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['scannetpp'],
        angles=[False]
    ),
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['scannetpp'],
        datasets_weights=[1],
        # 必填：即使不使用旋转框，也要给一个占位
        bbox_loss_simple=dict(
            type='UniDet3DAxisAlignedIoULoss',
            mode='diou',
            reduction='none'
        ),
        bbox_loss_rotated=dict(
            type='UniDet3DAxisAlignedIoULoss',
            mode='diou',
            reduction='none'
        ),
        matcher=dict(
            type='UniMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(
                    type='BboxCostJointTraining',
                    weight=2.0,
                    loss_simple=dict(
                        type='UniDet3DAxisAlignedIoULoss',
                        mode='diou',
                        reduction='none'
                    ),
                    loss_rotated=dict(
                        type='UniDet3DAxisAlignedIoULoss',
                        mode='diou',
                        reduction='none'
                    )
                )
            ]
        ),
        loss_weight=[0.5, 1.0],
        non_object_weight=0.1,
        topk=[3],
        iter_matcher=True
    ),
    train_cfg=dict(topk=3),
    test_cfg=dict(
        low_sp_thr=0.0,
        up_sp_thr=1.0,
        topk_insts=200,
        score_thr=0.01,
        iou_thr=[0.55]
    )
)

# ---------------------------------------------------------------------------- #
# Dataset & Pipelines
# ---------------------------------------------------------------------------- #
dataset_type = 'Scannetpp_'
data_root = 'data/scannetpp/bins'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points_spt'
)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH',
         shift_height=False, use_color=True,
         load_dim=6, use_dim=[0,1,2,3,4,5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=True, with_bbox_3d=True,
         with_sp_mask_3d=True),
    dict(type='PointSample_', num_points=200000),
    dict(type='RandomFlip3D', sync_2d=False,
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[0,0],
         scale_ratio_range=[0.9,1.1],
         translation_std=[0.1,0.1,0.1],
         shift_height=False),
    dict(type='NormalizePointsColor_',
         color_mean=[127.5,127.5,127.5]),
    dict(type='ElasticTransfrom',
         gran=[6,20], mag=[40,160],
         voxel_size=voxel_size, p=-1),
    dict(type='Pack3DDetInputs_',
         keys=['points','elastic_coords',
               'gt_bboxes_3d','gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH',
         shift_height=False, use_color=True,
         load_dim=6, use_dim=[0,1,2,3,4,5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=False, with_bbox_3d=False,
         with_sp_mask_3d=True),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333,800), pts_scale_ratio=1,
         flip=False,
         transforms=[
            dict(type='PointSample_', num_points=200000),
            dict(type='NormalizePointsColor_',
                 color_mean=[127.5,127.5,127.5])
         ]),
    dict(type='Pack3DDetInputs_', keys=['points','sp_pts_mask'])
]

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='scannetpp_infos_train.pkl',
        data_prefix=data_prefix,
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False
    )
)

val_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='scannetpp_infos_val.pkl',
        data_prefix=data_prefix,
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# ---------------------------------------------------------------------------- #
# Checkpoint & Evaluation
# ---------------------------------------------------------------------------- #
load_from = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'
work_dir = './work_dirs/unidet3d_1xb8_scannetpp'

test_evaluator = dict(
    type='IndoorMetric_',
    datasets=['scannetpp'],
    datasets_classes=[classes_scannetpp]
)
val_evaluator = test_evaluator

# ---------------------------------------------------------------------------- #
# Optimizer & Scheduler & Hooks
# ---------------------------------------------------------------------------- #
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2)
)

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9)

custom_hooks = [
    dict(type='RenameSPMaskHook', priority='VERY_HIGH'),
    dict(type='EmptyCacheHook', after_iter=True),
]
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5)
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024,
    dynamic_intervals=[(1, 16), (1024-16, 1)]
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
