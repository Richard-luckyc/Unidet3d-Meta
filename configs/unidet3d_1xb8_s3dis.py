
_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(
    imports=[
        'unidet3d',
        # 'custom_hooks.save_above_threshold_hook'
    ],
    allow_failed_imports=False
)

classes_s3dis_model = ['table', 'chair', 'sofa', 'bookcase', 'board']
classes_s3dis_eval = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter', 'unlabeled'
    ]
num_semantic_classes_eval = len(classes_s3dis_eval) - 1
metainfo_s3dis = dict(classes=classes_s3dis_model)
dataset_type_s3dis = 'S3DISSegDetDataset'
data_root_s3dis = 'data/s3dis/'
data_prefix_s3dis = dict(
    pts='points', pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask', sp_pts_mask='super_points')
train_area = [1, 2, 3, 4, 6]
test_area = 5

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
    bbox_by_mask=[True],
    target_by_distance=[False],
    use_superpoints=[True],
    fast_nms=[False],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_s3dis_model],
        datasets=['s3dis'],
        angles=[False],
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        ),
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['s3dis'],
        datasets_weights=[1],
        loss_weight=[0.5, 1.0],
        non_object_weight=0.1,
        iter_matcher=True,
        topk=[6],
        bbox_loss_simple=dict(type='UniDet3DAxisAlignedIoULoss', mode='diou', reduction='none'),
        bbox_loss_rotated=dict(type='UniDet3DRotatedIoU3DLoss', mode='diou', reduction='none'),
        matcher=dict(
            type='UniMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='BboxCostJointTraining', weight=2.0,
                     loss_simple=dict(type='UniDet3DAxisAlignedIoULoss', mode='diou', reduction='none'),
                     loss_rotated=dict(type='UniDet3DRotatedIoU3DLoss', mode='diou', reduction='none'))]
            )
        ),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18, up_sp_thr=0.81, topk_insts=1000, score_thr=0,
        iou_thr=[0.55])
    )

train_pipeline_s3dis = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_', with_label_3d=False, with_bbox_3d=False, with_mask_3d=True, with_seg_3d=True, with_sp_mask_3d=True),
    dict(type='PointSample_', num_points=180000),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[0.9, 1.1], translation_std=[.1, .1, .1], shift_height=False),
    dict(type='PointDetClassMappingS3DIS', classes=[7, 8, 9, 10, 11]),
    dict(type='NormalizePointsColor_', color_mean=[127.5, 127.5, 127.5]),
    dict(type='ElasticTransfrom', gran=[6, 20], mag=[40, 160], voxel_size=voxel_size, p=-1),
    dict(type='Pack3DDetInputs_', keys=['points', 'elastic_coords', 'gt_labels_3d', 'sp_pts_mask', 'gt_sp_masks', 'pts_semantic_mask', 'pts_instance_mask'])
]
test_pipeline_s3dis = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_', with_bbox_3d=False, with_label_3d=False, with_mask_3d=True, with_seg_3d=True, with_sp_mask_3d=True),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PointSample_', num_points=180000),
             dict(type='NormalizePointsColor_', color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
         type='ConcatDataset', datasets=[
             dict(type=dataset_type_s3dis, data_root=data_root_s3dis, ann_file=f's3dis_sp_infos_Area_{i}.pkl',
                  metainfo=metainfo_s3dis, pipeline=train_pipeline_s3dis, filter_empty_gt=True,
                  data_prefix=data_prefix_s3dis, box_type_3d='Depth', test_mode=False)
             for i in train_area ]
     ))
val_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_s3dis, data_root=data_root_s3dis, ann_file=f's3dis_sp_infos_Area_{test_area}.pkl',
        metainfo=metainfo_s3dis, pipeline=test_pipeline_s3dis, test_mode=True,
        data_prefix=data_prefix_s3dis, box_type_3d='Depth'))
test_dataloader = val_dataloader

load_from = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'

resume = False


val_evaluator = dict(
    type='IndoorMetric_',
    datasets=['s3dis'],
    datasets_classes=[classes_s3dis_model])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9) 

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=16))


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=96,
    dynamic_intervals=[(1, 8), (160 - 8, 1)]) 
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
