# config file for training UniDet3D on ARKitScenes dataset only

_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'unidet3d',
        'custom_hooks.save_above_threshold_hook',
        # Add import for ARKitScenes dataset class if it's custom and not auto-imported
        # e.g., 'path.to.your.arkitscenes_dataset_file'
    ],
    allow_failed_imports=False
)

# Define classes specifically for ARKitScenes
classes_arkitscenes = ['cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
                       'sink', 'washer', 'toilet', 'bathtub', 'oven',
                       'dishwasher', 'fireplace', 'stool', 'chair', 'table',
                       'tv_monitor', 'sofa']

# model settings
num_channels=32
voxel_size=0.02 # Check if this voxel size is appropriate for ARKitScenes

model = dict(
    type='UniDet3D_Meta',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=[True], # Assuming model works with masks or pipeline converts bbox to mask
    target_by_distance=[False],
    use_superpoints=[True], # Assumes superpoints are generated for ARKitScenes
    fast_nms=[True],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_arkitscenes], # Only ARKitScenes classes
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['arkitscenes'], # Only ARKitScenes dataset identifier
        angles=[False]), # Assuming axis-aligned boxes for ARKitScenes
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['arkitscenes'], # Only ARKitScenes
        datasets_weights=[1], # Weight for ARKitScenes loss
        bbox_loss_simple=dict( # Axis-aligned loss
            type='UniDet3DAxisAlignedIoULoss',
            mode='diou',
            reduction='none'),
        bbox_loss_rotated=dict( # Rotated loss (kept for compatibility)
            type='UniDet3DRotatedIoU3DLoss',
            mode='diou',
            reduction='none'),
        matcher=dict(
            type='UniMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='BboxCostJointTraining',
                     weight=2.0,
                     loss_simple=dict(
                         type='UniDet3DAxisAlignedIoULoss',
                         mode='diou',
                         reduction='none'),
                     loss_rotated=dict(
                         type='UniDet3DRotatedIoU3DLoss',
                         mode='diou',
                         reduction='none'))]),
        loss_weight=[0.5, 1.0], # Weights for classification and bbox losses
        non_object_weight=0.1,
        topk=[6],
        iter_matcher=True),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18, # These thresholds might need tuning for ARKitScenes
        up_sp_thr=0.81,  # These thresholds might need tuning for ARKitScenes
        topk_insts=1000,
        score_thr=0,
        iou_thr=[0.5])) # Adjust iou_thr if needed for ARKitScenes eval (e.g., [0.25, 0.5])

# ARKitScenes dataset settings
metainfo_arkitscenes = dict(classes=classes_arkitscenes)
data_root_arkitscenes = 'data/arkitscenes' # Check your data path
dataset_type_arkitscenes = 'ARKitScenesOfflineDataset' # Check registered dataset name
data_prefix_arkitscenes = dict(
    pts='offline_prepared_data', # Check directory name under data_root
    sp_pts_mask='super_points', # Check superpoint key/filename pattern
    # Add other prefixes if needed by LoadAnnotations3D_ or other pipeline steps
    # e.g., gt_bboxes_3d='path/to/bboxes', gt_labels_3d='path/to/labels'
)

train_pipeline_arkitscenes = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH', # Or 'CAMERA' depending on ARKitScenes format
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_', # Ensure this loads gt_bboxes_3d, gt_labels_3d
         with_bbox_3d=True,
         with_label_3d=True,
         with_mask_3d=False, # Set based on whether masks are needed/available
         with_seg_3d=False,  # Set based on whether segs are needed/available
         with_sp_mask_3d=True), # Assumes superpoint masks are loaded
    dict(type='PointSample_', num_points=100000), # Adjust point count if needed
    # Keep the specific color normalization sequence for ARKitScenes
    dict(
        type='DenormalizePointsColor', # Specific to ARKitScenes?
        color_mean=[0, 0, 0],
        color_std=[255, 255, 255]),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.5, 0.5], # Check if rotation range is suitable
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    # ElasticTransfrom is inactive (p=-1), so elastic_coords not needed in Pack
    # dict(
    #     type='ElasticTransfrom',
    #     gran=[6, 20],
    #     mag=[40, 160],
    #     voxel_size=voxel_size,
    #     p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'sp_pts_mask'])
        # Removed 'elastic_coords'
]
test_pipeline_arkitscenes = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH', # Or 'CAMERA'
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_', # Load GT for evaluation
         with_bbox_3d=True,
         with_label_3d=True,
         with_mask_3d=False,
         with_seg_3d=False,
         with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800), # Seems irrelevant for point clouds
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=100000), # Consistent sampling
            # Keep the specific color normalization sequence
            dict(
                type='DenormalizePointsColor',
                color_mean=[0, 0, 0],
                color_std=[255, 255, 255]),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    # Include GT keys needed by the evaluator
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask', 'gt_bboxes_3d', 'gt_labels_3d'])
]


# run settings
# Adjust batch size based on GPU memory and ARKitScenes data size
# 8 might be too large, starting with 4.
train_dataloader = dict(
    batch_size=4, # Adjusted from 8
    num_workers=4, # Adjust based on batch size and CPU cores
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict( # Directly configure the ARKitScenes dataset
        type=dataset_type_arkitscenes,
        data_root=data_root_arkitscenes,
        data_prefix=data_prefix_arkitscenes,
        ann_file='arkitscenes_offline_infos_train.pkl', # Check actual train annotation file name
        pipeline=train_pipeline_arkitscenes,
        metainfo=metainfo_arkitscenes,
        test_mode=False,
        # filter_empty_gt=True, # Optional: filter scenes with no GT boxes
        # box_type_3d='Depth' # Set if needed, default is Depth in original RScan class
        ))

val_dataloader = dict(
    batch_size=1, # Typically 1 for validation/testing
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict( # Directly configure the ARKitScenes dataset
        type=dataset_type_arkitscenes,
        data_root=data_root_arkitscenes,
        data_prefix=data_prefix_arkitscenes,
        ann_file='arkitscenes_offline_infos_val.pkl', # Check actual validation annotation file name
        pipeline=test_pipeline_arkitscenes, # Use test pipeline for validation
        metainfo=metainfo_arkitscenes,
        test_mode=True,
        # box_type_3d='Depth' # Set if needed
        ))

# Use the same dataloader config for testing
test_dataloader = val_dataloader

# Pre-trained model path (optional, update if needed)
load_from = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'

# Evaluator settings for ARKitScenes
test_evaluator = dict(
    type='IndoorMetric_', # Ensure this evaluator works for ARKitScenes format
    datasets=['arkitscenes'],
    datasets_classes=[classes_arkitscenes],
    # Specify metric options if needed, e.g., IoU thresholds
    # metric_options=dict(iou_thrs=[0.25, 0.5])
    )

# Use the same evaluator for validation
val_evaluator = test_evaluator

# Optimizer and Scheduler settings (kept from original)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05), # Adjust LR if needed
    clip_grad=dict(max_norm=10, norm_type=2))

# Adjust scheduler length if max_epochs changes
param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9) # End should match max_epochs



# Default hooks (kept from original, adjust interval/max_keep_ckpts if needed)
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=16)) # Maybe save less frequently for large datasets?

# Training loop configuration (kept from original, adjust epochs/interval)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024, # Adjust as needed for ARKitScenes convergence
    val_interval=16, # Set validation interval
    )
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')