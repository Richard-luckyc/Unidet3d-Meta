# configs/unidet3d_1xb8_multiscan.py
# 基础配置，包含默认的运行时设置
_base_ = ['mmdet3d::_base_/default_runtime.py']
# 自定义导入项
custom_imports = dict(
    imports=[
        'unidet3d',  # UniDet3D 模型相关导入
        'custom_hooks.save_above_threshold_hook'  # 自定义保存钩子
    ],
    allow_failed_imports=False
)

# --- MultiScan 数据集特定设置 ---
classes_multiscan = [
    'door', 'table',  'chair',  'cabinet',  'window',  'sofa',  'microwave',  'pillow',
    'tv_monitor',  'curtain',  'trash_can',  'suitcase',  'sink',  'backpack',  'bed',
    'refrigerator',  'toilet'
]
metainfo_multiscan = dict(classes=classes_multiscan) # 元信息
data_root_multiscan = 'data/multiscan/bins' # 数据集根目录
dataset_type_multiscan = 'MultiScan_' # 数据集类型
data_prefix_multiscan = dict( # 数据文件前缀
    pts='points',
    pts_instance_mask='instance_mask', # 注意：MultiScan原配置加载这些，但pipeline未使用
    pts_semantic_mask='semantic_mask',# 注意：MultiScan原配置加载这些，但pipeline未使用
    sp_pts_mask='super_points' # 注意：MultiScan 使用 'super_points'
)

# --- 模型设置 ---
# (与ScanNet++配置基本一致，除了数据集相关部分)
num_channels=32
voxel_size=0.02

model = dict(
    type='UniDet3D_Meta',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=[True], # MultiScan 加载 bbox，此设置可能依赖模型实现
    target_by_distance=[False],
    use_superpoints=[True],
    fast_nms=[True],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_multiscan], # !!! 修改为 MultiScan 类别
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['multiscan'], # !!! 修改为 'multiscan'
        angles=[False]), # 假设 MultiScan 使用轴对齐框
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['multiscan'], # !!! 修改为 'multiscan'
        datasets_weights=[1],
        bbox_loss_simple=dict(
            type='UniDet3DAxisAlignedIoULoss',
            mode='diou',
            reduction='none'),
        bbox_loss_rotated=dict(
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
            loss_weight=[0.5, 1.0],
            non_object_weight=0.1,
            topk=[6],
            iter_matcher=True),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18, # 这些阈值可能需要为 MultiScan 调整
        up_sp_thr=0.81,
        topk_insts=1000,
        score_thr=0,
        iou_thr=[0.5])) # 评估 IoU 阈值

# --- MultiScan 数据处理流水线 ---
# (基于原始配置中的 train_pipeline_multiscan 和 test_pipeline_multiscan)
train_pipeline_multiscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
        with_label_3d=True,
        with_bbox_3d=True, # MultiScan 加载 BBox
        with_mask_3d=False, # MultiScan 原配置不加载 Mask
        with_seg_3d=False,  # MultiScan 原配置不加载 Seg
        with_sp_mask_3d=True), # 加载超点 Mask
    dict(type='PointSample_',
        num_points=100000), # !!! MultiScan 使用 100k 点
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0], # !!! MultiScan 原配置无旋转增强
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1), # 与原配置保持一致
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'elastic_coords', 'gt_bboxes_3d', # 打包 BBox
              'gt_labels_3d', 'sp_pts_mask'])
]

test_pipeline_multiscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_', # 测试时不加载真值标注
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_sp_mask_3d=True), # 加载超点 Mask
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=100000), # !!! MultiScan 使用 100k 点
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask']) # 打包测试输入
]

# --- Dataloader 设置 ---
train_dataloader = dict(
    batch_size=8, # 根据 GPU 显存调整
    num_workers=8, # 根据 CPU 核心数调整
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict( # 直接定义 MultiScan 数据集
        type=dataset_type_multiscan,
        ann_file='multiscan_infos_train.pkl', # !!! MultiScan 训练标注文件
        data_prefix=data_prefix_multiscan,
        data_root=data_root_multiscan,
        metainfo=metainfo_multiscan,
        pipeline=train_pipeline_multiscan, # !!! 使用 MultiScan 训练流水线
        test_mode=False)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict( # 直接定义 MultiScan 验证集
        type=dataset_type_multiscan,
        ann_file='multiscan_infos_val.pkl', # !!! MultiScan 验证标注文件
        data_prefix=data_prefix_multiscan,
        data_root=data_root_multiscan,
        metainfo=metainfo_multiscan,
        pipeline=test_pipeline_multiscan, # !!! 使用 MultiScan 测试流水线
        test_mode=True)
)

test_dataloader = val_dataloader # 测试 Dataloader 与验证 Dataloader 相同

# --- 评估器设置 ---
test_evaluator = dict(type='IndoorMetric_', # 假设 IndoorMetric_ 适用
                      datasets=['multiscan'], # !!! 指定数据集为 multiscan
                      datasets_classes=[classes_multiscan]) # !!! 指定类别为 multiscan 类别

val_evaluator = test_evaluator

# --- 优化器和学习率调度器 ---
# (保持与ScanNet++一致，可根据MultiScan实验调整)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

# (总 epoch 数可根据 MultiScan 调整)
max_epochs = 1024 # !!! 可根据需要调整 MultiScan 的训练轮数
param_scheduler = dict(
    type='PolyLR',
    begin=0,
    end=max_epochs,
    power=0.9)

# --- 训练、验证、测试循环配置 ---
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    dynamic_intervals=[(1, 16), (max_epochs - 16, 1)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认 Hook 设置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=16),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# --- 加载预训练模型 ---
# (如果你想从 ScanNet 预训练模型开始，保持不变；否则设为 None)
load_from = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'
# resume = False