# config file for training UniDet3D on 3rscan dataset only

_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'unidet3d',
        'custom_hooks.save_above_threshold_hook',
        # 确保包含你的数据集代码路径，如果它不在默认搜索路径中
        # 'path.to.your.rscan_dataset_file' # 例如 'unidet3d.rscan_dataset' 如果文件在unidet3d包内
    ],
    allow_failed_imports=False
)

# --- 修改点 1: 使用代码中定义的精确类别列表 ---
# 使用与 ThreeRScan_.METAINFO['classes'] 完全相同的元组
classes_3rscan = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                  'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                  'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')

# model settings (保持不变)
num_channels=32
voxel_size=0.02
model = dict(
    type='UniDet3D_Meta',
    # ... (模型其他部分保持不变，确保 datasets_classes 使用更新后的 classes_3rscan)
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_3rscan], # 使用更新后的列表/元组
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['3rscan'],
        angles=[False]),
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['3rscan'],
        datasets_weights=[1],
        # ... (criterion 其他部分保持不变)
        ),
    # ... (模型其他部分保持不变)
    )

# --- 修改点 2: 更新 metainfo 和 dataset_type ---
metainfo_3rscan = dict(classes=classes_3rscan)
data_root_3rscan = 'data/3rscan'
dataset_type_3rscan = 'ThreeRScan_' # 确保与 @DATASETS.register_module() 装饰的类名匹配
data_prefix_3rscan = dict(
    pts='points',
    pts_instance_mask='instance_mask', # 确认 pipeline 是否需要
    pts_semantic_mask='semantic_mask', # 确认 pipeline 是否需要
    gt_bboxes_3d='gt_bboxes_3d',
    gt_labels_3d='gt_labels_3d',
    sp_pts_mask='super_points_spt') # 与 parse_data_info 中的 'super_pts_path' 逻辑对应

# Pipelines (train_pipeline_3rscan, test_pipeline_3rscan) 保持不变
# 只需要确保 LoadAnnotations3D_ 加载了 'bbox_label_3d' 字段
# 并且 Pack3DDetInputs_ 包含了模型和评估所需的字段（points, gt_labels_3d, gt_bboxes_3d, sp_pts_mask 等）
train_pipeline_3rscan = [
    dict(type='LoadPointsFromFile', ...),
    dict(type='LoadAnnotations3D_', # 确保加载 bbox_label_3d
         with_bbox_3d=True,
         with_label_3d=True, # 需要 bbox_label_3d 字段
         with_mask_3d=False,
         with_seg_3d=False,
         with_sp_mask_3d=True),
    dict(type='PointSample_', num_points=100000),
    dict(type='RandomFlip3D', ...),
    dict(type='GlobalRotScaleTrans', ...),
    dict(type='NormalizePointsColor_', ...),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'sp_pts_mask'])
]
test_pipeline_3rscan = [
    dict(type='LoadPointsFromFile', ...),
    dict(type='LoadAnnotations3D_', # 确保加载 bbox_label_3d
         with_bbox_3d=True,
         with_label_3d=True, # 需要 bbox_label_3d 字段
         with_mask_3d=False,
         with_seg_3d=False,
         with_sp_mask_3d=True),
    dict(type='MultiScaleFlipAug3D', ...,
         transforms=[
             dict(type='PointSample_', num_points=100000),
             dict(type='NormalizePointsColor_', ...)
         ]),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'sp_pts_mask', 'gt_bboxes_3d', 'gt_labels_3d']) # 评估器需要 GT
]


# --- 修改点 3: 在 dataloader 中使用正确的 dataset_type 和 metainfo ---
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_3rscan, # 使用 'ThreeRScan_'
        data_root=data_root_3rscan,
        data_prefix=data_prefix_3rscan,
        ann_file='3rscan_infos_train.pkl',
        pipeline=train_pipeline_3rscan,
        metainfo=metainfo_3rscan, # 使用更新后的 metainfo
        test_mode=False,
        # partition=1.0 # 可以根据需要添加 partition 参数，默认为 1
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_3rscan, # 使用 'ThreeRScan_'
        data_root=data_root_3rscan,
        data_prefix=data_prefix_3rscan,
        ann_file='3rscan_infos_val.pkl',
        pipeline=test_pipeline_3rscan,
        metainfo=metainfo_3rscan, # 使用更新后的 metainfo
        test_mode=True,
        ))

test_dataloader = val_dataloader

# Evaluator (确保 datasets_classes 使用更新后的列表)
test_evaluator = dict(
    type='IndoorMetric_',
    datasets=['3rscan'],
    datasets_classes=[classes_3rscan], # 使用更新后的列表/元组
    )
val_evaluator = test_evaluator

# 其他部分 (load_from, optim_wrapper, param_scheduler, custom_hooks, default_hooks, train_cfg, etc.) 保持不变
load_from = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9)


default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=16))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024,
    val_interval=16,
    )
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')