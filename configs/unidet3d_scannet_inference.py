custom_imports = dict(
    imports=['unidet3d'],
    allow_failed_imports=False)
voxel_size = 0.02
num_channels = 256  # 关键：根据之前的 size mismatch 错误，特征维度应为 256

# ScanNet 的类别列表，用于 metainfo
classes_scannet = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'otherfurniture'
]
metainfo = dict(classes=classes_scannet)

# --- 第二部分：从您的训练配置中完整复制的模型定义 ---
# 关键修改：
# 1. model.type 设置为 'unidet3d' (与您代码注册的名称一致)
# 2. 所有与通道数相关的参数都使用 num_channels=256
# 3. decoder.num_classes 设置为 100，以匹配您的预训练权重
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
    fast_nms=[True],
    backbone=dict(
        type='SpConvUNet',
        # 使用一个更标准的U-Net通道配置，确保最后一层输出为 num_channels (256)
        num_planes=[32, 64, 128, 256, 256],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        # 关键修改：强制指定总类别数为100，以匹配权重文件
        num_classes=100,
        datasets_classes=[classes_scannet], # 保留这个用于可能的内部类别映射
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['scannet'],
        angles=[False]),
    # 对于推理，criterion, train_cfg 等不是必需的，可以省略以保持配置文件整洁
    criterion=dict(
        type='UniDet3DCriterion',
            datasets=['scannet', 's3dis', 'multiscan', '3rscan', 
                      'scannetpp', 'arkitscenes'],
            datasets_weights=[1, 1, 1, 1, 1, 1],
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
            topk=[6, 6, 3, 3, 3, 3],
            iter_matcher=True),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18,
        up_sp_thr=0.81,
        topk_insts=1000,
        score_thr=0,
        iou_thr=[0.5]
    )
)


# --- 第三部分：定义一个干净、简单的、专用于推理的 test_dataloader ---
# 这是解决之前所有问题的关键所在

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    # Inferencer 在推理时不需要加载真值标注
    # dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False),
    dict(type='Pack3DDetInputs', keys=['points'])
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ScanNetDetDataset',
        # data_root 和 ann_file 在我们的脚本中不是必需的，因为我们直接传入数据
        # 但为了初始化不出错，可以保留
        data_root='data/scannet/',
        ann_file='scannet_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo, # 使用上面定义的 metainfo
        test_mode=True
    )
)

# 确保 val_dataloader 也被覆盖，以防万一
val_dataloader = test_dataloader