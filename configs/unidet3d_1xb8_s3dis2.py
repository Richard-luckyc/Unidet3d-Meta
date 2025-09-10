# config_unidet3d_meta_s3dis_test.py (示例文件名)

# 基础运行时配置和必要的自定义导入
_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['unidet3d'], allow_failed_imports=False)

# --- S3DIS 数据集定义 ---
# 用于模型输出和检测评估的 5 个类别
classes_s3dis_model = ['table', 'chair', 'sofa', 'bookcase', 'board']
metainfo_s3dis = dict(classes=classes_s3dis_model) # 元信息使用这 5 个类
dataset_type_s3dis = 'S3DISSegDetDataset' # 数据集类型
data_root_s3dis = 'data/s3dis/' # 数据集根目录
# 数据文件的前缀定义
data_prefix_s3dis = dict(
    pts='points', pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask', sp_pts_mask='super_points')
test_area = 5 # 指定 Area 5 作为测试区域
# --- S3DIS 定义结束 ---

# --- 模型定义 ---
num_channels = 32
voxel_size = 0.02
model = dict(
    type='UniDet3D_Meta', # 模型类型
    data_preprocessor=dict(type='Det3DDataPreprocessor_'), # 数据预处理器
    in_channels=6, # 输入点云维度 (x,y,z,r,g,b)
    num_channels=num_channels, # UNet 基础通道数
    voxel_size=voxel_size, # 体素大小
    min_spatial_shape=128, # 最小空间形状
    query_thr=3000, # 查询阈值
    bbox_by_mask=[True], # 是否通过 Mask 计算 BBox
    target_by_distance=[False], # 是否根据距离分配目标
    use_superpoints=[True], # 是否使用超点
    fast_nms=[False], # 是否使用快速 NMS (根据你的原始配置)
    backbone=dict( # 主干网络 (3D UNet)
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict( # 解码器/检测头
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_s3dis_model], # 告知解码器输出类别
        datasets=['s3dis'], # 数据集标识
        angles=[False], # 是否预测角度 (轴对齐框为 False)
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
    ),
    # --- 注意: criterion 和 matcher 在 test.py 运行时不用于计算损失, 但定义完整无害 ---
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['s3dis'], datasets_weights=[1], loss_weight=[0.5, 1.0],
        non_object_weight=0.1, iter_matcher=True, topk=[6],
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
    # --- train_cfg 在 test.py 运行时被忽略 ---
    # train_cfg=dict(topk=6),
    # --- test_cfg 定义测试时模型的行为 ---
    test_cfg=dict(
        low_sp_thr=0.18, up_sp_thr=0.81, topk_insts=1000, score_thr=0,
        iou_thr=[0.55] # 测试时 NMS 等操作的 IoU 阈值 (根据你的原始配置)
    )
)
# --- 模型定义结束 ---

# --- 测试数据处理流水线定义 ---
# 这个流水线只包含测试所需步骤，无数据增强
test_pipeline_s3dis = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    # 加载标注信息 - 评估器 IndoorMetric_ 可能需要用到真值来进行匹配和计算指标
    dict(type='LoadAnnotations3D_', with_bbox_3d=False, with_label_3d=False, with_mask_3d=True, with_seg_3d=True, with_sp_mask_3d=True),
    # 多尺度翻转增强（这里 flip=False, 实际只应用内部的 transforms）
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PointSample_', num_points=180000), # 点采样
             dict(type='NormalizePointsColor_', color_mean=[127.5, 127.5, 127.5]) # 颜色归一化
         ]),
    # 打包测试时模型前向传播所需的输入
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]
# --- 流水线定义结束 ---

# --- 测试数据加载器定义 ---
# 使用 Area 5 作为测试集
test_dataloader = dict(
    batch_size=1, # 测试时 batch_size 通常为 1
    num_workers=2, # 根据你的机器配置调整
    persistent_workers=True, # 保持 worker 进程
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False), # !!! 测试时必须 shuffle=False !!!
    dataset=dict(
        type=dataset_type_s3dis,
        data_root=data_root_s3dis,
        ann_file=f's3dis_sp_infos_Area_{test_area}.pkl', # 指定测试区域的标注文件
        metainfo=metainfo_s3dis, # 数据集元信息
        pipeline=test_pipeline_s3dis, # !!! 使用测试流水线 !!!
        test_mode=True, # !!! 必须是 True !!!
        data_prefix=data_prefix_s3dis,
        box_type_3d='Depth'
    )
)
# --- 数据加载器定义结束 ---


# --- 评估器定义 ---
# 假设目标是评估 5 个类别的目标检测性能 (mAP)
test_evaluator = dict(
    type='IndoorMetric_', # 适用于室内场景 3D 检测评估
    datasets=['s3dis'], # 数据集名称
    datasets_classes=[classes_s3dis_model] # 评估的类别列表
)
# --- 评估器定义结束 ---

# --- 测试循环定义 ---
test_cfg = dict(type='TestLoop')
# --- 测试循环定义结束 ---

# --- 运行时相关设置 (可选但推荐) ---
# 定义默认的 hooks，主要用于日志和可视化（如果使用 --show）
default_hooks = dict(
    # timer=dict(type='IterTimerHook'), # test 时通常不需要
    logger=dict(type='LoggerHook', interval=50), # 日志记录频率
    # param_scheduler=dict(type='ParamSchedulerHook'), # test 时不需要
    checkpoint=dict(type='CheckpointHook', interval=-1), # test 时不保存 checkpoint
    sampler_seed=dict(type='DistSamplerSeedHook'), # 分布式测试时设置种子
    visualization=dict(type='Det3DVisualizationHook', draw=False)) # 可视化 hook，默认不绘制

# 环境设置，有助于可复现性
env_cfg = dict(
    cudnn_benchmark=False, # 设置为 False 以增加可复现性（可能牺牲一点速度）
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # 多进程设置
    dist_cfg=dict(backend='nccl'), # 分布式后端
)

# 可以设置工作目录，但 test.py 的 --work-dir 参数优先级更高
# work_dir = './work_dirs/s3dis_test_results'

# test.py 会通过命令行参数传入 checkpoint 路径，以下 load_from 在 test.py 中会被覆盖
# load_from = None
# resume = False # 测试时不需要 resume
# --- 运行时设置结束 ---