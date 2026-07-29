# -*- coding: utf-8 -*-
"""
加载使用 MMDetection3D (及 unidet3d 扩展) 训练的 .pth 模型，
对单个 ScanNet 点云文件 (.bin) 进行推理，并使用 Open3D 可视化预测的 3D 边界框。
代码根据提供的配置文件进行适配。
修改：手动构建输入数据样本以适配 UniDet3D 的自定义 predict 方法，避免 inference_detector 的兼容性问题。
额外修改：添加 sys.path 和显式导入以确保模型注册。针对 UniDet3D 项目结构优化，确保在 init_model 前导入 unidet3d.py 以触发注册。
最新修改：使用 PointData 替换 DummyGtPtsSeg 以匹配框架期望。将 points_for_inference 转换为 torch.tensor (N, 6) float32，并移动到 device 以修复 spconv CUDA 要求。override test_cfg.topk_insts to smaller value (5) to avoid topk out of range for dummy inference.
"""
import torch
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import sys


# --- MMDetection3D/MMEngine 相关导入 ---
try:
    from mmdet3d.apis import init_model
    from mmdet3d.structures import Det3DDataSample, PointData
    from mmdet3d.registry import MODELS
    print("✅ 成功导入 MMDetection3D API")
except ImportError as e:
    print(f"❌ 导入 MMDetection3D 失败: {e}")
    print(f"当前 Python 解释器: {sys.executable}")
    print("请确认你运行的是 conda 环境 (mm3d)，并且 mmdet3d 已正确安装。")
    exit()


# --- 配置区 ---
config_file = '/root/autodl-tmp/unidet3d-master/configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
checkpoint_file = '/root/autodl-tmp/unidet3d-master/work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1022.pth'
target_scene_name = "scene0015_00"
point_cloud_file = f'/root/autodl-tmp/unidet3d-master/data/scannet/points/{target_scene_name}.bin'

device = 'cuda:0'
score_threshold = 0.3
bin_dtype = np.float32

# --- 确保模型注册（UniDet3D 是一个自定义模型文件，不是库） ---
project_root = '/root/autodl-tmp/unidet3d-master'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"项目根目录已添加到 sys.path: {project_root}")

# 手动导入 unidet3d.py 以触发 @MODELS.register_module()
try:
    # 假设 unidet3d.py 在项目根目录（根据您提供的文档）
    import unidet3d
    # 强制导入模型类（如果需要）
    from unidet3d import UniDet3D
    print("✅ 成功导入 unidet3d 和 UniDet3D,注册已触发")
except ImportError as e:
    print(f"❌ 导入 unidet3d 失败: {e}")
    print("请确认 unidet3d.py 文件在项目根目录，并检查其内容是否包含 @MODELS.register_module()")
    exit()

# 调试：检查注册表中是否包含 UniDet3D
try:
    if 'UniDet3D' in MODELS.module_dict:
        print("✅ UniDet3D 已成功注册到 MODELS")
    else:
        print("❌ UniDet3D 未在注册表中，尝试手动注册")
        # 如果失败，手动注册（作为备选）
        from mmdet3d.registry import MODELS
        MODELS.register_module()(UniDet3D)
        print("✅ 手动注册 UniDet3D")
except Exception as e:
    print(f"注册检查失败: {e}")
    exit()

# --- 点云加载函数 ---
def load_point_cloud_for_visualization(file_path, dtype=np.float32):
    print(f"加载点云文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None, None
    raw = np.fromfile(file_path, dtype=dtype)
    if raw.size == 0:
        print("错误: 点云文件为空。")
        return None, None

    # 自动推断维度
    if raw.size % 6 == 0:
        num_attributes = 6
        contains_color = True
    elif raw.size % 3 == 0:
        num_attributes = 3
        contains_color = False
    else:
        print("错误: 点云数据既不是 N×3 也不是 N×6。")
        return None, None

    points_data = raw.reshape(-1, num_attributes)
    points = points_data[:, :3].astype(np.float64)
    colors = None
    if contains_color:
        colors = points_data[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0).astype(np.float64)
    print(f"  成功加载 {points.shape[0]} 个点，维度 {num_attributes}。")
    return points, colors

# --- 可视化函数 ---
def visualize_predictions(points, colors, predictions, score_threshold=0.3, model_meta=None, window_title="Prediction Visualization"):
    if points is None:
        print("错误: 没有点云数据。")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.8, 0.8, 0.8])

    if not hasattr(predictions, 'pred_instances_3d'):
        print("警告: 没有检测结果，显示点云。")
        o3d.visualization.draw_geometries([pcd], window_name=window_title)
        return

    pred_instances = predictions.pred_instances_3d
    boxes_3d = pred_instances.bboxes_3d
    scores_3d = pred_instances.scores_3d
    labels_3d = pred_instances.labels_3d

    keep_indices = scores_3d > score_threshold
    if keep_indices.sum().item() == 0:
        print("没有边界框满足阈值，只显示点云。")
        o3d.visualization.draw_geometries([pcd], window_name=window_title)
        return

    boxes_to_draw = boxes_3d[keep_indices]
    labels_to_draw = labels_3d[keep_indices]
    scores_to_draw = scores_3d[keep_indices]

    class_names = model_meta.get('CLASSES', None) if model_meta else None
    valid_labels = labels_to_draw.cpu().numpy()
    num_classes = max(valid_labels) + 1 if len(valid_labels) > 0 else 1
    palette = plt.get_cmap("tab20")
    colors_O3D = [palette(i % 20)[:3] for i in range(num_classes)]

    o3d_bboxes = []
    for i, box in enumerate(boxes_to_draw):
        try:
            # 对于 DepthInstance3DBoxes，无 yaw，box_dim=6
            tensor = box.tensor.cpu().numpy()  # (1,6) 或直接 (6,)
            if len(tensor.shape) > 1:
                tensor = tensor[0]
            center = tensor[:3]
            extent = tensor[3:]  # [w, l, h]

            # 无旋转，identity matrix
            rotation_matrix = np.eye(3)

            o3d_box = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
            label = valid_labels[i]
            if 0 <= label < len(colors_O3D):
                o3d_box.color = colors_O3D[label]
            else:
                o3d_box.color = [1, 0, 0]

            o3d_bboxes.append(o3d_box)
        except Exception as e:
            print(f"绘制边界框失败: {e}")

    o3d.visualization.draw_geometries([pcd] + o3d_bboxes, window_name=window_title)

# --- 主程序 ---
if __name__ == "__main__":
    print("--- 检查路径 ---")
    for f in [config_file, checkpoint_file, point_cloud_file]:
        if not os.path.exists(f):
            print(f"错误: 文件不存在 {f}")
            exit()

    print("--- 初始化模型 ---")
    try:
        model = init_model(config_file, checkpoint_file, device=device)
        model.eval()
    # Override topk_insts to 1 for dummy inference to avoid out-of-range error
        model.test_cfg.topk_insts = 1
        print("✅ 模型初始化成功")
        print("注意: Checkpoint 中的图像分支键被忽略（模型有融合模块，但可能未训练图像部分）。")
        print("test_cfg.topk_insts overridden to 1 for dummy inference.")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        exit()

    print("--- 加载点云 ---")
    points, colors = load_point_cloud_for_visualization(point_cloud_file, dtype=bin_dtype)
    if points is None:
        exit()

# 子采样到50k点加速dummy推理（生产时移除）
    # 子采样到10k点加速
    subsample_idx = np.random.choice(len(points), min(10000, len(points)), replace=False)
    points = points[subsample_idx]
    if colors is not None:
        colors = colors[subsample_idx]
    print(f"子采样到 {len(points)} 个点")

# 准备推理输入：points 为 torch.tensor (N, 6) float32，以匹配 in_channels=6（x,y,z,r,g,b），并移动到 device
    points_data_np = np.hstack([points.astype(np.float32), (colors * 255).astype(np.float32)]) if colors is not None else  np.hstack([points.astype(np.float32), np.zeros((len(points), 3), dtype=np.float32)])
    points_for_inference = torch.from_numpy(points_data_np).float().to(device)
    N = points_for_inference.shape[0]
    print(f"准备推理：{N} 个点 (通道: {points_for_inference.shape[1]})")

    # 构建