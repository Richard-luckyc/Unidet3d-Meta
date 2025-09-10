# -*- coding: utf-8 -*-
"""
通用推理脚本：
使用 Open3D 可视化在【联合数据集】上训练的模型的预测结果。
支持通过修改顶部的配置区，在不同数据集（如 ScanNet, S3DIS 等）之间切换。
"""
import open3d as o3d
import numpy as np
import os
import torch
import pickle
from mmdet3d.apis import LidarDet3DInferencer

# --- 点云加载函数 (已更新，更通用) ---
def load_point_cloud(file_path, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5], has_color=True):
    """
    从文件加载点云。目前主要支持 .bin 文件。
    
    Args:
        file_path (str): 点云文件路径。
        load_dim (int): 文件中每个点的属性总数。
        use_dim (list): 要使用的属性的索引列表。
        has_color (bool): 是否包含颜色信息。

    Returns:
        tuple: (open3d.geometry.PointCloud, np.ndarray | None)
    """
    if not os.path.exists(file_path):
        print(f"错误：点云文件未找到 {file_path}")
        return None, None
    try:
        raw_data = np.fromfile(file_path, dtype=np.float32)
        points_all_dims = raw_data.reshape(-1, load_dim)
        points_data = points_all_dims[:, use_dim] # 按需选取维度
        
        num_points = points_data.shape[0]
        print(f"成功加载点云: {file_path} ({num_points} 点)")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])

        if has_color and points_data.shape[1] >= 6:
            colors_rgb = points_data[:, 3:6]
            if np.max(colors_rgb) > 1.0:
                colors_rgb = colors_rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb, 0.0, 1.0))

        # 返回 Open3D 对象和用于模型输入的 NumPy 数组
        return pcd, points_all_dims

    except Exception as e:
        print(f"加载点云 '{file_path}' 时发生错误: {e}")
        return None, None

# --- 主程序 ---
if __name__ == "__main__":
    # 1. --- !!! 核心配置区：在这里切换您想测试的数据集和场景 !!! ---

    # ... (DATASET_TYPE = 'scannet' and other configs) ...
    DATASET_TYPE = 'scannet' 
    # --- 联合训练的类别列表 (自动构建) ---
    # 1. 定义从您配置文件中复制过来的各个数据集的类别列表
    classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                       'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
                       'toilet', 'sink', 'bathtub', 'otherfurniture','unannotated']
    classes_s3dis = ['table', 'chair', 'sofa', 'bookcase', 'board']
    classes_multiscan = ['door', 'table',  'chair',  'cabinet',  'window',  'sofa',  'microwave',  'pillow',
                         'tv_monitor',  'curtain',  'trash_can',  'suitcase',  'sink',  'backpack',  'bed',
                         'refrigerator',  'toilet']
    classes_3rscan = classes_scannet
    classes_scannetpp = ['table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 'window', 'box',
                         'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can', 'book', 'plant', 'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket',
                         'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes',
                         'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 'cup', 'coat hanger', 'light switch',
                         'speaker', 'table lamp', 'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel',
                         'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker']
    classes_arkitscenes = ['cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
                           'sink', 'washer', 'toilet', 'bathtub', 'oven',
                           'dishwasher', 'fireplace', 'stool', 'chair', 'table',
                           'tv_monitor', 'sofa']

    # 2. 自动合并所有类别并去重，以生成最终的联合类别列表
    all_classes_lists = [
        classes_scannet,
        classes_s3dis,
        classes_multiscan,
        classes_3rscan,
        classes_scannetpp,
        classes_arkitscenes
    ]
    
    # 使用集合(set)来自动处理去重，并用sorted()来保证顺序固定
    unique_classes = set()
    for class_list in all_classes_lists:
        unique_classes.update(class_list)
    
    JOINT_CLASSES = sorted(list(unique_classes))

    # 3. 验证类别总数是否为100 (这与 unidet3d 论文中的数量一致)
    print(f"自动构建的联合类别列表包含 {len(JOINT_CLASSES)} 个唯一类别。")
    if len(JOINT_CLASSES) != 100:
        print(f"警告：最终类别数不为100，这可能与您的权重文件 'unidet3d.pth' 不匹配！")
        print("请确认您的类别列表是否与原始 unidet3d 项目完全一致。")


    # --- 项目和模型路径 ---
    PROJECT_ROOT = "/root/autodl-tmp/unidet3d-master"
    PATH_TO_MODEL_CONFIG = os.path.join(
    PROJECT_ROOT, 
    "configs/unidet3d_scannet_inference.py" # <-- 使用这个新文件
)
    PATH_TO_MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1024.pth") # 您的权重文件

    # --- 设备 ---
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # --- 特定场景配置 ---
    pcd_file_path = ''
    info_file_path = ''
    scene_id = ''
    axis_align_matrix = np.eye(4) # 默认使用单位矩阵（即不变换）
    load_dim, use_dim, has_color = 6, [0, 1, 2, 3, 4, 5], True # 默认ScanNet格式

    if DATASET_TYPE == 'scannet':
        scene_id = "scene0011_00"
        pcd_file_path = os.path.join(PROJECT_ROOT, f"data/scannet/points/{scene_id}.bin")
        info_file_path = os.path.join(PROJECT_ROOT, "data/scannet/scannet_infos_val.pkl")
        print(f"配置为 [ScanNet] 场景: {scene_id}")

    elif DATASET_TYPE == 's3dis':
        # --- 这是一个示例，请根据您的 S3DIS 文件结构修改 ---
        scene_id = "Area_5_office_1" # S3DIS 场景示例
        pcd_file_path = os.path.join(PROJECT_ROOT, f"data/s3dis/points/{scene_id}.bin") # 假设路径
        info_file_path = os.path.join(PROJECT_ROOT, "data/s3dis/s3dis_infos_Area_5.pkl") # 假设路径
        # S3DIS 也可能有轴对齐矩阵，加载逻辑与 ScanNet 类似
        print(f"配置为 [S3DIS] 场景: {scene_id}")

    # 您可以在这里添加更多 elif 来支持 nuScenes, Waymo 等其他数据集的配置
    # elif DATASET_TYPE == 'nuscenes':
    #     scene_id = "n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590"
    #     pcd_file_path = os.path.join(PROJECT_ROOT, f"data/nuscenes/samples/LIDAR_TOP/{scene_id}.bin")
    #     info_file_path = os.path.join(PROJECT_ROOT, "data/nuscenes/nuscenes_infos_val.pkl")
    #     load_dim, use_dim, has_color = 5, [0, 1, 2, 3], False # nuScenes 点云格式 (x,y,z,intensity,ring)
    #     # nuScenes 通常不需要轴对齐
    #     print(f"配置为 [nuScenes] 场景: {scene_id}")
    
    else:
        raise ValueError(f"未知的数据集类型: {DATASET_TYPE}")

    # --- 配置区结束 ---


    # 2. 初始化 Inferencer
    print("正在初始化 LidarDet3DInferencer...")
    inferencer = LidarDet3DInferencer(
        model=PATH_TO_MODEL_CONFIG,
        weights=PATH_TO_MODEL_CHECKPOINT,
        device=DEVICE
    )
    print("Inferencer 初始化成功！")

    # 3. 加载点云
    pcd, pcd_numpy = load_point_cloud(pcd_file_path, load_dim, use_dim, has_color)

    if pcd and pcd_numpy is not None:
        # 4. 获取轴对齐矩阵 (如果需要)
        # 这段逻辑主要针对 ScanNet 和 S3DIS
        if DATASET_TYPE in ['scannet', 's3dis'] and os.path.exists(info_file_path):
            print(f"正在从 {info_file_path} 寻找元数据...")
            with open(info_file_path, 'rb') as f:
                pkl_data = pickle.load(f)
                data_list = pkl_data.get('data_list', pkl_data) if isinstance(pkl_data, dict) else pkl_data
            
            matrix_found = False
            for scene_info in data_list:
                # 尝试多种可能的路径键
                path_key_options = ['lidar_path', 'pts_path']
                current_path = ''
                for key in path_key_options:
                     current_path = scene_info.get('lidar_points', {}).get(key, scene_info.get(key, ''))
                     if current_path: break
                
                if scene_id in current_path:
                    # 尝试多种可能的矩阵键
                    matrix_key_options = ['axis_align_matrix', 'annos']
                    matrix_data = None
                    for key in matrix_key_options:
                        if key == 'annos': # 兼容旧格式
                            matrix_data = scene_info.get(key, {}).get('axis_align_matrix')
                        else:
                            matrix_data = scene_info.get(key)
                        if matrix_data is not None: break

                    if matrix_data is not None:
                        axis_align_matrix = np.array(matrix_data)
                        print(f"成功为场景 '{scene_id}' 获取了轴对齐矩阵。")
                        matrix_found = True
                    break
            if not matrix_found:
                 print(f"警告：未能在 {info_file_path} 中找到场景 '{scene_id}' 的轴对齐矩阵，将使用单位矩阵。")
        
        # 5. 执行推理
        print("正在执行模型推理...")
        inputs_dict = dict(points=pcd_numpy, axis_align_matrix=axis_align_matrix)
        results = inferencer(inputs_dict)
        print("推理完成。")

        # 6. 解析结果并转换为 Open3D 格式
        o3d_bboxes = []
        if 'predictions' in results and len(results['predictions']) > 0:
            pred_instances = results['predictions'][0]['pred_instances_3d']
            bboxes_tensor = pred_instances['bboxes_3d'].tensor
            scores_tensor = pred_instances['scores_3d']
            labels_tensor = pred_instances['labels_3d']

            print("\n--- 检测结果详情 ---")
            colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
            
            for i in range(len(bboxes_tensor)):
                score = scores_tensor[i].item()
                if score < 0.3: continue # 可以适当调整分数阈值
                
                label_idx = labels_tensor[i].item()
                label_name = JOINT_CLASSES[label_idx] if label_idx < len(JOINT_CLASSES) else f"未知类别({label_idx})"
                
                bbox = bboxes_tensor[i].cpu().numpy()
                if len(bbox) < 7: continue
                    
                center, size, yaw = bbox[:3], bbox[3:6], bbox[6]
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
                o3d_bbox = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
                o3d_bbox.color = colors[label_idx % len(colors)]
                o3d_bboxes.append(o3d_bbox)
                
                print(f"  - 目标 {i+1}: 类别 = {label_name}, 置信度 = {score:.4f}")
            print("--------------------\n")

        # 7. 可视化
        print("正在准备可视化...")
        # 对点云应用变换以匹配预测框的坐标系
        pcd.transform(axis_align_matrix)
        
        geometries_to_draw = [pcd] + o3d_bboxes
        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name=f"[{DATASET_TYPE}] {scene_id} - Model Prediction",
            width=1280, height=720
        )
        print("可视化结束。")
    else:
        print(f"无法加载点云文件 {pcd_file_path}，可视化取消。")