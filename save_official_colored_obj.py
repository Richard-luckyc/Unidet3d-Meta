# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
from mmengine.config import Config
from mmdet3d.apis import init_model
from mmengine.dataset import Compose
import traceback

# --- 1. 配置区 ---
TARGET_SCENE_ID = "scene0015_00" 
DATASET_TYPE = "scannet" 
PROJECT_ROOT = "/root/autodl-tmp/unidet3d-master"
CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py")
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1022.pth")
SCORE_THR = 0.3 
OUTPUT_OBJ_FILE = os.path.join(PROJECT_ROOT, f"work_dirs/{TARGET_SCENE_ID}_pred_official_color.obj")
# -----------------

# --- 2. 颜色定义 ---
SCANNET_COLOR_MAP = [
    [31, 119, 180], [255, 187, 120], [188, 189, 34], [140, 86, 75], [255, 152, 150],
    [214, 39, 40], [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207],
    [178, 76, 76], [247, 182, 210], [66, 188, 102], [219, 219, 141], [140, 57, 197],
    [202, 185, 52], [51, 176, 203], [200, 54, 131]
]
S3DIS_COLOR_MAP = [[170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100], [200, 200, 200]]
GENERIC_COLOR_MAP = [[31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120], [44, 160, 44]]

def get_palette(dataset_type):
    if dataset_type == 'scannet': return SCANNET_COLOR_MAP
    elif dataset_type == 's3dis': return S3DIS_COLOR_MAP
    else: return GENERIC_COLOR_MAP

def get_box_corners(center, size, heading):
    l, w, h = float(size[0]), float(size[1]), float(size[2])
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    c = np.cos(float(heading)); s = np.sin(float(heading))
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = np.dot(rot_mat, corners)
    corners[0,:] += float(center[0]); corners[1,:] += float(center[1]); corners[2,:] += float(center[2])
    return corners.T

def write_colored_obj(bboxes, labels, scores, output_path, palette):
    # 强制转换为 numpy
    if isinstance(bboxes, torch.Tensor): bboxes = bboxes.cpu().numpy()
    elif isinstance(bboxes, list): bboxes = np.array(bboxes)
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    elif isinstance(labels, list): labels = np.array(labels)
    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
    elif isinstance(scores, list): scores = np.array(scores)

    with open(output_path, 'w') as f:
        f.write(f"# Created by inference script\n")
        total_vertices = 0
        count = 0
        for i in range(len(bboxes)):
            score = float(scores[i])
            if score < SCORE_THR: continue
            count += 1
            label_id = int(labels[i])
            rgb_255 = palette[label_id] if label_id < len(palette) else palette[label_id % len(palette)]
            r, g, b = [c / 255.0 for c in rgb_255]
            
            box = bboxes[i]
            corners = get_box_corners(box[:3], box[3:6], box[6])
            for corner in corners:
                f.write(f"v {corner[0]:.4f} {corner[1]:.4f} {corner[2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")
            edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
            start_idx = total_vertices + 1
            for edge in edges:
                f.write(f"l {start_idx + edge[0]} {start_idx + edge[1]}\n")
            total_vertices += 8
    print(f"成功写入 {count} 个预测框到: {output_path}")

def force_symlink(src, dst):
    if os.path.exists(dst): return
    if not os.path.exists(src):
        print(f"[警告] 源文件不存在: {src}")
        return
    try:
        os.symlink(src, dst)
    except Exception as e:
        print(f"[警告] 创建链接失败: {e}")

def cleanup_symlink(dst):
    if os.path.islink(dst):
        os.remove(dst)

def main():
    paths = {}
    if DATASET_TYPE == 'scannet':
        paths['pkl'] = os.path.join(PROJECT_ROOT, 'data/scannet/scannet_infos_val.pkl')
        paths['bin_dir'] = os.path.join(PROJECT_ROOT, 'data/scannet/points')
    elif DATASET_TYPE == 's3dis':
        paths['pkl'] = os.path.join(PROJECT_ROOT, 'data/s3dis/s3dis_infos_Area_5.pkl')
        paths['bin_dir'] = os.path.join(PROJECT_ROOT, 'data/s3dis/points')

    print(f"正在加载元数据...")
    with open(paths['pkl'], 'rb') as f:
        pkl_data = pickle.load(f)
        data_list = pkl_data.get('data_list', pkl_data)
    
    target_info = None
    temp_links = []

    for info in data_list:
        lidar_path = info.get('lidar_points', {}).get('lidar_path', '')
        base_name = os.path.basename(lidar_path)
        scene_id = os.path.splitext(base_name)[0]
        
        if TARGET_SCENE_ID in scene_id: 
            target_info = info
            target_info['lidar_points']['lidar_path'] = os.path.join(paths['bin_dir'], base_name)
            
            # --- [修复 1] 转换 axis_align_matrix 为 numpy 数组 ---
            if 'axis_align_matrix' in target_info:
                target_info['axis_align_matrix'] = np.array(target_info['axis_align_matrix'], dtype=np.float32)
            # ----------------------------------------------------

            if DATASET_TYPE == 'scannet':
                scannet_root = os.path.dirname(paths['bin_dir']) 
                abs_sp_path = os.path.join(scannet_root, 'super_points', base_name)
                target_info['sp_pts_mask_path'] = abs_sp_path
                
                local_link = os.path.join(os.getcwd(), base_name)
                force_symlink(abs_sp_path, local_link)
                temp_links.append(local_link)

                target_info['pts_instance_mask_path'] = os.path.join(scannet_root, 'instance_mask', base_name)
                target_info['pts_semantic_mask_path'] = os.path.join(scannet_root, 'semantic_mask', base_name)
            break
            
    if target_info is None:
        print(f"错误：未找到场景 {TARGET_SCENE_ID}")
        return

    print(f"正在加载模型...")
    cfg = Config.fromfile(CONFIG_FILE)
    if hasattr(cfg.model, 'test_cfg'):
        cfg.model.test_cfg.score_thr = [SCORE_THR] * 6 
    model = init_model(cfg, CHECKPOINT_FILE, device='cuda:0')
    
    print("正在运行推理...")
    dataset_cfg = cfg.test_dataloader.dataset
    if 'pipeline' in dataset_cfg: pipeline_cfg = dataset_cfg.pipeline
    elif 'datasets' in dataset_cfg:
        if 'pipeline' in dataset_cfg: pipeline_cfg = dataset_cfg.pipeline
        else: pipeline_cfg = dataset_cfg.datasets[0].pipeline
    else: raise AttributeError("无法找到 pipeline 配置")

    test_pipeline = Compose(pipeline_cfg)
    if 'sample_idx' not in target_info: target_info['sample_idx'] = TARGET_SCENE_ID
    
    try:
        data = test_pipeline(target_info)
        data = {'inputs': [data['inputs']], 'data_samples': [data['data_samples']]}
        with torch.no_grad():
            result = model.test_step(data)[0]

        pred_instances = result.pred_instances_3d
        pred_bboxes = pred_instances.bboxes_3d.tensor.cpu().numpy()
        pred_scores = pred_instances.scores_3d.cpu().numpy()
        pred_labels = pred_instances.labels_3d.cpu().numpy()

        official_palette = get_palette(DATASET_TYPE)
        write_colored_obj(pred_bboxes, pred_labels, pred_scores, OUTPUT_OBJ_FILE, official_palette)
        print(f"\n完成！请下载 {OUTPUT_OBJ_FILE} 并将其拖入本地 MeshLab。")
        
    except Exception as e:
        traceback.print_exc()
    finally:
        for link in temp_links:
            cleanup_symlink(link)

if __name__ == "__main__":
    main()