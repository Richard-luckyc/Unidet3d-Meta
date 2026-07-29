import argparse
import os
import os.path as osp
import json
import numpy as np
import open3d as o3d
import imageio  # pip install imageio if needed
from typing import List, Dict, Any

from mmengine import load
from mmdet3d.structures import Det3DDataSample

def parse_args():
    parser = argparse.ArgumentParser(description='Headless visualization for MMDet3D results')
    parser.add_argument('--results-file', default='work_dirs/results.json', help='Path to test results JSON')
    parser.add_argument('--points-dir', default='data/scannet/points', help='Directory of point cloud .bin files')
    parser.add_argument('--show-dir', default='work_dirs/vis', help='Output directory for PNG/GIF')
    parser.add_argument('--score-thr', type=float, default=0.0, help='Bbox score threshold')
    parser.add_argument('--scene-name', type=str, default='scene0015_00', help='Specific scene to visualize')
    args = parser.parse_args()
    return args

def visualize_headless(result: Det3DDataSample, points: np.ndarray, args, scene_id: str):
    """Headless visualization for a single result: save PNG and GIF."""
    os.makedirs(args.show_dir, exist_ok=True)
    
    pred_instances = result.pred_instances_3d
    if pred_instances is None:
        print(f"No predictions for {scene_id}")
        return

    bboxes = pred_instances.bboxes_3d
    scores = pred_instances.scores_3d
    labels = pred_instances.labels_3d

    # Filter low scores
    keep = scores > args.score_thr
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Handle points (N,3 or N,6)
    if points.shape[1] == 6:
        colors = points[:, 3:6] / 255.0
        points_vis = points[:, :3]
    else:
        colors = None
        points_vis = points

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vis)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create bounding boxes
    o3d_bboxes = []
    for bbox, score, label in zip(bboxes, scores, labels):
        tensor = bbox.tensor.cpu().numpy()[0] if len(bbox.tensor.shape) > 1 else bbox.tensor.cpu().numpy()
        center = tensor[:3]
        extent = tensor[3:6]  # w, l, h
        rotation_matrix = np.eye(3)  # Assume no yaw
        o3d_box = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
        # Color by label (simple)
        o3d_box.color = [1.0, 0.0, 0.0] if label == 0 else [0.0, 1.0, 0.0]
        o3d_bboxes.append(o3d_box)

    # Headless rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=768)  # Offscreen
    vis.add_geometry(pcd)
    for bbox in o3d_bboxes:
        vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()

    # Save static PNG
    img = vis.capture_screen_float_buffer(do_render=True)
    img_path = osp.join(args.show_dir, f'{scene_id}.png')
    o3d.io.write_image(img_path, img)
    print(f"PNG saved: {img_path}")

    # Save rotating GIF (72 frames, 360 deg)
    frames = []
    ctr = vis.get_view_control()
    for angle in range(0, 360, 5):
        ctr.rotate(angle * 0.01)  # Rotate view
        vis.poll_events()
        vis.update_renderer()
        frames.append(vis.capture_screen_float_buffer(do_render=True))
    gif_path = osp.join(args.show_dir, f'{scene_id}.gif')
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"GIF saved: {gif_path}")

    vis.destroy_window()

def main():
    args = parse_args()

    # Load test results (assume JSON from --format-only)
    if not osp.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}. Run test.py with --format-only first.")

    with open(args.results_file, 'r') as f:
        results_data = json.load(f)  # List of dicts with 'pred_instances_3d'

    # Load specific scene point cloud
    points_file = osp.join(args.points_dir, f'{args.scene_name}.bin')
    if not osp.exists(points_file):
        raise FileNotFoundError(f"Point cloud not found: {points_file}")
    raw = np.fromfile(points_file, dtype=np.float32)
    points = raw.reshape(-1, 6)  # Assume (N,6) for ScanNet

    # Find result for scene (assume results_data has 'lidar_path' or index matching scene)
    result = None
    for res in results_data:
        if args.scene_name in res.get('lidar_path', '') or res.get('scene_id', '') == args.scene_name:
            result = Det3DDataSample.from_dict(res)  # Reconstruct DataSample
            break
    if result is None:
        print(f"No result for {args.scene_name}. Available scenes: {[r.get('lidar_path', 'unknown') for r in results_data[:5]]}")
        return

    # Visualize
    visualize_headless(result, points, args, args.scene_name)


if __name__ == '__main__':
    main()
