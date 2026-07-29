import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
from torch import Tensor

from mmcv.ops import nms3d, nms3d_normal
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.models import Base3DDetector
from mmdet3d.models.layers.box3d_nms import aligned_3d_nms
from mmdet3d.structures import rotation_3d_in_axis

from .criterion import _bbox_to_loss
from .structures import InstanceData_

# Meta Learning
from torch.autograd import Variable
import torch.nn.init as init
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if curr_module is None:
            curr_module = self
        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, grad in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name_, mod in curr_mod.named_children():
                if module_name == name_:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

# ===================================================================
# 已移除 MetaConv2d, ResNet101_Backbone, 和 NeighborhoodCrossAttentionFusion 类
# 因为它们只被已移除的图像分支所使用。
# ===================================================================

@MODELS.register_module()
class UniDet3D_Meta(MetaModule, Base3DDetector):
    r"""UniDet3D_Meta 的精简版本，用于消融实验。
    此版本完全移除了独立的图像分支和交叉注意力融合模块。
    """
    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 min_spatial_shape,
                 query_thr,
                 use_superpoints,
                 bbox_by_mask, 
                 target_by_distance,
                 fast_nms,
                 use_sync_bn=True,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 # 关键改动：移除了 image_backbone_cfg 参数
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if backbone is not None:
            self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.use_superpoints = use_superpoints
        self.bbox_by_mask = bbox_by_mask
        self.target_by_distance = target_by_distance 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_sync_bn = use_sync_bn
        self.fast_nms = fast_nms
        
        # 关键改动：移除了 image_backbone 和 cross_modal_fusion 的初始化
        
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        if self.use_sync_bn:
            self.output_layer = spconv.SparseSequential(
                torch.nn.SyncBatchNorm(num_channels, eps=1e-4, momentum=0.1),
                torch.nn.ReLU(inplace=True))
        else:
            self.output_layer = spconv.SparseSequential(
                torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
                torch.nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """
        精简版的特征提取函数，只处理点云数据。
        移除了 image_input 参数和相关的融合逻辑。
        """
        # LiDAR 特征提取
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        lidar_feat = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        
        # 将 LiDAR 特征划分为 batch 内的各样本
        lidar_feats = []
        for i in range(len(batch_offsets) - 1):
            lidar_feats.append(lidar_feat[batch_offsets[i]: batch_offsets[i + 1]])
        
        return lidar_feats

    def collate(self, points, elastic_points=None):
        """Collate a batch of points into a sparse tensor."""
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(self, *args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def _select_queries(self, x, gt_instances):
        """Select queries for the training pass."""
        queries = []
        sp_centers = []
        for i in range(len(x)):
            if len(x[i]) > self.query_thr:
                ids = torch.randperm(len(x[i]))[:self.query_thr].to(x[i].device)
                queries.append(x[i][ids])
                sp_centers.append(gt_instances[i].sp_centers[ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
                gt_instances[i].sp_centers = gt_instances[i].sp_centers[ids]
            else:
                queries.append(x[i])
                sp_centers.append(gt_instances[i].sp_centers)
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        return queries, sp_centers, gt_instances

    def get_bboxes_by_masks(self, masks, points):
        """Generate 3D bounding boxes from masks."""
        boxes = []
        for mask in masks:
            object_points = points[mask]
            xyz_min = object_points.min(dim=0).values
            xyz_max = object_points.max(dim=0).values
            center = (xyz_max + xyz_min) / 2
            size = xyz_max - xyz_min
            box = torch.cat((center, size))
            boxes.append(box)
        if len(boxes) == 0:
            bboxes = DepthInstance3DBoxes(
                masks.new_zeros(0, 6), with_yaw=False, 
                box_dim=6, origin=(0.5, 0.5, 0.5))
        else:
            boxes = torch.stack(boxes)
            bboxes = DepthInstance3DBoxes(
                boxes, with_yaw=False, box_dim=6, origin=(0.5, 0.5, 0.5))
        return bboxes
    
    def get_gt_inst_masks(self, masks_src):
        """Create ground truth instance masks."""
        masks = masks_src.clone()
        if torch.sum(masks == -1) != 0:
            masks[masks == -1] = torch.max(masks) + 1
            masks = torch.nn.functional.one_hot(masks)[:, :-1]
        else:
            masks = torch.nn.functional.one_hot(masks)
        return masks.bool()

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples."""
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        sp_centers = []
        
        if batch_inputs_dict.get('elastic_coords') is not None:
            points = [(point - point.min(0)[0]) * self.voxel_size for point in \
                batch_inputs_dict['elastic_coords']]
            shifts = [point.min(0)[0] * self.voxel_size for point in \
                batch_inputs_dict['elastic_coords']]
        else:
            points = [point[:, :3] - point[:, :3].min(0)[0] for point in \
                batch_inputs_dict['points']]
            shifts = [point[:, :3].min(0)[0] for point in \
                batch_inputs_dict['points']]

        datasets_names = []
        for i in range(len(batch_data_samples)):
            datasets_names.append(self.get_dataset(
                            batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            dataset = self.decoder.datasets.index(datasets_names[i])
            if self.bbox_by_mask[dataset]:
                gt_masks = self.get_gt_inst_masks(gt_pts_seg.pts_instance_mask)
                batch_data_samples[i].gt_instances_3d.bboxes_3d = \
                                            self.get_bboxes_by_masks(gt_masks.T,
                                                                    points[i])
            else:
                center = batch_data_samples[i].gt_instances_3d.\
                                    bboxes_3d.gravity_center - \
                                    shifts[i]
                bboxes = torch.cat((center,
                                    batch_data_samples[i].gt_instances_3d.\
                                    bboxes_3d.tensor[:, 3:]),
                                    dim=1)
                batch_data_samples[i].gt_instances_3d.bboxes_3d = \
                    DepthInstance3DBoxes(
                        bboxes, 
                        with_yaw=batch_data_samples[i].gt_instances_3d.\
                                                    bboxes_3d.with_yaw, 
                        box_dim=bboxes.shape[1], origin=(0.5, 0.5, 0.5))
            
            batch_data_samples[i].gt_instances_3d.sp_centers = \
                scatter_mean(points[i], gt_pts_seg.sp_pts_mask, dim=0)
            if self.target_by_distance[dataset]:
                batch_data_samples[i].gt_instances_3d.sp_masks = \
                    self.get_targets(batch_data_samples[i].gt_instances_3d.\
                                        sp_centers,
                                     batch_data_samples[i].gt_instances_3d.\
                                        bboxes_3d,
                                     self.train_cfg.topk)
            sp_centers.append(batch_data_samples[i].gt_instances_3d.sp_centers)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        
        # 关键改动：调用不带 image_input 参数的精简版 extract_feat
        x = self.extract_feat(
            x,sp_pts_masks, inverse_mapping, batch_offsets)

        queries, sp_centers_queries, sp_gt_instances = \
                    self._select_queries(x, sp_gt_instances)
        x = self.decoder(queries, sp_centers_queries, [self.get_dataset(ds) for ds in [d.lidar_path for d in batch_data_samples]])
        loss = self.criterion(x, sp_gt_instances, [self.get_dataset(ds) for ds in [d.lidar_path for d in batch_data_samples]])

        return loss

    def get_dataset(self, lidar_path):
        for dataset in self.decoder.datasets:
            if dataset in lidar_path.split('/'):
                return dataset

    def get_targets(self, points, gt_bboxes, topk):
        """Compute targets for final locations for a single scene."""
        float_max = points[0].new_tensor(1e8)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        boxes = torch.cat((gt_bboxes.gravity_center, 
                           gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.expand(n_points, n_boxes, boxes.shape[1])
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)

        topk_distances = torch.topk(
            center_distances,
            min(topk + 1, len(center_distances)),
            largest=False,
            dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)
        center_distances = torch.where(topk_condition, center_distances,
                                        float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, n_boxes)
        min_dist_condition = torch.nn.functional.one_hot(
            min_inds, num_classes=n_boxes + 1)[:, :-1].bool()

        return min_dist_condition.T

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples."""
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        sp_centers = []
        datasets_names = []
        sp_pts_masks_src = []
        points_src = []
        for i in range(len(batch_data_samples)):
            datasets_names.append(self.get_dataset(
                            batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            points = batch_inputs_dict['points'][i][:, :3]
            points_src.append(points)
            sp_centers.append(scatter_mean(points, 
                                           gt_pts_seg.sp_pts_mask, dim=0))
            sp_pts_masks_src.append(gt_pts_seg.sp_pts_mask)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)

        x = self.decoder(x, sp_centers, datasets_names)

        results_list = self.predict_by_feat(x, sp_pts_masks_src, 
                                            points_src, datasets_names)
        
        for i, data_sample in enumerate(batch_data_samples):
            if i < len(results_list):
                bboxes, labels, scores = results_list[i]
                data_sample.pred_instances_3d = InstanceData_(
                    bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                    points=batch_inputs_dict['points'][0])
        
        return batch_data_samples

    def predict_by_feat(self, out, sp_pts_masks, points, 
                        datasets_names):
        """Predict bounding boxes and labels from model outputs."""
        cls_preds = out['cls_preds'][0]
        pred_bboxes = out['bboxes'][0]
        sp_pts_mask = sp_pts_masks[0] 
        point = points[0]
        dataset_name = datasets_names[0]
    
        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        num_classes = scores.shape[1]
        labels = torch.arange(
            num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        
        total_scores = scores.numel()
        k = min(self.test_cfg.topk_insts, total_scores)
        if k <= 0: return []

        scores, topk_idx = scores.flatten(0, 1).topk(k, sorted=True)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, num_classes, rounding_mode='floor')
        pred_bboxes = pred_bboxes[topk_idx]

        fast_nms = self.fast_nms[self.decoder.\
                                        datasets.index(dataset_name)]
        iou_thr = self.test_cfg.iou_thr[self.decoder.\
                                        datasets.index(dataset_name)]
        nms_bboxes, nms_scores, nms_labels = \
                self._single_scene_multiclass_nms(pred_bboxes,
                                                  scores, 
                                                  labels,
                                                  fast_nms, 
                                                  iou_thr)
        if not self.use_superpoints[
            self.decoder.datasets.index(dataset_name)]:
            return [(DepthInstance3DBoxes(
                nms_bboxes, 
                with_yaw=nms_bboxes.shape[1] == 7, 
                box_dim=nms_bboxes.shape[1], 
                origin=(0.5, 0.5, 0.5)),
                nms_labels, nms_scores)]
        else:
            return self.trim_bboxes_by_superpoints(sp_pts_mask, point, 
                                                   nms_bboxes, nms_labels,
                                                   nms_scores)

    def trim_bboxes_by_superpoints(self, sp_pts_mask, point, 
                                   bboxes, labels, scores):
        """Trim bounding boxes based on superpoint masks."""
        n_points = point.shape[0]
        n_boxes = bboxes.shape[0]
        if n_boxes == 0:
            return [(DepthInstance3DBoxes(bboxes.new_zeros((0, 6)), with_yaw=False, box_dim=6, origin=(0.5, 0.5, 0.5)), labels, scores)]

        point = point.unsqueeze(1).expand(n_points, n_boxes, 3)
        if bboxes.shape[1] == 6:
            bboxes = torch.cat(
                (bboxes, torch.zeros_like(bboxes[:, :1])),
                dim=1)
        bboxes_expanded = bboxes.unsqueeze(0).expand(n_points, n_boxes, 
                                            bboxes.shape[1])
        face_distances = get_face_distances(point, bboxes_expanded)

        inside_bbox = face_distances.min(dim=-1).values > 0
        inside_bbox = inside_bbox.T
        sp_inside = scatter_mean(inside_bbox.float(), 
                                        sp_pts_mask, dim=-1)
        sp_del = sp_inside < self.test_cfg.low_sp_thr
        inside_bbox[sp_del[:, sp_pts_mask]] = False

        sp_add = sp_inside > self.test_cfg.up_sp_thr
        inside_bbox[sp_add[:, sp_pts_mask]] = True

        points_for_max = point.clone()
        points_for_min = point.clone()
        points_for_max[~inside_bbox.T.bool()] = float('-inf')
        points_for_min[~inside_bbox.T.bool()] = float('inf')
        bboxes_max = points_for_max.max(axis=0)[0]
        bboxes_min = points_for_min.min(axis=0)[0]
        bboxes_sizes = bboxes_max - bboxes_min
        bboxes_centers = (bboxes_max + bboxes_min) / 2
        bboxes_out = torch.hstack((bboxes_centers, bboxes_sizes))
        bboxes_out = DepthInstance3DBoxes(bboxes_out, with_yaw=False, 
                                      box_dim=6, origin=(0.5, 0.5, 0.5))       
        return [(bboxes_out, labels, scores)]

    def _single_scene_multiclass_nms(self, bboxes, scores, 
                                     labels, fast_nms, iou_thr):
        """Multi-class nms for a single scene."""
        classes = labels.unique()
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for class_id in classes:
            ids = scores[labels == class_id] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[labels == class_id][ids]
            class_bboxes = bboxes[labels == class_id][ids]
            class_labels = labels[labels == class_id][ids]
            if with_yaw:
                nms_ids = nms3d(class_bboxes, class_scores, iou_thr)
            else:
                if fast_nms:
                    class_bboxes_for_nms = torch.cat(
                        (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                        dim=1)
                    nms_ids = nms3d_normal(class_bboxes_for_nms, class_scores, iou_thr)
                else:
                    nms_ids = aligned_3d_nms(_bbox_to_loss(class_bboxes), 
                                class_scores, class_labels, iou_thr)

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(class_labels[nms_ids])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        return nms_bboxes, nms_scores, nms_labels

def get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
    """Calculate distances from point to box faces."""
    shift = torch.stack(
        (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
            points[..., 2] - boxes[..., 2]),
        dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(
        shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = boxes[..., :3] + shift
    dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
    dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
    dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
    dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
    return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                        dim=-1)

# Meta-Learning 训练接口
def meta_train_step(model: UniDet3D_Meta, support_set, query_set, lr_inner):
    original_params = {name: param.clone() for name, param in model.named_params()}
    loss_support = model.compute_loss(support_set)
    grads = torch.autograd.grad(loss_support, model.parameters(), create_graph=True)
    model.update_params(lr_inner, first_order=False, source_params=grads)
    loss_query = model.compute_loss(query_set)
    for name, param in model.named_params():
        model.set_param(model, name, original_params[name])
    return loss_query

def compute_loss_example(model: UniDet3D_Meta, batch):
    batch_inputs_dict, batch_data_samples = batch
    loss = model.loss(batch_inputs_dict, batch_data_samples)
    return loss

