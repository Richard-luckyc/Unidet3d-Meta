# unidet3d_final_method_a.py
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

# Meta Learning Utils
from torch.autograd import Variable
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

# ---------------- ModuleA: Multi-scale Feature Enhancement Block ----------------
class ModuleA(nn.Module):
    def __init__(self, channels, reduction=8, dilations=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        mid = max(16, channels // 4)
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid)
        self.convs = nn.ModuleList([
            nn.Conv2d(mid, mid, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.bn_convs = nn.ModuleList([nn.BatchNorm2d(mid) for _ in dilations])
        self.fuse = nn.Conv2d(mid * len(dilations), channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x
        x = self.act(self.bn_reduce(self.reduce(x)))
        outs = []
        for conv, bn in zip(self.convs, self.bn_convs):
            outs.append(self.act(bn(conv(x))))
        x = torch.cat(outs, dim=1)
        x = self.act(self.bn_fuse(self.fuse(x)))
        s = self.global_pool(x)
        s = self.act(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        out = orig * s + orig
        return out

# ---------------- [Method A Modification] Single-Modal Fusion Module ----------------
class NeighborhoodCrossAttentionFusion(nn.Module):
    """
    修改说明：
    该模块已从 'Image-LiDAR Fusion' 修改为 'Latent-LiDAR Interaction'。
    我们移除了图像输入，改用一组可学习的参数 (Learnable Token) 作为 Key/Value。
    这既保留了 Cross-Attention 的计算结构，又使其适应单模态点云输入。
    """
    def __init__(self, embed_dim, num_heads=8, kernel_size=7, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dropout = dropout

        # [Method A] 移除 Image Projection，改为可学习的 Latent Token
        # Shape: [1, embed_dim, 1, 1] -> 将被广播到任意 H, W
        self.learnable_token = nn.Parameter(torch.randn(1, embed_dim, 1, 1))
        # 使用 Xavier 初始化，保证训练初期的稳定性
        nn.init.xavier_uniform_(self.learnable_token)

        # 自适应权重 MLP
        self.adapt_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 2),  # 输出 [w_lidar, w_latent]
            nn.Softmax(dim=-1)
        )
        
        self.scale = embed_dim ** -0.5
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, lidar_feat):
        # [Method A] Input signature changed: removes image_feat
        B, C, H, W = lidar_feat.shape
        
        # 1. 构造 Latent Feature Map (模拟 Image Feature)
        # 将 [1, C, 1, 1] 扩展为 [B, C, H, W]
        # 这样每个点云特征都在查询同样的全局上下文，但位置编码(隐含在Conv中)和Unfold会带来局部差异
        latent_feat = self.learnable_token.expand(B, -1, H, W)
        
        pad = self.kernel_size // 2
        unfold = nn.Unfold(kernel_size=self.kernel_size, padding=pad)
        
        # 2. 对 Latent Token 进行 Unfold (提取邻域)
        latent_unfold = unfold(latent_feat)  # [B, embed_dim * K*K, L]
        
        lidar_flat = lidar_feat.view(B, C, -1).permute(0, 2, 1)  # [B, L, embed_dim]
        K2 = self.kernel_size * self.kernel_size
        
        latent_unfold = latent_unfold.view(B, C, K2, H*W).permute(0, 3, 2, 1)  # [B, L, K*K, embed_dim]
        
        # 3. Cross Attention: Query=LiDAR, Key/Value=Latent
        query = lidar_flat.unsqueeze(2).expand(-1, -1, K2, -1)
        
        # 计算注意力分数
        attn_scores = (query * latent_unfold).sum(dim=-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, L, K*K, 1]
        
        # 聚合特征
        fused = (attn_weights * latent_unfold).sum(dim=2)  # [B, L, embed_dim]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        fused = self.out_proj(fused)
        
        # 4. 自适应融合 (Adaptive Weighting)
        # 以前是 image_feat.mean(), 现在用 learnable_token.squeeze()
        # 注意：这里需要 expand 到 Batch 维度
        token_vec = self.learnable_token.view(1, C).expand(B, -1) 
        feat_concat = torch.cat([lidar_feat.mean([2,3]), token_vec], dim=1)  # [B, 2*C]
        
        weights = self.adapt_mlp(feat_concat).unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]
        
        # ResNet-style fusion
        fused = weights[:, 0] * fused + weights[:, 1] * lidar_feat 
        
        # FFN & Norm
        fused_flat = fused.view(B, C, -1).permute(0, 2, 1)
        fused_flat = self.norm(fused_flat)
        fused = fused_flat.permute(0, 2, 1).view(B, C, H, W)
        
        fused_ffn = self.ffn(fused.view(B, C, -1).transpose(1,2))
        fused_ffn = fused_ffn.transpose(1,2).view(B, C, H, W)
        fused = fused + fused_ffn
        
        return fused

# ---------------- UniDet3D_Meta ----------------
@MODELS.register_module()
class UniDet3D_Meta(MetaModule, Base3DDetector):
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
                 image_backbone_cfg=None, # 保留接口但不使用
                 module_a_cfg=None,
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

        # [Method A] Image backbone removed to save memory since we are single-modal
        self.image_backbone = None 

        # [Method A] 使用修改后的 Latent Fusion Module
        self.cross_modal_fusion = NeighborhoodCrossAttentionFusion(embed_dim=num_channels,
                                                                   num_heads=8,
                                                                   kernel_size=7,
                                                                   dropout=0.1)
        # ModuleA Setup
        if module_a_cfg is None:
            self.module_a = ModuleA(num_channels)
        else:
            ch = module_a_cfg.get('channels', num_channels)
            red = module_a_cfg.get('reduction', 8)
            dils = module_a_cfg.get('dilations', (1,2,4))
            self.module_a = ModuleA(ch, reduction=red, dilations=tuple(dils))

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

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets, image_input=None):
        # ---------------- LiDAR 特征提取 ----------------
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        lidar_feat = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)

        # ---------------- 按 batch 分割 ----------------
        lidar_feats = []
        for i in range(len(batch_offsets) - 1):
            s = batch_offsets[i]
            e = batch_offsets[i + 1]
            lidar_feats.append(lidar_feat[s:e])

        # ---------------- ModuleA 特征增强 & [Method A] Latent Fusion ----------------
        processed_feats = []
        for sample_idx, lf in enumerate(lidar_feats):
            N_i = lf.shape[0]
            if N_i == 0:
                processed_feats.append(lf)
                continue
            C = lf.shape[1]

            # 动态 reshape
            H = int((N_i) ** 0.5)
            if H <= 0: H = 1
            W = int((N_i + H - 1) // H)
            pad_n = H * W - N_i

            if pad_n > 0:
                pad_tensor = lf.new_zeros((pad_n, C))
                lf_padded = torch.cat([lf, pad_tensor], dim=0)
            else:
                lf_padded = lf

            try:
                # 1. ModuleA 增强
                lf_2d = lf_padded.transpose(0, 1).contiguous().view(1, C, H, W)
                lf_enh = self.module_a(lf_2d)  # [1, C, H, W]

                # 2. [Method A] Latent Fusion (替代原来的 Multi-modal Fusion)
                # 直接传入 enhanced feature，不需要图像
                lf_fused = self.cross_modal_fusion(lf_enh)

                # 3. 还原回 1D
                lf_final_1d = lf_fused.view(C, -1).transpose(0, 1).contiguous()
                lf_final_1d = lf_final_1d[:N_i, :]
                processed_feats.append(lf_final_1d)

            except Exception as e:
                print(f"[Feature WARNING] Failed on sample {sample_idx}: {e}")
                processed_feats.append(lf)

        lidar_feats = processed_feats
        
        # [Method A] Removed original "if image_input..." fusion block entirely
        
        return lidar_feats

    # [以下保持原样，省略未变动部分以节省空间]
    def collate(self, points, elastic_points=None):
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

    def _forward(*args, **kwargs):
        pass

    def _select_queries(self, x, gt_instances):
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
        masks = masks_src.clone()
        if torch.sum(masks == -1) != 0:
            masks[masks == -1] = torch.max(masks) + 1
            masks = torch.nn.functional.one_hot(masks)[:, :-1]
        else:
            masks = torch.nn.functional.one_hot(masks)
        return masks.bool()

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
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
        
        # 注意：这里不再传入 image_input
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets, image_input=None)

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
        # 保持原有的 predict 逻辑基本不变，只是 extract_feat 调用变了
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
        
        # 不再传入图像
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)

        x = self.decoder(x, sp_centers, datasets_names)

        results_list = self.predict_by_feat(x, sp_pts_masks_src, 
                                            points_src, datasets_names)
        
        for i, data_sample in enumerate(batch_data_samples):
            bboxes, labels, scores = results_list[i]
            data_sample.pred_instances_3d = InstanceData_(
                bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                points=batch_inputs_dict['points'][0])
        
        return batch_data_samples

    # 剩余的辅助函数（predict_by_feat, trim_bboxes..., _single_scene_multiclass_nms, get_face_distances）
    # 请保持原样，未做修改。
    # 这里为了代码完整性，建议将上面原始代码的 predict_by_feat 等函数直接复制粘贴到此处。
    # ... (Keep the rest of the functions exactly as in original code) ...
    def predict_by_feat(self, out, sp_pts_masks, points, 
                        datasets_names):
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
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=True)
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
        n_points = point.shape[0]
        n_boxes = bboxes.shape[0]
        point = point.unsqueeze(1).expand(n_points, n_boxes, 3)
        if bboxes.shape[1] == 6:
            bboxes = torch.cat(
                (bboxes, torch.zeros_like(bboxes[:, :1])),
                dim=1)
        bboxes = bboxes.unsqueeze(0).expand(n_points, n_boxes, 
                                            bboxes.shape[1])
        face_distances = get_face_distances(point, bboxes)

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
        bboxes = torch.hstack((bboxes_centers, bboxes_sizes))
        bboxes = DepthInstance3DBoxes(bboxes, with_yaw=False, 
                                      box_dim=6, origin=(0.5, 0.5, 0.5))       
        return [(bboxes, labels, scores)]

    def _single_scene_multiclass_nms(self, bboxes, scores, 
                                     labels, fast_nms, iou_thr):
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
                    class_bboxes = torch.cat(
                        (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                        dim=1)
                    nms_ids = nms3d_normal(class_bboxes, class_scores, iou_thr)
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

# utils
def get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
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

# Meta-Learning 训练接口（保持原样）
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