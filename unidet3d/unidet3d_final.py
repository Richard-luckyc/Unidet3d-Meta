# unidet3d_final.py
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

# 以 MetaModule 为基础实现 MetaConv2d
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size

        self.register_buffer('weight', to_var(conv.weight.data, requires_grad=True))
        if conv.bias is not None:
            self.register_buffer('bias', to_var(conv.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 影像分支类定义保留，但不会被初始化使用
import torchvision.models as models

class ResNet101_Backbone(MetaModule):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet101(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# ---------------- ModuleA: Multi-scale Feature Enhancement Block ----------------
class ModuleA(nn.Module):
    """
    ModuleA: Multi-scale convolutional enhancement + SE-style channel attention.
    Lightweight, plug-and-play. Input expected as [B, C, H, W].
    """
    def __init__(self, channels, reduction=8, dilations=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        mid = max(16, channels // 4)
        # 1x1 reduce
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid)
        # parallel convs with different receptive fields (dilations)
        self.convs = nn.ModuleList([
            nn.Conv2d(mid, mid, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.bn_convs = nn.ModuleList([nn.BatchNorm2d(mid) for _ in dilations])
        # fuse
        self.fuse = nn.Conv2d(mid * len(dilations), channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(channels)
        # SE-style channel gating
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, H, W]
        orig = x
        x = self.act(self.bn_reduce(self.reduce(x)))  # [B, mid, H, W]
        outs = []
        for conv, bn in zip(self.convs, self.bn_convs):
            outs.append(self.act(bn(conv(x))))
        x = torch.cat(outs, dim=1)
        x = self.act(self.bn_fuse(self.fuse(x)))  # [B, C, H, W]
        # SE gating
        s = self.global_pool(x)  # [B, C, 1, 1]
        s = self.act(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        out = orig * s + orig  # gated residual
        return out

# ---------------- NeighborhoodAdaptiveContext (NAC) ----------------
class NeighborhoodAdaptiveContext(nn.Module):
    """
    Neighborhood Adaptive Context (NAC) Module.
    
    [修改说明]: 
    1. 类名已从 NeighborhoodCrossAttentionFusion 改为 NeighborhoodAdaptiveContext。
    2. 这是一个“模态内自注意力”版本，仅接受几何特征作为输入。
    3. 通过 Unfold 操作提取几何特征自身的邻域作为上下文。
    """
    def __init__(self, embed_dim, num_heads=8, kernel_size=7, scale=1.0):
        super().__init__()
        # [修改 1] 移除了 image_channels/context_channels 参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = scale

        # [修改 2] 上下文投影层 (Context Projection)
        # 输入通道为 embed_dim (LiDAR)
        self.context_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # 自适应门控网络 (Adaptive Gating Network)
        # 负责计算原始特征和邻域上下文的融合权重
        self.adapt_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # FFN (Feed-Forward Network)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, lidar_feat):
        """
        修改forward 函数仅接受 lidar_feat 一个参数。
        """
        B, C, H, W = lidar_feat.shape
        
        # --- 1. 构建邻域上下文 (Context Construction) ---
        # [修改 4] 直接投影几何特征作为上下文源
        ctx_feat = self.context_proj(lidar_feat)  # [B, C, H, W]

        # 提取 7x7 局部邻域
        pad = self.kernel_size // 2
        unfold = nn.Unfold(kernel_size=self.kernel_size, padding=pad)
        
        # 形状: [B, C*K*K, L] -> [B, L, K*K, C]
        ctx_unfold = unfold(ctx_feat).view(B, C, -1, H*W).permute(0, 3, 2, 1) 

        # --- 2. 准备查询向量 (Query Preparation) ---
        # 几何特征作为 Query: [B, C, H, W] -> [B, L, 1, C]
        query = lidar_feat.view(B, C, -1).permute(0, 2, 1).unsqueeze(2)

        # --- 3. 邻域自注意力 (Neighborhood Self-Attention) ---
        # 计算相似度: Query (中心点) vs Neighbors (周围点)
        attn_scores = (query * ctx_unfold).sum(dim=-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, L, K*K, 1]

        # 聚合特征 (Weighted Sum)
        aggregated = (attn_weights * ctx_unfold).sum(dim=2)  # [B, L, C]

        # 恢复 2D 空间结构
        aggregated = aggregated.permute(0, 2, 1).view(B, C, H, W)
        aggregated = self.out_proj(aggregated)

        # --- 4. 自适应门控 (Adaptive Gating) ---
        # 计算全局统计信息 (Global Average Pooling)
        feat_concat = torch.cat([lidar_feat.mean([2,3]), aggregated.mean([2,3])], dim=1)
        
        # 生成权重 [alpha, beta]
        weights = self.adapt_mlp(feat_concat).unsqueeze(-1).unsqueeze(-1)

        # 融合：alpha * 邻域信息 + beta * 原始信息
        fused = weights[:, 0] * aggregated + weights[:, 1] * lidar_feat

        # --- 5. 前馈网络增强 (FFN Refinement) ---
        fused_flat = fused.view(B, C, -1).permute(0, 2, 1)
        fused_flat = self.norm(fused_flat)
        fused = fused_flat.permute(0, 2, 1).view(B, C, H, W)
        
        fused_ffn = self.ffn(fused.view(B, C, -1).transpose(1,2))
        fused_ffn = fused_ffn.transpose(1,2).view(B, C, H, W)
        
        # 残差连接
        fused = fused + fused_ffn
        
        return fused

# ---------------- UniDet3D_Meta (主类) ----------------
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
                 image_backbone_cfg=None,
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

        # [修改] 移除/注释掉 image backbone 初始化
        # if image_backbone_cfg is not None:
        #     self.image_backbone = MODELS.build(image_backbone_cfg)
        # else:
        #     self.image_backbone = ResNet101_Backbone(pretrained=True)

        # [修改] 使用 NeighborhoodAdaptiveContext 替代 CrossAttention
        self.nac_module = NeighborhoodAdaptiveContext(embed_dim=num_channels,
                                                      num_heads=8,
                                                      kernel_size=7)

        # ModuleA
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
        """
        修改后的 extract_feat：
          - 纯点云流程
          - ModuleA 增强
          - NAC 模态内增强 (强制执行)
        """
        # 1. LiDAR 特征提取
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        lidar_feat = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        # [新增逻辑] 获取超点的 3D 质心坐标，用于计算空间排序索引
        sp_coords_all = scatter_mean(x.indices[:, 1:].float()[inverse_mapping], superpoints, dim=0)
        
        # 2. 按 batch 分割
        lidar_feats = []
        for i in range(len(batch_offsets) - 1):
            s = batch_offsets[i]
            e = batch_offsets[i + 1]
            lidar_feats.append(lidar_feat[s:e])

        # 3. ModuleA 特征增强
        enhanced_feats = []
        for sample_idx, lf in enumerate(lidar_feats):
            N_i = lf.shape[0]
            if N_i == 0:
                enhanced_feats.append(lf)
                continue

            s = batch_offsets[sample_idx]
            e = batch_offsets[sample_idx + 1]
            coords = sp_coords_all[s:e]
            
            # 将坐标归一化并量化为 8-bit (0-255)
            coord_norm = (coords - coords.min(0)[0]) / (coords.max(0)[0] - coords.min(0)[0] + 1e-6)
            quantized = (coord_norm * 255).int()
            
            # 简单的位移打包操作（起到类似 Z-Curve 空间哈希的作用）
            morton_like = (quantized[:, 0] << 16) | (quantized[:, 1] << 8) | quantized[:, 2]
            
            # 获取排序索引和逆排序索引（Unsort）
            sort_idx = torch.argsort(morton_like)
            unsort_idx = torch.argsort(sort_idx)  # 核心：用于在卷积后恢复原序
            
            # 对特征进行空间局部性排序
            lf_sorted = lf[sort_idx]
            C = lf.shape[1]

            H = int((N_i) ** 0.5)
            if H <= 0: H = 1
            W = int((N_i + H - 1) // H)
            pad_n = H * W - N_i

            # 注意：这里改用 lf_sorted 进行 padding
            if pad_n > 0:
                pad_tensor = lf_sorted.new_zeros((pad_n, C))
                lf_padded = torch.cat([lf_sorted, pad_tensor], dim=0)
            else:
                lf_padded = lf_sorted

            try:
                lf_2d = lf_padded.transpose(0, 1).contiguous().view(1, C, H, W)
                lf_enh = self.module_a(lf_2d)
                lf_enh_1d = lf_enh.view(C, -1).transpose(0, 1).contiguous()
                lf_enh_1d = lf_enh_1d[:N_i, :]
                # [新增逻辑] 恢复原始顺序，对齐下游 Transformer 和 GT 标签
                lf_enh_1d = lf_enh_1d[unsort_idx]
                enhanced_feats.append(lf_enh_1d)
            except Exception as e:
                print(f"[ModuleA WARNING] Sample {sample_idx} failed: {e}")
                enhanced_feats.append(lf)

        lidar_feats = enhanced_feats

        # 4. [关键修改] NAC 模块调用 (Intra-modal Self-Attention)
        # 不再检查 image_input，对每个样本进行自适应增强
        nac_processed_feats = []
        for sample_idx, lf in enumerate(lidar_feats):
            N_i = lf.shape[0]
            if N_i == 0:
                nac_processed_feats.append(lf)
                continue
            
            C = lf.shape[1]
            # 重新计算 H, W (因为需要转为 2D 进行 Unfold)
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
                # [1, C, H, W]
                lf_2d = lf_padded.transpose(0, 1).contiguous().view(1, C, H, W)
                
                # 调用 NAC 模块 (输入输出均为几何特征)
                lf_nac = self.nac_module(lf_2d)
                
                # 转回 1D
                lf_nac_1d = lf_nac.view(C, -1).transpose(0, 1).contiguous()
                lf_nac_1d = lf_nac_1d[:N_i, :]
                nac_processed_feats.append(lf_nac_1d)
            except Exception as e:
                print(f"[NAC WARNING] Sample {sample_idx} failed: {e}")
                nac_processed_feats.append(lf)

        lidar_feats = nac_processed_feats

        return lidar_feats

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
        x = self.extract_feat(
            x,sp_pts_masks, inverse_mapping, batch_offsets,image_input=batch_inputs_dict.get('images', None))

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
            bboxes, labels, scores = results_list[i]
            data_sample.pred_instances_3d = InstanceData_(
                bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                points=batch_inputs_dict['points'][0])
        
        return batch_data_samples

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