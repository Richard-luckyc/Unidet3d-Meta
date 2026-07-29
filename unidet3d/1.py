```python
# unidet3d_cross_with_moduleA.py
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

        # 注册为 buffer 以便于 meta 参数更新逻辑复用（轻量实现）
        self.register_buffer('weight', to_var(conv.weight.data, requires_grad=True))
        if conv.bias is not None:
            self.register_buffer('bias', to_var(conv.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 影像分支（保留，以免影响加载已有 ckpt）
import torchvision.models as models

class ResNet101_Backbone(MetaModule):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet101(pretrained=pretrained)
        # 去掉平均池化和全连接层，只保留 conv1 ~ layer4
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
        return x  # [B, 2048, H', W']

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

# ---------------- NeighborhoodCrossAttentionFusion (保留, 添加自适应权重) ----------------
class NeighborhoodCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, kernel_size=7, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dropout = dropout

        # 投影图像特征到 embed_dim
        self.img_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        # 新增: 自适应融合权重 MLP
        self.adapt_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),  # 拼接LiDAR+img feat
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 2),  # 输出 [w_lidar, w_img]
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
    
    def forward(self, lidar_feat, image_feat):
        B, C, H, W = lidar_feat.shape
        img_feat_proj = self.img_proj(image_feat)  # [B, embed_dim, H, W]
        pad = self.kernel_size // 2
        unfold = nn.Unfold(kernel_size=self.kernel_size, padding=pad)
        img_unfold = unfold(img_feat_proj)  # [B, embed_dim * K*K, L]
        lidar_flat = lidar_feat.view(B, C, -1).permute(0, 2, 1)  # [B, L, embed_dim]
        K2 = self.kernel_size * self.kernel_size
        img_unfold = img_unfold.view(B, C, K2, H*W).permute(0, 3, 2, 1)  # [B, L, K*K, embed_dim]
        query = lidar_flat.unsqueeze(2).expand(-1, -1, K2, -1)
        attn_scores = (query * img_unfold).sum(dim=-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, L, K*K, 1]
        fused = (attn_weights * img_unfold).sum(dim=2)  # [B, L, embed_dim]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        fused = self.out_proj(fused)
        # 新增: 自适应权重融合
        feat_concat = torch.cat([lidar_feat.mean([2,3]), img_feat_proj.mean([2,3])], dim=1)  # [B, 2*C]
        weights = self.adapt_mlp(feat_concat).unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]
        fused = weights[:, 0] * fused + weights[:, 1] * lidar_feat  # 自适应残差
        fused_flat = fused.view(B, C, -1).permute(0, 2, 1)
        fused_flat = self.norm(fused_flat)
        fused = fused_flat.permute(0, 2, 1).view(B, C, H, W)
        fused_ffn = self.ffn(fused.view(B, C, -1).transpose(1,2))
        fused_ffn = fused_ffn.transpose(1,2).view(B, C, H, W)
        fused = fused + fused_ffn
        return fused

# ---------------- UniDet3D_Meta (主类，集成 ModuleA) ----------------
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
        # 注意：调用 Base3DDetector 的初始化保持不变（以兼容 mmengine）
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

        # image backbone（保留）
        if image_backbone_cfg is not None:
            self.image_backbone = MODELS.build(image_backbone_cfg)
        else:
            # 可能会引入意外的 ckpt keys（请按需开/关）
            self.image_backbone = ResNet101_Backbone(pretrained=True)

        # cross-modal fusion (保留)
        self.cross_modal_fusion = NeighborhoodCrossAttentionFusion(embed_dim=num_channels,
                                                                   num_heads=8,
                                                                   kernel_size=7,
                                                                   dropout=0.1)
        # ModuleA: 如果 module_a_cfg 为 None，使用默认参数；否则可从 cfg 中解析
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
        稳健版 extract_feat：
          - LiDAR 特征提取（原始）
          - 将超点特征按样本切分
          - 对每个样本用 ModuleA 进行增强（reshape -> ModuleA -> 裁回）
          - 可选的图像融合（如果 image_input & image_backbone 存在）
        """

        # ---------------- LiDAR 特征提取 ----------------
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)

        # 聚合超点特征
        lidar_feat = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)

        # ---------------- 按 batch 分割 ----------------
        lidar_feats = []
        for i in range(len(batch_offsets) - 1):
            s = batch_offsets[i]
            e = batch_offsets[i + 1]
            lidar_feats.append(lidar_feat[s:e])

        # ---------------- ModuleA 特征增强 ----------------
        enhanced_feats = []
        for sample_idx, lf in enumerate(lidar_feats):
            N_i = lf.shape[0]
            if N_i == 0:
                enhanced_feats.append(lf)
                continue
            C = lf.shape[1]

            # 自动估计 H, W
            H = int((N_i) ** 0.5)
            if H <= 0:
                H = 1
            W = int((N_i + H - 1) // H)
            pad_n = H * W - N_i

            if pad_n > 0:
                pad_tensor = lf.new_zeros((pad_n, C))
                lf_padded = torch.cat([lf, pad_tensor], dim=0)
            else:
                lf_padded = lf

            try:
                # reshape to [1, C, H, W]
                lf_2d = lf_padded.transpose(0, 1).contiguous().view(1, C, H, W)
                lf_enh = self.module_a(lf_2d)  # [1, C, H, W]
                lf_enh_1d = lf_enh.view(C, -1).transpose(0, 1).contiguous()
                lf_enh_1d = lf_enh_1d[:N_i, :]
                enhanced_feats.append(lf_enh_1d)
            except Exception as e:
                print(f"[ModuleA WARNING] Failed on sample {sample_idx} with {N_i} pts: H={H}, W={W}, pad={pad_n}, err={e}")
                enhanced_feats.append(lf)

        lidar_feats = enhanced_feats

        # ---------------- 可选图像融合 ----------------
        if image_input is not None and hasattr(self, 'image_backbone'):
            try:
                image_feats = self.image_backbone(image_input)
            except Exception as e:
                print(f"[Fusion WARNING] image_backbone failed: {e}")
                image_feats = None

            if image_feats is not None:
                fused_feats = []
                B = image_feats.shape[0]
                n_iter = min(B, len(lidar_feats))
                for i in range(n_iter):
                    lf_i = lidar_feats[i]
                    N_i = lf_i.shape[0]
                    if N_i == 0:
                        fused_feats.append(lf_i)
                        continue

                    Hg = int((N_i) ** 0.5)
                    if Hg <= 0:
                        Hg = 1
                    Wg = int((N_i + Hg - 1) // Hg)
                    pad_n = Hg * Wg - N_i

                    if pad_n > 0:
                        pad_tensor = lf_i.new_zeros((pad_n, lf_i.shape[1]))
                        lf_pad = torch.cat([lf_i, pad_tensor], dim=0)
                    else:
                        lf_pad = lf_i

                    try:
                        lf_2d = lf_pad.transpose(0, 1).contiguous().view(1, lf_pad.shape[1], Hg, Wg)
                        img_feat_i = image_feats[i].unsqueeze(0)
                        fused_2d = self.cross_modal_fusion(lf_2d, img_feat_i)
                        fused_1d = fused_2d.view(fused_2d.shape[1], -1).transpose(0, 1).contiguous()
                        fused_1d = fused_1d[:N_i, :]
                        fused_feats.append(fused_1d)
                    except Exception as e:
                        print(f"[Fusion WARNING] sample {i}: fusion failed: N={N_i}, Hg={Hg}, Wg={Wg}, pad_n={pad_n}, err={e}")
                        fused_feats.append(lf_i)

                if len(lidar_feats) > n_iter:
                    fused_feats.extend(lidar_feats[n_iter:])

                lidar_feats = fused_feats

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

    def predict_by_feat(self, out, sp_pts_masks_src, points_src, datasets_names):
        # 修改：支持batch处理，每个样本独立计算
        results_list = []
        num_classes = out['cls_preds'][0].shape[1] - 1  # 假设所有样本共享num_classes
        for i in range(len(sp_pts_masks_src)):
            cls_preds = out['cls_preds'][i]
            pred_bboxes = out['bboxes'][i]
            sp_pts_mask = sp_pts_masks_src[i] 
            point = points_src[i]
            dataset_name = datasets_names[i]
        
            scores = F.softmax(cls_preds, dim=-1)[:, :-1]
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
                result = [(DepthInstance3DBoxes(
                    nms_bboxes, 
                    with_yaw=nms_bboxes.shape[1] == 7, 
                    box_dim=nms_bboxes.shape[1], 
                    origin=(0.5, 0.5, 0.5)),
                    nms_labels, nms_scores)]
            else:
                result = self.trim_bboxes_by_superpoints(sp_pts_mask, point, 
                                                         nms_bboxes, nms_labels,
                                                         nms_scores)
            results_list.append(result[0])  # 取第一个（单场景）
        return results_list

    def trim_bboxes_by_superpoints(self, sp_pts_mask, point, 
                                   bboxes, labels, scores):
        n_points = point.shape[0]
        n_boxes = bboxes.shape[0]
        device = point.device

        if n_boxes == 0:
            # 处理空bboxes情况：返回空结果，避免scatter_mean错误
            empty_bboxes = DepthInstance3DBoxes(
                torch.zeros((0, 6), device=device), 
                with_yaw=False, box_dim=6, origin=(0.5, 0.5, 0.5)
            )
            return [(empty_bboxes, torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device))]

        # 扩展point和bboxes
        point = point.unsqueeze(1).expand(n_points, n_boxes, 3)
        if bboxes.shape[1] == 6:
            bboxes = torch.cat(
                (bboxes, torch.zeros_like(bboxes[:, :1])),
                dim=1)
        bboxes = bboxes.unsqueeze(0).expand(n_points, n_boxes, 
                                            bboxes.shape[1])
        face_distances = get_face_distances(point, bboxes)

        inside_bbox = face_distances.min(dim=-1).values > 0
        inside_bbox = inside_bbox.T  # [n_boxes, n_points]

        # 修正：scatter_mean的dim=1（对points维度聚合），并指定dim_size避免空case
        num_sp = int(sp_pts_mask.max().item()) + 1
        sp_inside = scatter_mean(inside_bbox.float(), 
                                 sp_pts_mask.unsqueeze(0).expand(n_boxes, -1),  # broadcast index to [n_boxes, n_points]
                                 dim=1, dim_size=num_sp)  # 输出 [n_boxes, num_sp]

        sp_del = sp_inside < self.test_cfg.low_sp_thr  # [n_boxes, num_sp]
        # 扩展sp_del到points：sp_del[box_idx, sp_id] -> mask for points in that sp
        sp_del_points = sp_del[torch.arange(n_boxes).unsqueeze(1), sp_pts_mask]  # [n_boxes, n_points]
        inside_bbox[sp_del_points] = False

        sp_add = sp_inside > self.test_cfg.up_sp_thr  # [n_boxes, num_sp]
        sp_add_points = sp_add[torch.arange(n_boxes).unsqueeze(1), sp_pts_mask]  # [n_boxes, n_points]
        inside_bbox[sp_add_points] = True

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
```