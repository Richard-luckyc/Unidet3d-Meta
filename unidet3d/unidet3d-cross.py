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

# 影像分支及融合模块
import torchvision.models as models

# 我们直接采用 ResNet101 作为图像特征提取 backbone，去掉最后全连接层
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
        # 输入 x: [B, 3, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # 输出尺寸通常为 [B, C, H', W']

class NeighborhoodCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, kernel_size=7, dropout=0.1):
        """
        Args:
            embed_dim (int): 融合后的特征维度，与 LiDAR 特征通道数一致。
            num_heads (int): 多头注意力头数。
            kernel_size (int): 局部注意力窗口大小（例如 7 表示 7x7）。
            dropout (float): dropout 概率。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dropout = dropout

        # 投影图像特征到 embed_dim
        self.img_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        # 交叉注意力：我们不直接调用全局 MultiheadAttention，而是只在局部区域计算
        # 为了实现局部注意力，可以采用 nn.Unfold 提取局部块，再计算点积注意力
        self.scale = embed_dim ** -0.5
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, lidar_feat, image_feat):
        """
        Args:
            lidar_feat: Tensor [B, embed_dim, H, W]，来自 LiDAR 分支的 BEV 特征
            image_feat: Tensor [B, 2048, H, W]，图像分支的特征（需与 LiDAR 对齐）
        Returns:
            fused_feat: Tensor [B, embed_dim, H, W]，融合后的特征
        """
        B, C, H, W = lidar_feat.shape
        # 将图像特征投影到 embed_dim
        img_feat_proj = self.img_proj(image_feat)  # [B, embed_dim, H, W]
        print(image_feat)
        
        # 提取局部邻域特征（使用 nn.Unfold）
        pad = self.kernel_size // 2
        unfold = nn.Unfold(kernel_size=self.kernel_size, padding=pad)
        # 将动态分支（图像）展开，得到局部块，形状: [B, embed_dim * (kernel_size^2), H*W]
        img_unfold = unfold(img_feat_proj)  # [B, embed_dim * K*K, L]，L = H*W
        
        # 将 LiDAR 特征作为 query，reshape 为 [B, embed_dim, H*W]
        lidar_flat = lidar_feat.view(B, C, -1)  # [B, embed_dim, L]
        # 转置为 [B, L, embed_dim]
        lidar_flat = lidar_flat.permute(0, 2, 1)  # [B, L, embed_dim]
        # 将展开后的图像特征 reshape 为 [B, L, K*K, embed_dim]
        K2 = self.kernel_size * self.kernel_size
        img_unfold = img_unfold.view(B, C, K2, H*W).permute(0, 3, 2, 1)  # [B, L, K*K, embed_dim]

        # 计算 query 与局部 key 的点积注意力
        # lidar_flat: [B, L, embed_dim]，expand为 [B, L, K*K, embed_dim]
        query = lidar_flat.unsqueeze(2).expand(-1, -1, K2, -1)
        # 注意力得分: [B, L, K*K]
        attn_scores = (query * img_unfold).sum(dim=-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, K*K]
        attn_weights = attn_weights.unsqueeze(-1)  # [B, L, K*K, 1]
        # 计算局部融合结果：对局部区域的值加权求和
        fused = (attn_weights * img_unfold).sum(dim=2)  # [B, L, embed_dim]
        # 将融合结果 reshape 回 [B, embed_dim, H, W]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        # 输出融合后的特征通过一个 1x1 卷积
        fused = self.out_proj(fused)
        # 残差连接：与原始 LiDAR 特征相加，并进行归一化与前馈
        fused = fused + lidar_feat
        # 归一化注意，此处使用 LayerNorm需要先 reshape
        fused_flat = fused.view(B, C, -1).permute(0, 2, 1)
        fused_flat = self.norm(fused_flat)
        fused = fused_flat.permute(0, 2, 1).view(B, C, H, W)
        # 前馈网络
        fused_ffn = self.ffn(fused.view(B, C, -1).transpose(1,2))
        fused_ffn = fused_ffn.transpose(1,2).view(B, C, H, W)
        fused = fused + fused_ffn
        return fused

@MODELS.register_module()
class UniDet3D_Meta(MetaModule,Base3DDetector):
    r"""UniDet3D for unifed 3D object detection.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select min(query_thr, n_queries) queries
            for training and testing.
        use_superpoints (bool): Flag to indicate whether to use superpoints
            for improved detection.
        bbox_by_mask (bool): Whether to derive bounding boxes from masks.
        target_by_distance (bool): Whether to use targets based on distance 
            to bbox center.
        fast_nms (bool): Flag for using fast Non-Maximum Suppression.
        use_sync_bn (bool, optional): Flag to use synchronized 
            batch normalization. Defaults to True.
        backbone (ConfigDict, optional): Config dict of the backbone. 
            Defaults to None.
        decoder (ConfigDict, optional): Config dict of the decoder. 
            Defaults to None.
        criterion (ConfigDict, optional): Config dict of the criterion. 
            Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters. 
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process 
            config of :class:BaseDataPreprocessor.
            It usually includes:
                - ``pad_size_divisor``
                - ``pad_value``
                - ``mean``
                - ``std``.
        init_cfg (dict or ConfigDict, optional): The config to control the 
            initialization. Defaults to None.
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
                 image_backbone_cfg=None,
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
        # 新增图像分支，采用 ResNet101，利用 ResNet101_Backbone 封装
        if image_backbone_cfg is not None:
            self.image_backbone = MODELS.build(image_backbone_cfg)
        else:
            self.image_backbone = ResNet101_Backbone(pretrained=True)
        # 融合模块：假设我们将 LiDAR 特征投影到 BEV 后尺寸与图像特征一致
        # 这里假设 LiDAR 输出 channels = num_channels, 图像分支输出 channels = 2048（ResNet101 layer4 的输出）
        self.cross_modal_fusion = NeighborhoodCrossAttentionFusion(embed_dim=num_channels,
                                                                   num_heads=8,
                                                                   kernel_size=7,
                                                                   dropout=0.1)
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

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets,image_input=None):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        #LiDAR 特征提取
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        lidar_feat = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        # 将 LiDAR 特征划分为 batch 内的各样本
        lidar_feats = []
        for i in range(len(batch_offsets) - 1):
            lidar_feats.append(lidar_feat[batch_offsets[i]: batch_offsets[i + 1]])
        
        # 如果提供了图像输入，则进行图像特征提取与融合
        if image_input is not None and hasattr(self,'image_backbone'):
            # 提取图像特征：输出尺寸通常为 [B, 2048, H_img, W_img]
            image_feats = self.image_backbone(image_input)
            # 假设已有一个投影模块将 LiDAR 特征转换为 2D 格式，与图像特征尺寸对齐
            # 此处简单模拟，将每个超点特征 reshape 成 2D 格式
            # 注意：实际中需要设计投影模块，这里仅作示例
            B = image_feats.shape[0]
            # 假设每个样本对应一个特征图尺寸 H_img x W_img，且 H_img * W_img == lidar_feats[i].shape[0]
            fused_feats = []
            for i in range(B):
                lidar_feat_i = lidar_feats[i]  # [N_i, C]
                # 将 lidar_feat_i reshape 成 [1, C, H_img, W_img]
                # 此处假设 H_img, W_img 已知（例如通过配置传入），这里简单用 sqrt 估计
                N_i = lidar_feat_i.shape[0]
                H_img = W_img = int(N_i**0.5)
                lidar_2d = lidar_feat_i.transpose(0,1).view(1, -1, H_img, W_img)
                # 对图像特征取对应样本
                image_feat_i = image_feats[i].unsqueeze(0)  # [1, 2048, H_img, W_img]
                # 融合
                fused_2d = self.cross_modal_fusion(lidar_2d, image_feat_i)
                # 将融合后的 2D 特征 reshape 回 [N_i, C]
                fused_1d = fused_2d.view(fused_2d.shape[1], -1).transpose(0,1)
                fused_feats.append(fused_1d)
            lidar_feats = fused_feats
        
        return lidar_feats

    def collate(self, points, elastic_points=None):
        """Collate a batch of points into a sparse tensor.

        Args:
            points (List[Tensor]): A batch of point tensors. Each tensor
                should contain points in the format (N, 3 + num_features),
                where N is the number of points.
            elastic_points (List[Tensor], optional): A batch of transformed
                point tensors (if any) after elastic point augmentation. 
                Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: 
                - coordinates (Tensor): The sparse tensor coordinates after 
                quantization and normalization.
                - features (Tensor): The features corresponding to the points.
                - inverse_mapping (Tensor): A mapping of points to their 
                indices in the original tensor.
                - spatial_shape (Tensor): The spatial shape of the sparse tensor,
                clipped to the minimum spatial shape.
        """
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
        """Implement abstract method of Base3DDetector."""
        pass

    def _select_queries(self, x, gt_instances):
        """Select queries for the training pass.

        Args:
            x (List[Tensor]): A list of tensors of length `batch_size`, 
                where each tensor has the shape (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): A list of ground truth 
                instances of length `batch_size`, where each instance may 
                contain:
                    - labels of shape (n_gts_i,)
                    - sp_masks of shape (n_gts_i, n_points_i).

        Returns:
            Tuple[List[Tensor], List[Tensor], List[InstanceData_]]:
                - queries (List[Tensor]): A list of queries of length 
                `batch_size`, where each query has the shape 
                (n_queries_i, n_channels).
                - sp_centers (List[Tensor]): A list of tensors representing 
                spatial centers for the selected queries.
                - updated_gt_instances (List[InstanceData_]): A list of ground 
                truth instances (same length as `gt_instances`), 
                each updated with query_masks of shape (n_gts_i, n_queries_i).
        """
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
        """Generate 3D bounding boxes from masks.

        Args:
            masks (Tensor): A tensor of boolean masks, of shape 
                (n, n_points) indicating which points belong to each object.
            points (Tensor): A tensor of shape (n_points, 3) representing 
                the 3D coordinates of the points.

        Returns:
            DepthInstance3DBoxes: A set of 3D bounding boxes, where each box 
            is represented as a tensor of shape (6,) containing:
                - Center coordinates (x, y, z)
                - Dimensions (width, height, depth)
            
            If no masks are provided, an empty `DepthInstance3DBoxes` instance 
            will be returned.

        """
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
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_points, 1).
        
        
        Returns:
            mask (Tensor): instance masks of shape (n_points, num_inst_obj).
        """
        masks = masks_src.clone()
        if torch.sum(masks == -1) != 0:
            masks[masks == -1] = torch.max(masks) + 1
            masks = torch.nn.functional.one_hot(masks)[:, :-1]
        else:
            masks = torch.nn.functional.one_hot(masks)

        return masks.bool()

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
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
        """Compute targets for final locations for a single scene.

        Args:
            points (Tensor): Final locations for level.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            topk (int): The number of nearest ground truth boxes 
                to consider for target assignment.

        Returns:
            Tensor: A tensor indicating which ground truth boxes each 
                point is assigned to, where the shape is (n_points, n_boxes).        
        """
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
        """Predict results from a batch of inputs and data samples
                with post-processing.

        Args:
            batch_inputs_dict (dict): A dictionary containing model inputs, 
                which must include 'points' key.
            batch_data_samples (List[:obj:Det3DDataSample]): A list of Data 
                Samples. Each Data Sample includes information such as
                superpoints (gt_pts_seg.sp_pts_mask).

        Returns:
            List[:obj:Det3DDataSample]: Detection results for the input 
                samples. Each Det3DDataSample contains 'pred_instances_3d' 
                with the following keys:
                    - bboxes_3d (Tensor): 3D bounding boxes of detected 
                    instances, shape (num_instances, 6).
                    - scores_3d (Tensor): Classification scores for each 
                    detected instance, shape (num_instances,).
                    - labels_3d (Tensor): Labels of instances, shape 
                    (num_instances,).
        """
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
        """Predict bounding boxes and labels from model outputs.

        Args:
            out (dict): A dictionary containing model outputs with the 
                following keys:
                - 'cls_preds': Tensor of shape (n_bboxes, num_classes) 
                containing classification scores for each point.
                - 'bboxes': Tensor of shape (n_bboxes, 7) containing 
                predicted bounding boxes.
            sp_pts_masks (List[Tensor]): A list of superpoint masks.
            points (List[Tensor]): A list of point tensors containing 
                the 3D coordinates of the points being evaluated.

            datasets_names (List[str]): A list of dataset names 
                corresponding to the input samples.

        Returns:
            List[Tuple[DepthInstance3DBoxes, Tensor, Tensor]]: A list containing 
            tuples of predicted bounding boxes and their associated 
            labels and scores.
        """
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
        """Trim bounding boxes based on superpoint masks.

        Args:
            sp_pts_mask (Tensor): A boolean tensor indicating the valid points 
                for each superpoint.
            point (Tensor): A tensor of shape (n_points, 3) representing the 
                3D coordinates of the points.
            bboxes (Tensor): A tensor of predicted bounding boxes, with shape 
                (n_boxes, 6) or (n_boxes, 7) if yaw is included.
            labels (Tensor): A tensor of shape (n_boxes,) containing the 
                predicted labels for each bounding box.
            scores (Tensor): A tensor of shape (n_boxes,) containing the 
                classification scores for each bounding box.

        Returns:
            List[Tuple[DepthInstance3DBoxes, Tensor, Tensor]]: A list 
                containing a tuple of trimmed bounding boxes, 
                labels, and scores.
        """
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
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted bounding boxes of shape (N_boxes, 6) 
                or (N_boxes, 7), where each box represents (x, y, z, length, 
                width, height) and optionally yaw.
            scores (Tensor): Predicted scores for the bounding boxes of 
                shape (N_boxes,), representing confidence scores.
            labels (Tensor): Predicted labels for each bounding box, 
                shape (N_boxes,).
            fast_nms (bool): Flag indicating whether to use the fast NMS 
                implementation.
            iou_thr (float): IoU threshold for NMS to filter overlapping boxes.

        Returns:
            tuple[Tensor, ...]: Predicted bboxes, scores and labels.
        """
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

def get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
    """Calculate distances from point to box faces.

    Args:
        points (Tensor): Final locations of shape (N_points, N_boxes, 3).
        boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

    Returns:
        Tensor: Face distances of shape (N_points, N_boxes, 6),
        (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
    """
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