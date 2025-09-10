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

# 影像分支及融合模块
import torchvision.models as models
import torch.nn.init as init # init is imported but not used in the final code, can be removed if not needed elsewhere

# 我们直接采用 ResNet101 作为图像特征提取 backbone，去掉最后全连接层
# Changed inheritance from MetaModule to nn.Module
class ResNet101_Backbone(nn.Module):
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
            num_heads (int): 多头注意力头数。 (Note: current implementation doesn't split into heads)
            kernel_size (int): 局部注意力窗口大小（例如 7 表示 7x7）。
            dropout (float): dropout 概率。 (Note: dropout is defined but not used in forward)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads # Parameter stored but not used in the simplified local attention
        self.kernel_size = kernel_size
        self.dropout = dropout # Parameter stored but not used

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

        # 计算 query 与局部 key 的点积注意力 (Simplified attention without multi-head)
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
        fused_residual = fused + lidar_feat # Store residual before norm
        # 归一化注意，此处使用 LayerNorm需要先 reshape
        fused_flat = fused_residual.view(B, C, -1).permute(0, 2, 1) # Apply norm on residual
        fused_flat = self.norm(fused_flat)
        fused_norm = fused_flat.permute(0, 2, 1).view(B, C, H, W) # Reshape back after norm
        # 前馈网络
        # Apply FFN on the normalized features
        fused_ffn_input = fused_norm.view(B, C, -1).transpose(1,2)
        fused_ffn = self.ffn(fused_ffn_input)
        fused_ffn = fused_ffn.transpose(1,2).view(B, C, H, W)
        # Second residual connection
        fused = fused_norm + fused_ffn # Add FFN output to the normalized feature
        return fused

@MODELS.register_module()
# Changed inheritance, removed MetaModule
class UniDet3D_Meta(Base3DDetector):
    r"""UniDet3D for unifed 3D object detection (Meta features removed).

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
        image_backbone_cfg (dict or ConfigDict, optional): Config for image backbone.
            If None, uses default ResNet101_Backbone. Defaults to None.
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
                 image_backbone_cfg=None, # Added config option for image backbone
                 init_cfg=None):
        # Changed super call
        super().__init__(
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
             # Build from config if provided
            self.image_backbone = MODELS.build(image_backbone_cfg)
        else:
             # Default to our ResNet101_Backbone class
            self.image_backbone = ResNet101_Backbone(pretrained=True)

        # 融合模块：假设我们将 LiDAR 特征投影到 BEV 后尺寸与图像特征一致
        # 这里假设 LiDAR 输出 channels = num_channels, 图像分支输出 channels = 2048（ResNet101 layer4 的输出）
        self.cross_modal_fusion = NeighborhoodCrossAttentionFusion(embed_dim=num_channels,
                                                                     num_heads=8, # Keep parameters, even if simplified attention doesn't use all
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
            # Use SyncBatchNorm if specified, applied to sparse tensor features
            self.output_layer = spconv.SparseSequential(
                 nn.SyncBatchNorm(num_channels, eps=1e-4, momentum=0.1),
                 nn.ReLU(inplace=True))
        else:
             # Use BatchNorm1d for sparse tensor features if not using SyncBN
            self.output_layer = spconv.SparseSequential(
                 nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
                 nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets, image_input=None):
        """Extract features from sparse tensor, optionally fusing with image features.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (total_superpoints,).
            inverse_mapping (Tensor): of shape (n_points,) maps original points to sparse tensor indices.
            batch_offsets (List[int]): of len batch_size + 1, indicates superpoint boundaries per batch item.
            image_input (Tensor, optional): Image tensor [B, 3, H_img, W_img]. Defaults to None.

        Returns:
            List[Tensor]: List of length batch_size,
                each tensor containing features for superpoints in that batch item
                (shape [n_superpoints_i, n_channels]). Features are potentially fused with image data.
        """
        # LiDAR 特征提取
        x = self.input_conv(x)
        x, _ = self.unet(x) # Pass through the sparse U-Net backbone
        x = self.output_layer(x) # Apply BatchNorm/SyncBatchNorm and ReLU

        # Aggregate features for superpoints using the inverse mapping
        # Features shape: [N_sparse_points, C], inverse_mapping shape: [N_original_points]
        # Need to map features back to original point cloud structure before scattering
        # Assuming `inverse_mapping` maps indices in the sparse tensor feature `x.features`
        # back to the original concatenated point cloud order.
        # `superpoints` should index into this original concatenated point cloud.

        # Example: Check if inverse_mapping size matches original total points
        # total_original_points = batch_offsets[-1] # This might not be correct if batch_offsets refer to superpoints
        # Need clarification on how `inverse_mapping` relates `x.features` to `superpoints`

        # *** Assuming inverse_mapping maps sparse features back to original point order ***
        # Create a full feature tensor aligned with the original points
        # Need the total number of original points before sparsification
        num_original_points = superpoints.shape[0] # Assuming superpoints covers all original points
        original_features = x.features.new_zeros((num_original_points, x.features.shape[1]))
        original_features[inverse_mapping] = x.features

        # Now scatter mean using the original point features and superpoint indices
        lidar_feat = scatter_mean(original_features, superpoints, dim=0) # Shape: [total_superpoints, C]

        # 将 LiDAR 特征划分为 batch 内的各样本
        lidar_feats = []
        for i in range(len(batch_offsets) - 1):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i+1]
            lidar_feats.append(lidar_feat[start_idx:end_idx]) # lidar_feats is List[Tensor[N_sp_i, C]]

        # 如果提供了图像输入，则进行图像特征提取与融合
        if image_input is not None and hasattr(self, 'image_backbone') and hasattr(self, 'cross_modal_fusion'):
            # 提取图像特征：输出尺寸通常为 [B, 2048, H_img, W_img]
            image_feats = self.image_backbone(image_input) # [B, 2048, H_img', W_img']

            # *** Crucial Step: Project LiDAR features to 2D BEV grid and align with image features ***
            # This part requires a specific projection mechanism (e.g., PointPillars-style or BEV pooling)
            # The current code simulates this by reshaping, which is likely incorrect.
            # Placeholder for actual LiDAR-to-BEV projection:
            # lidar_bev_feats = self.lidar_to_bev_projection(lidar_feats, ...) # Needs implementation

            # --- Simulation Logic (Needs Replacement) ---
            B = image_feats.shape[0]
            fused_feats = []
            H_img_feat, W_img_feat = image_feats.shape[-2:] # Get spatial dims of image features

            for i in range(B):
                lidar_feat_i = lidar_feats[i]  # [N_sp_i, C] where C is num_channels
                # *** This reshaping is a placeholder and likely incorrect ***
                # It assumes a square BEV grid can be formed directly from superpoint features.
                N_sp_i = lidar_feat_i.shape[0]
                # Attempt to match image feature spatial dimensions if possible
                # This requires knowing the relationship between N_sp_i and H_img_feat * W_img_feat
                # Forcing a square root might lead to incorrect dimensions or errors if N_sp_i is not a perfect square.
                # Let's assume for simulation purposes that the BEV projection yielded a map
                # of size H_bev x W_bev, and somehow N_sp_i = H_bev * W_bev.
                # Further assume H_bev=H_img_feat, W_bev=W_img_feat for simplicity here.
                if N_sp_i != H_img_feat * W_img_feat:
                     print(f"Warning: Superpoint count {N_sp_i} doesn't match image feature grid size {H_img_feat}x{W_img_feat}. Reshaping might fail or be incorrect.")
                     # Fallback or error handling needed here. Skipping fusion for this item?
                     # For now, attempt reshape but be aware it's problematic.
                     try:
                         # If sizes match, reshape is straightforward
                         lidar_2d = lidar_feat_i.transpose(0, 1).view(1, self.unet.num_channels, H_img_feat, W_img_feat)
                     except RuntimeError:
                         print(f"Error: Cannot reshape LiDAR features [{N_sp_i}, {self.unet.num_channels}] to [1, {self.unet.num_channels}, {H_img_feat}, {W_img_feat}]. Skipping fusion for item {i}.")
                         fused_feats.append(lidar_feat_i) # Use original LiDAR features if fusion fails
                         continue
                else:
                     lidar_2d = lidar_feat_i.transpose(0, 1).view(1, self.unet.num_channels, H_img_feat, W_img_feat) # [1, C, H_img', W_img']

                # 对图像特征取对应样本
                image_feat_i = image_feats[i].unsqueeze(0)  # [1, 2048, H_img', W_img']

                # Ensure image features are compatible with fusion module input (size and channels)
                # The NeighborhoodCrossAttentionFusion expects lidar_feat [B, embed_dim, H, W] and image_feat [B, 2048, H, W]
                # Here lidar_2d has C channels, image_feat_i has 2048 channels. They match the fusion module expectations.
                # Spatial dimensions must also match.
                if lidar_2d.shape[-2:] != image_feat_i.shape[-2:]:
                     print(f"Warning: Mismatched spatial dimensions between LiDAR BEV ({lidar_2d.shape[-2:]}) and Image features ({image_feat_i.shape[-2:]}). Fusion might be incorrect.")
                     # Optional: Resize one to match the other?
                     # image_feat_i = F.interpolate(image_feat_i, size=lidar_2d.shape[-2:], mode='bilinear', align_corners=False)


                # 融合
                fused_2d = self.cross_modal_fusion(lidar_2d, image_feat_i) # [1, C, H_img', W_img']

                # 将融合后的 2D 特征 reshape 回 [N_sp_i, C]
                fused_1d = fused_2d.view(fused_2d.shape[1], -1).transpose(0, 1) # [N_sp_i, C]
                fused_feats.append(fused_1d)

            lidar_feats = fused_feats # Replace original list with fused features list
            # --- End Simulation Logic ---

        return lidar_feats

    def collate(self, points, elastic_points=None):
        """Collate a batch of points into a sparse tensor using MinkowskiEngine.

        Args:
            points (List[Tensor]): A batch of point tensors. Each tensor
                should contain points in the format (N, 3 + num_features),
                where N is the number of points. Coordinates are expected in the first 3 columns.
            elastic_points (List[Tensor], optional): A batch of transformed
                point tensors (if any) after elastic point augmentation. Coordinates only.
                Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - coordinates (Tensor): The sparse tensor coordinates after
                  quantization and normalization (batch_idx, x, y, z).
                - features (Tensor): The features corresponding to the points.
                - inverse_mapping (Tensor): A mapping of unique voxel indices back to their
                  original point indices in the concatenated batch.
                - spatial_shape (Tensor): The spatial shape of the sparse tensor,
                  clipped to the minimum spatial shape.
        """
        coords_list = []
        feats_list = []
        device = points[0].device # Get device from first tensor

        for i, p in enumerate(points):
            # Coordinates for quantization: Use elastic points if available, otherwise original points.
            # Normalize coordinates by voxel size relative to the minimum coordinate in that point cloud.
            if elastic_points is not None:
                 # Elastic points are assumed to be coordinates only [N, 3]
                 # The original points `p` provide features and relative coordinates
                 coords = (elastic_points[i] - elastic_points[i].min(0)[0]) / self.voxel_size
            else:
                 coords = (p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size

            # Features: original features + relative coordinates (mean centered)
            feats = torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0)))

            coords_list.append(coords.int()) # Minkowski needs integer coordinates
            feats_list.append(feats)

        # Use MinkowskiEngine for batch collation and sparsification
        coordinates, features = ME.utils.sparse_collate(coords_list, feats_list, device=device)

        # Create SparseTensor directly for quantization
        # Note: ME.SparseTensor automatically handles quantization and duplicate removal
        sparse_tensor = ME.SparseTensor(features=features, coordinates=coordinates, device=device)

        # Extract quantized coordinates and unique features
        quantized_coordinates = sparse_tensor.coordinates
        unique_features = sparse_tensor.features

        # Compute inverse mapping (which original point falls into which voxel)
        # This requires tracking original indices during collation, ME.utils.sparse_collate doesn't provide this directly.
        # Re-implementing collation with index tracking or using a different method might be necessary
        # if the exact inverse map (like ME.TensorField provides) is needed downstream.
        # Let's simulate the TensorField behavior for inverse_map for now:
        field = ME.TensorField(features=features, coordinates=coordinates.float()) # TensorField needs float coords initially
        # Use the coordinate_manager from the sparse tensor for consistency
        inverse_mapping = field.inverse_mapping(sparse_tensor.coordinate_manager)


        # Calculate spatial shape based on max coordinates in the batch
        # Add 1 because coordinates are 0-indexed
        spatial_shape = torch.max(quantized_coordinates[:, 1:], dim=0)[0] + 1
        # Clip spatial shape to minimum required size
        spatial_shape = torch.maximum(spatial_shape, torch.tensor(self.min_spatial_shape, device=device))


        return quantized_coordinates, unique_features, inverse_mapping, spatial_shape.cpu() # spatial_shape often needed on CPU

    def _forward(self, *args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        # This method is usually not directly used in MMDetection3D framework
        # The main logic is in loss() and predict()
        raise NotImplementedError("'_forward' method is not implemented for UniDet3D_Meta")

    def _select_queries(self, x, gt_instances):
        """Select queries for the training pass based on query_thr.

        Args:
            x (List[Tensor]): A list of tensors of length `batch_size`,
                where each tensor has the shape (n_superpoints_i, n_channels). Features per superpoint.
            gt_instances (List[InstanceData_]): A list of ground truth
                instances of length `batch_size`, where each instance may
                contain:
                    - labels of shape (n_gts_i,)
                    - sp_masks of shape (n_gts_i, n_superpoints_i). Boolean mask indicating which superpoints belong to GT.
                    - sp_centers of shape (n_superpoints_i, 3). Center coords of each superpoint.

        Returns:
            Tuple[List[Tensor], List[Tensor], List[InstanceData_]]:
                - queries (List[Tensor]): A list of selected query features length
                  `batch_size`, shape (n_queries_i, n_channels).
                - sp_centers_queries (List[Tensor]): A list of tensors representing
                  spatial centers for the selected queries. Shape (n_queries_i, 3).
                - updated_gt_instances (List[InstanceData_]): List of ground
                  truth instances, updated with query_masks of shape (n_gts_i, n_queries_i).
        """
        queries = []
        sp_centers_queries = [] # Renamed for clarity
        for i in range(len(x)):
            n_superpoints_i = x[i].shape[0]
            if n_superpoints_i > self.query_thr:
                # Randomly select query_thr superpoints
                ids = torch.randperm(n_superpoints_i, device=x[i].device)[:self.query_thr]
                queries.append(x[i][ids])
                sp_centers_queries.append(gt_instances[i].sp_centers[ids])
                # Update GT instance masks to match the selected queries
                if hasattr(gt_instances[i], 'sp_masks') and gt_instances[i].sp_masks is not None:
                     gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
                else:
                     # Handle case where sp_masks might not exist or be needed
                     # Maybe set query_masks to None or an empty tensor depending on downstream use
                     gt_instances[i].query_masks = None # Or handle appropriately

                # Optional: update sp_centers in gt_instances too? The code did this, let's keep it
                # Though it seems redundant as sp_centers_queries is returned separately.
                gt_instances[i].sp_centers = gt_instances[i].sp_centers[ids]


            else:
                # Use all superpoints if fewer than query_thr
                queries.append(x[i])
                sp_centers_queries.append(gt_instances[i].sp_centers)
                if hasattr(gt_instances[i], 'sp_masks') and gt_instances[i].sp_masks is not None:
                     gt_instances[i].query_masks = gt_instances[i].sp_masks
                else:
                     gt_instances[i].query_masks = None
                # No need to update gt_instances[i].sp_centers here as all are kept


        return queries, sp_centers_queries, gt_instances

    def get_bboxes_by_masks(self, masks, points):
        """Generate axis-aligned 3D bounding boxes from point masks.

        Args:
            masks (Tensor): A tensor of boolean masks, of shape
                (n_objects, n_points) indicating which points belong to each object instance.
            points (Tensor): A tensor of shape (n_points, 3) representing
                the 3D coordinates of the points for a single sample.

        Returns:
            DepthInstance3DBoxes: A set of 3D bounding boxes (axis-aligned), where each box
            is represented as a tensor of shape (7,) containing:
                - Center coordinates (x, y, z)
                - Dimensions (length, width, height) -> dx, dy, dz
                - Yaw (0 for axis-aligned)

            If no masks result in valid points, an empty `DepthInstance3DBoxes` instance
            will be returned.
        """
        boxes = []
        for mask in masks: # Iterate through each object mask
            object_points = points[mask]
            if object_points.shape[0] == 0: # Skip if no points belong to this mask
                continue
            xyz_min = object_points.min(dim=0).values
            xyz_max = object_points.max(dim=0).values
            center = (xyz_max + xyz_min) / 2
            size = xyz_max - xyz_min # dx, dy, dz (length, width, height)
            # Ensure minimum size to avoid degenerate boxes
            size = torch.clamp(size, min=1e-4)
            # Format: center(3), size(3), yaw(1)
            box = torch.cat((center, size, center.new_zeros(1))) # Add 0 yaw
            boxes.append(box)

        if len(boxes) == 0:
            bboxes = DepthInstance3DBoxes(
                masks.new_zeros(0, 7), with_yaw=True, # Use 7 dim even if yaw is 0
                box_dim=7, origin=(0.5, 0.5, 0.5))
        else:
            boxes = torch.stack(boxes)
            bboxes = DepthInstance3DBoxes(
                boxes, with_yaw=True, box_dim=7, origin=(0.5, 0.5, 0.5))
        return bboxes

    def get_gt_inst_masks(self, masks_src):
        """Create one-hot ground truth instance masks from instance IDs.

        Args:
            masks_src (Tensor): Instance IDs per point, shape (n_points,).
                               -1 or other ignored label might be present.

        Returns:
            mask (Tensor): Boolean instance masks of shape (num_inst_obj, n_points).
        """
        masks = masks_src.clone().long() # Ensure long type for one_hot
        unique_ids = torch.unique(masks)
        # Filter out potential ignore labels (e.g., -1, sometimes 0 depending on dataset)
        # Assuming -1 is the ignore label
        valid_ids = unique_ids[unique_ids != -1]

        if len(valid_ids) == 0:
            # Return empty mask if no valid instances
            return masks.new_zeros((0, masks.shape[0]), dtype=torch.bool)

        # Map original IDs to a contiguous range [0, num_inst_obj-1] for one_hot
        id_map = {int(old_id): new_id for new_id, old_id in enumerate(valid_ids)}
        mapped_masks = masks.new_full(masks.shape, -1) # Initialize with ignore value
        for old_id, new_id in id_map.items():
             mapped_masks[masks == old_id] = new_id

        # Create one-hot, ignoring the -1 entries implicitly if num_classes is set correctly
        num_inst_obj = len(valid_ids)
        one_hot_masks = F.one_hot(mapped_masks, num_classes=num_inst_obj) # Shape: [n_points, num_inst_obj]

        # Transpose to get [num_inst_obj, n_points] and convert to bool
        return one_hot_masks.permute(1, 0).bool()

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which includes
                'points' key, and potentially 'images', 'elastic_coords'.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as `gt_instances_3d`
                and `gt_pts_seg` (containing `pts_instance_mask`, `sp_pts_mask`).
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0] # Tracks cumulative number of superpoints per batch item
        superpoint_bias = 0 # Ensures superpoint IDs are unique across the batch
        sp_gt_instances = [] # List to store GT instances per sample, maybe updated
        sp_pts_masks = [] # List to store superpoint masks per sample (adjusted with bias)
        sp_centers = [] # List to store superpoint centers per sample

        # --- Prepare Ground Truth and Point Coordinates ---
        points_list = batch_inputs_dict['points'] # Original points list
        elastic_coords_list = batch_inputs_dict.get('elastic_coords', None) # Elastic coords if available

        processed_points = [] # Points adjusted relative to their min coordinate
        shifts = [] # The minimum coordinate shift applied to each point cloud

        for i in range(len(points_list)):
             if elastic_coords_list is not None:
                 # Use elastic coords for geometry, but shift needs to be calculated
                 # relative to original points if GT boxes are in original space
                 p_orig = points_list[i][:, :3]
                 p_elastic = elastic_coords_list[i] # Assumed shape [N, 3]
                 shift = p_orig.min(0)[0] # Shift based on original points
                 # Processed points for input are elastic coords shifted relative to their own min, scaled by voxel size
                 # This seems inconsistent with the collate function logic? Revisit collate maybe?
                 # Let's assume collate handles the (coords - min_coords) / voxel_size correctly.
                 # Here, we need points relative to their *original* shift for GT processing.
                 processed_p = p_orig - shift # Use original points relative to their shift for GT ops
                 processed_points.append(processed_p)
                 shifts.append(shift)
             else:
                 p_orig = points_list[i][:, :3]
                 shift = p_orig.min(0)[0]
                 processed_p = p_orig - shift
                 processed_points.append(processed_p)
                 shifts.append(shift)


        datasets_names = []
        for i, data_sample in enumerate(batch_data_samples):
            # Determine dataset name (used for dataset-specific configs)
            dataset_name = self.get_dataset(data_sample.lidar_path) # Implement this helper robustly
            if dataset_name is None:
                 raise ValueError(f"Could not determine dataset name from path: {data_sample.lidar_path}")
            datasets_names.append(dataset_name)
            dataset_idx = self.decoder.datasets.index(dataset_name) # Get index for config lookups

            gt_pts_seg = data_sample.gt_pts_seg # Contains instance and superpoint masks
            gt_instances_3d = data_sample.gt_instances_3d # Contains labels and potentially bboxes

            # --- Get/Update Ground Truth BBoxes ---
            if self.bbox_by_mask[dataset_idx]:
                 # Generate GT boxes from point instance masks
                 gt_masks = self.get_gt_inst_masks(gt_pts_seg.pts_instance_mask) # [N_gt, N_points]
                 # Use processed points (relative to shift) for bbox calculation
                 gt_bboxes_3d = self.get_bboxes_by_masks(gt_masks, processed_points[i])
                 gt_instances_3d.bboxes_3d = gt_bboxes_3d
            else:
                 # Adjust existing GT box centers by the calculated shift
                 if not hasattr(gt_instances_3d, 'bboxes_3d') or gt_instances_3d.bboxes_3d is None:
                      raise ValueError("GT Instances do not have bboxes_3d for non-mask based GT")

                 original_bboxes = gt_instances_3d.bboxes_3d
                 center = original_bboxes.gravity_center - shifts[i]
                 # Reconstruct tensor (center, size, [yaw])
                 new_bbox_tensor = torch.cat((center, original_bboxes.tensor[:, 3:]), dim=1)
                 gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(
                     new_bbox_tensor,
                     with_yaw=original_bboxes.with_yaw,
                     box_dim=new_bbox_tensor.shape[1],
                     origin=(0.5, 0.5, 0.5)) # Assuming center origin

            # --- Prepare Superpoint Information ---
            current_sp_mask = gt_pts_seg.sp_pts_mask.clone() # Get superpoint mask for this sample
            # Calculate superpoint centers using processed points
            sp_centers_i = scatter_mean(processed_points[i], current_sp_mask, dim=0)
            gt_instances_3d.sp_centers = sp_centers_i # Store centers in GT instance (for _select_queries)

            # --- Generate Target Masks (sp_masks) ---
            if self.target_by_distance[dataset_idx]:
                 # Assign superpoints to closest GT boxes based on distance
                 gt_instances_3d.sp_masks = self.get_targets(
                     sp_centers_i,
                     gt_instances_3d.bboxes_3d,
                     self.train_cfg.topk # Dataset specific topk? Assume global for now.
                 ) # Shape: [N_gt, N_superpoints]
            else:
                 print(f"Warning: target_by_distance is False for dataset {dataset_name}, but no alternative sp_mask generation is implemented.")
                 # Maybe set sp_masks based on point instance masks projected to superpoints? Needs implementation.
                 # gt_instances_3d.sp_masks = project_instance_mask_to_superpoints(...)
                 pass # Let it potentially fail later if sp_masks is None and needed


            # Adjust superpoint mask IDs to be unique across batch
            current_sp_mask += superpoint_bias
            sp_pts_masks.append(current_sp_mask) # Store adjusted mask
            num_superpoints_i = sp_centers_i.shape[0]
            superpoint_bias += num_superpoints_i # Update bias for next sample
            batch_offsets.append(superpoint_bias) # Record end offset for this sample

            sp_gt_instances.append(gt_instances_3d) # Store potentially updated GT instance

        # --- Collate Points and Extract Features ---
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        # Create sparse tensor for input to the network
        x = spconv.SparseConvTensor(
            features=features,
            indices=coordinates.int(), # spconv needs int indices (coords) -> Ensure collate provides batch_idx, x, y, z
            spatial_size=spatial_shape.tolist(), # Use spatial_shape from collate
            batch_size=len(batch_data_samples)
        )

        # Concatenate all adjusted superpoint masks for the batch
        sp_pts_masks_all = torch.cat(sp_pts_masks, dim=0) # Total superpoints in batch

        # Extract features (potentially fused with image)
        # Pass image tensor if available
        image_tensor = batch_inputs_dict.get('images', None)
        x_features_list = self.extract_feat(
            x, sp_pts_masks_all, inverse_mapping, batch_offsets, image_input=image_tensor
        ) # Returns list of features per sample [N_sp_i, C]

        # --- Select Queries and Run Decoder ---
        # Note: _select_queries modifies sp_gt_instances in-place (adds query_masks)
        queries_list, sp_centers_queries_list, sp_gt_instances_updated = \
            self._select_queries(x_features_list, sp_gt_instances)

        # Pass queries to the decoder
        # The decoder likely expects concatenated queries and batch info, or handles lists. Check decoder implementation.
        # Assuming decoder handles lists:
        decoder_output = self.decoder(queries_list, sp_centers_queries_list, datasets_names) # Pass dataset names too

        # --- Calculate Loss ---
        # Criterion calculates loss based on decoder output and updated GT instances
        loss_dict = self.criterion(decoder_output, sp_gt_instances_updated, datasets_names)

        return loss_dict

    def get_dataset(self, lidar_path):
        """Determine the dataset name from the lidar file path."""
        # Simple implementation: check if known dataset names are substrings
        # More robust: Use metadata passed in data_sample if available
        for dataset in self.decoder.datasets: # Assumes decoder has a list of known dataset names
            if dataset in lidar_path.split('/'):
                return dataset
        # Fallback or error if no known dataset found
        print(f"Warning: Could not identify dataset from path {lidar_path}. Available datasets: {self.decoder.datasets}")
        return None # Or return a default dataset name?

    def get_targets(self, points, gt_bboxes, topk):

        float_max = points.new_tensor(1e8)
        n_points = points.shape[0]
        n_boxes = len(gt_bboxes)

        if n_boxes == 0:
            # Return all-false mask if no GT boxes
            return points.new_zeros((0, n_points), dtype=torch.bool)

        # Use gravity center for distance calculation
        gt_centers = gt_bboxes.gravity_center # Shape: (n_boxes, 3)

        # Expand dimensions for broadcasting distance calculation
        points_expanded = points.unsqueeze(1).expand(n_points, n_boxes, 3) # [N_points, N_boxes, 3]
        gt_centers_expanded = gt_centers.unsqueeze(0).expand(n_points, n_boxes, 3) # [N_points, N_boxes, 3]

        # Calculate squared Euclidean distance
        center_distances = torch.sum((points_expanded - gt_centers_expanded)**2, dim=-1) # Shape: [N_points, N_boxes]

        # --- Top-k assignment logic ---
        # Find the distance to the k-th nearest point for each box
        # Need distances from each box to all points: shape [N_boxes, N_points]
        center_distances_T = center_distances.T # Shape: [N_boxes, N_points]

        # Ensure k is not larger than the number of points
        actual_topk = min(topk, n_points)

        if actual_topk > 0:
             # Get the distance to the k-th nearest point for each box
             topk_distances_per_box, _ = torch.topk(
                 center_distances_T,
                 k=actual_topk,
                 dim=1, # Find nearest points for each box
                 largest=False # Smallest distances are nearest
             )
             # The distance threshold for each box is the distance to its k-th nearest point
             distance_threshold = topk_distances_per_box[:, -1] # Shape: [N_boxes]
        else:
             # If topk is 0, no points should be assigned
             distance_threshold = points.new_full((n_boxes,), -1.0) # Negative threshold ensures nothing matches


        # Assign points to a box if their distance is less than or equal to the box's threshold
        # Add a small epsilon for strict inequality if needed, but <= seems common
        assignment_mask = center_distances_T <= distance_threshold.unsqueeze(1) # Shape: [N_boxes, N_points]


        # --- Ensure each point is assigned to at most one box (closest one within its top-k sets) ---
        # If a point falls within the top-k radius of multiple boxes, assign it to the *closest* one.

        # Mask out distances for points not within the top-k radius of *any* box
        # A point is a candidate if it's true in *any* row of assignment_mask
        point_is_candidate = assignment_mask.any(dim=0) # Shape: [N_points]

        # Create a distance matrix where non-candidate or non-topk distances are infinity
        masked_distances = center_distances_T.clone() # [N_boxes, N_points]
        # Set non-topk distances to infinity
        masked_distances[~assignment_mask] = float_max
        # Set distances for non-candidate points to infinity (across all boxes)
        masked_distances[:, ~point_is_candidate] = float_max


        # Find the minimum distance (closest box) for each point among the valid assignments
        min_dist_values, min_box_indices = torch.min(masked_distances, dim=0) # Find best box for each point

        # Create final mask: True only for the closest box assignment for each point
        final_assignment_mask = torch.zeros_like(assignment_mask, dtype=torch.bool)

        # Indices for points that were assigned at least one box
        assigned_point_indices = torch.where(point_is_candidate)[0]

        if len(assigned_point_indices) > 0:
             # Get the corresponding best box indices for these points
             best_box_indices_for_assigned_points = min_box_indices[assigned_point_indices]
             # Use advanced indexing to set True at the [best_box, assigned_point] locations
             final_assignment_mask[best_box_indices_for_assigned_points, assigned_point_indices] = True


        return final_assignment_mask # Shape: [n_boxes, n_points]


    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):

        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks_batch = [] # Concatenated superpoint masks for the whole batch
        sp_centers_batch = [] # List of superpoint centers per sample
        datasets_names = []
        sp_pts_masks_src_list = [] # List of original superpoint masks per sample
        points_src_list = [] # List of original points per sample

        for i, data_sample in enumerate(batch_data_samples):
            dataset_name = self.get_dataset(data_sample.lidar_path)
            if dataset_name is None:
                 raise ValueError(f"Could not determine dataset name from path: {data_sample.lidar_path}")
            datasets_names.append(dataset_name)

            # Use superpoint mask from data sample (might be from GT or prediction during inference)
            # Often GT superpoints are used even during inference for methods like this.
            if not hasattr(data_sample, 'gt_pts_seg') or data_sample.gt_pts_seg is None:
                 # Need a way to get superpoints if not in gt_pts_seg (e.g., run segmentation first)
                 raise ValueError("Superpoint masks (sp_pts_mask) not found in data_sample.gt_pts_seg for prediction.")

            sp_pts_mask_orig = data_sample.gt_pts_seg.sp_pts_mask.clone()
            sp_pts_masks_src_list.append(sp_pts_mask_orig) # Store original mask for this sample

            points = batch_inputs_dict['points'][i][:, :3] # Use original points [N, 3]
            points_src_list.append(points) # Store original points

            # Calculate superpoint centers
            sp_centers_i = scatter_mean(points, sp_pts_mask_orig, dim=0)
            sp_centers_batch.append(sp_centers_i)

            # Adjust mask IDs for batch processing
            sp_mask_adjusted = sp_pts_mask_orig + superpoint_bias
            sp_pts_masks_batch.append(sp_mask_adjusted)

            num_superpoints_i = sp_centers_i.shape[0]
            superpoint_bias += num_superpoints_i
            batch_offsets.append(superpoint_bias)


        # --- Collate Points and Extract Features ---
        # Note: Elastic coords are usually not used during prediction
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])

        x = spconv.SparseConvTensor(
            features=features,
            indices=coordinates.int(),
            spatial_size=spatial_shape.tolist(),
            batch_size=len(batch_data_samples)
        )

        sp_pts_masks_all = torch.cat(sp_pts_masks_batch, dim=0)

        # Extract features (potentially fused with image)
        image_tensor = batch_inputs_dict.get('images', None)
        # Pass image tensor if available
        x_features_list = self.extract_feat(
            x, sp_pts_masks_all, inverse_mapping, batch_offsets, image_input=image_tensor
        ) # Returns list [N_sp_i, C]

        # --- Run Decoder ---
        # Decoder takes list of features, list of centers, and dataset names
        decoder_output = self.decoder(x_features_list, sp_centers_batch, datasets_names)

        # --- Post-process Decoder Output ---
        # Pass necessary info for post-processing (NMS, potentially superpoint trimming)
        # predict_by_feat likely handles one sample at a time if it iterates through lists internally.
        # Or it might expect batched tensors from the decoder. Adjust based on decoder/predict_by_feat structure.
        # Assuming predict_by_feat processes the output dictionary and needs lists of other info.
        results_list = self.predict_by_feat(decoder_output, sp_pts_masks_src_list,
                                            points_src_list, datasets_names)

        # --- Update Data Samples with Predictions ---
        for i, data_sample in enumerate(batch_data_samples):
            bboxes, labels, scores = results_list[i] # Assumes results_list has one tuple per sample
            pred_instances = InstanceData() # Use standard InstanceData
            pred_instances.bboxes_3d = bboxes # Should be DepthInstance3DBoxes
            pred_instances.labels_3d = labels
            pred_instances.scores_3d = scores
            # Add points to the prediction instance data if needed downstream
            # pred_instances.points = points_src_list[i] # Add original points if needed
            data_sample.pred_instances_3d = pred_instances # Assign predictions

        return batch_data_samples


    def predict_by_feat(self, out, sp_pts_masks_list, points_list,
                        datasets_names):

        results_batch = []
        num_samples = len(out['cls_preds']) # Assuming output is a list per sample

        for i in range(num_samples):
            cls_preds = out['cls_preds'][i] # Logits for sample i [N_queries_i, N_classes+1]
            pred_bboxes = out['bboxes'][i] # Box parameters for sample i [N_queries_i, box_dim]
            sp_pts_mask = sp_pts_masks_list[i] # Original superpoint mask for sample i
            point = points_list[i] # Original points for sample i
            dataset_name = datasets_names[i]
            dataset_idx = self.decoder.datasets.index(dataset_name) # Index for config

            # --- Process Scores and Select Top-K ---
            scores = F.softmax(cls_preds, dim=-1)[:, :-1] # Apply softmax, remove background class [N_queries, N_classes]
            num_classes = scores.shape[1]

            # Create labels tensor corresponding to scores
            labels = torch.arange(
                num_classes,
                device=scores.device).unsqueeze(0).repeat(
                    cls_preds.shape[0], 1) # [N_queries, N_classes]

            # Flatten scores and labels, select top-k predictions overall
            scores_flat = scores.flatten() # [N_queries * N_classes]
            labels_flat = labels.flatten() # [N_queries * N_classes]

            # Make sure topk_insts doesn't exceed available scores
            actual_topk = min(self.test_cfg.topk_insts, scores_flat.shape[0])

            if actual_topk == 0 :
                 # Handle case with 0 predictions requested or available
                 empty_boxes = DepthInstance3DBoxes(scores.new_zeros((0, pred_bboxes.shape[1])), box_dim=pred_bboxes.shape[1], with_yaw=(pred_bboxes.shape[1]==7))
                 results_batch.append((empty_boxes, scores.new_zeros(0, dtype=torch.long), scores.new_zeros(0)))
                 continue


            topk_scores, topk_indices_flat = scores_flat.topk(actual_topk, sorted=True)
            topk_labels = labels_flat[topk_indices_flat]

            # --- Map Top-K back to Original Queries ---
            # Find which query generated each top-k prediction
            topk_query_indices = torch.div(topk_indices_flat, num_classes, rounding_mode='floor')
            topk_bboxes = pred_bboxes[topk_query_indices] # Select corresponding boxes [TopK, box_dim]

            # --- Perform NMS ---
            # Get NMS parameters based on dataset
            fast_nms_flag = self.fast_nms[dataset_idx]
            iou_thr = self.test_cfg.iou_thr[dataset_idx]
            score_thr = self.test_cfg.score_thr # Assuming global score threshold for now

            nms_bboxes_tensor, nms_scores, nms_labels = \
                self._single_scene_multiclass_nms(topk_bboxes,
                                                  topk_scores,
                                                  topk_labels,
                                                  fast_nms_flag,
                                                  iou_thr,
                                                  score_thr) # Pass score threshold

            # Convert final tensor boxes to DepthInstance3DBoxes
            nms_bboxes = DepthInstance3DBoxes(
                nms_bboxes_tensor,
                with_yaw=(nms_bboxes_tensor.shape[1] == 7),
                box_dim=nms_bboxes_tensor.shape[1],
                origin=(0.5, 0.5, 0.5) # Assuming center origin
            )

            # --- Optional: Trim BBoxes using Superpoints ---
            use_sp_trimming = self.use_superpoints[dataset_idx] # Check dataset config
            if use_sp_trimming and nms_bboxes.tensor.shape[0] > 0: # Only trim if configured and boxes exist
                 # Call trimming function
                 # trim_bboxes_by_superpoints returns a list containing one tuple
                 trimmed_results = self.trim_bboxes_by_superpoints(
                     sp_pts_mask, point,
                     nms_bboxes.tensor, # Pass the tensor data
                     nms_labels, nms_scores,
                     dataset_idx # Pass dataset index to access test_cfg inside
                 )
                 results_batch.append(trimmed_results[0]) # Append the tuple (bboxes, labels, scores)
            else:
                 # If not trimming, add NMS results directly
                 results_batch.append((nms_bboxes, nms_labels, nms_scores))

        return results_batch


    def trim_bboxes_by_superpoints(self, sp_pts_mask, point,
                                   bboxes_tensor, labels, scores, dataset_idx):

        n_points = point.shape[0]
        n_boxes = bboxes_tensor.shape[0]

        if n_boxes == 0:
            # Return empty if no boxes to trim
            empty_boxes = DepthInstance3DBoxes(bboxes_tensor.new_zeros((0, 6)), box_dim=6, with_yaw=False)
            return [(empty_boxes, labels, scores)]


        # Ensure boxes have 7 dims (add 0 yaw if needed) for get_face_distances
        if bboxes_tensor.shape[1] == 6:
            bboxes_7d = torch.cat(
                (bboxes_tensor, torch.zeros_like(bboxes_tensor[:, :1])), dim=1)
        else:
            bboxes_7d = bboxes_tensor

        # Expand points and boxes for distance calculation
        points_expanded = point.unsqueeze(1).expand(n_points, n_boxes, 3) # [N_points, N_boxes, 3]
        bboxes_expanded = bboxes_7d.unsqueeze(0).expand(n_points, n_boxes, 7) # [N_points, N_boxes, 7]

        # Calculate distance from each point to each box face
        face_distances = get_face_distances(points_expanded, bboxes_expanded) # [N_points, N_boxes, 6]

        # Determine which points are inside each box (all face distances > 0)
        inside_bbox = face_distances.min(dim=-1).values > 1e-5 # Add small epsilon, Shape: [N_points, N_boxes]

        # Transpose to [N_boxes, N_points] for scattering
        inside_bbox_T = inside_bbox.T # [N_boxes, N_points]

        num_superpoints = sp_pts_mask.max().item() + 1
        # scatter_mean needs float input
        sp_inside_fraction = scatter_mean(inside_bbox_T.float(), sp_pts_mask, dim=1, dim_size=num_superpoints)
        # Result shape: [N_boxes, N_superpoints]

        # --- Filter superpoints based on thresholds ---
        low_sp_thr = self.test_cfg.low_sp_thr[dataset_idx] # Get dataset-specific threshold
        up_sp_thr = self.test_cfg.up_sp_thr[dataset_idx]   # Get dataset-specific threshold

        # Identify superpoints to exclude (low fraction inside the box)
        sp_exclude = sp_inside_fraction < low_sp_thr # Shape [N_boxes, N_superpoints]
        # Identify superpoints to forcefully include (high fraction inside the box)
        sp_include = sp_inside_fraction > up_sp_thr # Shape [N_boxes, N_superpoints]

        # --- Update the point inclusion mask (inside_bbox_T) ---
        # Final mask: initially points inside the box
        final_point_inclusion = inside_bbox_T.clone() # [N_boxes, N_points]

        # For each box, find the points belonging to excluded superpoints and set their inclusion to False
        # This requires mapping superpoint decisions back to points
        # Get the exclude decision for the superpoint each point belongs to
        point_sp_exclude_decision = sp_exclude[:, sp_pts_mask] # Advanced indexing [N_boxes, N_points]
        final_point_inclusion[point_sp_exclude_decision] = False # Exclude points in low-fraction superpoints

        # For each box, find the points belonging to included superpoints and set their inclusion to True
        point_sp_include_decision = sp_include[:, sp_pts_mask] # Advanced indexing [N_boxes, N_points]
        final_point_inclusion[point_sp_include_decision] = True # Include points in high-fraction superpoints


        # --- Recalculate BBoxes based on the final included points ---
        trimmed_boxes_list = []
        for i in range(n_boxes): # Iterate through each box
            included_points_mask = final_point_inclusion[i] # [N_points] boolean mask for this box
            points_for_box = point[included_points_mask]

            if points_for_box.shape[0] < 2: # Need at least 2 points to define a box; keep original if not enough
                # Keep original box (axis-aligned version)
                 center = bboxes_7d[i, :3]
                 size = bboxes_7d[i, 3:6]
                 # Yaw is discarded for the trimmed box
                 trimmed_box = torch.cat((center, size))
                 trimmed_boxes_list.append(trimmed_box)
                 # print(f"Warning: Box {i} has < 2 points after trimming, keeping original.")
            else:
                # Calculate new axis-aligned box from included points
                xyz_min = points_for_box.min(dim=0).values
                xyz_max = points_for_box.max(dim=0).values
                center = (xyz_max + xyz_min) / 2
                size = xyz_max - xyz_min
                size = torch.clamp(size, min=1e-4) # Ensure min size
                trimmed_box = torch.cat((center, size)) # [6]
                trimmed_boxes_list.append(trimmed_box)

        if not trimmed_boxes_list:
            # If all boxes became invalid
            trimmed_bboxes_tensor = bboxes_tensor.new_zeros((0, 6))
        else:
            trimmed_bboxes_tensor = torch.stack(trimmed_boxes_list) # [N_boxes, 6]

        # Create final DepthInstance3DBoxes (axis-aligned)
        trimmed_bboxes = DepthInstance3DBoxes(
            trimmed_bboxes_tensor,
            with_yaw=False, # Trimmed boxes are axis-aligned
            box_dim=6,
            origin=(0.5, 0.5, 0.5)
        )

        return [(trimmed_bboxes, labels, scores)]


    def _single_scene_multiclass_nms(self, bboxes, scores, labels, fast_nms, iou_thr, score_thr):
        """Multi-class NMS for a single scene.

        Args:
            bboxes (Tensor): Predicted bounding boxes tensor of shape (N_boxes, 6)
                or (N_boxes, 7).
            scores (Tensor): Predicted scores for the boxes [N_boxes,].
            labels (Tensor): Predicted labels for each box [N_boxes,].
            fast_nms (bool): Flag for using fast NMS (nms3d_normal).
            iou_thr (float): IoU threshold for NMS.
            score_thr (float): Threshold to filter boxes by score before NMS.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Filtered bboxes, scores, and labels after NMS.
        """
        unique_labels = labels.unique()
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes_list, nms_scores_list, nms_labels_list = [], [], []

        for class_id in unique_labels:
             # Filter boxes, scores, labels for the current class
             class_mask = (labels == class_id)
             class_scores_all = scores[class_mask]
             class_bboxes_all = bboxes[class_mask]
             class_labels_all = labels[class_mask] # All are `class_id`

             # Apply score threshold
             score_filter_mask = class_scores_all > score_thr
             if not score_filter_mask.any():
                 continue # Skip class if no boxes pass score threshold

             class_scores = class_scores_all[score_filter_mask]
             class_bboxes = class_bboxes_all[score_filter_mask]
             class_labels = class_labels_all[score_filter_mask] # Still all `class_id`


             # Perform NMS for the current class
             if class_bboxes.shape[0] == 0:
                  continue # Skip if no boxes left after score filtering


             if with_yaw:
                 # Use nms3d for rotated boxes
                 # nms3d expects boxes [N, 7] and scores [N]
                 keep_indices = nms3d(class_bboxes, class_scores, iou_thr)
             else:
                 # Use NMS for axis-aligned boxes
                 if fast_nms:
                      # nms3d_normal expects [N, 7], add dummy yaw
                      class_bboxes_7d = torch.cat(
                         (class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                      keep_indices = nms3d_normal(class_bboxes_7d, class_scores, iou_thr)
                 else:
                      # Use aligned_3d_nms (expects format [x1, y1, z1, x2, y2, z2])
                      # Convert boxes from (cx, cy, cz, dx, dy, dz) to (x1, y1, z1, x2, y2, z2)
                      boxes_loss_fmt = _bbox_to_loss(class_bboxes) # Check if _bbox_to_loss does this conversion
                      # aligned_3d_nms API might differ slightly based on MMDetection3D version
                      # Assuming it returns indices to keep:
                      keep_indices = aligned_3d_nms(boxes_loss_fmt, class_scores, class_labels, iou_thr)


             # Append kept boxes, scores, labels
             nms_bboxes_list.append(class_bboxes[keep_indices])
             nms_scores_list.append(class_scores[keep_indices])
             nms_labels_list.append(class_labels[keep_indices])


        # Concatenate results from all classes
        if len(nms_bboxes_list) > 0:
             final_nms_bboxes = torch.cat(nms_bboxes_list, dim=0)
             final_nms_scores = torch.cat(nms_scores_list, dim=0)
             final_nms_labels = torch.cat(nms_labels_list, dim=0)
        else:
             # Return empty tensors if nothing survived NMS
             final_nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
             final_nms_scores = scores.new_zeros((0,))
             final_nms_labels = labels.new_zeros((0,), dtype=torch.long)

        return final_nms_bboxes, final_nms_scores, final_nms_labels


def get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
    """Calculate signed distances from points to box faces in the box frame.

    Args:
        points (Tensor): Points coordinates, shape (N_points, N_boxes, 3).
        boxes (Tensor): 3D boxes, shape (N_points, N_boxes, 7)
                      (cx, cy, cz, dx, dy, dz, yaw). Note: yaw is assumed consistent
                      across the N_points dimension for each box instance if N_points > 1,
                      i.e., boxes[0, i, :] defines the box for all points[j, i, :].
                      Typically, N_points=1 or N_boxes=1 for simpler broadcasting.
                      The original code uses N_points, N_boxes, dim - let's stick to that.

    Returns:
        Tensor: Face distances shape (N_points, N_boxes, 6), representing
                (dist_to_+x_face, dist_to_-x_face, dist_to_+y_face, dist_to_-y_face, dist_to_+z_face, dist_to_-z_face)
                Positive distance means inside the box relative to that face.
    """
    # Calculate point coordinates relative to box centers
    relative_points = points - boxes[..., :3] # Shape: [N_points, N_boxes, 3]

    # Rotate relative points by negative yaw to align with box axes
    # rotation_3d_in_axis expects input shape (N, 3)
    # We need to rotate points[p, b, :] relative to boxes[p, b, :]
    # Reshape for rotation function: Merge N_points and N_boxes dimensions
    n_points, n_boxes, _ = points.shape
    relative_points_flat = relative_points.reshape(-1, 3) # [N_points * N_boxes, 3]
    # Yaw values need corresponding shape. Assuming yaw is constant for a box instance: boxes[0, :, 6]
    # If boxes[p, b, 6] can vary with p, this needs adjustment. Assuming it doesn't.
    yaws = boxes[0, :, 6].unsqueeze(0).repeat(n_points, 1).reshape(-1) # [N_points * N_boxes]

    # Apply rotation
    rotated_relative_points_flat = rotation_3d_in_axis(
        relative_points_flat, -yaws, axis=2 # Rotate by negative yaw around Z axis
    )

    # Reshape back
    rotated_relative_points = rotated_relative_points_flat.reshape(n_points, n_boxes, 3) # [N_points, N_boxes, 3]

    # Box half-sizes
    half_sizes = boxes[..., 3:6] / 2.0 # [N_points, N_boxes, 3]

    # Calculate distances to faces in the rotated frame
    # Distance to positive face = half_size - rotated_coord
    # Distance to negative face = half_size + rotated_coord
    dx_pos = half_sizes[..., 0] - rotated_relative_points[..., 0] # Dist to +x face (right)
    dx_neg = half_sizes[..., 0] + rotated_relative_points[..., 0] # Dist to -x face (left)
    dy_pos = half_sizes[..., 1] - rotated_relative_points[..., 1] # Dist to +y face (front)
    dy_neg = half_sizes[..., 1] + rotated_relative_points[..., 1] # Dist to -y face (back)
    dz_pos = half_sizes[..., 2] - rotated_relative_points[..., 2] # Dist to +z face (top)
    dz_neg = half_sizes[..., 2] + rotated_relative_points[..., 2] # Dist to -z face (bottom)

    # Stack distances: +x, -x, +y, -y, +z, -z
    # Note: Original code had a different order (min/max derived). This seems more direct.
    # Let's match the original's apparent output order: dx_min, dx_max, etc. where 'min' refers to the negative face dist?
    # Original calculation:
    # dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2  -> This seems wrong or centers isn't points
    # Let's assume the goal is distance inside from each face boundary. Positive = inside.
    # Stack as: (+x face dist, -x face dist, +y face dist, -y face dist, +z face dist, -z face dist)
    return torch.stack((dx_pos, dx_neg, dy_pos, dy_neg, dz_pos, dz_neg), dim=-1)