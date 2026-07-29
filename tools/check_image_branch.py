# tools/check_image_branch.py
"""
外部验证脚本（不修改模型文件）
 - 检查 checkpoint 是否包含 image/fusion 参数
 - 注册 hooks 在 image_backbone / cross_modal_fusion（若存在）
 - 使用 model.collate + extract_feat 做有/无图像的对比
 - 输出诊断信息以判断 image branch 是否参与运算
"""
import os
import sys
import copy
import torch
import traceback
import types
from mmengine.config import Config
from mmdet3d.registry import MODELS, DATASETS
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS as MMENGINE_MODELS
from unidet3d.data_preprocessor import Det3DDataPreprocessor_

# 配置（按需修改）
CONFIG_PATH = 'configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
CHECKPOINT_PATH = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MMENGINE_MODELS.register_module(name='Det3DDataPreprocessor_', module=Det3DDataPreprocessor_)
print("⚡ 已手动注册 Det3DDataPreprocessor_ 到 mmengine.MODELS")
def print_header(s):
    print("\n" + ("=" * 10) + " " + s + " " + ("=" * 10))

def safe_load_model_from_cfg(cfg_path, device):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()
    print("Built model from cfg:", cfg_path)
    # try load checkpoint permissively
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint (strict=False):", CHECKPOINT_PATH)
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # just report keys related to image/fusion
        image_keys = [k for k in sd.keys() if ('image_backbone' in k or 'img' in k or 'cross_attn' in k or 'cross_modal' in k or 'fusion' in k)]
        print(f"Found {len(image_keys)} candidate image/fusion keys in checkpoint (showing up to 20):")
        for k in image_keys[:20]:
            print("  ", k)
        # load permissively, print mismatch summary
        res = model.load_state_dict(sd, strict=False)
        print("load_state_dict result:", res)
    else:
        print("Checkpoint not found at", CHECKPOINT_PATH)
        res = None
    return model, cfg, res

class HookCollector:
    def __init__(self):
        self.hooks = []
        self.data = {}

    def add_hook_by_name(self, model, name_substr):
        for n, m in model.named_modules():
            if name_substr in n:
                self._register_single(n, m)
                return True
        return False

    def _register_single(self, name, module):
        def hook(mod, inp, out):
            try:
                if isinstance(out, torch.Tensor):
                    val = float(out.detach().cpu().norm().item())
                elif isinstance(out, (list, tuple)):
                    # try first tensor
                    t = None
                    for o in out:
                        if isinstance(o, torch.Tensor):
                            t = o; break
                    if t is not None:
                        val = float(t.detach().cpu().norm().item())
                    else:
                        val = str(type(out))
                else:
                    val = str(type(out))
            except Exception:
                val = "hook-fail"
            self.data.setdefault(name, []).append(val)
        try:
            h = module.register_forward_hook(hook)
            self.hooks.append(h)
            self.data[name] = []
            print(f"Registered hook on module: {name}")
        except Exception as e:
            print("Failed to register hook on", name, ":", e)

    def remove_all(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

def build_dummy_batch(num_points=1024, num_superpoints=64, device='cpu'):
    # create one sample: points Nx6 (x,y,z, r,g,b)
    points = [torch.rand((num_points, 6), device=device)]
    # images: (B,3,224,224)
    imgs = torch.rand((1, 3, 224, 224), device=device)
    # create simple data_sample-like object (only need lidar_path, gt_pts_seg.sp_pts_mask)
    try:
        from mmdet3d.structures import Det3DDataSample, PointData, InstanceData
        ds = Det3DDataSample()
        pd = PointData()
        mask = torch.randint(0, num_superpoints, (num_points,), device=device)
        pd.sp_pts_mask = mask
        ds.gt_pts_seg = pd
        inst = InstanceData()
        inst.labels_3d = torch.arange(num_superpoints, dtype=torch.long, device=device)
        inst.bboxes_3d = torch.zeros((num_superpoints, 7), device=device)
        ds.gt_instances_3d = inst
        ds.lidar_path = 'dummy_dataset'
    except Exception:
        # fallback simple namespace
        pd = types.SimpleNamespace()
        pd.sp_pts_mask = torch.randint(0, num_superpoints, (num_points,), device=device)
        ds = types.SimpleNamespace()
        ds.gt_pts_seg = pd
        ds.gt_instances_3d = types.SimpleNamespace()
        ds.lidar_path = 'dummy_dataset'
    return {'points': points, 'images': imgs}, [ds]

def main():
    print_header("START CHECK")
    print("device:", device)
    if not os.path.exists(CONFIG_PATH):
        print("Config not found:", CONFIG_PATH)
        return

    model, cfg, load_res = safe_load_model_from_cfg(CONFIG_PATH, device)

    # Check if model has image_backbone and cross_modal_fusion
    has_img = hasattr(model, 'image_backbone')
    has_fusion = hasattr(model, 'cross_modal_fusion') or any('fusion' in n for n, _ in model.named_modules())
    print("Model has image_backbone:", has_img)
    print("Model has cross_modal_fusion / fusion-like modules:", has_fusion)

    # report param counts for candidate modules
    def param_stats(prefix):
        ps = [p for n, p in model.named_parameters() if prefix in n]
        if not ps:
            return None
        total = sum(p.numel() for p in ps)
        nz = sum((p.detach().abs() > 1e-12).sum().item() for p in ps)
        return {'count': len(ps), 'total_params': total, 'nonzero': int(nz)}
    print("image_backbone param stats:", param_stats('image_backbone'))
    print("cross_modal_fusion param stats:", param_stats('cross_modal_fusion'))
    print("cross_attn param stats (decoder):", param_stats('cross_attn') or param_stats('cross_attn_layers'))

    # Hook registration: try to attach to commonly named modules (non-invasive)
    hooks = HookCollector()
    registered = 0
    if has_img:
        # try directly image_backbone module
        try:
            hooks._register_single('image_backbone', model.image_backbone)
            registered += 1
        except Exception:
            pass
    # try to find fusion-like names
    for name, mod in model.named_modules():
        lname = name.lower()
        if 'fusion' in lname or 'cross_attn' in lname or 'cross_modal' in lname:
            hooks._register_single(name, mod)
            registered += 1
            if registered >= 6:
                break
    print("Total hooks registered:", len(hooks.hooks))

    # Try to get a real sample from dataset (best) otherwise dummy
    inputs = None
    data_samples = None
    try:
        dataset_cfg = cfg.get('train_dataloader', {}).get('dataset', None)
        if dataset_cfg:
            ds = DATASETS.build(dataset_cfg)
            sample = ds[0]
            if isinstance(sample, dict) and 'inputs' in sample and 'data_samples' in sample:
                inputs = sample['inputs']
                dsamples = sample['data_samples']
                data_samples = dsamples if isinstance(dsamples, (list, tuple)) else [dsamples]
                print("Using real dataset[0] sample for test.")
    except Exception:
        print("Failed to build dataset or read sample; will use dummy.")

    if inputs is None:
        inputs, data_samples = build_dummy_batch(num_points=1024, num_superpoints=64, device=device)
        print("Using dummy batch for test.")

    # move tensors to device
    inputs = copy.deepcopy(inputs)
    if 'points' in inputs:
        pts = inputs['points']
        inputs['points'] = [p.to(device) if isinstance(p, torch.Tensor) else torch.as_tensor(p, device=device) for p in pts]
    if 'images' in inputs and isinstance(inputs['images'], torch.Tensor):
        inputs['images'] = inputs['images'].to(device)

    # We will call model.collate + extract_feat directly (bypass high-level forward)
    try:
        # collate -> coordinates, features, inverse_mapping, spatial_shape
        coords, feats, inv_map, spatial_shape = model.collate(inputs['points'])
        # build sparse conv tensor
        batch_size = len(inputs['points'])
        x = spconv.SparseConvTensor(feats, coords, spatial_shape, batch_size)
        # build superpoints vector (length == number of points). We will make it consistent with data_samples gt
        # prefer using gt_pts_seg if available:
        sp_mask = None
        try:
            if hasattr(data_samples[0], 'gt_pts_seg') and getattr(data_samples[0].gt_pts_seg, 'sp_pts_mask', None) is not None:
                sp_mask = data_samples[0].gt_pts_seg.sp_pts_mask.to(device)
                print("Using sp_pts_mask from data_samples (len=%d)" % sp_mask.numel())
            else:
                raise Exception("no gt_pts_seg.sp_pts_mask")
        except Exception:
            # fallback: evenly assign points to 64 superpoints
            num_points = inputs['points'][0].shape[0]
            sp_mask = torch.randint(0, 64, (num_points,), device=device)
            print("Using synthetic sp_mask (len=%d, unique=%d)" % (sp_mask.numel(), sp_mask.unique().numel()))
        # inverse mapping inv_map is used in the code as index into x.features
        # create batch_offsets: values [0, num_sp]
        num_sp = int(sp_mask.max().item()) + 1
        batch_offsets = [0, num_sp]
    except Exception as e:
        print("Failed to prepare collate/extract inputs:", e)
        traceback.print_exc()
        hooks.remove_all()
        return

    # Now run extract_feat without images and with images (to exercise fusion)
    print_header("RUN extract_feat WITHOUT image_input")
    try:
        lidar_feats_no = model.extract_feat(x, sp_mask, inv_map, batch_offsets, image_input=None)
        print("extract_feat (no image) returned list of length:", len(lidar_feats_no))
    except Exception:
        print("extract_feat (no image) raised exception:")
        traceback.print_exc()
        lidar_feats_no = None

    print_header("RUN extract_feat WITH image_input")
    try:
        imgs = inputs.get('images', None)
        lidar_feats_with = model.extract_feat(x, sp_mask, inv_map, batch_offsets, image_input=imgs)
        print("extract_feat (with image) returned list of length:", len(lidar_feats_with))
    except Exception:
        print("extract_feat (with image) raised exception:")
        traceback.print_exc()
        lidar_feats_with = None

    # collect hook data
    print_header("HOOK RESULTS")
    for k, v in hooks.data.items():
        print(f"{k}: {v[:5]}... (len={len(v)})")

    # Compare lidar_feats if both available
    print_header("COMPARE LIDAR FEATS")
    if lidar_feats_no is not None and lidar_feats_with is not None:
        try:
            # compare first sample features
            a = lidar_feats_no[0]
            b = lidar_feats_with[0]
            print("no_img feat shape:", getattr(a, 'shape', type(a)))
            print("with_img feat shape:", getattr(b, 'shape', type(b)))
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                # if shapes equal compute norm diff
                if a.shape == b.shape:
                    diff = (a - b).abs().mean().item()
                    print("Mean abs diff between lidar_feats (with/without image):", diff)
                    if diff > 1e-6:
                        print("=> Image branch CHANGES extracted lidar features (fusion active).")
                    else:
                        print("=> Extracted lidar features nearly identical (fusion may be inactive or ineffective).")
                else:
                    print("Feature shapes differ -> fusion changed shape or content.")
            else:
                print("Returned lidar_feats are not tensors; types:", type(a), type(b))
        except Exception:
            print("Error while comparing lidar_feats:")
            traceback.print_exc()
    else:
        print("One or both extract_feat calls failed, cannot compare lidar_feats reliably.")

    # cleanup hooks
    hooks.remove_all()
    print_header("FINISHED")
    print("If the script showed that (a) checkpoint contains image/fusion keys, (b) model has image_backbone & fusion module, and (c) lidar_feats differ with/without image -> then fusion IS being used.")
    print("Caveat: if extract_feat succeeded only for particular superpoint counts (perfect squares), your fusion reshape code may be brittle; use caution.")

if __name__ == "__main__":
    main()
