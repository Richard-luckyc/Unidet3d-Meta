# unidet3d/verify.py
"""
验证脚本（增强版）
- 自动扫描 cfg 中的数据集 pipeline，注册缺失/不兼容的 transforms（尝试兼容 mmdet3d 中的原始类）
- 若 dataset 构建失败则回退到 dummy batch（dummy batch 尽量包含 lidar_path、metainfo、points、images）
- 分别执行 predict(with/without image) 和 loss(with/without image) 做对比，判断图像分支是否在起作用
"""

import os
import copy
import inspect
import traceback
import torch
from mmengine.config import Config
from mmengine.registry import MODELS as MMENGINE_MODELS, TRANSFORMS as MMENGINE_TRANSFORMS
from mmdet3d.registry import MODELS, DATASETS, TRANSFORMS as MMDET3D_TRANSFORMS
from mmengine.runner import load_checkpoint
from mmdet3d.structures import Det3DDataSample

# 自定义 data_preprocessor（你的实现）
import unidet3d.data_preprocessor
from unidet3d.data_preprocessor import Det3DDataPreprocessor_

# 注册自定义 preprocessor 到 mmengine MODELS（防止 cfg 中使用该类型时报错）
if 'Det3DDataPreprocessor_' not in MMENGINE_MODELS.module_dict:
    MMENGINE_MODELS.register_module(name='Det3DDataPreprocessor_', module=Det3DDataPreprocessor_)
    print("⚡ Registered Det3DDataPreprocessor_ into mmengine MODELS")

# 配置、权重路径（按需修改）
CONFIG_PATH = 'configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
CHECKPOINT_PATH = 'autodl-tmp/work_dirs/tmp/epoch_880.pth'


# ------------------------------
# 工具：为缺失 transform 注册兼容 wrapper / dummy
# ------------------------------
def register_transform_if_missing(name, cls):
    """注册到 mmengine 和 mmdet3d TRANSFORMS（若尚未注册）"""
    try:
        if name not in MMENGINE_TRANSFORMS.module_dict:
            MMENGINE_TRANSFORMS.register_module(name=name, module=cls)
            print(f"⚡ Registered into mmengine TRANSFORMS: {name}")
    except Exception:
        pass
    try:
        if name not in MMDET3D_TRANSFORMS.module_dict:
            MMDET3D_TRANSFORMS.register_module(name=name, module=cls)
    except Exception:
        pass


def make_compat_wrapper(orig_cls, alias_name=None, key_map=None):
    """
    为 orig_cls 生成一个兼容的 wrapper 类，__init__ 会：
      - 接收任意 kwargs
      - 根据 orig_cls.__init__ 的参数签名筛选允许的参数
      - 对传入的 kwargs 做简单映射（key_map 字典），再传给 orig_cls.__init__
    返回一个新的类对象（继承 orig_cls）
    """
    key_map = key_map or {}

    sig = None
    try:
        sig = inspect.signature(orig_cls.__init__)
        allowed = set(sig.parameters.keys()) - {'self', 'args', 'kwargs'}
    except Exception:
        allowed = None

    class Compat(orig_cls):
        def __init__(self, *args, **kwargs):
            # 做拷贝不改变外部 kwargs
            k = dict(kwargs)
            # 先做 known key mapping
            for old_key, new_key in key_map.items():
                if old_key in kwargs and new_key not in k:
                    k[new_key] = kwargs[old_key]
            if allowed is not None:
                filtered = {kk: vv for kk, vv in k.items() if kk in allowed}
            else:
                # 若无法拿到签名，尝试直接传入所有 kwargs（原类可能接受 **kwargs）
                filtered = k
            # 调用父类的 __init__
            try:
                super(Compat, self).__init__(*args, **filtered)
            except TypeError:
                # 若仍然报类型错误（参数名仍不匹配），尽量尝试仅传 args
                super(Compat, self).__init__(*args)
    # 给类一个有意义的名字
    if alias_name:
        Compat.__name__ = alias_name
    else:
        Compat.__name__ = orig_cls.__name__ + "_Compat"
    return Compat


def ensure_pipeline_transforms_registered(cfg):
    """
    扫描 cfg 中 train/val/test dataloader 的 dataset pipeline，
    对缺失的 transform 尝试做以下处理：
      1) 在 mmdet3d 的常见模块中寻找原始类，若找到则注册兼容 wrapper（会过滤未知 kwargs）
      2) 若找不到任何候选，实现一个 NoOp transform 注册（直接返回输入）
    """
    candidate_modules = [
        'mmdet3d.datasets.transforms.loading',
        'mmdet3d.datasets.transforms.formating',
        'mmdet3d.datasets.transforms.formatting',
        'mmdet3d.datasets.transforms.transforms_3d',
        'mmdet3d.datasets.transforms.color',
        'mmdet3d.datasets.transforms',
    ]

    # collect dataset dicts from train/val/test dataloaders if present
    dataset_cfgs = []
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        entry = cfg.get(key, None)
        if not entry:
            continue
        ds = entry.get('dataset', None)
        if ds is None:
            continue
        if isinstance(ds, dict) and 'datasets' in ds:
            dataset_cfgs.extend(ds['datasets'])
        elif isinstance(ds, list):
            dataset_cfgs.extend(ds)
        else:
            dataset_cfgs.append(ds)

    missing = set()
    for d_cfg in dataset_cfgs:
        pipeline = d_cfg.get('pipeline', [])
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            tname = step.get('type', None)
            if not tname:
                continue
            if tname not in MMENGINE_TRANSFORMS.module_dict:
                missing.add(tname)

    if not missing:
        print("✅ pipeline 中的 transforms 都已在 mmengine TRANSFORMS 中注册（或暂未检测到缺失）")
        return

    print("⚠️ pipeline 中缺少以下 transforms，将尝试自动注册/兼容：", missing)

    # try to locate original class for each missing transform
    for t in sorted(missing):
        found = False
        for mod_path in candidate_modules:
            try:
                m = __import__(mod_path, fromlist=['*'])
            except Exception:
                continue
            # try exact name
            if hasattr(m, t):
                orig_cls = getattr(m, t)
                # wrap with compat wrapper (no key mapping)
                wrapper = make_compat_wrapper(orig_cls, alias_name=t, key_map={'with_sp_mask_3d': 'with_sp_mask'})
                register_transform_if_missing(t, wrapper)
                print(f"⚡ Found and registered {t} from {mod_path}")
                found = True
                break
            # try without trailing underscore (e.g., config used LoadAnnotations3D_ but module has LoadAnnotations3D)
            if t.endswith('_'):
                tn = t[:-1]
                if hasattr(m, tn):
                    orig_cls = getattr(m, tn)
                    wrapper = make_compat_wrapper(orig_cls, alias_name=t, key_map={'with_sp_mask_3d': 'with_sp_mask'})
                    register_transform_if_missing(t, wrapper)
                    print(f"⚡ Found {tn} in {mod_path} and registered compat as {t}")
                    found = True
                    break
            # try adding trailing underscore
            tn2 = t + '_'
            if hasattr(m, tn2):
                orig_cls = getattr(m, tn2)
                wrapper = make_compat_wrapper(orig_cls, alias_name=t)
                register_transform_if_missing(t, wrapper)
                print(f"⚡ Found {tn2} in {mod_path} and registered as {t}")
                found = True
                break
        if not found:
            # register a NoOp placeholder
            class _NoOpTransform:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, results):
                    return results
            register_transform_if_missing(t, _NoOpTransform)
            print(f"⚡ Registered NoOp placeholder for missing transform: {t}")

# ------------------------------
# 其它工具函数
# ------------------------------
def safe_load_checkpoint(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    state_dict = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    # remove common mismatched keys (分类头等)
    rem_suffixes = ['decoder.outs_cls.2.weight', 'decoder.outs_cls.2.bias']
    for k in list(state_dict.keys()):
        if any(k.endswith(s) for s in rem_suffixes):
            state_dict.pop(k, None)
    return model.load_state_dict(state_dict, strict=False)


def dict_to_scalar(d):
    out = {}
    if not isinstance(d, dict):
        return {"type": str(type(d))}
    for k, v in d.items():
        try:
            if isinstance(v, torch.Tensor):
                out[k] = float(v.detach().cpu().item())
            elif isinstance(v, (float, int)):
                out[k] = float(v)
            else:
                out[k] = str(type(v))
        except Exception:
            out[k] = str(type(v))
    return out


def make_dummy_batch(device='cpu', batch_size=1, num_points=512):
    """构造 dummy batch，尽量包含 (x,y,z,r,g,b) 格式的点、images 和带 lidar_path 的 DataSample"""
    pts_list = []
    for _ in range(batch_size):
        pts = torch.rand((num_points, 6), device=device)  # (x,y,z,r,g,b)
        pts_list.append(pts)
    imgs = torch.rand((batch_size, 3, 224, 224), device=device)
    samples = []
    for i in range(batch_size):
        ds = Det3DDataSample()
        # 直接设置属性 lidar_path（避免放到 metainfo 导致字段冲突）
        try:
            ds.lidar_path = f'dummy_scene_{i}.bin'
        except Exception:
            # fallback: some DataSample 可能限制属性设置 -> 忽略
            pass
        # 安全地设置可用的 metainfo 字段（不要覆盖已有数据字段）
        try:
            if hasattr(ds, 'set_metainfo'):
                ds.set_metainfo({'batch_input_shape': (224, 224), 'pad_shape': (224, 224)})
        except Exception:
            # 忽略 set_metainfo 失败
            pass
        samples.append(ds)
    inputs = {'points': pts_list, 'images': imgs, 'imgs': imgs, 'imgs_norm': imgs}
    return inputs, samples


def try_adjust_topk_and_retry(model, fn, *args, **kwargs):
    """若预测中出现 topk 超范围的 RuntimeError，尝试临时减小 topk 重试一次"""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if 'selected index k out of range' in msg or 'out of range' in msg or 'k out of range' in msg:
            try:
                orig_k = getattr(model.test_cfg, 'topk_insts', None)
                if orig_k is None:
                    orig_k = getattr(model.test_cfg, 'topk', None)
                if orig_k is None:
                    raise e
                new_k = min(256, max(1, int(orig_k // 10) if isinstance(orig_k, int) else 256))
                print(f"⚠️ topk 超范围，临时将 model.test_cfg.topk_insts 从 {orig_k} 调整为 {new_k} 重试")
                # 尝试设置多个可能的字段名
                if hasattr(model.test_cfg, 'topk_insts'):
                    model.test_cfg.topk_insts = new_k
                elif hasattr(model.test_cfg, 'topk'):
                    model.test_cfg.topk = new_k
                return fn(*args, **kwargs)
            except Exception:
                raise e
        else:
            raise e

# ------------------------------
# 主流程
# ------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    if not os.path.exists(CONFIG_PATH):
        print(f"❌ 配置文件不存在: {CONFIG_PATH}")
        return
    cfg = Config.fromfile(CONFIG_PATH)
    print("Loaded cfg")

    # 先尝试自动注册/兼容 pipeline 中的 transforms（避免 DATASETS.build 直接因找不到 transform 而 Crash）
    try:
        ensure_pipeline_transforms_registered(cfg)
    except Exception:
        print("⚠️ ensure_pipeline_transforms_registered 失败，继续（会在构建 dataset 时回退到 dummy）")
        traceback.print_exc()

    # 确保 model.data_preprocessor 指向已注册的类型
    try:
        if 'data_preprocessor' in cfg.model:
            dp = cfg.model['data_preprocessor']['type']
            if dp not in MODELS.module_dict:
                print(f"⚠️ cfg.model.data_preprocessor {dp} 未注册，替换为 Det3DDataPreprocessor_")
                cfg.model['data_preprocessor']['type'] = 'Det3DDataPreprocessor_'
    except Exception:
        pass

    # 构建模型
    print("Building model...")
    try:
        model = MODELS.build(cfg.model)
    except Exception:
        print("❌ 构建模型失败，traceback:")
        traceback.print_exc()
        return

    # 加载 checkpoint（若存在）
    if os.path.exists(CHECKPOINT_PATH):
        try:
            missing_unexp = safe_load_checkpoint(model, CHECKPOINT_PATH)
            print(f"Loads checkpoint by local backend from path: {CHECKPOINT_PATH}")
            print("Loaded checkpoint (strict=False). missing_keys:", getattr(missing_unexp, 'missing_keys', None))
        except Exception:
            print("⚠️ 加载 checkpoint 失败，继续（可能仅做结构验证）")
            traceback.print_exc()
    else:
        print("⚠️ checkpoint 未找到，继续（使用随机/未初始化权重）")

    model.to(device)
    model.eval()
    print("✅ 模型构建成功，开始验证图像分支...")

    # 尝试用真实 dataset 的第一个样本；若失败回退到 dummy
    inputs = None
    data_samples = None
    dataset_cfg = cfg.get('train_dataloader', {}).get('dataset', None)
    used_real_sample = False
    if dataset_cfg:
        try:
            dataset = DATASETS.build(dataset_cfg)
            print(f"✅ Dataset built: {type(dataset)} length={len(dataset)}")
            sample = dataset[0]
            if isinstance(sample, dict) and 'inputs' in sample and 'data_samples' in sample:
                inputs = sample['inputs']
                ds = sample['data_samples']
                if isinstance(ds, (list, tuple)):
                    data_samples = [d.to(device) if hasattr(d, 'to') else d for d in ds]
                else:
                    try:
                        data_samples = [ds.to(device)]
                    except Exception:
                        data_samples = [ds]
                # ensure each data_sample has lidar_path attribute (some model code relies on它)
                for s in data_samples:
                    try:
                        if not hasattr(s, 'lidar_path'):
                            s.lidar_path = getattr(sample, 'lidar_path', 'dataset_sample.bin')
                    except Exception:
                        pass
                used_real_sample = True
                print("✅ 使用真实 dataset[0] 作为验证样本")
            else:
                print("⚠️ dataset[0] 格式不符合预期，回退到 dummy")
        except Exception:
            print("⚠️ Failed to build/read dataset — fallback to dummy sample.")
            traceback.print_exc()

    if not used_real_sample:
        print("⚠️ 使用 dummy 输入代替真实数据（注意：dummy 不一定能触发真实 fusion）")
        inputs, data_samples = make_dummy_batch(device=device, batch_size=1, num_points=1024)
        # move data_samples to device if possible
        try:
            data_samples = [s.to(device) if hasattr(s, 'to') else s for s in data_samples]
        except Exception:
            pass

    # 尝试 predict 无图像
    print("\n--- 1) Predict WITHOUT images ---")
    inputs_no_img = {k: v for k, v in inputs.items() if k not in ('images', 'imgs', 'img', 'imgs_norm')}
    # 把点和图像 Tensor 移到 device
    if isinstance(inputs_no_img.get('points'), list):
        inputs_no_img['points'] = [p.to(device) if isinstance(p, torch.Tensor) else p for p in inputs_no_img['points']]
    elif isinstance(inputs_no_img.get('points'), torch.Tensor):
        inputs_no_img['points'] = inputs_no_img['points'].to(device)

    # ensure data_samples on device
    try:
        data_samples = [ds.to(device) if hasattr(ds, 'to') else ds for ds in data_samples]
    except Exception:
        pass

    pred_no = None
    try:
        results_no = try_adjust_topk_and_retry(model, model.forward, inputs=inputs_no_img, data_samples=data_samples, mode='predict')
        if results_no and hasattr(results_no[0], 'pred_instances_3d'):
            pred_no = results_no[0].pred_instances_3d
            print("predict(no_img): bbox count =", len(pred_no.bboxes_3d))
        else:
            print("predict(no_img): no pred_instances_3d / empty")
    except Exception:
        print("❌ predict(no_img) 出错：")
        traceback.print_exc()
        pred_no = None

    # loss with image (训练路径)
    print("\n--- 2) Loss WITH images (training path) ---")
    model.train()
    loss_with = None
    try:
        inputs_with = copy.deepcopy(inputs)
        # move image tensors to device
        for key in ('images', 'imgs', 'imgs_norm'):
            if key in inputs_with and isinstance(inputs_with[key], torch.Tensor):
                inputs_with[key] = inputs_with[key].to(device)
        loss_with = model.forward(inputs=inputs_with, data_samples=data_samples, mode='loss')
        print("loss(with_img):", dict_to_scalar(loss_with))
    except Exception:
        print("❌ loss(with_img) 出错：")
        traceback.print_exc()
        loss_with = None
    model.eval()

    # predict with image (可选)
    print("\n--- 3) Predict WITH images (optional) ---")
    pred_with = None
    try:
        inputs_with_pred = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        results_with = try_adjust_topk_and_retry(model, model.forward, inputs=inputs_with_pred, data_samples=data_samples, mode='predict')
        if results_with and hasattr(results_with[0], 'pred_instances_3d'):
            pred_with = results_with[0].pred_instances_3d
            print("predict(with_img): bbox count =", len(pred_with.bboxes_3d))
        else:
            print("predict(with_img): no pred_instances_3d / empty")
    except Exception:
        print("⚠️ predict(with images) 出错：")
        traceback.print_exc()
        pred_with = None

    # final decision
    print("\n--- 4) Final decision ---")
    success = False
    try:
        if pred_no is not None and pred_with is not None and len(getattr(pred_no, 'bboxes_3d', [])) > 0 and len(getattr(pred_with, 'bboxes_3d', [])) > 0:
            c_no = pred_no.bboxes_3d.tensor[0, :3]
            c_with = pred_with.bboxes_3d.tensor[0, :3]
            diff = (c_with - c_no).abs().mean().item()
            print(f"bbox center avg abs diff: {diff:.6f}")
            if diff > 1e-6:
                print("✅ Image branch affects predictions (predict path).")
                success = True
            else:
                print("❌ predict path 有预测但几乎无差异（可能融合未生效或样本不敏感）")
        else:
            print("predict path 无法直接比较（有一侧为空或没有 pred），将使用 loss fallback")
    except Exception:
        print("⚠️ 比较 predict 输出时出错：")
        traceback.print_exc()

    if not success and loss_with is not None:
        print("\n--- 使用 loss 做 fallback 比较 ---")
        model.train()
        try:
            loss_no = None
            try:
                loss_no = model.forward(inputs=inputs_no_img, data_samples=data_samples, mode='loss')
            except Exception:
                print("⚠️ 计算 loss(no_img) 失败（有时需要 gt），异常如下：")
                traceback.print_exc()
            model.eval()
            if loss_no is not None:
                lw = dict_to_scalar(loss_with)
                ln = dict_to_scalar(loss_no)
                print("loss WITH image:", lw)
                print("loss WITHOUT image:", ln)
                diffs = {k: abs(lw.get(k, 0) - ln.get(k, 0)) for k in lw.keys() if isinstance(lw.get(k, None), float)}
                if any(v > 1e-6 for v in diffs.values()):
                    print("✅ loss 在有/无图像间存在差异，说明图像分支在训练路径参与计算。")
                    success = True
                else:
                    print("❌ loss 在有/无图像间差异很小（或无），图像分支可能未生效或样本不敏感。")
            else:
                print("⚠️ 无法得到 loss(no_img) 进行比较")
        except Exception:
            print("❌ loss fallback 比较时发生异常：")
            traceback.print_exc()

    if success:
        print("\n==== 最终结论: ✅ 图像分支已在模型的某些路径/样本上被激活 ====")
    else:
        print("\n==== 最终结论: ❌ 未能确认图像分支对预测/loss 有显著影响 ====")
        print("建议：")
        print(" - 使用真实 batch（不是单样本或 dummy）进行多样本平均比较")
        print(" - 在模型融合处插入中间特征打印或 hook 以直接检查 image-feature 是否被融合")
        print(" - 若 dataset 构建失败，检查 pipeline 中自定义 transforms 的参数签名（脚本做了兼容尝试，但并非覆盖所有情况）")

if __name__ == '__main__':
    main()

