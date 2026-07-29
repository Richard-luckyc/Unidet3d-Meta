import torch
from mmdet3d.registry import MODELS, DATASETS # 导入 DATASETS 注册表
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.structures import Det3DDataSample
from mmengine.registry import MODELS as MMENGINE_MODELS
from unidet3d.data_preprocessor import Det3DDataPreprocessor_
import unidet3d.unidet3d  # ⚡ 确保触发模型类注册
import inspect
print("注册表里 UniDet3D_Meta:", 'UniDet3D_Meta' in MODELS.module_dict)

MMENGINE_MODELS.register_module(name='Det3DDataPreprocessor_', module=Det3DDataPreprocessor_)
print("⚡ 已手动注册 Det3DDataPreprocessor_ 到 mmengine.MODELS")

print("Det3DDataPreprocessor_ 定义位置：", inspect.getfile(unidet3d.data_preprocessor.Det3DDataPreprocessor_))
print("Det3DDataPreprocessor_ 源码：")
print(inspect.getsource(unidet3d.data_preprocessor.Det3DDataPreprocessor_))
print("注册表里 Det3DDataPreprocessor_:", 'Det3DDataPreprocessor_' in MODELS.module_dict)
print("注册表里 Det3DDataPreprocessor:", 'Det3DDataPreprocessor' in MODELS.module_dict)

try:
    from unidet3d.unidet3d import UniDet3D_Meta
    from unidet3d.data_preprocessor import Det3DDataPreprocessor_
    from unidet3d.concat_dataset import ConcatDataset_
    print("✅ Successfully imported core custom classes.")
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import a core class. Error: {e}")
    print("   Please check if the class names and file paths are correct.")
    # 如果核心类都找不到，后续无法进行
    sys.exit(1)
def main():
    # 1️⃣ 读取配置
    cfg = Config.fromfile("configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py")
    print("cfg.model.data_preprocessor:", cfg.model.get('data_preprocessor', None))

    # 2️⃣ 构造 dummy 输入
    # 点云: batch=2, 每个点云 100 个点，每点 3维 (x,y,z)
    dummy_points = [torch.rand((100, 3)) for _ in range(2)]
    # 图像: batch=2, RGB 3通道, 224x224
    dummy_imgs = torch.rand((2, 3, 224, 224))

    # 构造假的 data_sample（必须是 list，不能是 None）
    fake_samples = [Det3DDataSample() for _ in range(2)]

    batch_inputs_with_img = {
        'inputs': {
            'points': dummy_points,
            'imgs': dummy_imgs
        },
        'data_samples': fake_samples
    }
    batch_inputs_without_img = {
        'inputs': {
            'points': dummy_points
        },
        'data_samples': fake_samples
    }

    # 3️⃣ 构造模型
    model = MODELS.build(cfg.model)

    checkpoint = "autodl-tmp/work_dirs/tmp/epoch_880.pth"
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"已加载预训练权重: {checkpoint}")
    print("⚠️ 加载权重报告:", msg)

    # 4️⃣ 分别测试有/无图像分支
    print("\n=== 测试无图像分支 ===")
    out_no_img = model.test_step([batch_inputs_without_img])
    print("无图像分支输出 keys:", out_no_img[0].keys())

    print("\n=== 测试有图像分支 ===")
    out_with_img = model.test_step([batch_inputs_with_img])
    print("有图像分支输出 keys:", out_with_img[0].keys())

    # 5️⃣ 简单对比差异
    print("\n=== 差异对比 ===")
    for key in out_with_img[0].keys():
        if key in out_no_img[0]:
            try:
                diff = (out_with_img[0][key] - out_no_img[0][key]).abs().mean()
                print(f"{key}: 有/无图像分支差异均值 = {diff.item():.6f}")
            except Exception:
                print(f"{key}: 无法直接对比 (可能是结构化数据)")
        else:
            print(f"{key}: 仅存在于有图像分支输出中")

if __name__ == "__main__":
    main()
