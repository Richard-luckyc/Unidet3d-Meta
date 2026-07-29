import torch
from mmengine import Config
from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint

# 🔑 强制 import，确保自定义的 Det3DDataPreprocessor_ 被注册
import unidet3d.data_preprocessor
import unidet3d
import inspect
from mmengine.registry import MODELS as MMENGINE_MODELS
from unidet3d.data_preprocessor import Det3DDataPreprocessor_

# 手动把类塞进 mmengine 的 MODELS
MMENGINE_MODELS.register_module(name='Det3DDataPreprocessor_', module=Det3DDataPreprocessor_)
print("⚡ 已手动注册 Det3DDataPreprocessor_ 到 mmengine.MODELS")

print("Det3DDataPreprocessor_ 定义位置：", inspect.getfile(unidet3d.data_preprocessor.Det3DDataPreprocessor_))
print("Det3DDataPreprocessor_ 源码：")
print(inspect.getsource(unidet3d.data_preprocessor.Det3DDataPreprocessor_))
print("注册表里 Det3DDataPreprocessor_:", 'Det3DDataPreprocessor_' in MODELS.module_dict)
print("注册表里 Det3DDataPreprocessor:", 'Det3DDataPreprocessor' in MODELS.module_dict)


def check_dataset_rgb(cfg):
    """检查哪些数据集包含 RGB 信息 (use_color=True)"""
    rgb_datasets = []
    all_datasets = [cfg.train_dataloader['dataset'],
                    cfg.val_dataloader['dataset'],
                    cfg.test_dataloader['dataset']]
    for dataset_cfg in all_datasets:
        datasets = dataset_cfg['datasets'] if 'datasets' in dataset_cfg else [dataset_cfg]
        for d in datasets:
            for step in d['pipeline']:
                if step['type'] == 'LoadPointsFromFile' and step.get('use_color', False):
                    rgb_datasets.append(d['type'])
    return list(set(rgb_datasets))


def main():
    # 1. 加载 config
    cfg = Config.fromfile('configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py')

    # 🔎 打印 data_preprocessor 配置
    print("cfg.model.data_preprocessor:", cfg.model.get('data_preprocessor', None))

    # 🔎 打印所有已注册的模块，方便确认
    print("当前 MODELS 注册表包含的模块:", list(MODELS.module_dict.keys())[:50], "...")

    # 2. 检查并修正 data_preprocessor
    if 'data_preprocessor' in cfg.model:
        dp_type = cfg.model['data_preprocessor']['type']
        if dp_type not in MODELS:
            print(f"⚠️  data_preprocessor 类型 {dp_type} 不在注册表里，尝试替换为 Det3DDataPreprocessor_")
            cfg.model['data_preprocessor']['type'] = 'Det3DDataPreprocessor_'

    # 3. 检查数据集 RGB 信息
    rgb_datasets = check_dataset_rgb(cfg)
    print("具有RGB信息的数据集:", rgb_datasets)

    # 4. 构建模型
    model = MODELS.build(cfg.model)

    # 如果有预训练权重，就加载
    if hasattr(cfg, 'load_from') and cfg.load_from is not None:
        load_checkpoint(model, cfg.load_from, map_location='cpu')
        print(f"已加载预训练权重: {cfg.load_from}")

    print("✅ 模型构建成功，可以继续验证图像分支效果。")


if __name__ == '__main__':
    main()

