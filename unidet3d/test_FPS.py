import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if sys.path[0] == script_dir:
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import time
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmdet3d.utils import setup_multi_processes
from mmengine.runner import Runner
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 导入注册表初始化所需的方法
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

def parse_args():
    parser = argparse.ArgumentParser(description='UniDet3D-Meta Benchmark')
    parser.add_argument('config', help='配置文件路径，例如 configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py')
    parser.add_argument('--checkpoint', help='模型权重路径', default=None)
    parser.add_argument('--samples', type=int, default=100, help='用于测试 Latency 的样本数量 (默认: 100)')
    parser.add_argument('--warmup', type=int, default=20, help='GPU 预热次数 (默认: 20)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载配置
    cfg = Config.fromfile(args.config)
    try:
        setup_multi_processes(cfg)
    except (AttributeError, KeyError):
        pass
        
    # ====================================================
    # 初始化注册表作用域 (Scope)
    # ====================================================
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    
    # 2. 构建数据加载器
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)
    dataset_iterator = iter(val_dataloader)
    
    # 3. 构建模型
    model = MODELS.build(cfg.model)
    if args.checkpoint is not None:
        from mmengine.runner.checkpoint import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().eval()
    
    # 构建预处理器
    data_preprocessor = MODELS.build(cfg.model.data_preprocessor).cuda()

    print("\n" + "="*50)
    print("1. 参数量分析 (Parameter Analysis)")
    print("="*50)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量 (Params): {total_params / 1e6:.2f} M")
    print("(注: 因稀疏卷积的动态哈希特性，标准工具无法准确追踪 FLOPs，请以 Latency 为准)")

    print("\n" + "="*50)
    print("2. 推理延迟测试 (Latency Benchmark)")
    print("="*50)
    print(f"正在进行预热 (Warmup {args.warmup} iterations)...")
    
    with torch.no_grad():
        for _ in range(args.warmup):
            # 确保持续获取数据以防止迭代器耗尽报错
            try:
                batch_data = next(dataset_iterator)
            except StopIteration:
                dataset_iterator = iter(val_dataloader)
                batch_data = next(dataset_iterator)
                
            # 【修复点】正确提取预处理后的数据
            processed_data = data_preprocessor(batch_data, training=False)
            batch_inputs = processed_data['inputs']
            batch_data_samples = processed_data['data_samples']
            
            _ = model.predict(batch_inputs, batch_data_samples)

    print(f"开始测速 (Benchmarking {args.samples} samples)...")
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for i in range(args.samples):
            try:
                batch_data = next(dataset_iterator)
            except StopIteration:
                dataset_iterator = iter(val_dataloader)
                batch_data = next(dataset_iterator)
                
            # 【修复点】正确提取预处理后的数据
            processed_data = data_preprocessor(batch_data, training=False)
            batch_inputs = processed_data['inputs']
            batch_data_samples = processed_data['data_samples']
            
            starter.record()
            _ = model.predict(batch_inputs, batch_data_samples)
            ender.record()
            
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
                
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    fps = 1000.0 / mean_syn
    
    print(f"端到端平均延迟 (Average Latency): {mean_syn:.2f} ms (+/- {std_syn:.2f} ms)")
    print(f"吞吐量 (FPS): {fps:.2f} 帧/秒")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()