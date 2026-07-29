import torch

def fix_checkpoint(src_path, dst_path):
    print(f"正在加载旧权重: {src_path}")
    checkpoint = torch.load(src_path, map_location='cpu')
    
    # 获取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    
    print("开始转换权重键名...")
    for k, v in state_dict.items():
        new_key = k
        
        # 1. 映射 ModuleA (a -> module_a)
        if k.startswith('a.'):
            new_key = k.replace('a.', 'module_a.', 1)
            
        # 2. 映射 NAC 模块 (cross_modal_fusion -> nac_module)
        elif k.startswith('cross_modal_fusion.'):
            # 先替换前缀
            new_key = k.replace('cross_modal_fusion.', 'nac_module.', 1)
            
            # 3. 映射 NAC 内部层 (img_proj -> context_proj)
            # 这一步非常关键！不仅前缀要改，里面的层名你也改了
            if 'img_proj' in new_key:
                new_key = new_key.replace('img_proj', 'context_proj')
        
        # 4. 移除不再需要的 image_backbone 权重 (可选，保持清爽)
        if 'image_backbone' in k:
            continue
            
        new_state_dict[new_key] = v
        
    # 覆盖原 state_dict
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict
        
    print(f"保存修复后的权重至: {dst_path}")
    torch.save(checkpoint, dst_path)
    print("转换完成！")

if __name__ == '__main__':
    # 请修改这里的路径为你实际的路径
    src = 'work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1024.pth'
    dst = 'epoch_1024_renamed.pth'
    
    fix_checkpoint(src, dst)