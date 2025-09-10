import torch

def load_partial_state_dict(model, checkpoint_path):
    """
    仅加载 checkpoint 中与当前模型匹配的部分权重（例如只加载 LiDAR backbone 及公共模块的权重），
    忽略新增模块（例如图像分支、交叉注意力融合模块或分类头不匹配部分）的参数。

    Args:
        model (nn.Module): 当前模型实例。
        checkpoint_path (str): checkpoint 文件路径。
    """
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 如果 checkpoint 存在 "state_dict" 键，则使用其中的 state_dict，否则直接使用 checkpoint
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 取得当前模型的 state_dict
    model_state = model.state_dict()
    new_state_dict = {}

    # 遍历 checkpoint 中的每个参数
    for key, weight in state_dict.items():
        if key in model_state:
            if model_state[key].size() == weight.size():
                new_state_dict[key] = weight
            else:
                print(f"Skipping key {key} due to size mismatch: model {model_state[key].size()} vs checkpoint {weight.size()}")
        else:
            # 也可以对部分不匹配的键进行过滤；例如，如果你要加载的是 LiDAR 分支权重，
            # 可增加判断：只加载那些以 'unet.' 开头的参数
            if key.startswith('unet.'):
                # 如果模型中使用的前缀与 checkpoint 不一致，可以进行调整
                new_key = key  # 此处如果需要修改前缀，做相应修改
                if new_key in model_state and model_state[new_key].size() == weight.size():
                    new_state_dict[new_key] = weight
                else:
                    print(f"Ignoring key {key} (with adjusted key {new_key}) due to mismatch")
            else:
                print(f"Ignoring unexpected key from checkpoint: {key}")
    # 加载筛选后的权重（strict=False 允许忽略缺失或额外参数）
    load_result = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded partial state dict:")
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)
