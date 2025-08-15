import torch

# 读取 .npy 文件
obj = torch.load("data/amass/CMU-g1_retargeted/01/01_06_stageii.npy")

# 打印形状和部分数据
print(type(obj))
if torch.is_tensor(obj):
    print(obj.shape)
# 如果是字典或其他结构
elif isinstance(obj, dict):
    for k, v in obj.items():
        if torch.is_tensor(v):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v))