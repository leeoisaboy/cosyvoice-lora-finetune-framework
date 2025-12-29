# Copyright (c) 2024
# LoRA (Low-Rank Adaptation) module for Flow model fine-tuning
#
# LoRA 优势：
# 1. 大幅减少训练参数量（原模型的 0.1%-1%）
# 2. 防止过拟合，适合小数据集
# 3. 显存占用小
# 4. 可以保存小体积的适配器权重

import math
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA adapted Linear layer

    原理：
    原始: y = Wx
    LoRA: y = Wx + (BA)x

    其中 B: (out_features, r), A: (r, in_features)
    r << min(in_features, out_features)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 冻结原始权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 初始化
        # A 使用 kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 使用小随机值初始化，而不是 0
        # 这样 lora_A 从第一步就能获得梯度
        nn.init.normal_(self.lora_B, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        result = self.original_layer(x)

        # LoRA 增量: (BA)x * scaling
        lora_output = self.lora_dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # x @ A^T
        lora_output = F.linear(lora_output, self.lora_B)  # (x @ A^T) @ B^T

        return result + lora_output * self.scaling


class LoRAConv1d(nn.Module):
    """LoRA adapted Conv1d layer

    对于 1x1 卷积，等价于 Linear
    """

    def __init__(
        self,
        original_layer: nn.Conv1d,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size[0] if isinstance(original_layer.kernel_size, tuple) else original_layer.kernel_size

        # 冻结原始权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA: 使用 1x1 卷积实现低秩分解
        # A: (r, in_channels, 1)
        # B: (out_channels, r, 1)
        self.lora_A = nn.Conv1d(in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(r, out_channels, kernel_size=1, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B 使用小随机值初始化，让 A 从第一步就有梯度
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        result = self.original_layer(x)

        # LoRA 增量
        lora_output = self.lora_dropout(x)
        lora_output = self.lora_A(lora_output)
        lora_output = self.lora_B(lora_output)

        return result + lora_output * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, int]:
    """将 LoRA 应用到模型的指定层

    Args:
        model: 原始模型
        r: LoRA 秩（越小参数越少，但表达能力越弱）
        lora_alpha: LoRA 缩放因子
        lora_dropout: LoRA dropout 概率
        target_modules: 目标模块名称列表，None 则使用默认配置

    Returns:
        统计信息字典
    """

    # 默认目标模块（针对 Flow 模型的注意力层）
    if target_modules is None:
        target_modules = [
            # Attention 层的 QKV 投影
            'to_q', 'to_k', 'to_v',
            'linear_q', 'linear_k', 'linear_v',
            # Attention 输出投影
            'linear_out',
            # FFN 层
            'w_1', 'w_2',
            # 其他重要投影
            'linear_pos',
        ]

    target_modules_set = set(target_modules)

    replaced_count = 0
    total_lora_params = 0
    original_params = sum(p.numel() for p in model.parameters())

    # 遍历所有模块
    def replace_with_lora(parent_module: nn.Module, prefix: str = ''):
        nonlocal replaced_count, total_lora_params

        for name, child in list(parent_module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # 检查是否是目标模块
            should_replace = any(target in name for target in target_modules_set)

            if should_replace:
                if isinstance(child, nn.Linear):
                    # 替换 Linear 层
                    lora_layer = LoRALinear(
                        child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                    )
                    setattr(parent_module, name, lora_layer)
                    replaced_count += 1
                    total_lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                    # print(f"  [LoRA] {full_name}: Linear({child.in_features}, {child.out_features})")

                elif isinstance(child, nn.Conv1d) and child.kernel_size[0] == 1:
                    # 替换 1x1 Conv1d 层
                    lora_layer = LoRAConv1d(
                        child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                    )
                    setattr(parent_module, name, lora_layer)
                    replaced_count += 1
                    total_lora_params += (
                        lora_layer.lora_A.weight.numel() +
                        lora_layer.lora_B.weight.numel()
                    )
                    # print(f"  [LoRA] {full_name}: Conv1d({child.in_channels}, {child.out_channels})")

            # 递归处理子模块
            replace_with_lora(child, full_name)

    replace_with_lora(model)

    # 冻结非 LoRA 参数
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    # 统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'replaced_layers': replaced_count,
        'original_params': original_params,
        'lora_params': total_lora_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / original_params * 100,
    }


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """提取模型中的 LoRA 参数"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data.clone()
    return lora_state_dict


def save_lora_weights(model: nn.Module, path: str):
    """保存 LoRA 权重（仅保存适配器，文件很小）"""
    lora_state_dict = get_lora_state_dict(model)
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA weights: {len(lora_state_dict)} tensors to {path}")


def load_lora_weights(model: nn.Module, path: str):
    """加载 LoRA 权重"""
    lora_state_dict = torch.load(path, map_location='cpu')

    # 加载到模型
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)

    print(f"Loaded LoRA weights: {len(lora_state_dict)} tensors from {path}")


def merge_lora_weights(model: nn.Module):
    """将 LoRA 权重合并到原始权重中（用于推理加速）

    合并后可以直接使用原始模型结构，无需 LoRA 层
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 合并: W_new = W + BA * scaling
            with torch.no_grad():
                delta_w = module.lora_B @ module.lora_A * module.scaling
                module.original_layer.weight.add_(delta_w)

        elif isinstance(module, LoRAConv1d):
            # 合并 1x1 卷积
            with torch.no_grad():
                # A: (r, in_channels, 1), B: (out_channels, r, 1)
                # delta_w: (out_channels, in_channels, 1)
                a_weight = module.lora_A.weight  # (r, in_channels, 1)
                b_weight = module.lora_B.weight  # (out_channels, r, 1)
                delta_w = torch.einsum('ori,ric->oic', b_weight, a_weight) * module.scaling
                module.original_layer.weight.add_(delta_w)

    print("LoRA weights merged into original model")


def get_merged_state_dict(model: nn.Module) -> dict:
    """
    获取合并 LoRA 后的原始格式 state_dict

    这个函数会：
    1. 先调用 merge_lora_weights 合并 LoRA 权重
    2. 然后提取原始格式的 state_dict（不包含 lora_A, lora_B, original_layer 等）

    返回的 state_dict 可以直接被 CosyVoice 原始模型加载
    """
    # 先合并 LoRA 权重
    merge_lora_weights(model)

    # 构建原始格式的 state_dict
    original_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            # 对于 LoRA 层，只保存合并后的原始权重
            # 将 xxx.original_layer.weight -> xxx.weight
            # 将 xxx.original_layer.bias -> xxx.bias
            prefix = name
            original_state_dict[f"{prefix}.weight"] = module.original_layer.weight.data.clone()
            if module.original_layer.bias is not None:
                original_state_dict[f"{prefix}.bias"] = module.original_layer.bias.data.clone()

    # 添加非 LoRA 层的权重
    for name, param in model.named_parameters():
        # 跳过 LoRA 相关的参数
        if 'lora_A' in name or 'lora_B' in name or 'original_layer' in name:
            continue
        original_state_dict[name] = param.data.clone()

    # 添加 buffer（如 running_mean, running_var 等）
    for name, buf in model.named_buffers():
        if 'lora_' not in name and 'original_layer' not in name:
            original_state_dict[name] = buf.clone()

    print(f"Exported merged state_dict with {len(original_state_dict)} keys")
    return original_state_dict
