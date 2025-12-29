#!/usr/bin/env python3
"""
LoRA 权重合并脚本

将 LoRA 权重合并到预训练模型，生成可直接加载的完整权重文件。

使用方法:
    # 从最新的 checkpoint 合并
    python merge_weights.py

    # 从指定的 checkpoint 合并
    python merge_weights.py --ckpt output/flow_lora/flow_best_xxx.ckpt

    # 指定输出路径
    python merge_weights.py --output my_merged_weights.pt

输出:
    合并后的权重可以直接被 CosyVoice 加载，无需 LoRA 结构。
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PRETRAINED_MODEL_DIR, OUTPUT_DIR, LORA_CONFIG


def find_latest_checkpoint(output_dir):
    """找到最新的 checkpoint 文件"""
    ckpts = [f for f in os.listdir(output_dir) if f.endswith('.ckpt')]
    if not ckpts:
        return None
    # 按修改时间排序
    ckpts_with_time = [(f, os.path.getmtime(os.path.join(output_dir, f))) for f in ckpts]
    ckpts_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(output_dir, ckpts_with_time[0][0])


def merge_from_checkpoint(ckpt_path: str, output_path: str):
    """从 Lightning checkpoint 导出合并后的权重"""
    from flow_model import build_flow_model
    from lora import apply_lora_to_model, get_merged_state_dict

    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 构建模型
    print("构建 Flow 模型...")
    flow_model = build_flow_model(pretrained_path=PRETRAINED_MODEL_DIR, device='cpu')

    # 应用 LoRA 结构
    print("应用 LoRA 结构...")
    apply_lora_to_model(
        flow_model,
        r=LORA_CONFIG['lora_r'],
        lora_alpha=LORA_CONFIG['lora_alpha'],
        lora_dropout=LORA_CONFIG['lora_dropout'],
        target_modules=LORA_CONFIG['target_modules'],
    )

    # 加载权重
    print("加载训练后的权重...")
    model_state = flow_model.state_dict()
    loaded = 0
    for key, value in state_dict.items():
        # 处理 Lightning 的 'model.' 前缀
        clean_key = key.replace('model.', '') if key.startswith('model.') else key
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            model_state[clean_key] = value
            loaded += 1

    flow_model.load_state_dict(model_state)
    print(f"加载了 {loaded} 个参数")

    # 导出合并后的权重
    print("合并 LoRA 权重到原始模型...")
    merged_state_dict = get_merged_state_dict(flow_model)

    torch.save(merged_state_dict, output_path)
    print(f"\n✓ 合并后的权重已保存: {output_path}")

    # 验证格式
    print("\n验证权重格式...")
    check_keys = [
        'encoder.encoders.0.self_attn.linear_q.weight',
        'decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight',
    ]
    for key in check_keys:
        if key in merged_state_dict:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ 缺失: {key}")

    return merged_state_dict


def merge_from_lora_weights(lora_path: str, output_path: str):
    """从单独的 LoRA 权重文件合并"""
    from flow_model import build_flow_model
    from lora import apply_lora_to_model, load_lora_weights, get_merged_state_dict

    print(f"加载 LoRA 权重: {lora_path}")

    # 构建模型
    print("构建 Flow 模型...")
    flow_model = build_flow_model(pretrained_path=PRETRAINED_MODEL_DIR, device='cpu')

    # 应用 LoRA 结构
    print("应用 LoRA 结构...")
    apply_lora_to_model(
        flow_model,
        r=LORA_CONFIG['lora_r'],
        lora_alpha=LORA_CONFIG['lora_alpha'],
        lora_dropout=LORA_CONFIG['lora_dropout'],
        target_modules=LORA_CONFIG['target_modules'],
    )

    # 加载 LoRA 权重
    load_lora_weights(flow_model, lora_path)

    # 导出合并后的权重
    print("合并 LoRA 权重到原始模型...")
    merged_state_dict = get_merged_state_dict(flow_model)

    torch.save(merged_state_dict, output_path)
    print(f"\n✓ 合并后的权重已保存: {output_path}")

    return merged_state_dict


def main():
    parser = argparse.ArgumentParser(description='合并 LoRA 权重')
    parser.add_argument('--ckpt', type=str, help='Lightning checkpoint 路径')
    parser.add_argument('--lora', type=str, help='LoRA 权重文件路径 (flow_lora.pt)')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出路径')

    args = parser.parse_args()

    print("=" * 60)
    print("LoRA 权重合并工具")
    print("=" * 60)

    output_path = args.output or os.path.join(OUTPUT_DIR, 'flow_merged.pt')

    if args.lora:
        # 从 LoRA 权重文件合并
        if not os.path.exists(args.lora):
            print(f"错误: LoRA 权重文件不存在: {args.lora}")
            return
        merge_from_lora_weights(args.lora, output_path)

    elif args.ckpt:
        # 从指定 checkpoint 合并
        if not os.path.exists(args.ckpt):
            print(f"错误: checkpoint 不存在: {args.ckpt}")
            return
        merge_from_checkpoint(args.ckpt, output_path)

    else:
        # 自动查找最新 checkpoint
        latest_ckpt = find_latest_checkpoint(OUTPUT_DIR)
        if latest_ckpt:
            print(f"使用最新 checkpoint: {latest_ckpt}")
            merge_from_checkpoint(latest_ckpt, output_path)
        else:
            # 尝试查找 flow_lora.pt
            lora_path = os.path.join(OUTPUT_DIR, 'flow_lora.pt')
            if os.path.exists(lora_path):
                print(f"使用 LoRA 权重: {lora_path}")
                merge_from_lora_weights(lora_path, output_path)
            else:
                print("错误: 未找到 checkpoint 或 LoRA 权重文件")
                print(f"请确保以下文件存在:")
                print(f"  - {OUTPUT_DIR}/*.ckpt")
                print(f"  - {OUTPUT_DIR}/flow_lora.pt")
                return

    print("\n完成!")
    print(f"\n现在可以使用推理脚本:")
    print(f"  python inference.py")


if __name__ == '__main__':
    main()
