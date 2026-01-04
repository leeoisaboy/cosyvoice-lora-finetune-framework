#!/usr/bin/env python3
"""
Joint 训练 LoRA 权重合并脚本

将 LLM 和 Flow 的 LoRA 权重分别合并到预训练模型，生成可直接加载的完整权重文件。

使用方法:
    # 从最新的 joint checkpoint 合并（同时合并 LLM 和 Flow）
    python merge_joint_weights.py

    # 只合并 LLM
    python merge_joint_weights.py --llm-only

    # 只合并 Flow
    python merge_joint_weights.py --flow-only

    # 从指定的 checkpoint 合并
    python merge_joint_weights.py --ckpt output/joint_joint_epoch=26_train_loss=2.8127.ckpt

    # 指定输出路径
    python merge_joint_weights.py --llm-output my_llm.pt --flow-output my_flow.pt

输出:
    - llm_merged.pt: 合并后的 LLM 权重
    - flow_merged.pt: 合并后的 Flow 权重
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PRETRAINED_MODEL_DIR, OUTPUT_DIR, JOINT_TRAINING_CONFIG


def find_latest_joint_checkpoint(output_dir: str, mode: str = None) -> str:
    """找到最新的 joint checkpoint 文件

    Args:
        output_dir: 输出目录
        mode: 训练模式过滤 ('joint', 'llm_only', 'flow_only')，None 则不过滤
    """
    ckpts = [f for f in os.listdir(output_dir) if f.endswith('.ckpt')]

    # 按模式过滤
    if mode:
        ckpts = [f for f in ckpts if f'joint_{mode}' in f or f'{mode}' in f]
    else:
        # 优先选择 joint 模式的 checkpoint
        joint_ckpts = [f for f in ckpts if 'joint_joint' in f]
        if joint_ckpts:
            ckpts = joint_ckpts

    if not ckpts:
        return None

    # 按修改时间排序
    ckpts_with_time = [(f, os.path.getmtime(os.path.join(output_dir, f))) for f in ckpts]
    ckpts_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(output_dir, ckpts_with_time[0][0])


def merge_llm_from_checkpoint(ckpt_path: str, output_path: str):
    """从 checkpoint 合并 LLM LoRA 权重"""
    from llm_flow_model import build_joint_model
    from lora import get_merged_state_dict

    print(f"\n{'='*60}")
    print("合并 LLM LoRA 权重")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")

    # 加载 checkpoint
    print("\n加载 checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 构建模型（只需要 LLM）
    print("构建模型...")
    model = build_joint_model(
        pretrained_path=PRETRAINED_MODEL_DIR,
        device='cpu',
        training_mode='llm_only',
        llm_lora_config=JOINT_TRAINING_CONFIG.get('llm_lora'),
        flow_lora_config=None,  # 不需要 Flow
    )

    # 加载权重到 LLM
    print("加载训练后的 LLM 权重...")
    llm_state = model.llm.state_dict()
    loaded = 0

    for key, value in state_dict.items():
        # 处理 Lightning 的前缀
        clean_key = key
        for prefix in ['model.llm.', 'llm.']:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break

        if clean_key in llm_state and llm_state[clean_key].shape == value.shape:
            llm_state[clean_key] = value
            loaded += 1

    model.llm.load_state_dict(llm_state)
    print(f"加载了 {loaded} 个 LLM 参数")

    # 合并 LoRA 权重
    print("合并 LoRA 权重...")
    merged_state_dict = get_merged_state_dict(model.llm)

    # 保存
    torch.save(merged_state_dict, output_path)
    print(f"\nLLM 权重已保存: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return merged_state_dict


def merge_flow_from_checkpoint(ckpt_path: str, output_path: str):
    """从 checkpoint 合并 Flow LoRA 权重"""
    from llm_flow_model import build_joint_model
    from lora import get_merged_state_dict

    print(f"\n{'='*60}")
    print("合并 Flow LoRA 权重")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")

    # 加载 checkpoint
    print("\n加载 checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 构建模型（只需要 Flow）
    print("构建模型...")
    model = build_joint_model(
        pretrained_path=PRETRAINED_MODEL_DIR,
        device='cpu',
        training_mode='flow_only',
        llm_lora_config=None,  # 不需要 LLM
        flow_lora_config=JOINT_TRAINING_CONFIG.get('flow_lora'),
    )

    # 加载权重到 Flow
    print("加载训练后的 Flow 权重...")
    flow_state = model.flow.state_dict()
    loaded = 0

    for key, value in state_dict.items():
        # 处理 Lightning 的前缀
        clean_key = key
        for prefix in ['model.flow.', 'flow.']:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break

        if clean_key in flow_state and flow_state[clean_key].shape == value.shape:
            flow_state[clean_key] = value
            loaded += 1

    model.flow.load_state_dict(flow_state)
    print(f"加载了 {loaded} 个 Flow 参数")

    # 合并 LoRA 权重
    print("合并 LoRA 权重...")
    merged_state_dict = get_merged_state_dict(model.flow)

    # 保存
    torch.save(merged_state_dict, output_path)
    print(f"\nFlow 权重已保存: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return merged_state_dict


def merge_both_from_checkpoint(ckpt_path: str, llm_output: str, flow_output: str):
    """从 checkpoint 同时合并 LLM 和 Flow 权重"""
    from llm_flow_model import build_joint_model
    from lora import get_merged_state_dict

    print(f"\n{'='*60}")
    print("合并 LLM + Flow LoRA 权重")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")

    # 加载 checkpoint
    print("\n加载 checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 构建完整模型
    print("构建模型...")
    model = build_joint_model(
        pretrained_path=PRETRAINED_MODEL_DIR,
        device='cpu',
        training_mode='joint',
        llm_lora_config=JOINT_TRAINING_CONFIG.get('llm_lora'),
        flow_lora_config=JOINT_TRAINING_CONFIG.get('flow_lora'),
    )

    # 加载所有权重
    print("加载训练后的权重...")

    # LLM 权重
    llm_state = model.llm.state_dict()
    llm_loaded = 0
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ['model.llm.', 'llm.']:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break
        if clean_key in llm_state and llm_state[clean_key].shape == value.shape:
            llm_state[clean_key] = value
            llm_loaded += 1
    model.llm.load_state_dict(llm_state)
    print(f"加载了 {llm_loaded} 个 LLM 参数")

    # Flow 权重
    flow_state = model.flow.state_dict()
    flow_loaded = 0
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ['model.flow.', 'flow.']:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break
        if clean_key in flow_state and flow_state[clean_key].shape == value.shape:
            flow_state[clean_key] = value
            flow_loaded += 1
    model.flow.load_state_dict(flow_state)
    print(f"加载了 {flow_loaded} 个 Flow 参数")

    # 合并并保存 LLM
    print("\n合并 LLM LoRA 权重...")
    llm_merged = get_merged_state_dict(model.llm)
    torch.save(llm_merged, llm_output)
    print(f"LLM 权重已保存: {llm_output}")
    print(f"文件大小: {os.path.getsize(llm_output) / 1024 / 1024:.1f} MB")

    # 需要重新构建 Flow 模型（因为 LLM 已经被 merge 修改了）
    print("\n重新构建 Flow 模型...")
    model_flow = build_joint_model(
        pretrained_path=PRETRAINED_MODEL_DIR,
        device='cpu',
        training_mode='flow_only',
        llm_lora_config=None,
        flow_lora_config=JOINT_TRAINING_CONFIG.get('flow_lora'),
    )

    # 重新加载 Flow 权重
    flow_state = model_flow.flow.state_dict()
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ['model.flow.', 'flow.']:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break
        if clean_key in flow_state and flow_state[clean_key].shape == value.shape:
            flow_state[clean_key] = value
    model_flow.flow.load_state_dict(flow_state)

    # 合并并保存 Flow
    print("合并 Flow LoRA 权重...")
    flow_merged = get_merged_state_dict(model_flow.flow)
    torch.save(flow_merged, flow_output)
    print(f"Flow 权重已保存: {flow_output}")
    print(f"文件大小: {os.path.getsize(flow_output) / 1024 / 1024:.1f} MB")

    return llm_merged, flow_merged


def main():
    parser = argparse.ArgumentParser(description='合并 Joint 训练的 LoRA 权重')
    parser.add_argument('--ckpt', type=str, help='指定 checkpoint 路径')
    parser.add_argument('--llm-only', action='store_true', help='只合并 LLM')
    parser.add_argument('--flow-only', action='store_true', help='只合并 Flow')
    parser.add_argument('--llm-output', type=str, default=None, help='LLM 输出路径')
    parser.add_argument('--flow-output', type=str, default=None, help='Flow 输出路径')

    args = parser.parse_args()

    print("=" * 60)
    print("Joint 训练 LoRA 权重合并工具")
    print("=" * 60)

    # 确定输出路径
    llm_output = args.llm_output or os.path.join(OUTPUT_DIR, 'llm_merged.pt')
    flow_output = args.flow_output or os.path.join(OUTPUT_DIR, 'flow_merged.pt')

    # 确定 checkpoint 路径
    if args.ckpt:
        ckpt_path = args.ckpt
        if not os.path.exists(ckpt_path):
            print(f"错误: checkpoint 不存在: {ckpt_path}")
            return
    else:
        # 自动查找最新的 checkpoint
        if args.llm_only:
            ckpt_path = find_latest_joint_checkpoint(OUTPUT_DIR, 'llm_only')
            if not ckpt_path:
                ckpt_path = find_latest_joint_checkpoint(OUTPUT_DIR, 'joint')
        elif args.flow_only:
            ckpt_path = find_latest_joint_checkpoint(OUTPUT_DIR, 'flow_only')
            if not ckpt_path:
                ckpt_path = find_latest_joint_checkpoint(OUTPUT_DIR, 'joint')
        else:
            ckpt_path = find_latest_joint_checkpoint(OUTPUT_DIR)

        if not ckpt_path:
            print("错误: 未找到 checkpoint 文件")
            print(f"请确保 {OUTPUT_DIR} 目录下有 .ckpt 文件")
            return

        print(f"使用最新 checkpoint: {ckpt_path}")

    # 执行合并
    if args.llm_only:
        merge_llm_from_checkpoint(ckpt_path, llm_output)
    elif args.flow_only:
        merge_flow_from_checkpoint(ckpt_path, flow_output)
    else:
        merge_both_from_checkpoint(ckpt_path, llm_output, flow_output)

    print("\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)

    if not args.flow_only:
        print(f"\nLLM 权重: {llm_output}")
    if not args.llm_only:
        print(f"Flow 权重: {flow_output}")

    print(f"\n下一步：使用 inference_joint.py 进行推理")
    print(f"  python inference_joint.py --text \"你要合成的文本\"")


if __name__ == '__main__':
    main()
