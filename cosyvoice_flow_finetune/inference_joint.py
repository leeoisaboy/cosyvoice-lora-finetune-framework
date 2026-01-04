#!/usr/bin/env python3
"""
LLM + Flow 联合微调后的无 Prompt 推理脚本

特点：
1. 完全不需要参考音频
2. LLM 和 Flow 都已学习到说话人的音色和吟诵风格
3. 直接输入文本即可生成语音

用法:
    # 基本用法（使用联合训练的权重）
    python inference_joint.py --text "你好世界"

    # 指定 LLM 和 Flow 权重
    python inference_joint.py --llm llm_merged.pt --flow flow_merged.pt --text "你好"

    # 调整语速
    python inference_joint.py --text "你好" --speed 0.9

前提条件：
    需要先运行 train_joint.py 进行联合训练
"""

import os
import sys
import gc
import time
import argparse
from pathlib import Path
from functools import wraps

import torch
import torchaudio

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
COSYVOICE_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, COSYVOICE_ROOT)

from config import PRETRAINED_MODEL_DIR, OUTPUT_DIR, MEL_MEAN, MEL_STD


def clear_gpu_cache():
    """清空 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """将原始 mel 归一化"""
    return (mel - MEL_MEAN) / MEL_STD


def denormalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """将归一化的 mel 还原"""
    return mel * MEL_STD + MEL_MEAN


def inference_no_prompt_joint(
    text: str,
    llm_weight_path: str,
    flow_weight_path: str,
    output_path: str = None,
    speed: float = 1.0,
    n_timesteps: int = 10,
):
    """
    无 Prompt 推理 - 使用联合训练后的权重

    这是真正的无 prompt 推理：
    - LLM 不需要 prompt speech tokens，直接从文本生成 speech tokens
    - Flow 不需要 prompt mel，直接从 speaker embedding 生成 mel

    Args:
        text: 要合成的文本
        llm_weight_path: 合并后的 LLM 权重路径
        flow_weight_path: 合并后的 Flow 权重路径
        output_path: 输出音频路径
        speed: 语速调整
        n_timesteps: Flow 的 ODE 步数
    """
    from cosyvoice.cli.cosyvoice import CosyVoice
    import torch.nn.functional as F

    print("=" * 60)
    print("CosyVoice 无 Prompt 推理（联合训练版）")
    print("=" * 60)

    # 检查权重文件
    if not os.path.exists(llm_weight_path):
        print(f"错误：LLM 权重不存在: {llm_weight_path}")
        return None
    if not os.path.exists(flow_weight_path):
        print(f"错误：Flow 权重不存在: {flow_weight_path}")
        return None

    print(f"\nLLM 权重: {os.path.basename(llm_weight_path)}")
    print(f"Flow 权重: {os.path.basename(flow_weight_path)}")

    # 清理显存
    clear_gpu_cache()
    time.sleep(1)

    # 加载 CosyVoice
    print("\n加载 CosyVoice...")
    cosyvoice = CosyVoice(PRETRAINED_MODEL_DIR, load_jit=False, load_trt=False)
    print(f"  采样率: {cosyvoice.sample_rate}")

    # 加载微调后的 LLM 权重
    print("\n加载 LLM 权重...")
    llm_state_dict = torch.load(llm_weight_path, map_location='cpu')
    cosyvoice.model.llm.load_state_dict(llm_state_dict, strict=True)
    del llm_state_dict
    clear_gpu_cache()
    print("  LLM 权重加载成功")

    # 加载微调后的 Flow 权重
    print("\n加载 Flow 权重...")
    flow_state_dict = torch.load(flow_weight_path, map_location='cpu')
    cosyvoice.model.flow.load_state_dict(flow_state_dict, strict=True)
    del flow_state_dict
    clear_gpu_cache()
    print("  Flow 权重加载成功")

    # Patch Flow 的 inference 方法，添加归一化处理
    original_flow_inference = cosyvoice.model.flow.inference

    @wraps(original_flow_inference)
    @torch.inference_mode()
    def patched_flow_inference(token, token_len, prompt_token, prompt_token_len,
                                prompt_feat, prompt_feat_len, embedding, flow_cache=None):
        # 对 prompt_feat 进行归一化（虽然是空的，但保持接口一致）
        if prompt_feat.shape[1] > 0:
            prompt_feat = normalize_mel(prompt_feat)

        result, cache = original_flow_inference(
            token, token_len, prompt_token, prompt_token_len,
            prompt_feat, prompt_feat_len, embedding, flow_cache
        )

        # 反归一化
        result = denormalize_mel(result)

        return result, cache

    cosyvoice.model.flow.inference = patched_flow_inference
    print("  Flow 已 patch")

    # 准备推理
    print(f"\n文本: {text}")

    # 使用 frontend 处理文本
    device = cosyvoice.model.device

    # 无 prompt 推理的关键：
    # 1. prompt_text 为空
    # 2. prompt_speech_token 为空
    # 3. prompt_speech_feat 为空
    # 4. 使用训练数据的平均 speaker embedding（或者可以从训练数据中提取）

    # 文本编码 - 直接使用 cosyvoice.frontend
    frontend = cosyvoice.frontend

    # 处理文本
    text_normalized = frontend.text_normalize(text, split=False)
    text_tokens = frontend.tokenizer.encode(text_normalized, allowed_special=frontend.allowed_special)
    text_tokens = torch.tensor([text_tokens], dtype=torch.int32)
    text_len = torch.tensor([text_tokens.shape[1]], dtype=torch.int32)

    # 空的 prompt
    prompt_text = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech_feat = torch.zeros(1, 0, 80)

    # Speaker embedding - 使用零向量（联合训练时应该已经学到了固定的说话人表示）
    # 如果效果不好，可以考虑从训练数据中提取平均 embedding
    embedding = torch.zeros(1, 192)

    print("\n开始推理...")
    clear_gpu_cache()

    output_audio = None

    try:
        with torch.inference_mode():
            # 使用 tts 方法（它会自动处理 LLM 和 Flow 的调用）
            for result in cosyvoice.model.tts(
                text=text_tokens,
                flow_embedding=embedding,
                llm_embedding=embedding,
                prompt_text=prompt_text,
                llm_prompt_speech_token=prompt_speech_token,
                flow_prompt_speech_token=prompt_speech_token,
                prompt_speech_feat=prompt_speech_feat,
                stream=False,
                speed=speed,
            ):
                audio = result['tts_speech']
                print(f"\n输出结果:")
                print(f"  形状: {audio.shape}")
                print(f"  长度: {audio.shape[-1] / cosyvoice.sample_rate:.2f}s")

                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)

                output_audio = audio

                # 保存
                if output_path is None:
                    os.makedirs(os.path.join(OUTPUT_DIR, 'inference'), exist_ok=True)
                    output_path = os.path.join(OUTPUT_DIR, 'inference', 'joint_output.wav')

                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                torchaudio.save(output_path, output_audio, cosyvoice.sample_rate)
                print(f"\n已保存到: {output_path}")

    except Exception as e:
        print(f"\n推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        clear_gpu_cache()

    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)

    return output_audio


def main():
    parser = argparse.ArgumentParser(
        description='CosyVoice 无 Prompt 推理（联合训练版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认权重
  python inference_joint.py --text "你好世界"

  # 指定权重文件
  python inference_joint.py --llm llm_merged.pt --flow flow_merged.pt --text "你好"

  # 调整语速
  python inference_joint.py --text "床前明月光" --speed 0.9

前提条件:
  需要先运行 train_joint.py 进行联合训练，生成 llm_merged_joint.pt 和 flow_merged_joint.pt
        """
    )

    parser.add_argument('--text', '-t', type=str, required=True,
                        help='要合成的文本')
    parser.add_argument('--llm', type=str, default=None,
                        help='LLM 权重路径（默认: output/llm_merged_joint.pt）')
    parser.add_argument('--flow', type=str, default=None,
                        help='Flow 权重路径（默认: output/flow_merged_joint.pt）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出音频路径')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='语速调整（默认 1.0）')

    args = parser.parse_args()

    # 默认权重路径
    llm_weight = args.llm or os.path.join(OUTPUT_DIR, 'llm_merged_joint.pt')
    flow_weight = args.flow or os.path.join(OUTPUT_DIR, 'flow_merged_joint.pt')

    inference_no_prompt_joint(
        text=args.text,
        llm_weight_path=llm_weight,
        flow_weight_path=flow_weight,
        output_path=args.output,
        speed=args.speed,
    )


if __name__ == '__main__':
    main()
