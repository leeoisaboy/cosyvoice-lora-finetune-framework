#!/usr/bin/env python3
"""
微调后的 Flow 模型推理脚本

使用方法:
    python inference.py                           # 使用默认文本和参考音频
    python inference.py --text "你好世界"          # 指定合成文本
    python inference.py --prompt audio.wav        # 指定参考音频
    python inference.py --weight flow_merged.pt   # 指定权重文件

关键修复:
    1. 参考音频截取前 5 秒（避免注意力分散）
    2. mel 归一化/反归一化（匹配训练时的处理）
    3. GPU 缓存管理（避免 OOM 和 CUDA 错误）
    4. 裁剪输出开头的 prompt 残留
"""

import os
import sys
import gc
import argparse
from functools import wraps
from pathlib import Path

import torch
import torchaudio

# 添加路径 - 使用项目内的 cosyvoice 和 matcha
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    PRETRAINED_MODEL_DIR, RAW_AUDIO_DIR, OUTPUT_DIR,
    MEL_MEAN, MEL_STD, INFERENCE_CONFIG
)


# ============================================================
# GPU 缓存管理
# ============================================================

def clear_gpu_cache():
    """清空 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def reset_cuda():
    """强力重置 CUDA"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


# ============================================================
# Mel 归一化
# ============================================================

def normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """将原始 mel 归一化（与训练时一致）"""
    return (mel - MEL_MEAN) / MEL_STD


def denormalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """将归一化的 mel 还原"""
    return mel * MEL_STD + MEL_MEAN


def patch_flow_for_finetuned(flow_model, verbose=True):
    """
    Patch flow 模型的 inference 方法，添加归一化/反归一化处理

    训练时：target = (mel_raw - MEL_MEAN) / MEL_STD
    推理时：
      1. prompt_feat 需要归一化（与训练时的 conds 一致）
      2. 模型输出需要反归一化（还原给 HIFIGAN）
    """
    original_inference = flow_model.inference

    @wraps(original_inference)
    @torch.inference_mode()
    def patched_inference(token, token_len, prompt_token, prompt_token_len,
                          prompt_feat, prompt_feat_len, embedding, flow_cache=None):
        if verbose:
            print(f"    [Patch] prompt_feat 原始: mean={prompt_feat.mean():.2f}, "
                  f"range=[{prompt_feat.min():.2f}, {prompt_feat.max():.2f}]")

        # 对 prompt_feat 进行归一化
        prompt_feat_normalized = normalize_mel(prompt_feat)

        if verbose:
            print(f"    [Patch] prompt_feat 归一化后: mean={prompt_feat_normalized.mean():.2f}, "
                  f"range=[{prompt_feat_normalized.min():.2f}, {prompt_feat_normalized.max():.2f}]")

        # 调用原始 inference
        result, cache = original_inference(
            token, token_len, prompt_token, prompt_token_len,
            prompt_feat_normalized, prompt_feat_len, embedding, flow_cache
        )

        if verbose:
            print(f"    [Patch] flow 输出: mean={result.mean():.2f}, "
                  f"range=[{result.min():.2f}, {result.max():.2f}]")

        # 对输出进行反归一化
        result_denormalized = denormalize_mel(result)

        if verbose:
            print(f"    [Patch] 反归一化后: mean={result_denormalized.mean():.2f}, "
                  f"range=[{result_denormalized.min():.2f}, {result_denormalized.max():.2f}]")

        return result_denormalized, cache

    flow_model.inference = patched_inference
    print("  ✓ Flow 模型已 patch：推理时自动进行 mel 归一化/反归一化")


# ============================================================
# 主推理函数
# ============================================================

def inference(
    text: str,
    prompt_wav_path: str,
    prompt_text: str = None,
    weight_path: str = None,
    output_path: str = None,
):
    """
    执行 TTS 推理

    Args:
        text: 要合成的文本
        prompt_wav_path: 参考音频路径
        prompt_text: 参考音频对应的文本（可选，会从文件名解析）
        weight_path: 微调权重路径（默认使用 flow_merged.pt）
        output_path: 输出音频路径
    """
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav

    print("=" * 60)
    print("微调模型推理")
    print("=" * 60)

    # 默认权重路径
    if weight_path is None:
        weight_path = os.path.join(OUTPUT_DIR, 'flow_merged.pt')

    # 检查权重是否存在
    if not os.path.exists(weight_path):
        print(f"错误：微调权重不存在: {weight_path}")
        print(f"请先运行训练脚本或使用 merge_weights.py 合并权重")
        return None

    # 检查参考音频
    if not os.path.exists(prompt_wav_path):
        print(f"错误：参考音频不存在: {prompt_wav_path}")
        return None

    # 清空 GPU 缓存
    reset_cuda()
    print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 加载模型
    print("\n1. 加载 CosyVoice...")
    reset_cuda()
    cosyvoice = CosyVoice(PRETRAINED_MODEL_DIR, load_jit=False, load_trt=False)
    print(f"  采样率: {cosyvoice.sample_rate}")

    # 加载微调权重
    print(f"\n2. 加载微调权重: {os.path.basename(weight_path)}")
    clear_gpu_cache()
    finetuned_state_dict = torch.load(weight_path, map_location='cpu')
    cosyvoice.model.flow.load_state_dict(finetuned_state_dict, strict=True)
    del finetuned_state_dict
    clear_gpu_cache()
    print("  ✓ 微调权重加载成功")

    # 应用归一化 patch
    print("\n3. 应用归一化 patch...")
    patch_flow_for_finetuned(cosyvoice.model.flow)

    # 加载并截取参考音频
    print(f"\n4. 参考音频: {os.path.basename(prompt_wav_path)}")
    prompt_speech_16k = load_wav(prompt_wav_path, 16000)
    original_len = prompt_speech_16k.shape[1] / 16000
    print(f"  原始长度: {original_len:.2f}s")

    max_samples = int(INFERENCE_CONFIG['max_prompt_seconds'] * 16000)
    if prompt_speech_16k.shape[1] > max_samples:
        prompt_speech_16k = prompt_speech_16k[:, :max_samples]
        print(f"  截取后长度: {prompt_speech_16k.shape[1] / 16000:.2f}s ✓")

    # 准备参考文本
    if prompt_text is None:
        filename = Path(prompt_wav_path).stem
        parts = filename.split('_')
        for part in parts:
            if len(part) > 3 and not part.isdigit():
                prompt_text = part
                break
        if prompt_text is None:
            prompt_text = "参考音频"
    if len(prompt_text) > 20:
        prompt_text = prompt_text[:20]

    print(f"\n5. 文本:")
    print(f"  参考文本: {prompt_text}")
    print(f"  合成文本: {text}")

    # 推理
    print("\n6. 开始推理...")
    reset_cuda()
    torch.cuda.synchronize()

    output_audio = None

    try:
        with torch.inference_mode():
            for i, result in enumerate(cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=False
            )):
                audio = result['tts_speech']
                print(f"\n7. 输出结果:")
                print(f"  原始形状: {audio.shape}")
                print(f"  原始长度: {audio.shape[-1] / cosyvoice.sample_rate:.2f}s")

                # 裁剪开头的 prompt 残留
                prompt_output_samples = int(prompt_speech_16k.shape[1] * (cosyvoice.sample_rate / 16000))
                trim_samples = int(prompt_output_samples * INFERENCE_CONFIG['trim_ratio'])

                if trim_samples > 0:
                    print(f"  裁剪开头: {trim_samples} 样本 ({trim_samples / cosyvoice.sample_rate:.2f}s)")
                    if audio.dim() == 1:
                        audio = audio[trim_samples:]
                    else:
                        audio = audio[:, trim_samples:]
                    print(f"  裁剪后长度: {audio.shape[-1] / cosyvoice.sample_rate:.2f}s")

                print(f"  范围: [{audio.min():.4f}, {audio.max():.4f}]")

                # 保存
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)

                output_audio = audio.cpu()

                if output_path:
                    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                    torchaudio.save(output_path, output_audio, cosyvoice.sample_rate)
                    print(f"\n✓ 已保存到: {output_path}")

    finally:
        clear_gpu_cache()

    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)

    return output_audio


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='CosyVoice 微调模型推理')
    parser.add_argument('--text', '-t', type=str,
                        default="床前明月光，疑是地上霜。举头望明月，低头思故乡。",
                        help='要合成的文本')
    parser.add_argument('--prompt', '-p', type=str, default=None,
                        help='参考音频路径')
    parser.add_argument('--prompt-text', type=str, default=None,
                        help='参考音频对应的文本')
    parser.add_argument('--weight', '-w', type=str, default=None,
                        help='微调权重路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出音频路径')

    args = parser.parse_args()

    # 查找参考音频
    prompt_wav_path = args.prompt
    if prompt_wav_path is None:
        wav_files = list(Path(RAW_AUDIO_DIR).glob('*.wav'))
        if wav_files:
            prompt_wav_path = str(wav_files[0])
            print(f"使用默认参考音频: {prompt_wav_path}")
        else:
            print(f"错误：未找到参考音频，请使用 --prompt 指定")
            return

    # 默认输出路径
    output_path = args.output
    if output_path is None:
        os.makedirs(os.path.join(OUTPUT_DIR, 'inference'), exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, 'inference', 'output.wav')

    # 执行推理
    inference(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=args.prompt_text,
        weight_path=args.weight,
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
