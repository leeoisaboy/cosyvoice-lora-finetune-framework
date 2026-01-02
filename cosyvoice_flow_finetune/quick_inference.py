#!/usr/bin/env python3
"""
快速推理脚本 - 直接从 checkpoint 文件进行推理

特点：
1. 支持直接加载 .ckpt (Lightning checkpoint) 或 .pt (合并后的权重)
2. 自动完成 LoRA 权重提取 → 合并 → 推理 流程
3. 无需先运行 merge_weights.py

用法:
    # 从 checkpoint 推理
    python quick_inference.py --ckpt output/flow_best_epoch=23_train_loss=0.4901.ckpt --text "你好世界"

    # 从合并后的权重推理
    python quick_inference.py --weight output/flow_merged.pt --text "你好世界"

    # 指定参考音频
    python quick_inference.py --ckpt xxx.ckpt --text "你好" --prompt raw_audio/ref.wav --prompt-text "参考文本"

    # 指定输出路径
    python quick_inference.py --ckpt xxx.ckpt --text "你好" --output my_output.wav
"""

import os
import sys
import gc
import time
import argparse
import subprocess
from pathlib import Path
from functools import wraps

import torch
import torchaudio

# ============================================================
# 在导入任何其他模块之前，强制清空 GPU 显存
# 包括杀掉其他占用 GPU 的 Python 进程
# ============================================================

def _kill_other_python_gpu_processes():
    """杀掉其他占用 GPU 的 Python 进程（仅限当前用户）"""
    if not torch.cuda.is_available():
        return

    current_pid = os.getpid()
    killed = []

    try:
        # 使用 nvidia-smi 获取占用 GPU 的进程
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0].strip())
                        proc_name = parts[1].strip().lower()

                        # 只杀 python 进程，且不杀自己
                        if pid != current_pid and 'python' in proc_name:
                            print(f"[清理] 发现占用 GPU 的 Python 进程: PID={pid}")
                            if sys.platform == 'win32':
                                subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                                              capture_output=True, timeout=5)
                            else:
                                subprocess.run(['kill', '-9', str(pid)],
                                              capture_output=True, timeout=5)
                            killed.append(pid)
                    except (ValueError, subprocess.TimeoutExpired):
                        pass

        if killed:
            print(f"[清理] 已终止 {len(killed)} 个占用 GPU 的 Python 进程: {killed}")
            time.sleep(2)  # 等待进程完全释放

    except FileNotFoundError:
        print("[警告] 未找到 nvidia-smi，跳过进程清理")
    except subprocess.TimeoutExpired:
        print("[警告] nvidia-smi 超时，跳过进程清理")
    except Exception as e:
        print(f"[警告] 进程清理失败: {e}")


def _force_release_gpu_memory():
    """强制释放 GPU 显存"""
    if not torch.cuda.is_available():
        return

    # 多次 gc 确保 Python 对象被回收
    for _ in range(5):
        gc.collect()

    # 清空 CUDA 缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 重置内存统计
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except:
        pass

    # 再次 gc
    for _ in range(3):
        gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _early_gpu_cleanup(kill_other_processes=True):
    """在导入前强制清空 GPU 缓存"""
    print("[启动] 开始 GPU 显存清理...")

    # 检查系统内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[启动] 系统内存: {mem.available / 1024**3:.1f} GB 可用 / {mem.total / 1024**3:.1f} GB 总计")
        if mem.available < 4 * 1024**3:
            print("[警告] 系统可用内存不足 4GB，可能导致 bad allocation 错误")
    except ImportError:
        pass

    if not torch.cuda.is_available():
        print("[启动] CUDA 不可用，跳过 GPU 清理")
        return

    # 获取 GPU 信息
    device = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device).total_memory

    # 第一次检查显存使用
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)

    print(f"[启动] GPU: {torch.cuda.get_device_name(device)}")
    print(f"[启动] 清理前 - 已分配: {allocated_before / 1024**2:.1f} MB, 已保留: {reserved_before / 1024**2:.1f} MB")

    # 如果显存占用过高，尝试杀掉其他进程
    if kill_other_processes and reserved_before > 500 * 1024 * 1024:  # > 500MB
        print("[启动] 检测到显存占用较高，尝试清理其他进程...")
        _kill_other_python_gpu_processes()

    # 强制释放显存
    _force_release_gpu_memory()

    # 清理后的状态
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    free_mem = total_mem - reserved_after

    print(f"[启动] 清理后 - 已分配: {allocated_after / 1024**2:.1f} MB, 已保留: {reserved_after / 1024**2:.1f} MB")
    print(f"[启动] GPU 可用显存: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")

    # 如果显存仍然不足，给出警告
    if free_mem < 2 * 1024**3:  # < 2GB
        print("[警告] GPU 可用显存不足 2GB，可能会 OOM！")
        print("[建议] 请手动关闭其他占用 GPU 的程序，或重启电脑")


# 执行早期清理
_early_gpu_cleanup(kill_other_processes=True)

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 使用主项目的 cosyvoice 源码
COSYVOICE_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, COSYVOICE_ROOT)
MATCHA_PATH = os.path.join(COSYVOICE_ROOT, 'Matcha-TTS-main')
if os.path.exists(MATCHA_PATH):
    sys.path.insert(0, MATCHA_PATH)

from config import (
    PRETRAINED_MODEL_DIR, RAW_AUDIO_DIR, OUTPUT_DIR,
    MEL_MEAN, MEL_STD, INFERENCE_CONFIG, LORA_CONFIG
)

# 显示检测到的配置
print(f"[配置] 预训练模型: {PRETRAINED_MODEL_DIR}")
print(f"[配置] 原始音频目录: {RAW_AUDIO_DIR}")


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
    """强力重置 CUDA 和系统内存"""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.collect()


def force_cleanup(verbose=False):
    """强制清理所有内存（GPU + 系统）"""
    import ctypes

    if verbose and torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1024**2
        print(f"  [force_cleanup] 清理前: {before:.1f} MB")

    # 多轮 gc
    for _ in range(5):
        gc.collect()

    if torch.cuda.is_available():
        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 重置统计
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass

    # 再次 gc
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Windows: 释放工作集内存
    try:
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except:
        pass

    gc.collect()

    if verbose and torch.cuda.is_available():
        after = torch.cuda.memory_allocated() / 1024**2
        print(f"  [force_cleanup] 清理后: {after:.1f} MB")


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
    Patch CosyVoice 的 flow 模型，添加归一化/反归一化处理

    【重要说明】
    quick_inference.py 使用的是 CosyVoice 原版的模型结构（不是我们的 flow_model.py），
    只是替换了权重。因此需要外部 patch 来处理归一化。

    训练时：feat 被归一化到 N(0,1)，模型学习在这个分布下生成
    推理时：CosyVoice 原始代码不做归一化，所以需要：
    1. 对 prompt_feat 进行归一化（输入）
    2. 对输出进行反归一化（输出）
    """
    original_inference = flow_model.inference

    @wraps(original_inference)
    @torch.inference_mode()
    def patched_inference(token, token_len, prompt_token, prompt_token_len,
                          prompt_feat, prompt_feat_len, embedding, flow_cache=None):
        if verbose:
            print(f"    [Patch] prompt_feat 原始: mean={prompt_feat.mean():.2f}, "
                  f"range=[{prompt_feat.min():.2f}, {prompt_feat.max():.2f}]")

        # 对 prompt_feat 进行归一化（与训练时一致）
        prompt_feat_normalized = normalize_mel(prompt_feat)

        if verbose:
            print(f"    [Patch] prompt_feat 归一化后: mean={prompt_feat_normalized.mean():.2f}")

        # 调用原始 inference
        result, cache = original_inference(
            token, token_len, prompt_token, prompt_token_len,
            prompt_feat_normalized, prompt_feat_len, embedding, flow_cache
        )

        if verbose:
            print(f"    [Patch] flow 输出（归一化空间）: mean={result.mean():.2f}, "
                  f"range=[{result.min():.2f}, {result.max():.2f}]")

        # 对输出进行反归一化，还原到原始 mel 尺度
        result_denormalized = denormalize_mel(result)

        if verbose:
            print(f"    [Patch] 反归一化后: mean={result_denormalized.mean():.2f}, "
                  f"range=[{result_denormalized.min():.2f}, {result_denormalized.max():.2f}]")

        return result_denormalized, cache

    flow_model.inference = patched_inference
    print("  ✓ Flow 模型已 patch：推理时自动进行 mel 归一化/反归一化")


# ============================================================
# 从 Checkpoint 提取并合并 LoRA 权重
# ============================================================

def extract_and_merge_from_checkpoint(ckpt_path: str) -> dict:
    """
    从 Lightning checkpoint 提取 LoRA 权重并合并到原始模型

    Args:
        ckpt_path: checkpoint 文件路径

    Returns:
        合并后的 state_dict，可直接被 CosyVoice 加载
    """
    from flow_model import build_flow_model
    from lora import apply_lora_to_model, get_merged_state_dict

    # 清空 GPU 缓存
    reset_cuda()

    print(f"\n[1/4] 加载 checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 统计 LoRA 参数
    lora_keys = [k for k in state_dict.keys() if 'lora_' in k]
    print(f"  找到 {len(lora_keys)} 个 LoRA 参数")

    print(f"\n[2/4] 构建 Flow 模型并加载预训练权重...")
    clear_gpu_cache()
    flow_model = build_flow_model(pretrained_path=PRETRAINED_MODEL_DIR, device='cpu')

    print(f"\n[3/4] 应用 LoRA 结构...")
    clear_gpu_cache()
    apply_lora_to_model(
        flow_model,
        r=LORA_CONFIG['lora_r'],
        lora_alpha=LORA_CONFIG['lora_alpha'],
        lora_dropout=LORA_CONFIG['lora_dropout'],
        target_modules=LORA_CONFIG['target_modules'],
    )

    # 加载 checkpoint 中的权重
    model_state = flow_model.state_dict()
    loaded_count = 0
    for key, value in state_dict.items():
        # 处理 Lightning 的 'model.' 前缀
        clean_key = key.replace('model.', '') if key.startswith('model.') else key
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            model_state[clean_key] = value
            loaded_count += 1

    flow_model.load_state_dict(model_state)
    print(f"  加载了 {loaded_count} 个参数")

    print(f"\n[4/4] 合并 LoRA 权重到原始模型...")
    clear_gpu_cache()
    merged_state_dict = get_merged_state_dict(flow_model)

    # 释放模型内存
    del flow_model
    del ckpt
    del state_dict
    del model_state
    force_cleanup()

    return merged_state_dict


# ============================================================
# 推理函数
# ============================================================

def quick_inference(
    text: str,
    weight_path: str = None,
    ckpt_path: str = None,
    prompt_wav_path: str = None,
    prompt_text: str = None,
    output_path: str = None,
    auto_select: bool = True,
    use_semantic: bool = False,
):
    """
    快速推理

    Args:
        text: 要合成的文本
        weight_path: 合并后的权重路径 (.pt)
        ckpt_path: Lightning checkpoint 路径 (.ckpt)
        prompt_wav_path: 参考音频路径
        prompt_text: 参考音频对应的文本
        output_path: 输出音频路径
        auto_select: 是否自动选择最匹配的参考音频
        use_semantic: 是否使用语义相似度（需要 sentence-transformers）
    """
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav

    print("=" * 60)
    print("CosyVoice Flow LoRA 快速推理")
    print("=" * 60)

    # 查找参考音频（先做这步，不占内存）
    if prompt_wav_path is None:
        wav_files = list(Path(RAW_AUDIO_DIR).glob('*.wav'))
        if not wav_files:
            print(f"错误：未找到参考音频，请使用 --prompt 指定")
            return None

        if auto_select and len(wav_files) > 1:
            # 使用智能选择器自动匹配最佳参考音频
            try:
                from auto_prompt_selector import AutoPromptSelector

                print(f"\n[智能选择] 从 {len(wav_files)} 个音频中自动匹配...")
                selector = AutoPromptSelector(
                    RAW_AUDIO_DIR,
                    use_semantic=use_semantic,
                    cache_embeddings=True,
                )

                # 选择最佳匹配
                best_match = selector.select_best(
                    text,
                    top_k=1,
                    length_weight=0.35,
                    rhythm_weight=0.25,
                    semantic_weight=0.40,
                    verbose=True,
                )

                prompt_wav_path = best_match['path']
                if prompt_text is None:
                    prompt_text = best_match['text']

                print(f"\n[智能选择] 最佳匹配:")
                print(f"  参考文本: {best_match['text']}")
                print(f"  匹配得分: {best_match['score']:.3f}")
                print(f"  - 长度得分: {best_match['scores']['length']:.2f}")
                print(f"  - 节奏得分: {best_match['scores']['rhythm']:.2f}")
                print(f"  - 语义得分: {best_match['scores']['semantic']:.2f}")

            except ImportError:
                print("[警告] auto_prompt_selector 不可用，使用默认音频")
                prompt_wav_path = str(wav_files[0])
            except Exception as e:
                print(f"[警告] 智能选择失败: {e}，使用默认音频")
                prompt_wav_path = str(wav_files[0])
        else:
            # 不使用自动选择，用第一个音频
            prompt_wav_path = str(wav_files[0])
            print(f"\n使用默认参考音频: {os.path.basename(prompt_wav_path)}")

    if not os.path.exists(prompt_wav_path):
        print(f"错误：参考音频不存在: {prompt_wav_path}")
        return None

    # 确定权重路径（暂不加载）
    if ckpt_path:
        weight_source = ('ckpt', ckpt_path)
    elif weight_path:
        if not os.path.exists(weight_path):
            print(f"错误：权重文件不存在: {weight_path}")
            return None
        weight_source = ('weight', weight_path)
    else:
        default_weight = os.path.join(OUTPUT_DIR, 'flow_merged.pt')
        if os.path.exists(default_weight):
            weight_source = ('weight', default_weight)
        else:
            print("错误：未指定权重文件，也未找到默认权重")
            print(f"请使用 --ckpt 或 --weight 指定权重文件")
            return None

    print(f"\n权重来源: {weight_source[1]}")

    # 先清理内存，再加载 CosyVoice
    print("\n强制清理内存...")
    force_cleanup()
    time.sleep(1)
    force_cleanup()

    print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 加载 CosyVoice（此时还没加载微调权重）
    print("\n加载 CosyVoice...")
    print("  [DEBUG] 开始初始化 CosyVoice...")
    force_cleanup()
    try:
        cosyvoice = CosyVoice(PRETRAINED_MODEL_DIR, load_jit=False, load_trt=False)
        print("  [DEBUG] CosyVoice 初始化完成")
    except Exception as e:
        print(f"  [ERROR] CosyVoice 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    print(f"  采样率: {cosyvoice.sample_rate}")

    # 现在加载并替换 flow 权重
    print("\n加载并替换 Flow 权重...")
    clear_gpu_cache()

    if weight_source[0] == 'ckpt':
        print(f"  从 checkpoint 合并权重...")
        merged_state_dict = extract_and_merge_from_checkpoint(weight_source[1])
    else:
        print(f"  加载预合并权重: {os.path.basename(weight_source[1])}")
        merged_state_dict = torch.load(weight_source[1], map_location='cpu')

    cosyvoice.model.flow.load_state_dict(merged_state_dict, strict=True)
    del merged_state_dict
    clear_gpu_cache()
    print("  ✓ 微调权重加载成功")

    # 应用归一化 patch
    print("\n应用归一化 patch...")
    clear_gpu_cache()
    patch_flow_for_finetuned(cosyvoice.model.flow, verbose=False)

    # 加载并截取参考音频
    print(f"\n参考音频: {os.path.basename(prompt_wav_path)}")
    clear_gpu_cache()
    prompt_speech_16k = load_wav(prompt_wav_path, 16000)
    original_len = prompt_speech_16k.shape[1] / 16000
    print(f"  原始长度: {original_len:.2f}s")

    max_samples = int(INFERENCE_CONFIG['max_prompt_seconds'] * 16000)
    if prompt_speech_16k.shape[1] > max_samples:
        prompt_speech_16k = prompt_speech_16k[:, :max_samples]
        print(f"  截取后长度: {prompt_speech_16k.shape[1] / 16000:.2f}s ✓")

    # 准备参考文本
    if prompt_text is None:
        # 从文件名解析
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

    print(f"\n文本:")
    print(f"  参考文本: {prompt_text}")
    print(f"  合成文本: {text}")

    # 推理
    print("\n开始推理...")
    reset_cuda()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    output_audio = None

    try:
        with torch.inference_mode():
            clear_gpu_cache()
            for i, result in enumerate(cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=False
            )):
                audio = result['tts_speech']
                print(f"\n输出结果:")
                print(f"  形状: {audio.shape}")
                print(f"  长度: {audio.shape[-1] / cosyvoice.sample_rate:.2f}s")

                # 保存
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)

                output_audio = audio.cpu()

                # 注意：flow.inference() 和 model.py 的 token2wav() 已经处理了 prompt 切除
                # 这里不需要再次切除，否则会丢失有效内容

                # 旧的 trim_ratio 逻辑（保留兼容，但默认禁用）
                trim_ratio = INFERENCE_CONFIG.get('trim_ratio', 0.0)
                if trim_ratio > 0:
                    trim_samples = int(output_audio.shape[-1] * trim_ratio)
                    if trim_samples > 0:
                        output_audio = output_audio[:, trim_samples:]
                        print(f"  裁剪开头: {trim_samples} samples ({trim_ratio*100:.0f}%)")
                        print(f"  裁剪后长度: {output_audio.shape[-1] / cosyvoice.sample_rate:.2f}s")

                if output_path is None:
                    os.makedirs(os.path.join(OUTPUT_DIR, 'inference'), exist_ok=True)
                    output_path = os.path.join(OUTPUT_DIR, 'inference', 'quick_output.wav')

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
    parser = argparse.ArgumentParser(
        description='CosyVoice Flow LoRA 快速推理 - 直接从 checkpoint 推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 checkpoint 直接推理（自动选择最匹配的参考音频）
  python quick_inference.py --ckpt output/flow_best_epoch=23.ckpt --text "你好世界"

  # 从合并后的权重推理
  python quick_inference.py --weight output/flow_merged.pt --text "你好世界"

  # 使用语义相似度进行智能匹配（需要 sentence-transformers）
  python quick_inference.py --weight xxx.pt --text "床前明月光" --semantic

  # 禁用自动选择，手动指定参考音频
  python quick_inference.py --ckpt xxx.ckpt --text "你好" --no-auto --prompt ref.wav

  # 指定参考音频和输出
  python quick_inference.py --ckpt xxx.ckpt --text "你好" --prompt ref.wav --output out.wav
        """
    )

    parser.add_argument('--text', '-t', type=str, required=True,
                        help='要合成的文本')
    parser.add_argument('--ckpt', '-c', type=str, default=None,
                        help='Lightning checkpoint 路径 (.ckpt)')
    parser.add_argument('--weight', '-w', type=str, default=None,
                        help='合并后的权重路径 (.pt)')
    parser.add_argument('--prompt', '-p', type=str, default=None,
                        help='参考音频路径（指定后禁用自动选择）')
    parser.add_argument('--prompt-text', type=str, default=None,
                        help='参考音频对应的文本')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出音频路径')

    # 自动选择相关参数
    parser.add_argument('--no-auto', action='store_true',
                        help='禁用自动参考音频选择')
    parser.add_argument('--semantic', '-s', action='store_true',
                        help='启用语义相似度匹配（需要 sentence-transformers）')

    args = parser.parse_args()

    # 验证参数
    if args.ckpt is None and args.weight is None:
        # 检查是否有默认权重
        default_weight = os.path.join(OUTPUT_DIR, 'flow_merged.pt')
        if not os.path.exists(default_weight):
            parser.error("请指定 --ckpt 或 --weight 参数")

    # 如果手动指定了 prompt，则禁用自动选择
    auto_select = not args.no_auto and args.prompt is None

    # 执行推理
    quick_inference(
        text=args.text,
        weight_path=args.weight,
        ckpt_path=args.ckpt,
        prompt_wav_path=args.prompt,
        prompt_text=args.prompt_text,
        output_path=args.output,
        auto_select=auto_select,
        use_semantic=args.semantic,
    )


if __name__ == '__main__':
    main()
