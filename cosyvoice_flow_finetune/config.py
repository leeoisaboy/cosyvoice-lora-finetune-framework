"""
CosyVoice Flow LoRA 微调项目 - 配置文件

所有路径和参数配置都在这里，方便统一管理。
请根据你的环境修改下面的路径配置。
"""

import os

# ============================================================
# 路径配置 - 请根据你的环境修改
# ============================================================

# 项目根目录（自动检测，无需修改）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 【必须配置】预训练模型目录
# ============================================================

# 自动检测预训练模型位置
_local_model_dir = os.path.join(PROJECT_ROOT, "pretrained_models", "CosyVoice-300M")
_main_model_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "pretrained_models", "CosyVoice-300M")

if os.path.exists(os.path.join(_local_model_dir, "flow.pt")):
    PRETRAINED_MODEL_DIR = _local_model_dir
elif os.path.exists(os.path.join(_main_model_dir, "flow.pt")):
    PRETRAINED_MODEL_DIR = _main_model_dir
else:
    PRETRAINED_MODEL_DIR = _local_model_dir

# ============================================================
# 【必须配置】训练数据目录
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ============================================================
# 【必须配置】原始音频目录
# ============================================================
_local_raw_dir = os.path.join(PROJECT_ROOT, "raw_audio")
_main_raw_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "data_preparation", "short_raw_data")

if os.path.exists(_local_raw_dir) and any(f.endswith('.wav') for f in os.listdir(_local_raw_dir) if os.path.isfile(os.path.join(_local_raw_dir, f))):
    RAW_AUDIO_DIR = _local_raw_dir
elif os.path.exists(_main_raw_dir):
    RAW_AUDIO_DIR = _main_raw_dir
else:
    RAW_AUDIO_DIR = _local_raw_dir

# ============================================================
# 【可选配置】输出目录
# ============================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ============================================================
# 训练配置
# ============================================================

TRAIN_CONFIG = {
    # 训练参数
    'max_epochs': 100,
    'batch_size': 2,               # 降低 batch_size 减少内存
    'accumulate_grad_batches': 4,  # 有效 batch = 1 * 8 = 8
    'learning_rate': 1e-4,         # 降低学习率防止过拟合
    'min_learning_rate': 1e-6,
    'weight_decay': 0.01,          # 提高权重衰减
    'warmup_steps': 200,           # 增加预热步数

    # 显存优化
    # 【重要】max_feat_len 决定训练时的最大序列长度
    # 如果推理时序列长度远超此值，可能导致长序列质量下降
    # 建议设置为数据集中 95 分位数的长度
    # 400 帧 ≈ 4.6 秒，如果需要生成更长的音频，考虑增加到 600-800
    'max_feat_len': 600,           # 增加到 600 帧 ≈ 7 秒，支持更长序列
    'precision': '16-mixed',       # 混合精度
    'gradient_clip_val': 1.0,      # 梯度裁剪

    # 数据增强
    'augmentation': True,
}


# ============================================================
# LoRA 配置
# ============================================================

LORA_CONFIG = {
    'use_lora': True,
    'lora_r': 16,                  # LoRA 秩
    'lora_alpha': 16,               # 降低 LoRA 缩放因子，避免过拟合
    'lora_dropout': 0.05,           # 提高 Dropout

    # 目标模块 - 包含 Attention 和 FFN
    'target_modules': [
        'to_q', 'to_k', 'to_v',           # Attention QKV
        'linear_q', 'linear_k', 'linear_v', # Conformer Attention
        'linear_out',                      # Attention 输出
        'w_1', 'w_2',                      # FFN 层
    ],
}


# ============================================================
# 语义泄漏防护配置 (Semantic Leakage Prevention)
# ============================================================

ANTI_LEAKAGE_CONFIG = {
    # ========== 策略1: 静音隔离带 ==========
    # 【已禁用】不需要插入静音，会导致时间漂移
    'silence_padding_enabled': False,
    'silence_token_id': 0,
    'silence_min_tokens': 5,
    'silence_max_tokens': 10,
    'silence_mel_value': -11.5,

    # ========== 策略2: 动态 Prompt 长度 ==========
    # 随机选择不同长度的 prompt，增加泛化能力
    'dynamic_prompt_enabled': True,
    'prompt_min_ratio': 0.05,      # prompt 最小比例 (5%)  ← 降低，减少泄漏窗口
    'prompt_max_ratio': 0.20,      # prompt 最大比例 (20%) ← 降低

    # ========== 策略3: Prompt Dropout ==========
    # 增加无 prompt 训练的比例，增强鲁棒性
    'prompt_dropout_enabled': True,
    'prompt_dropout_prob': 0.25,   # 25% 概率无 prompt ← 提高

    # ========== 策略4: 边界 Loss 权重 ==========
    # 在 target 开头施加更高权重，确保干净生成
    'boundary_loss_enabled': True,
    'boundary_frames': 25,         # 边界区域的帧数 ← 增加
    'boundary_loss_weight': 5.0,   # 边界区域的 loss 权重倍数 ← 提高

    # ========== 策略5: 跨样本训练 ==========
    # 【核心策略】使用不同音频的 mel 作为 prompt，打破语义连续性
    # 提高概率是解决泄漏的关键！
    'cross_sample_enabled': True,
    'cross_sample_prob': 0.85,     # 85% 使用跨样本 ← 大幅提高

    # ========== 策略6: 文本侧致盲 ==========
    # 【核心策略】在 prompt 区域将 encoder 输出置零，切断语义泄漏路径
    'text_blinding_enabled': True,
    'text_blinding_prob': 0.95,    # 95% 概率启用文本致盲 ← 大幅提高
    'text_blinding_mode': 'zero',
}


# ============================================================
# 无 Prompt 训练模式配置
# ============================================================
# 启用此模式后，训练时完全不使用 prompt conditioning
# 模型将学习仅依赖 speaker embedding 和 LoRA 权重生成语音
# 这允许推理时完全不需要参考音频

NO_PROMPT_TRAINING_CONFIG = {
    # 是否启用无 prompt 训练模式
    'enabled': False,  # 设为 True 启用

    # 无 prompt 模式下的训练策略
    # 'full': 100% 样本都不使用 prompt（最激进）
    # 'mixed': 混合训练，一部分有 prompt，一部分无 prompt
    'mode': 'full',

    # mixed 模式下，无 prompt 样本的比例
    'no_prompt_ratio': 0.8,  # 80% 无 prompt

    # 是否使用固定的 speaker embedding（从训练数据计算均值）
    # 如果为 False，使用每个样本自己的 embedding
    'use_mean_embedding': False,
}


# ============================================================
# Mel 归一化参数
# ============================================================

# Flow Matching 期望目标分布接近 N(0, 1)
# 这些值是基于大规模语音数据的统计结果
#
# 【重要】为什么不需要针对不同数据集调整：
# 1. 预训练模型是在这个分布下训练的，改变会破坏预训练权重的兼容性
# 2. Log mel 的统计特性在不同人声数据集之间相对稳定（通常在 ±0.5 范围内）
# 3. LoRA 微调只调整少量参数，依赖预训练模型学到的分布
#
# 如果你的数据集非常特殊（如歌唱、耳语），可以在 prepare_data.py 中
# 计算统计值，但仍建议使用这些默认值以保持兼容性
MEL_MEAN = -6.0
MEL_STD = 2.0


# ============================================================
# 推理配置
# ============================================================

INFERENCE_CONFIG = {
    # 参考音频最大长度（秒）- 超过会截取
    'max_prompt_seconds': 5,

    # ========== 物理切除配置（绝对帧数）==========
    # 用于切除固定数量的帧，去除 prompt 泄漏
    'physical_trim_enabled': True,
    'physical_trim_mode': 'absolute',
    'physical_trim_frames': 80,    # 切除前 80 帧 ← 增加
    'physical_trim_extra_ms': 300, # 额外切除 300ms ← 增加

    # ========== 边界裁剪配置（按比例）==========
    # 用于 quick_inference.py 输出裁剪
    'trim_ratio': 0.08,  # 裁剪输出开头 8% ← 启用

    # 用于 model.py 的 token2wav
    # 按 prompt_mel_len * boundary_trim_ratio 裁剪
    # 用于去除 LoRA 微调模型中 prompt 语义内容泄漏到合成开头的问题
    'boundary_trim_ratio': 0.20,   # 裁剪 prompt 长度的 20% ← 增加
}


# ============================================================
# 模型参数（CosyVoice-300M 架构，请勿修改）
# ============================================================

MODEL_CONFIG = {
    'input_size': 512,
    'output_size': 80,
    'spk_embed_dim': 192,
    'vocab_size': 4096,
    'input_frame_rate': 50,
    'sample_rate': 22050,
}
