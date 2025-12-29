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
#
# 下载方式1 - HuggingFace:
#   pip install huggingface_hub
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='./pretrained_models/CosyVoice-300M')"
#
# 下载方式2 - 手动下载:
#   https://www.modelscope.cn/models/iic/CosyVoice-300M/files
#   https://huggingface.co/FunAudioLLM/CosyVoice-300M/tree/main
#
# 下载后目录结构应包含:
#   pretrained_models/CosyVoice-300M/
#   ├── flow.pt              # Flow 模型权重 (必须)
#   ├── hift.pt              # HiFi-GAN 声码器 (必须)
#   ├── llm.pt               # LLM 模型 (必须)
#   ├── campplus.onnx        # 说话人编码器 (必须)
#   ├── speech_tokenizer_v1.onnx  # 语音分词器 (必须)
#   └── cosyvoice.yaml       # 配置文件 (必须)
#
# ============================================================

# 自动检测预训练模型位置
# 优先级: 1. 本项目目录  2. 主项目目录
_local_model_dir = os.path.join(PROJECT_ROOT, "pretrained_models", "CosyVoice-300M")
_main_model_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "pretrained_models", "CosyVoice-300M")

if os.path.exists(os.path.join(_local_model_dir, "flow.pt")):
    PRETRAINED_MODEL_DIR = _local_model_dir
elif os.path.exists(os.path.join(_main_model_dir, "flow.pt")):
    PRETRAINED_MODEL_DIR = _main_model_dir
else:
    # 默认使用本项目目录（用户需要下载）
    PRETRAINED_MODEL_DIR = _local_model_dir

# ============================================================
# 【必须配置】训练数据目录
# 运行 data_prepare/prepare_data.py 后生成的 parquet 文件目录
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ============================================================
# 【必须配置】原始音频目录
# 包含 .wav 和对应 .txt 文件的目录，用于数据准备和推理
# ============================================================

# 自动检测原始音频目录
# 优先级: 1. 本项目目录  2. 主项目的 data_preparation/short_raw_data
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
# 训练产物（权重、日志）保存位置
# ============================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ============================================================
# 训练配置
# ============================================================

TRAIN_CONFIG = {
    # 训练参数
    'max_epochs': 100,
    'batch_size': 2,
    'accumulate_grad_batches': 4,  # 有效 batch = 2 * 4 = 8
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'weight_decay': 0.01,
    'warmup_steps': 200,

    # 显存优化
    'max_feat_len': 400,           # 最大帧数，超过会截断
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
    'lora_alpha': 16,              # LoRA 缩放因子
    'lora_dropout': 0.05,          # Dropout
    'target_modules': [            # 目标模块
        'to_q', 'to_k', 'to_v',           # Attention QKV
        'linear_q', 'linear_k', 'linear_v', # Conformer Attention
        'linear_out',                      # Attention 输出
        'w_1', 'w_2',                      # FFN 层
    ],
}


# ============================================================
# Mel 归一化参数（请勿修改）
# ============================================================

# 训练时使用的归一化参数
# Log mel 原始统计: mean ≈ -6.0, std ≈ 2.0
MEL_MEAN = -6.0
MEL_STD = 2.0


# ============================================================
# 推理配置
# ============================================================

INFERENCE_CONFIG = {
    # 参考音频最大长度（秒）- 超过会截取
    'max_prompt_seconds': 5,

    # 裁剪输出开头的比例（去除 prompt 残留）- 用于 quick_inference.py
    'trim_ratio': 0.0,  # 设为 0，因为 model.py 已经处理了边界裁剪

    # 边界裁剪比例 - 用于 model.py 的 token2wav
    # 这个值表示裁剪 prompt_mel_len * boundary_trim_ratio 帧
    # 用于去除 LoRA 微调模型中 prompt 语义内容泄漏到合成开头的问题
    'boundary_trim_ratio': 0.15,
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
