# CosyVoice Flow LoRA Finetune

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#faq">FAQ</a>
</p>

CosyVoice Flow 模型的 LoRA 微调工具，支持少量数据（10-50条）快速微调音色。

**完全独立**：包含所有依赖代码，无需额外安装 CosyVoice。

## Features

- 🚀 **少量数据微调**：10-50 条音频即可微调出高质量音色
- 🎯 **LoRA 高效训练**：仅训练少量参数，显存占用低
- 📦 **完全独立**：无需安装 CosyVoice，开箱即用
- ⚡ **快速推理**：支持直接从 checkpoint 推理，无需合并权重
- 🔧 **配置灵活**：统一配置文件，路径自动检测

## Project Structure

```
cosyvoice_flow_finetune/
├── config.py              # 配置文件（路径、训练参数）
├── train.py               # 训练脚本
├── quick_inference.py     # 快速推理（推荐）
├── inference.py           # 标准推理
├── merge_weights.py       # LoRA 权重合并工具
├── diagnose.py            # 诊断工具
├── data_prepare/          # 数据准备工具
│   ├── prepare_data.py    # 从音频+文本准备训练数据
│   └── mel_extractor.py   # Mel 频谱提取器
├── flow_model.py          # Flow 模型定义
├── modules.py             # 神经网络模块
├── lora.py                # LoRA 模块
├── dataset.py             # 数据集加载
├── utils.py               # 工具函数
├── cosyvoice/             # CosyVoice 核心代码（已集成）
├── matcha/                # Matcha-TTS 核心代码（已集成）
├── raw_audio/             # 原始音频目录（用户数据）
├── data/                  # 训练数据目录（自动生成）
├── output/                # 输出目录（自动生成）
└── pretrained_models/     # 预训练模型目录
```

## Quick Start

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/cosyvoice_flow_finetune.git
cd cosyvoice_flow_finetune

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

从以下地址下载 CosyVoice-300M 模型：

| 来源 | 链接 |
|-----|------|
| ModelScope | https://www.modelscope.cn/models/iic/CosyVoice-300M |
| HuggingFace | https://huggingface.co/FunAudioLLM/CosyVoice-300M |

**快速下载（HuggingFace）**：
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='./pretrained_models/CosyVoice-300M')"
```

下载后目录结构：
```
pretrained_models/CosyVoice-300M/
├── flow.pt              # Flow 模型权重
├── hift.pt              # HiFi-GAN 声码器
├── llm.pt               # LLM 模型
├── campplus.onnx        # 说话人编码器
├── speech_tokenizer_v1.onnx  # 语音分词器
└── cosyvoice.yaml       # 配置文件
```

### 3. 准备训练数据

将你的音频和对应文本放到 `raw_audio/` 目录：

```
raw_audio/
├── 001.wav
├── 001.txt      # 包含 001.wav 的文本内容
├── 002.wav
├── 002.txt
└── ...
```

**数据要求**：
- 音频格式：WAV 或 MP3
- 音频时长：0.5-30 秒
- 采样率：任意（会自动重采样到 22050Hz）
- 推荐数据量：10-50 条

### 4. 生成训练数据

```bash
python data_prepare/prepare_data.py
```

### 5. 开始训练

```bash
# 开始训练
python train.py

# 从断点恢复
python train.py --resume

# 强制从头开始
python train.py --fresh
```

### 6. 推理测试

```bash
# 快速推理（推荐）
python quick_inference.py \
    --ckpt output/flow_best_xxx.ckpt \
    --text "你好，世界"

# 指定参考音频
python quick_inference.py \
    --ckpt output/flow_best_xxx.ckpt \
    --text "要合成的文本" \
    --prompt raw_audio/reference.wav
```

## Usage

### 训练

```bash
# 基本训练
python train.py

# 从特定 checkpoint 恢复
python train.py --resume --ckpt output/flow_epoch=10.ckpt

# 使用不同配置
python train.py --batch_size 4 --lr 5e-5
```

### 推理

#### 方式 A：快速推理（推荐）

直接从 checkpoint 推理，无需合并权重：

```bash
python quick_inference.py \
    --ckpt output/flow_best_xxx.ckpt \
    --text "要合成的文本" \
    --prompt raw_audio/reference.wav \
    --output output/result.wav
```

#### 方式 B：标准推理

先合并权重再推理（适合多次推理）：

```bash
# 第一步：合并权重
python merge_weights.py --ckpt output/flow_best_xxx.ckpt

# 第二步：推理
python inference.py \
    --text "要合成的文本" \
    --prompt raw_audio/reference.wav \
    --weight output/flow_merged.pt \
    --output output/result.wav
```

### 诊断工具

```bash
# 检查环境和配置
python diagnose.py

# 检查分词器
python check_tokenizer.py
```

## Configuration

所有配置都在 `config.py` 中，支持自动路径检测。

### 路径配置

| 变量 | 说明 | 默认值 |
|-----|-----|-------|
| `PRETRAINED_MODEL_DIR` | 预训练模型目录 | 自动检测 |
| `DATA_DIR` | 训练数据目录 | `./data` |
| `RAW_AUDIO_DIR` | 原始音频目录 | `./raw_audio` |
| `OUTPUT_DIR` | 输出目录 | `./output` |

### 训练参数

```python
TRAIN_CONFIG = {
    'max_epochs': 100,           # 最大训练轮数
    'batch_size': 2,             # 批次大小
    'accumulate_grad_batches': 4, # 梯度累积
    'learning_rate': 1e-4,       # 学习率
    'max_feat_len': 400,         # 最大帧数
    'precision': '16-mixed',     # 混合精度训练
}
```

### LoRA 参数

```python
LORA_CONFIG = {
    'lora_r': 16,          # LoRA 秩（越大表达能力越强）
    'lora_alpha': 16,      # 缩放因子
    'lora_dropout': 0.05,  # Dropout
}
```

## Training Tips

### 显存优化

如果遇到 CUDA OOM，尝试：

1. 减小 `batch_size`（config.py）
2. 减小 `max_feat_len`（config.py）
3. 使用梯度累积（默认已启用）

### 训练监控

```bash
tensorboard --logdir output/logs
```

### 最佳实践

- 参考音频时长建议 3-5 秒
- 训练数据越多样化，泛化能力越好
- 建议训练 50-100 epochs

## FAQ

### Q: 输出是噪音
**A**: Mel 归一化不匹配。使用 `inference.py` 或 `quick_inference.py`，它们会自动处理归一化。

### Q: 输出音频很短
**A**: 参考音频过长。建议参考音频控制在 3-5 秒。

### Q: CUDA 内存不足
**A**: 减小 `batch_size` 或 `max_feat_len`。

### Q: 找不到模块
**A**: 确保 `config.py` 中的路径正确，Windows 路径使用原始字符串 `r"..."`。

### Q: 训练 loss 不下降
**A**:
1. 检查数据质量（音频是否清晰、文本是否准确）
2. 尝试调整学习率
3. 增加训练数据量

## Model Parameters

| 参数 | 值 | 说明 |
|-----|---|-----|
| `sample_rate` | 22050 | 输出音频采样率 |
| `input_frame_rate` | 50 | 每秒 mel 帧数 |
| mel 维度 | 80 | mel 频谱通道数 |
| `MEL_MEAN` | -6.0 | mel 归一化均值 |
| `MEL_STD` | 2.0 | mel 归一化标准差 |

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 阿里巴巴 FunAudioLLM 团队
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow Matching TTS

## License

本项目遵循 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的开源协议。

## Citation

如果本项目对你有帮助，请考虑引用：

```bibtex
@misc{cosyvoice_flow_finetune,
  title={CosyVoice Flow LoRA Finetune},
  year={2024},
  url={https://github.com/YOUR_USERNAME/cosyvoice_flow_finetune}
}
```
