# CosyVoice LLM + Flow Joint LoRA Finetune

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#faq">FAQ</a>
</p>

CosyVoice **LLM + Flow 联合微调**工具，支持少量数据（10-50条）快速微调，实现**无需参考音频**的推理。

## Features

- 🎯 **联合训练**：同时微调 LLM 和 Flow，学习音色 + 韵律风格
- 🚀 **无 Prompt 推理**：训练后推理无需参考音频，彻底解决语义泄漏问题
- 📦 **少量数据**：10-50 条音频即可微调出高质量音色
- ⚡ **LoRA 高效训练**：仅训练少量参数，8GB 显存可用
- 🔧 **防过拟合机制**：内置 LLM loss 阈值，自动防止过拟合

## Why Joint Training?

传统 Zero-Shot 方案只微调 Flow，需要参考音频才能推理，且存在**语义泄漏**问题（输出开头包含参考音频结尾内容）。

**联合训练方案**同时微调 LLM 和 Flow：
- LLM 学习目标说话人的韵律、节奏、吟诵风格
- Flow 学习目标说话人的音色特征
- 推理时无需参考音频，从根本上解决语义泄漏

## Project Structure

```
cosyvoice_flow_finetune/
├── config.py              # 统一配置文件
├── train_joint.py         # 联合训练脚本
├── inference_joint.py     # 无 Prompt 推理脚本
├── prepare_joint_data.py  # 数据准备脚本
├── llm_flow_model.py      # 联合模型定义
├── merge_joint_weights.py # 权重合并工具
├── dataset.py             # 数据集加载
├── lora.py                # LoRA 实现
├── utils.py               # 工具函数
├── modules.py             # 需要用到的模块
├── flow_model.py          # flow_model模块
├── cosyvoice/             # CosyVoice 核心代码（已集成）
├── matcha/                # Matcha-TTS 核心代码（已集成）
├── data/                  # 训练数据目录（自动生成）
└── output/                # 输出目录（模型权重）

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

下载后放置到上一级目录：
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

在上一级目录创建 `raw_audio/` 目录，放入音频和对应文本：

```
raw_audio/
├── 001.wav
├── 001.txt      # 包含 001.wav 的文本内容
├── 002.wav
├── 002.txt
└── ...
```

**数据要求**：
- 音频格式：WAV（推荐）或 MP3
- 音频时长：30秒以内（建议 2-5 秒）
- 采样率：任意（会自动重采样）
- 推荐数据量：20-50 条
- **重要**：文本与音频内容必须精确对应

### 4. 生成训练数据

```bash
python prepare_joint_data.py
```

这会在 `data/` 目录下生成 parquet 格式的训练数据。

### 5. 开始训练

```bash
# 联合训练（推荐）
python train_joint.py

# 从 checkpoint 恢复
python train_joint.py --resume output/joint_joint_last.ckpt
```

**训练会在以下条件时自动停止**：
- LLM loss 达到 1.5（防止过拟合）
- Flow loss 达到 0.3
- 连续 10 个 epoch 无改善（早停）

### 6. 推理测试

```bash
# 无 Prompt 推理（推荐）
python inference_joint.py \
    --llm output/llm_merged_joint.pt \
    --flow output/flow_merged_joint.pt \
    --text "要合成的文本"

# 指定输出路径
python inference_joint.py \
    --llm output/llm_merged_joint.pt \
    --flow output/flow_merged_joint.pt \
    --text "床前明月光，疑是地上霜" \
    --output output/result.wav
```

## Usage

### 训练模式

```bash
# 联合训练（推荐，同时学习韵律+音色）
python train_joint.py --mode joint

# 只训练 LLM（学习韵律风格）
python train_joint.py --mode llm_only

# 只训练 Flow（学习音色）
python train_joint.py --mode flow_only
```

### 手动合并权重

如果训练中断，可以手动合并 checkpoint 中的 LoRA 权重：

```bash
python merge_joint_weights.py --ckpt output/joint_joint_last.ckpt
```

### 推理参数

```bash
python inference_joint.py \
    --llm output/llm_merged_joint.pt \    # LLM 权重路径
    --flow output/flow_merged_joint.pt \  # Flow 权重路径
    --text "要合成的文本" \                # 合成文本
    --output output/result.wav \          # 输出路径（可选）
    --speed 1.0                           # 语速调节（可选）
```

## Configuration

所有配置都在 `config.py` 中。

### 联合训练参数

```python
JOINT_TRAINING_CONFIG = {
    'training_mode': 'joint',      # 训练模式

    # Loss 权重
    'llm_loss_weight': 2.0,        # LLM loss 权重（强化韵律学习）
    'flow_loss_weight': 1.0,       # Flow loss 权重

    # LLM LoRA 配置
    'llm_lora': {
        'lora_r': 8,               # LoRA 秩
        'lora_alpha': 16,          # 缩放因子
        'lora_dropout': 0.15,      # Dropout（防过拟合）
    },

    # Flow LoRA 配置
    'flow_lora': {
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
    },

    # 训练参数
    'learning_rate': 2e-4,
    'max_epochs': 100,
    'batch_size': 1,
    'accumulate_grad_batches': 16,
}
```

### Loss 阈值说明

| 指标 | 最佳范围 | 说明 |
|-----|---------|------|
| LLM loss | 1.5 ~ 2.5 | 过低会导致过拟合（自说自话） |
| Flow loss | 0.3 ~ 0.5 | 越低音色越清晰 |

## Training Tips

### 显存优化

8GB 显存推荐配置（已在 config.py 中设置）：
- `batch_size`: 1
- `accumulate_grad_batches`: 16
- `max_feat_len`: 250（约 1.7 秒）
- `lora_r`: 8（LLM）/ 16（Flow）

### 过拟合处理

如果出现"自说自话"现象（说完文本后继续生成）：
1. 检查 LLM loss，如果 < 1.5 说明过拟合
2. 使用更早的 checkpoint（LLM loss 在 1.5-2.0 的版本）
3. 增大 `llm_lora.lora_dropout`

### 训练监控

```bash
tensorboard --logdir output/joint_logs
```

## FAQ

### Q: 输出音频"自说自话"，说完文本后还在继续
**A**: LLM 过拟合。使用 LLM loss 在 1.5-2.0 范围内的 checkpoint，或增大 dropout。

### Q: 音色不像目标说话人
**A**: Flow 训练不足。可以单独用 `--mode flow_only` 继续训练 Flow。

### Q: 韵律太平淡，没有特色
**A**: LLM 训练不足。检查 LLM loss 是否还在 2.5 以上，可适当延长训练。

### Q: CUDA 内存不足
**A**:
1. 确保 `batch_size` 为 1
2. 减小 `max_feat_len`（config.py）
3. 减小 `lora_r`

### Q: 找不到模块
**A**: 确保从 `cosyvoice_flow_finetune` 目录运行脚本，或检查 `config.py` 中的路径。

## Model Architecture

```
文本 → [LLM + LoRA] → speech_tokens → [Flow + LoRA] → mel → HiFi-GAN → 音频
         ↑                                ↑
      学习韵律                          学习音色
```

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 阿里巴巴 FunAudioLLM 团队
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow Matching TTS

## License

本项目遵循 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的开源协议。
