<p align="center">
  <img src="removed_background_1767114134910.png" alt="CosyVoice Flow LoRA Finetune Logo" width="200"/>
</p>

<h1 align="center">CosyVoice Flow LoRA Finetune</h1>

<p align="center">
  <strong>一次与 AI 语音合成模型的深度调试之旅</strong><br>
  <em>从绝望的 Loss 30+ 到无 Prompt 推理的完整纪实</em>
</p>

<p align="center">
  <a href="#-探索历程">探索历程</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-调试纪实视频">调试纪实</a> •
  <a href="#-易懂指南">易懂指南</a>
</p>

---

## 项目全景

<p align="center">
  <img src="big_pic.png" alt="Project Overview" width="100%"/>
</p>

> 这张图展示了从 Zero-Shot 微调到 LLM + Flow 联合训练的完整技术演进路线。

---

## 调试纪实视频

<table>
<tr>
<td width="60%">

https://github.com/user-attachments/assets/YOUR_VIDEO_ID

</td>
<td width="40%">

### CosyVoice 案例：调试下一代 AI 语音

完整记录了从发现问题到解决的全过程：

- **发现问题**：语义泄漏现象
- **四关断路测试**：定位根因
- **联合训练方案**：LLM + Flow
- **无 Prompt 推理**：最终解决

> 点击左侧视频播放，或[下载完整视频](CosyVoice案例：调试下一代AI语音.mp4)

</td>
</tr>
</table>

<!--
上传说明：
1. 在 GitHub 上编辑此 README
2. 将视频文件拖拽到编辑框中
3. GitHub 会自动上传并生成链接
4. 用生成的链接替换上面的 YOUR_VIDEO_ID
-->

---

## 效果演示

| 音频 | 说明 |
|------|------|
| [床前明月光.wav](床前明月光.wav) | 床前明月光，疑似地上霜。举头望明月，低头思故乡。 |
| [两个黄鹂鸣翠柳.wav](两个黄鹂鸣翠柳.wav) | 两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。 |

---

## 探索历程

> *这是一段充满挑战与惊喜的技术探索之旅。从最初尝试 Zero-Shot 微调时遇到的"绝望 Loss"，到发现神秘的"语义泄漏"现象，再到最终实现"无 Prompt 推理"——每一步都充满了意外与收获。*

---

### 序章：起点与初心

**我的目标很简单**：用少量数据（10-50条音频，总共几分钟）微调 CosyVoice，让模型学会特定说话人的音色，用于"古诗吟诵"场景。

CosyVoice 是阿里巴巴开源的语音合成模型，提供了 Zero-Shot 模式——只需一段参考音频，模型就能模仿该音色。听起来很美好，对吧？

我最初的想法是：**通过 LoRA 微调 Flow 模型来强化音色学习**。

然而，这条路远比想象中曲折。

---

### 第一章：绝望之墙 —— Loss 卡在 20-30

#### 黑暗的开始

当我满怀期待地开始训练时，迎接我的是一堵冰冷的墙：

```
Epoch 1: Loss = 28.7
Epoch 5: Loss = 27.3
Epoch 10: Loss = 26.9
Epoch 50: Loss = 25.1
...
```

**Loss 死活降不下去**。无论我怎么调学习率、改 batch size，它就像被钉住了一样。

我开始怀疑人生：是数据有问题？是代码写错了？还是我根本不适合做这个？

#### 第一个转折：权重加载的魔鬼细节

经过无数次调试，我终于发现了第一个致命问题——**权重命名不匹配**。

CosyVoice 的预训练权重使用的命名规范是 `attn.linear_q`，而我的代码用的是 `attn1.linear_q`。看起来只是多了一个"1"，但就是这个微小的差异，导致整个模型的 Attention 权重完全没有加载进去！

```
预训练权重：estimator.encoder.encoders.0.self_attn.linear_q.weight
我的模型：  estimator.encoder.encoders.0.self_attn1.linear_q.weight
                                              ↑ 多了个 1！
```

这就像是买了一把钥匙，却发现锁孔差了零点一毫米——看起来几乎一样，但就是打不开。

#### 第二个魔鬼：激活函数的陷阱

修复命名问题后，Loss 从 25+ 降到了 15 左右。进步了！但仍然不理想。

继续深挖，我发现了第二个问题——**激活函数不匹配**。

原始 Matcha-TTS 使用的是 `snakebeta` 激活函数，但 CosyVoice 的预训练模型用的是标准的 `GELU`。我的代码还在用 `snakebeta`，这意味着模型的"思考方式"和预训练权重完全不兼容。

```
┌─────────────────────────────────────────────────────────────┐
│                激活函数不匹配的影响                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  预训练模型学到的：                                          │
│  输入 x → GELU(x) → 某种输出模式                             │
│                                                             │
│  我的模型在做：                                              │
│  输入 x → snakebeta(x) → 完全不同的输出模式                  │
│                                                             │
│  结果：预训练权重完全失效，模型从零开始学习                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 第三个隐藏 Boss：消失的 scale=1000

就在我以为问题都解决了的时候，Loss 仍然卡在 8-12。

这次的 bug 更隐蔽：**正弦位置编码缺少 scale=1000**。

CosyVoice 的位置编码实现中有一个关键参数 `scale=1000`，它让位置编码的值域扩大 1000 倍。我的代码漏掉了这个参数，导致位置信息太弱，模型无法正确理解时序关系。

这就像是 GPS 定位，原本精度是米级，突然变成了千米级——模型完全迷路了。

#### 第一章小结：Loss 下降的三把钥匙

| 问题 | Loss 范围 | 解决方案 |
|------|----------|---------|
| 权重命名不匹配 | 20-30 | `attn` → `attn1` 统一命名 |
| 激活函数错误 | 12-20 | `snakebeta` → `GELU` |
| 位置编码缺 scale | 8-12 | 添加 `scale=1000` |

---

### 第二章：幽灵般的语义泄漏

#### 新的曙光，新的问题

修复了所有权重对齐问题后，Loss 终于开始正常下降了！

```
Epoch 1: Loss = 8.2
Epoch 10: Loss = 3.5
Epoch 30: Loss = 1.2
Epoch 50: Loss = 0.6
```

我满心欢喜地进行推理测试，结果...

```
参考音频内容：「熊咆龙吟殷岩泉，栗深林兮惊层巅。」
目标文本：「我虽非官敢越爼，弟兄急难宜平章。」
实际输出：「栗深林兮惊层巅，弟兄急难宜平章。」
             ↑ 这是哪来的？！
```

**输出的开头居然包含了参考音频末尾的内容！**

我把这个诡异的现象命名为 **"语义泄漏"(Semantic Leakage)**。

#### 深入虎穴：四关断路测试法

为了找到问题根源，我开发了一套系统的诊断方法。

**第一关：底模 vs 微调模型对比**

我用同样的输入测试原始 CosyVoice 和微调后的模型，发现：
- 原始模型：泄漏相关性 ~0.88
- 微调模型：泄漏相关性 ~0.88

**震惊**：泄漏不是 LoRA 引入的，原始模型就有这个问题！

**第二关：语义坍塌测试**

当我用空白参考文本测试时，模型输出变成了胡言乱语。

这意味着 LoRA 学到了一个危险的模式：**过度依赖 prompt 的语义信息**，而不是专注于音色特征。

**第三关：物理裁剪测试**

我直接裁掉输出音频的前 80 帧（约 0.9 秒），泄漏消失了！

这证明泄漏确实存在于输出的开头部分。

**第四关：Token 边界分析**

通过分析 mel 频谱的相关性，我发现 Prompt 末尾和 Target 开头在声学特征上高度相关，尤其是中低频段。

#### 根因揭秘：泄漏的三条通道

深入阅读 CosyVoice 源码后，我终于找到了泄漏的根本原因：

```
┌───────────────────────────────────────────────────────────────┐
│                  CosyVoice Flow 的数据处理流程                 │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   输入：[prompt_mel | target_mel]  ← 两部分被拼接在一起        │
│           ↓                                                   │
│       Conformer Encoder                                       │
│           ↓                                                   │
│   内部：prompt 和 target 的信息可以互相"看见"                  │
│           ↓                                                   │
│       ConditionalCFM (Flow Matching)                          │
│           ↓                                                   │
│   输出：只返回 target 部分                                     │
│                                                               │
│   问题：target 的开头已经"污染"了 prompt 的信息                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**泄漏的三条通道**：

| 通道 | 机制 | 后果 |
|------|------|------|
| Self-Attention | Target 可以 attend 到 Prompt | 语义信息直接传递 |
| Conv1d (kernel=3) | 相邻帧互相影响 | 边界模糊 |
| Skip Connection | 保存了包含 prompt 的隐藏状态 | 信息残留 |

这是**架构级的问题**，不是简单调参能解决的。

---

### 第三章：六次失败的尝试

发现问题后，我开始了漫长的修复之旅。

#### 尝试 1：静音隔离带

**思路**：在 prompt 和 target 之间插入静音，物理隔离两部分。

**结果**：失败。静音被模型学会了，反而引入了新的伪影。

#### 尝试 2：动态 Prompt 长度

**思路**：随机改变 prompt 的长度，让模型不能"记住"固定的泄漏位置。

**结果**：略有改善，但泄漏仍然存在。

#### 尝试 3：Prompt Dropout

**思路**：训练时随机丢弃 prompt，强迫模型不依赖它。

**结果**：25% 的 dropout 率下，模型开始崩溃。

#### 尝试 4：边界 Loss 加权

**思路**：对 target 开头施加更高的 loss 权重，惩罚泄漏。

**结果**：泄漏减轻了，但音质下降。

#### 尝试 5：跨样本训练

**思路**：用不同音频的 mel 作为 prompt，打破语义连续性。

**结果**：有效果！泄漏明显减少，但没有完全消除。

#### 尝试 6：文本侧致盲

**思路**：在 prompt 区域将 encoder 输出置零，切断语义传递通道。

**结果**：配合跨样本训练，效果最好。但推理时仍有轻微残留。

#### 残酷的结论

经过六种策略的各种组合尝试，我不得不接受一个残酷的现实：

> **在 Zero-Shot 框架下，语义泄漏无法从根本上解决。**
>
> 这是架构决定的。只要 prompt 和 target 需要拼接、需要一起编码、需要共享 Attention，泄漏就不可避免。

---

### 第四章：转机 —— 联合训练的曙光

#### 问题的本质

回到最初的目标：我想让模型学会特定说话人的**音色**和**吟诵风格**。

Zero-Shot 模式的设计逻辑是：
- **推理时**：提供参考音频，模型实时提取音色
- **微调时**：强化音色学习

但这带来了两个问题：
1. **推理必须依赖参考音频**——每次都要提供，很麻烦
2. **语义泄漏无法避免**——架构决定的

#### 换一种思路

既然 Zero-Shot 行不通，为什么不换一种方式？

**新思路**：让模型"记住"目标说话人的音色和风格，推理时不需要参考音频。

这需要同时微调两个模块：
- **LLM**：学习韵律和节奏（吟诵风格的关键）
- **Flow**：学习音色特征

```
┌───────────────────────────────────────────────────────────────┐
│                    联合训练的优势                              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Zero-Shot (只训练 Flow)：                                    │
│   [需要参考音频] + [有语义泄漏] + [只能学音色]                  │
│                                                               │
│   联合训练 (LLM + Flow)：                                      │
│   [不需要参考音频] + [无泄漏] + [同时学音色和风格]              │
│                                                               │
│   关键突破：                                                   │
│   LLM 的 LoRA 权重"记住"了目标说话人的韵律模式                 │
│   Flow 的 LoRA 权重"记住"了目标说话人的音色特征                │
│   推理时直接使用，无需参考音频                                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

#### 显存的极限挑战

联合训练面临一个现实问题：**显存不够用**。

我使用的是 RTX 4060 Laptop 8GB，要同时加载 LLM 和 Flow 两个模型，还要计算梯度，这是极限操作。

```
显存分析：
├── LLM (~300M params, FP16): ~600 MB
├── Flow (~300M params, FP16): ~600 MB
├── 激活值 (batch=1, seq=250): ~2-3 GB
├── 梯度: ~1.2 GB
└── PyTorch 开销: ~1 GB
────────────────────────────────────
总计: ~6-7 GB (刚好能挤进 8GB)
```

通过极限优化配置，我成功在 8GB 显存上实现了联合训练。

---

### 第五章：新的陷阱 —— LLM 过拟合

#### "自说自话"现象

联合训练进展顺利，LLM Loss 一路下降：

```
Epoch 10: LLM Loss = 2.8, Flow Loss = 0.6
Epoch 30: LLM Loss = 1.5, Flow Loss = 0.5
Epoch 50: LLM Loss = 0.9, Flow Loss = 0.4
Epoch 70: LLM Loss = 0.5, Flow Loss = 0.4
```

当 LLM Loss 降到 0.5 时，我进行了推理测试...

```
输入文本：「床前明月光，疑是地上霜。」
实际输出：「床前明月光，疑是地上霜。举头望明月，低头思故乡。
          窗前一抹红，月下几重山...」
          ↑ 后面这些是哪来的？！
```

模型开始**"自说自话"**——说完目标文本后继续自由发挥，停不下来。

#### 根因分析

LLM 过拟合了。当 Loss 过低时，模型完全"记住"了训练数据的语义模式，以至于推理时按照这种模式继续生成。

**最佳 LLM Loss 范围**：1.5 ~ 2.5
- 低于 1.5：过拟合，自说自话
- 高于 2.5：欠拟合，韵律学习不足

#### 解决方案：阈值停止 + 正则化

```
┌───────────────────────────────────────────────────────────────┐
│                    防止 LLM 过拟合的措施                       │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   1. 设置 Loss 阈值自动停止                                    │
│      当 LLM Loss 达到 1.5 时，自动停止训练                     │
│                                                               │
│   2. 增加 LoRA Dropout                                        │
│      从 0.05 提高到 0.15，增强正则化                           │
│                                                               │
│   3. 降低 LoRA 容量                                           │
│      lora_r 从 16 降到 8，减少过拟合风险                       │
│                                                               │
│   4. 监控训练曲线                                              │
│      一旦发现 Loss 下降过快，及时停止                          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

### 终章：无 Prompt 推理的实现

经过所有努力，我终于实现了最初的目标——**真正的无 Prompt 推理**。

```
传统 Zero-Shot 推理：
├── 需要提供参考音频
├── 存在语义泄漏风险
└── 每次推理都要处理参考音频

联合训练后的无 Prompt 推理：
├── 不需要参考音频
├── 无语义泄漏
├── 音色特征存储在 LoRA 权重中
└── 韵律风格存储在 LLM LoRA 中
```

**最终效果**：

| 维度 | 状态 |
|------|------|
| 音色像目标说话人 | OK |
| 吟诵风格正确 | OK |
| 不需要参考音频 | OK |
| 无语义泄漏 | OK |

---

### 探索历程总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         技术探索路线图                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  第一章：绝望之墙 (Loss 卡在 20-30)                                      │
│    │                                                                    │
│    ├── 权重命名不匹配 (attn vs attn1)                                   │
│    ├── 激活函数错误 (snakebeta vs GELU)                                 │
│    └── 位置编码缺 scale=1000                                            │
│          ↓                                                              │
│  第二章：语义泄漏 (幽灵般的 bug)                                          │
│    │                                                                    │
│    ├── 四关断路测试定位根因                                              │
│    └── 发现是架构级问题                                                  │
│          ↓                                                              │
│  第三章：六次失败的尝试                                                   │
│    │                                                                    │
│    └── 结论：Zero-Shot 框架下无法根治                                    │
│          ↓                                                              │
│  第四章：联合训练的曙光                                                   │
│    │                                                                    │
│    ├── LLM + Flow 同时微调                                              │
│    └── 8GB 显存极限优化                                                  │
│          ↓                                                              │
│  第五章：LLM 过拟合陷阱                                                   │
│    │                                                                    │
│    └── Loss 阈值 + 正则化解决                                           │
│          ↓                                                              │
│  终章：无 Prompt 推理实现                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 致未来的探索者

如果你也在尝试微调 CosyVoice，希望这份记录能帮助你少走弯路：

1. **权重对齐是第一步** —— 检查命名、激活函数、位置编码
2. **语义泄漏是架构级问题** —— 不要在错误的方向上浪费时间
3. **联合训练是正确方向** —— 同时学习音色和风格
4. **LLM 过拟合要警惕** —— Loss 1.5-2.5 是最佳范围
5. **8GB 显存可以做到** —— 但需要极限优化

> *"调试的过程，就是与模型对话的过程。每一个错误都是一个线索，每一次失败都是一次学习。"*

---

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/cosyvoice_flow_finetune.git
cd cosyvoice_flow_finetune/cosyvoice_flow_finetune

# 安装依赖
pip install -r requirements.txt
```

### 下载预训练模型

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='./pretrained_models/CosyVoice-300M')"
```

### 准备数据

将音频和对应文本放入 `raw_audio/` 目录：

```
raw_audio/
├── 001.wav
├── 001.txt  (音频对应的文本)
├── 002.wav
├── 002.txt
└── ...
```

### 联合训练（推荐）

```bash
# 1. 生成训练数据
python prepare_joint_data.py

# 2. 开始联合训练
python train_joint.py --mode joint

# 3. 无 Prompt 推理
python inference_joint.py --text "床前明月光，疑是地上霜"
```

详细文档请参阅 [cosyvoice_flow_finetune/README.md](cosyvoice_flow_finetune/README.md)。

---

## 问题与解决方案

<details>
<summary><b>Q1: Loss 卡在 20-30 怎么办？</b></summary>

这是权重加载问题。检查以下三点：

1. **权重命名**：确保 `attn` / `attn1` 命名一致
2. **激活函数**：确保使用 `GELU` 而非 `snakebeta`
3. **位置编码**：确保 `SinusoidalPosEmb` 包含 `scale=1000`

</details>

<details>
<summary><b>Q2: 什么是"语义泄漏"？</b></summary>

微调后的模型在推理时，输出音频的开头会包含参考音频末尾的语义内容。

这是 CosyVoice Flow 架构的固有问题，因为 prompt 和 target 的 mel 特征会被拼接后一起编码，Self-Attention 机制允许信息跨越边界传递。

**解决方案**：使用 LLM + Flow 联合训练，实现无 Prompt 推理。

</details>

<details>
<summary><b>Q3: 模型"自说自话"怎么办？</b></summary>

这是 LLM 过拟合的表现。当 LLM Loss 低于 1.5 时，模型会"记住"训练数据的语义模式。

**解决方案**：
- 设置 `llm_loss_threshold = 1.5` 自动停止
- 增加 `lora_dropout` 到 0.15
- 使用 Loss 在 1.5-2.0 范围内的 checkpoint

</details>

<details>
<summary><b>Q4: CUDA 内存不足怎么办？</b></summary>

联合训练对显存要求较高。8GB 显存的优化配置：

- `batch_size`: 1
- `accumulate_grad_batches`: 16
- `max_feat_len`: 250（约 2.9 秒）
- `lora_r`: 8
- `precision`: '16-mixed'

如果仍然 OOM，可以分阶段训练：先 `--mode flow_only`，再 `--mode llm_only`。

</details>

---

## 易懂指南

> CosyVoice Flow 微调调试通关指南 —— 从 Loss 30+ 到完美收敛的系统化排错框架

<p align="center">
  <img src="docs/guide_images/page_01.png" alt="封面" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_02.png" alt="调试关卡地图" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_03.png" alt="第零关：数据是模型的基石" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_04.png" alt="第一关：攻克 Loss 20-30 的绝望之墙" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_05.png" alt="解锁第一关的钥匙：激活函数必须对齐" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_06.png" alt="第一关通关清单：结构对齐的魔鬼细节" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_07.png" alt="第二关：跨越 Loss 8-12 的收敛高原" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_08.png" alt="高原突破口：被遗忘的 scale=1000" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_09.png" alt="第三关：攻克最后的 Loss 壁垒" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_10.png" alt="隐藏关卡：警惕幽灵般的语义泄漏" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_11.png" alt="斩断泄漏：两种核心训练策略" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_12.png" alt="终极心法：用单批次过拟合快速定位问题" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_13.png" alt="CosyVoice Flow 微调通关总览" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_14.png" alt="结语" width="100%"/>
</p>

---

## 项目结构

```
cosyvoice_flow_finetune/
├── README.md                        # 本文件
├── big_pic.png                      # 项目全景图
├── CosyVoice案例：调试下一代AI语音.mp4  # 调试纪实视频
├── NotebookLM Mind Map.png          # 知识图谱
├── docs/guide_images/               # 易懂指南图片
│
└── cosyvoice_flow_finetune/         # 核心代码
    ├── config.py                    # 配置文件
    ├── train_joint.py               # 联合训练脚本
    ├── inference_joint.py           # 无 Prompt 推理
    ├── prepare_joint_data.py        # 数据准备
    ├── llm_flow_model.py            # 联合模型定义
    ├── merge_joint_weights.py       # 权重合并
    ├── lora.py                      # LoRA 模块
    ├── dataset.py                   # 数据集
    ├── utils.py                     # 工具函数
    └── cosyvoice/                   # CosyVoice 核心代码
```

---

## 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 阿里巴巴 FunAudioLLM 团队
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow Matching TTS
- [LoRA](https://arxiv.org/abs/2106.09685) - 低秩适配技术

---

## License

本项目遵循 [MIT License](cosyvoice_flow_finetune/LICENSE)，同时尊重 CosyVoice 和 Matcha-TTS 的原始协议。

---

<p align="center">
  <strong>如果这个项目对你有帮助，请给个 Star！</strong>
</p>
