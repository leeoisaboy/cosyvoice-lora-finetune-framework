<p align="center">
  <img src="removed_background_1767114134910.png" alt="CosyVoice Flow LoRA Finetune Logo" width="200"/>
</p>

<h1 align="center">CosyVoice Flow LoRA Finetune</h1>

<p align="center">
  <strong>A Deep Debugging Journey with AI Speech Synthesis Models</strong><br>
  <em>From Desperate Loss 30+ to Prompt-Free Inference: A Complete Documentary</em>
</p>

<p align="center">
  <a href="#exploration-journey">Exploration Journey</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#debugging-documentary-video">Documentary Video</a> •
  <a href="#visual-guide">Visual Guide</a>
</p>

---

## Project Overview

<p align="center">
  <img src="big_pic.png" alt="Project Overview" width="100%"/>
</p>

> This diagram illustrates the complete technical evolution from Zero-Shot fine-tuning to LLM + Flow joint training.

---

## Debugging Documentary Video

<table>
<tr>
<td width="60%">

https://github.com/user-attachments/assets/YOUR_VIDEO_ID

</td>
<td width="40%">

### CosyVoice Case Study: Debugging Next-Gen AI Voice

A complete record of the journey from problem discovery to solution:

- **Problem Discovery**: Semantic leakage phenomenon
- **Four-Gate Circuit Breaker Test**: Root cause identification
- **Joint Training Solution**: LLM + Flow
- **Prompt-Free Inference**: Final resolution

> Click the video on the left to play, or [download the full video](CosyVoice案例：调试下一代AI语音.mp4)

</td>
</tr>
</table>

<!--
Upload Instructions:
1. Edit this README on GitHub
2. Drag and drop the video file into the editor
3. GitHub will automatically upload and generate a link
4. Replace YOUR_VIDEO_ID above with the generated link
-->

---

## Demo Results

| Audio | Description |
|-------|-------------|
| [床前明月光.wav](床前明月光.wav) | "Moonlight before my bed, I suspect it's frost on the ground. I raise my head to gaze at the moon, then lower it, thinking of home." |
| [两个黄鹂鸣翠柳.wav](两个黄鹂鸣翠柳.wav) | "Two orioles sing in the green willows, a line of egrets flies into the blue sky. My window frames the thousand-year snows of the Western Mountains, my door harbors boats bound for distant Wu." |

---

## Exploration Journey

> *This is a journey filled with challenges and surprises. From the "desperate Loss" encountered during initial Zero-Shot fine-tuning attempts, to discovering the mysterious "semantic leakage" phenomenon, and finally achieving "prompt-free inference" — every step was filled with unexpected discoveries and learnings.*

---

### Prologue: The Starting Point and Original Intent

**My goal was simple**: Fine-tune CosyVoice with minimal data (10-50 audio clips, totaling a few minutes) to make the model learn a specific speaker's voice for "classical poetry recitation" scenarios.

CosyVoice is an open-source speech synthesis model from Alibaba that offers a Zero-Shot mode — just provide a reference audio, and the model can imitate that voice. Sounds great, right?

My initial idea was: **Enhance voice learning by LoRA fine-tuning the Flow model**.

However, this path proved far more treacherous than imagined.

---

### Chapter 1: The Wall of Despair — Loss Stuck at 20-30

#### A Dark Beginning

When I eagerly started training, I was greeted by an ice-cold wall:

```
Epoch 1: Loss = 28.7
Epoch 5: Loss = 27.3
Epoch 10: Loss = 26.9
Epoch 50: Loss = 25.1
...
```

**The Loss simply wouldn't go down**. No matter how I adjusted the learning rate or changed the batch size, it seemed pinned in place.

I began to doubt myself: Was the data problematic? Was my code wrong? Or was I simply not cut out for this?

#### The First Turning Point: The Devil in Weight Loading Details

After countless debugging sessions, I finally discovered the first fatal issue — **weight naming mismatch**.

CosyVoice's pretrained weights use the naming convention `attn.linear_q`, while my code used `attn1.linear_q`. It's just an extra "1", but this tiny difference meant the entire model's Attention weights were never loaded!

```
Pretrained weights: estimator.encoder.encoders.0.self_attn.linear_q.weight
My model:           estimator.encoder.encoders.0.self_attn1.linear_q.weight
                                                        ↑ Extra 1!
```

It's like buying a key only to find the keyhole is off by a tenth of a millimeter — looks almost identical, but just won't open.

#### The Second Devil: The Activation Function Trap

After fixing the naming issue, Loss dropped from 25+ to around 15. Progress! But still not ideal.

Digging deeper, I found the second problem — **activation function mismatch**.

The original Matcha-TTS uses `snakebeta` activation, but CosyVoice's pretrained model uses standard `GELU`. My code was still using `snakebeta`, meaning the model's "thinking pattern" was completely incompatible with the pretrained weights.

```
┌─────────────────────────────────────────────────────────────┐
│           Impact of Activation Function Mismatch            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  What the pretrained model learned:                         │
│  Input x → GELU(x) → certain output pattern                 │
│                                                             │
│  What my model was doing:                                   │
│  Input x → snakebeta(x) → completely different pattern      │
│                                                             │
│  Result: Pretrained weights rendered useless,               │
│          model learning from scratch                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### The Third Hidden Boss: The Missing scale=1000

Just when I thought all problems were solved, Loss remained stuck at 8-12.

This bug was even more subtle: **sinusoidal position encoding missing scale=1000**.

CosyVoice's position encoding implementation has a critical parameter `scale=1000` that expands the position encoding's value range by 1000x. My code missed this parameter, making position information too weak for the model to correctly understand temporal relationships.

It's like GPS positioning suddenly changing from meter-level to kilometer-level accuracy — the model was completely lost.

#### Chapter 1 Summary: Three Keys to Loss Reduction

| Problem | Loss Range | Solution |
|---------|------------|----------|
| Weight naming mismatch | 20-30 | Unify `attn` → `attn1` naming |
| Wrong activation function | 12-20 | `snakebeta` → `GELU` |
| Position encoding missing scale | 8-12 | Add `scale=1000` |

---

### Chapter 2: The Ghostly Semantic Leakage

#### New Dawn, New Problems

After fixing all weight alignment issues, Loss finally started dropping normally!

```
Epoch 1: Loss = 8.2
Epoch 10: Loss = 3.5
Epoch 30: Loss = 1.2
Epoch 50: Loss = 0.6
```

I joyfully ran inference tests, and the result...

```
Reference audio content: "Bears roar and dragons howl by rocky springs,
                          the deep forest shivers and high peaks tremble."
Target text: "Though not an official, I dare speak beyond my station,
             brothers in crisis should discuss openly."
Actual output: "The deep forest shivers and high peaks tremble,
               brothers in crisis should discuss openly."
               ↑ Where did this come from?!
```

**The beginning of the output contained content from the end of the reference audio!**

I named this bizarre phenomenon **"Semantic Leakage"**.

#### Into the Tiger's Den: Four-Gate Circuit Breaker Testing

To find the root cause, I developed a systematic diagnostic methodology.

**Gate 1: Base Model vs Fine-tuned Model Comparison**

I tested both the original CosyVoice and the fine-tuned model with identical inputs:
- Original model: Leakage correlation ~0.88
- Fine-tuned model: Leakage correlation ~0.88

**Shocking**: The leakage wasn't introduced by LoRA — the original model had this problem!

**Gate 2: Semantic Collapse Test**

When I tested with blank reference text, the model output became gibberish.

This meant LoRA had learned a dangerous pattern: **over-reliance on prompt semantic information** rather than focusing on voice characteristics.

**Gate 3: Physical Trimming Test**

I directly trimmed the first 80 frames (about 0.9 seconds) of the output audio, and the leakage disappeared!

This proved the leakage indeed existed at the beginning of the output.

**Gate 4: Token Boundary Analysis**

By analyzing mel spectrogram correlations, I found that Prompt endings and Target beginnings were highly correlated in acoustic features, especially in mid-low frequency bands.

#### Root Cause Revealed: Three Channels of Leakage

After deep-diving into CosyVoice source code, I finally found the fundamental cause of leakage:

```
┌───────────────────────────────────────────────────────────────┐
│            CosyVoice Flow Data Processing Pipeline            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Input: [prompt_mel | target_mel]  ← Concatenated together   │
│           ↓                                                   │
│       Conformer Encoder                                       │
│           ↓                                                   │
│   Internally: prompt and target can "see" each other          │
│           ↓                                                   │
│       ConditionalCFM (Flow Matching)                          │
│           ↓                                                   │
│   Output: Only return target portion                          │
│                                                               │
│   Problem: Target's beginning is already "contaminated"       │
│            with prompt information                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Three Channels of Leakage**:

| Channel | Mechanism | Consequence |
|---------|-----------|-------------|
| Self-Attention | Target can attend to Prompt | Direct semantic transfer |
| Conv1d (kernel=3) | Adjacent frames influence each other | Blurred boundaries |
| Skip Connection | Stores hidden states containing prompt | Information residue |

This is an **architectural issue**, not something simple parameter tuning can solve.

---

### Chapter 3: Six Failed Attempts

After identifying the problem, I began the long journey of fixing it.

#### Attempt 1: Silence Padding

**Idea**: Insert silence between prompt and target to physically isolate them.

**Result**: Failed. The model learned the silence pattern, introducing new artifacts instead.

#### Attempt 2: Dynamic Prompt Length

**Idea**: Randomly vary prompt length so the model can't "memorize" fixed leakage positions.

**Result**: Slight improvement, but leakage persisted.

#### Attempt 3: Prompt Dropout

**Idea**: Randomly drop prompts during training to force the model not to rely on them.

**Result**: At 25% dropout rate, the model began collapsing.

#### Attempt 4: Boundary Loss Weighting

**Idea**: Apply higher loss weight to target beginnings to penalize leakage.

**Result**: Leakage reduced, but audio quality degraded.

#### Attempt 5: Cross-Sample Training

**Idea**: Use mel from different audio as prompt to break semantic continuity.

**Result**: Effective! Leakage significantly reduced, but not completely eliminated.

#### Attempt 6: Text-Side Blinding

**Idea**: Zero out encoder outputs in prompt regions to cut off semantic transfer channels.

**Result**: Combined with cross-sample training, this worked best. But mild residue remained during inference.

#### The Cruel Conclusion

After trying various combinations of six strategies, I had to accept a cruel reality:

> **Within the Zero-Shot framework, semantic leakage cannot be fundamentally solved.**
>
> This is determined by the architecture. As long as prompt and target need to be concatenated, encoded together, and share Attention, leakage is inevitable.

---

### Chapter 4: The Turning Point — Dawn of Joint Training

#### The Essence of the Problem

Returning to the original goal: I wanted the model to learn a specific speaker's **voice** and **recitation style**.

The Zero-Shot mode's design logic:
- **During inference**: Provide reference audio, model extracts voice in real-time
- **During fine-tuning**: Enhance voice learning

But this brought two problems:
1. **Inference requires reference audio** — must provide one every time, inconvenient
2. **Semantic leakage is unavoidable** — determined by architecture

#### A Different Approach

Since Zero-Shot doesn't work, why not try a different way?

**New idea**: Let the model "remember" the target speaker's voice and style, requiring no reference audio during inference.

This requires fine-tuning two modules simultaneously:
- **LLM**: Learn prosody and rhythm (key to recitation style)
- **Flow**: Learn voice characteristics

```
┌───────────────────────────────────────────────────────────────┐
│               Advantages of Joint Training                    │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Zero-Shot (Flow only):                                      │
│   [Needs reference audio] + [Has leakage] + [Voice only]      │
│                                                               │
│   Joint Training (LLM + Flow):                                │
│   [No reference needed] + [No leakage] + [Voice + Style]      │
│                                                               │
│   Key Breakthrough:                                           │
│   LLM's LoRA weights "remember" target speaker's prosody      │
│   Flow's LoRA weights "remember" target speaker's voice       │
│   Use directly during inference, no reference audio needed    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

#### The Ultimate VRAM Challenge

Joint training faced a practical problem: **insufficient VRAM**.

I'm using an RTX 4060 Laptop 8GB. Loading both LLM and Flow models while computing gradients is an extreme operation.

```
VRAM Analysis:
├── LLM (~300M params, FP16): ~600 MB
├── Flow (~300M params, FP16): ~600 MB
├── Activations (batch=1, seq=250): ~2-3 GB
├── Gradients: ~1.2 GB
└── PyTorch overhead: ~1 GB
────────────────────────────────────
Total: ~6-7 GB (just barely fits in 8GB)
```

Through extreme optimization configuration, I successfully achieved joint training on 8GB VRAM.

---

### Chapter 5: A New Trap — LLM Overfitting

#### The "Talking to Itself" Phenomenon

Joint training progressed smoothly, with LLM Loss steadily decreasing:

```
Epoch 10: LLM Loss = 2.8, Flow Loss = 0.6
Epoch 30: LLM Loss = 1.5, Flow Loss = 0.5
Epoch 50: LLM Loss = 0.9, Flow Loss = 0.4
Epoch 70: LLM Loss = 0.5, Flow Loss = 0.4
```

When LLM Loss dropped to 0.5, I ran inference tests...

```
Input text: "Moonlight before my bed, I suspect it's frost on the ground."
Actual output: "Moonlight before my bed, I suspect it's frost on the ground.
               I raise my head to gaze at the moon, then lower it thinking of home.
               A touch of red by the window, mountains layered under moonlight..."
               ↑ Where did all this come from?!
```

The model started **"talking to itself"** — continuing to improvise after finishing the target text, unable to stop.

#### Root Cause Analysis

The LLM was overfitting. When Loss became too low, the model completely "memorized" the semantic patterns of training data, causing it to continue generating according to these patterns during inference.

**Optimal LLM Loss Range**: 1.5 ~ 2.5
- Below 1.5: Overfitting, talks to itself
- Above 2.5: Underfitting, insufficient prosody learning

#### Solution: Threshold Stopping + Regularization

```
┌───────────────────────────────────────────────────────────────┐
│            Measures to Prevent LLM Overfitting                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   1. Set Loss threshold for auto-stopping                     │
│      Stop training when LLM Loss reaches 1.5                  │
│                                                               │
│   2. Increase LoRA Dropout                                    │
│      From 0.05 to 0.15, enhancing regularization              │
│                                                               │
│   3. Reduce LoRA capacity                                     │
│      lora_r from 16 to 8, reducing overfitting risk           │
│                                                               │
│   4. Monitor training curves                                  │
│      Stop promptly if Loss drops too fast                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

### Finale: Achieving Prompt-Free Inference

After all the effort, I finally achieved my original goal — **true prompt-free inference**.

```
Traditional Zero-Shot Inference:
├── Requires reference audio
├── Risk of semantic leakage
└── Must process reference audio every time

Post-Joint-Training Prompt-Free Inference:
├── No reference audio needed
├── No semantic leakage
├── Voice characteristics stored in LoRA weights
└── Prosody style stored in LLM LoRA
```

**Final Results**:

| Dimension | Status |
|-----------|--------|
| Voice matches target speaker | OK |
| Correct recitation style | OK |
| No reference audio needed | OK |
| No semantic leakage | OK |

---

### Exploration Journey Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Technical Exploration Roadmap                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Chapter 1: Wall of Despair (Loss stuck at 20-30)                       │
│    │                                                                    │
│    ├── Weight naming mismatch (attn vs attn1)                           │
│    ├── Wrong activation function (snakebeta vs GELU)                    │
│    └── Position encoding missing scale=1000                             │
│          ↓                                                              │
│  Chapter 2: Semantic Leakage (A ghostly bug)                            │
│    │                                                                    │
│    ├── Four-gate circuit breaker test for root cause                    │
│    └── Discovered it's an architectural issue                           │
│          ↓                                                              │
│  Chapter 3: Six Failed Attempts                                         │
│    │                                                                    │
│    └── Conclusion: Cannot be cured within Zero-Shot framework           │
│          ↓                                                              │
│  Chapter 4: Dawn of Joint Training                                      │
│    │                                                                    │
│    ├── Simultaneous LLM + Flow fine-tuning                              │
│    └── Extreme 8GB VRAM optimization                                    │
│          ↓                                                              │
│  Chapter 5: LLM Overfitting Trap                                        │
│    │                                                                    │
│    └── Solved with Loss threshold + regularization                      │
│          ↓                                                              │
│  Finale: Prompt-Free Inference Achieved                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### To Future Explorers

If you're also trying to fine-tune CosyVoice, I hope this record helps you avoid detours:

1. **Weight alignment is step one** — Check naming, activation functions, position encoding
2. **Semantic leakage is an architectural issue** — Don't waste time in the wrong direction
3. **Joint training is the right path** — Learn voice and style simultaneously
4. **Watch out for LLM overfitting** — Loss 1.5-2.5 is the optimal range
5. **8GB VRAM can do it** — But requires extreme optimization

> *"The debugging process is a conversation with the model. Every error is a clue, every failure is a lesson."*

---

## Quick Start

### Environment Setup

```bash
# Clone the project
git clone https://github.com/YOUR_USERNAME/cosyvoice_flow_finetune.git
cd cosyvoice_flow_finetune/cosyvoice_flow_finetune

# Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Model

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='./pretrained_models/CosyVoice-300M')"
```

### Prepare Data

Place audio files and corresponding text files in the `raw_audio/` directory:

```
raw_audio/
├── 001.wav
├── 001.txt  (text corresponding to the audio)
├── 002.wav
├── 002.txt
└── ...
```

### Joint Training (Recommended)

```bash
# 1. Generate training data
python prepare_joint_data.py

# 2. Start joint training
python train_joint.py --mode joint

# 3. Prompt-free inference
python inference_joint.py --text "Your text here"
```

For detailed documentation, see [cosyvoice_flow_finetune/README.md](cosyvoice_flow_finetune/README.md).

---

## FAQ

<details>
<summary><b>Q1: What to do when Loss is stuck at 20-30?</b></summary>

This is a weight loading issue. Check these three points:

1. **Weight naming**: Ensure `attn` / `attn1` naming is consistent
2. **Activation function**: Ensure using `GELU` instead of `snakebeta`
3. **Position encoding**: Ensure `SinusoidalPosEmb` includes `scale=1000`

</details>

<details>
<summary><b>Q2: What is "Semantic Leakage"?</b></summary>

After fine-tuning, the model's output audio begins with semantic content from the end of the reference audio.

This is an inherent issue in CosyVoice Flow's architecture, because prompt and target mel features are concatenated and encoded together, and the Self-Attention mechanism allows information to transfer across boundaries.

**Solution**: Use LLM + Flow joint training to achieve prompt-free inference.

</details>

<details>
<summary><b>Q3: What to do when the model "talks to itself"?</b></summary>

This is a symptom of LLM overfitting. When LLM Loss drops below 1.5, the model "memorizes" the semantic patterns of training data.

**Solutions**:
- Set `llm_loss_threshold = 1.5` for auto-stopping
- Increase `lora_dropout` to 0.15
- Use checkpoints with Loss in the 1.5-2.0 range

</details>

<details>
<summary><b>Q4: What to do about CUDA out of memory?</b></summary>

Joint training has high VRAM requirements. Optimization configuration for 8GB VRAM:

- `batch_size`: 1
- `accumulate_grad_batches`: 16
- `max_feat_len`: 250 (about 2.9 seconds)
- `lora_r`: 8
- `precision`: '16-mixed'

If still OOM, train in stages: first `--mode flow_only`, then `--mode llm_only`.

</details>

---

## Visual Guide

> CosyVoice Flow Fine-tuning Debugging Walkthrough — A Systematic Troubleshooting Framework from Loss 30+ to Perfect Convergence

<p align="center">
  <img src="docs/guide_images/page_01.png" alt="Cover" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_02.png" alt="Debugging Level Map" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_03.png" alt="Level 0: Data is the Foundation" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_04.png" alt="Level 1: Conquering the Loss 20-30 Wall of Despair" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_05.png" alt="Key to Level 1: Activation Functions Must Align" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_06.png" alt="Level 1 Checklist: Devil in Structural Alignment Details" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_07.png" alt="Level 2: Crossing the Loss 8-12 Convergence Plateau" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_08.png" alt="Plateau Breakthrough: The Forgotten scale=1000" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_09.png" alt="Level 3: Conquering the Final Loss Barrier" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_10.png" alt="Hidden Level: Beware the Ghostly Semantic Leakage" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_11.png" alt="Cutting Off Leakage: Two Core Training Strategies" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_12.png" alt="Ultimate Technique: Single-Batch Overfitting for Quick Problem Location" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_13.png" alt="CosyVoice Flow Fine-tuning Complete Overview" width="100%"/>
</p>

---

<p align="center">
  <img src="docs/guide_images/page_14.png" alt="Epilogue" width="100%"/>
</p>

---

## Project Structure

```
cosyvoice_flow_finetune/
├── README.md                        # This file (Chinese)
├── README_EN.md                     # This file (English)
├── big_pic.png                      # Project overview diagram
├── CosyVoice案例：调试下一代AI语音.mp4  # Debugging documentary video
├── NotebookLM Mind Map.png          # Knowledge graph
├── docs/guide_images/               # Visual guide images
│
└── cosyvoice_flow_finetune/         # Core code
    ├── config.py                    # Configuration file
    ├── train_joint.py               # Joint training script
    ├── inference_joint.py           # Prompt-free inference
    ├── prepare_joint_data.py        # Data preparation
    ├── llm_flow_model.py            # Joint model definition
    ├── merge_joint_weights.py       # Weight merging
    ├── lora.py                      # LoRA module
    ├── dataset.py                   # Dataset
    ├── utils.py                     # Utility functions
    └── cosyvoice/                   # CosyVoice core code
```

---

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Alibaba FunAudioLLM Team
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow Matching TTS
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation

---

## License

This project follows the [MIT License](cosyvoice_flow_finetune/LICENSE), while respecting the original licenses of CosyVoice and Matcha-TTS.

---

<p align="center">
  <strong>If this project helps you, please give it a Star!</strong>
</p>
