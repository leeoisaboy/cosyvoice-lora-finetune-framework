# Copyright (c) 2024
# LLM + Flow 联合微调模块
#
# 为什么需要联合微调：
# 1. LLM 负责将文本转换为 speech tokens（决定韵律、节奏、吟诵风格）
# 2. Flow 负责将 speech tokens 转换为 mel spectrogram（决定音色）
# 3. 只微调 Flow 无法学习吟诵风格；只微调 LLM 无法学习音色
# 4. 联合微调可以同时学习音色和吟诵风格，实现无 prompt 推理

import os
import sys
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加 CosyVoice 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, COSYVOICE_ROOT)

from cosyvoice.utils.common import IGNORE_ID

from config import (
    PRETRAINED_MODEL_DIR, MEL_MEAN, MEL_STD,
    LORA_CONFIG, JOINT_TRAINING_CONFIG
)
from flow_model import build_flow_model
from lora import apply_lora_to_model


class JointLLMFlowModel(nn.Module):
    """
    LLM + Flow 联合微调模型

    架构：
    - LLM: TransformerLM（负责 text → speech tokens）
    - Flow: MaskedDiffWithXvec（负责 speech tokens → mel）

    训练模式：
    1. joint: 同时训练 LLM 和 Flow（推荐）
    2. llm_only: 只训练 LLM
    3. flow_only: 只训练 Flow（等价于原来的 train.py）

    无 Prompt 训练：
    - 不使用 prompt speech tokens，LLM 只根据文本生成 speech tokens
    - 不使用 prompt mel，Flow 只根据 speaker embedding 和 speech tokens 生成 mel
    - 推理时不需要参考音频
    """

    def __init__(
        self,
        llm: nn.Module,
        flow: nn.Module,
        training_mode: str = 'joint',  # 'joint', 'llm_only', 'flow_only'
        llm_loss_weight: float = 1.0,
        flow_loss_weight: float = 1.0,
        no_prompt_training: bool = True,  # 无 prompt 训练
    ):
        super().__init__()
        self.llm = llm
        self.flow = flow
        self.training_mode = training_mode
        self.llm_loss_weight = llm_loss_weight
        self.flow_loss_weight = flow_loss_weight
        self.no_prompt_training = no_prompt_training

        # Mel 归一化参数
        self.mel_mean = MEL_MEAN
        self.mel_std = MEL_STD

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """将 log mel 归一化到 N(0, 1)"""
        return (mel - self.mel_mean) / self.mel_std

    def forward(self, batch: dict, device: torch.device) -> Dict[str, Any]:
        """
        联合训练 forward pass

        无 Prompt 训练模式：
        - LLM: 不使用 prompt_speech_token，直接从文本生成 speech tokens
        - Flow: 不使用 prompt mel，只使用 speaker embedding
        """
        losses = {}

        # ========== LLM 训练 ==========
        if self.training_mode in ['joint', 'llm_only']:
            llm_result = self._forward_llm(batch, device)
            losses['llm_loss'] = llm_result['loss'] * self.llm_loss_weight
            if 'acc' in llm_result:
                losses['llm_acc'] = llm_result['acc']

        # ========== Flow 训练 ==========
        if self.training_mode in ['joint', 'flow_only']:
            flow_result = self._forward_flow(batch, device)
            losses['flow_loss'] = flow_result['loss'] * self.flow_loss_weight

        # 计算总 loss
        if self.training_mode == 'joint':
            losses['loss'] = losses['llm_loss'] + losses['flow_loss']
        elif self.training_mode == 'llm_only':
            losses['loss'] = losses['llm_loss']
        else:  # flow_only
            losses['loss'] = losses['flow_loss']

        return losses

    def _forward_llm(self, batch: dict, device: torch.device) -> Dict[str, Any]:
        """
        LLM forward pass - 无 Prompt 模式

        在无 Prompt 模式下：
        - 不使用 prompt_text 和 prompt_speech_token
        - LLM 只根据目标文本生成 speech tokens
        - 这样训练后，推理时不需要参考音频
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 获取 LLM 的数据类型
        dtype = next(self.llm.parameters()).dtype

        # 准备 LLM target（与原版相同）
        # target: [IGNORE, IGNORE, ..., speech_token_1, speech_token_2, ..., EOS]
        lm_target = [
            torch.tensor(
                [IGNORE_ID] * (2 + text_token_len[i]) +  # SOS + embedding + text
                speech_token[i, :speech_token_len[i]].tolist() +
                [self.llm.speech_token_size]  # EOS
            )
            for i in range(text_token.size(0))
        ]
        lm_target = torch.nn.utils.rnn.pad_sequence(
            lm_target, batch_first=True, padding_value=IGNORE_ID
        ).to(device)

        # 编码文本
        text_emb = self.llm.text_embedding(text_token)
        text_emb, text_emb_len = self.llm.encode(text_emb, text_token_len)

        # Speaker embedding
        embedding = F.normalize(embedding.to(dtype), dim=1)
        embedding = self.llm.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # SOS 和 Task ID embedding
        sos_eos_emb = self.llm.llm_embedding.weight[self.llm.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm.llm_embedding.weight[self.llm.task_id].reshape(1, 1, -1)

        # Speech token embedding
        speech_emb = self.llm.speech_embedding(speech_token)

        # 构建 LLM 输入（无 prompt）
        # 格式: [SOS, speaker_embedding, text, TASK_ID, speech_tokens]
        lm_input, lm_input_len = self.llm.pad_unpad_sequence(
            sos_eos_emb, embedding, text_emb, text_emb_len,
            task_id_emb, speech_emb, speech_token_len
        )

        # 运行 LLM
        lm_output, _ = self.llm.llm(lm_input, lm_input_len.to(device))
        logits = self.llm.llm_decoder(lm_output)

        # 计算 loss
        loss = self.llm.criterion_ce(logits, lm_target)

        # 计算准确率
        from cosyvoice.utils.common import th_accuracy
        acc = th_accuracy(
            logits.view(-1, self.llm.speech_token_size + 1),
            lm_target,
            ignore_label=IGNORE_ID
        )

        return {'loss': loss, 'acc': acc}

    def _forward_flow(self, batch: dict, device: torch.device) -> Dict[str, Any]:
        """
        Flow forward pass - 无 Prompt 模式

        在无 Prompt 模式下：
        - conditioning 全为零（不使用 prompt mel）
        - 模型只依赖 speaker embedding 和 text encoder 输出
        """
        # 获取模型精度
        dtype = self.flow.input_embedding.weight.dtype

        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device).to(dtype)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device).to(dtype)

        # Mel 归一化
        feat = self.normalize_mel(feat)

        # Speaker embedding
        embedding = F.normalize(embedding, dim=1)
        embedding = self.flow.spk_embed_affine_layer(embedding)

        # Token embedding
        from utils import make_pad_mask
        mask = (~make_pad_mask(token_len)).to(dtype).unsqueeze(-1).to(device)
        token_emb = self.flow.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        h, _ = self.flow.encoder(token_emb, token_len)
        h = self.flow.encoder_proj(h)
        h, _ = self.flow.length_regulator(h, feat_len)

        # 无 Prompt 模式：conditioning 全为零
        conds = torch.zeros(feat.shape, device=device, dtype=dtype)
        conds = conds.transpose(1, 2)

        # 计算 loss（使用原始 CosyVoice 的 compute_loss 接口）
        loss_mask = (~make_pad_mask(feat_len)).to(h)
        loss, _ = self.flow.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            loss_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
        )

        return {'loss': loss}


def build_joint_model(
    pretrained_path: str = PRETRAINED_MODEL_DIR,
    device: str = 'cuda',
    training_mode: str = 'joint',
    llm_lora_config: Optional[dict] = None,
    flow_lora_config: Optional[dict] = None,
) -> JointLLMFlowModel:
    """
    构建联合训练模型

    Args:
        pretrained_path: 预训练模型路径
        device: 设备
        training_mode: 训练模式 ('joint', 'llm_only', 'flow_only')
        llm_lora_config: LLM LoRA 配置
        flow_lora_config: Flow LoRA 配置

    Returns:
        JointLLMFlowModel 实例
    """
    # 加载预训练的 CosyVoice 模型
    from cosyvoice.cli.cosyvoice import CosyVoice

    print(f"[Joint] 加载预训练模型: {pretrained_path}")
    cosyvoice = CosyVoice(pretrained_path, load_jit=False, load_trt=False)

    llm = cosyvoice.model.llm
    flow = cosyvoice.model.flow

    # 应用 LoRA
    if training_mode in ['joint', 'llm_only'] and llm_lora_config:
        print(f"\n[Joint] 对 LLM 应用 LoRA...")
        llm_stats = apply_lora_to_model(
            llm,
            r=llm_lora_config.get('lora_r', 8),
            lora_alpha=llm_lora_config.get('lora_alpha', 16),
            lora_dropout=llm_lora_config.get('lora_dropout', 0.05),
            target_modules=llm_lora_config.get('target_modules', [
                'linear_q', 'linear_k', 'linear_v', 'linear_out',
                'w_1', 'w_2',
            ]),
        )
        print(f"  LLM LoRA: {llm_stats['replaced_layers']} 层, "
              f"{llm_stats['trainable_params']:,} 参数 ({llm_stats['trainable_ratio']:.2f}%)")

    if training_mode in ['joint', 'flow_only'] and flow_lora_config:
        print(f"\n[Joint] 对 Flow 应用 LoRA...")
        flow_stats = apply_lora_to_model(
            flow,
            r=flow_lora_config.get('lora_r', 16),
            lora_alpha=flow_lora_config.get('lora_alpha', 16),
            lora_dropout=flow_lora_config.get('lora_dropout', 0.05),
            target_modules=flow_lora_config.get('target_modules', [
                'to_q', 'to_k', 'to_v', 'linear_q', 'linear_k', 'linear_v',
                'linear_out', 'w_1', 'w_2',
            ]),
        )
        print(f"  Flow LoRA: {flow_stats['replaced_layers']} 层, "
              f"{flow_stats['trainable_params']:,} 参数 ({flow_stats['trainable_ratio']:.2f}%)")

    # 创建联合模型
    joint_config = JOINT_TRAINING_CONFIG
    model = JointLLMFlowModel(
        llm=llm,
        flow=flow,
        training_mode=training_mode,
        llm_loss_weight=joint_config.get('llm_loss_weight', 1.0),
        flow_loss_weight=joint_config.get('flow_loss_weight', 1.0),
        no_prompt_training=joint_config.get('no_prompt_training', True),
    )

    model = model.to(device)

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Joint] 总参数: {total:,}, 可训练: {trainable:,} ({trainable/total*100:.2f}%)")

    return model


def get_joint_merged_state_dict(model: JointLLMFlowModel) -> Dict[str, dict]:
    """
    获取合并 LoRA 后的 LLM 和 Flow state_dict

    Returns:
        {'llm': llm_state_dict, 'flow': flow_state_dict}
    """
    from lora import get_merged_state_dict

    result = {}

    # 检查 LLM 是否有 LoRA
    llm_has_lora = any('lora_' in name for name, _ in model.llm.named_parameters())
    if llm_has_lora:
        print("[Joint] 合并 LLM LoRA 权重...")
        result['llm'] = get_merged_state_dict(model.llm)

    # 检查 Flow 是否有 LoRA
    flow_has_lora = any('lora_' in name for name, _ in model.flow.named_parameters())
    if flow_has_lora:
        print("[Joint] 合并 Flow LoRA 权重...")
        result['flow'] = get_merged_state_dict(model.flow)

    return result
