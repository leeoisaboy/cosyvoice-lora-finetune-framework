# Copyright (c) 2024 Alibaba Inc
# Self-contained Flow model for fine-tuning
# No external CosyVoice/Matcha-TTS dependencies

import random
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_pad_mask
from modules import ConformerEncoder, InterpolateRegulator, ConditionalDecoder

# 导入语义泄漏防护配置
try:
    from config import ANTI_LEAKAGE_CONFIG, MEL_MEAN, MEL_STD, NO_PROMPT_TRAINING_CONFIG
except ImportError:
    # 默认配置
    ANTI_LEAKAGE_CONFIG = {
        'silence_padding_enabled': False,
        'silence_token_id': 0,
        'silence_min_tokens': 5,
        'silence_max_tokens': 10,
        'dynamic_prompt_enabled': True,
        'prompt_min_ratio': 0.10,
        'prompt_max_ratio': 0.30,
        'prompt_dropout_enabled': True,
        'prompt_dropout_prob': 0.10,
        'boundary_loss_enabled': True,
        'boundary_frames': 15,
        'boundary_loss_weight': 3.0,
        'cross_sample_enabled': True,
        'cross_sample_prob': 0.5,
        'text_blinding_enabled': True,
        'text_blinding_prob': 0.7,
        'text_blinding_mode': 'zero',
        'silence_mel_value': -11.5,
    }
    MEL_MEAN = -6.0
    MEL_STD = 2.0
    NO_PROMPT_TRAINING_CONFIG = {
        'enabled': False,
        'mode': 'full',
        'no_prompt_ratio': 0.8,
        'use_mean_embedding': False,
    }


class ConditionalCFM(nn.Module):
    """Conditional Flow Matching module"""

    def __init__(
        self,
        in_channels: int,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
        sigma_min: float = 1e-6,
        t_scheduler: str = 'cosine',
        training_cfg_rate: float = 0.2,
        inference_cfg_rate: float = 0.7,
        estimator: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=None):
        """Forward diffusion (inference)"""
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature

        if cache is not None and cache.shape[2] != 0:
            cache_size = cache.shape[2]
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]

        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2) if prompt_len > 0 else z[:, :, -34:]
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2) if prompt_len > 0 else mu[:, :, -34:]
        new_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * 3.14159265359)

        return self.solve_euler(z, t_span, mu, mask, spks, cond), new_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """Euler ODE solver"""
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)  # 80 is correct (after affine projection)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond

            assert self.estimator is not None
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, prompt_lens=None):
        """
        Compute flow matching loss with prompt region masking and Prompt Isolation.

        【重要修复】
        1. 将 prompt 区域的 Loss 权重设为 0，防止模型学习"复读" prompt
        2. 在 Attention 计算时使用 Prompt Isolation Mask，阻止 target 看到 prompt

        Args:
            x1: target mel spectrogram (B, 80, T)
            mask: padding mask (B, 1, T)
            mu: encoder output (B, 80, T)
            spks: speaker embedding (B, 80)
            cond: conditioning (B, 80, T)
            prompt_lens: list of prompt lengths for each sample, used for loss masking
        """
        b, _, _ = mu.shape

        # Random timestep
        t_step = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_step = 1 - torch.cos(t_step * 0.5 * 3.14159265359)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate
        y = (1 - (1 - self.sigma_min) * t_step) * z + t_step * x1
        u = x1 - (1 - self.sigma_min) * z

        # CFG dropout
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        # ========== 【Prompt Isolation】设置 prompt_len 用于 Attention 隔离 ==========
        # 使用 batch 中最大的 prompt_len（简化实现）
        assert self.estimator is not None
        if prompt_lens is not None and len(prompt_lens) > 0:
            max_prompt_len = max(prompt_lens)
            self.estimator.prompt_isolation_len = max_prompt_len
            self.estimator.prompt_isolation_enabled = True
        else:
            self.estimator.prompt_isolation_len = 0

        pred = self.estimator(y, mask, mu, t_step.view(b), spks, cond)

        # 重置 prompt_isolation_len
        self.estimator.prompt_isolation_len = 0

        # ========== 【核心修复】创建 Loss 掩码，屏蔽 prompt 区域 ==========
        loss_mask = mask.clone()

        if prompt_lens is not None:
            boundary_frames = ANTI_LEAKAGE_CONFIG.get('boundary_frames', 15)
            boundary_weight = ANTI_LEAKAGE_CONFIG.get('boundary_loss_weight', 3.0)

            for i, prompt_len in enumerate(prompt_lens):
                if prompt_len > 0:
                    # 【关键】将 prompt 区域的权重设为 0
                    loss_mask[i, :, :prompt_len] = 0

                    # 在边界区域施加更高权重
                    if ANTI_LEAKAGE_CONFIG.get('boundary_loss_enabled', True):
                        end_idx = min(prompt_len + boundary_frames, loss_mask.shape[2])
                        loss_mask[i, :, prompt_len:end_idx] = boundary_weight

        # 计算带掩码的 Loss
        diff = (pred - u) * loss_mask
        valid_elements = torch.sum(loss_mask) * u.shape[1]
        if valid_elements > 0:
            loss = (diff ** 2).sum() / valid_elements
        else:
            loss = torch.tensor(0.0, device=mu.device, requires_grad=True)

        return loss, y


class MaskedDiffWithXvec(nn.Module):
    """Main Flow model with speaker embedding"""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        encoder: Optional[nn.Module] = None,
        length_regulator: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        self.encoder = encoder
        assert self.encoder is not None
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator

        # Mel 归一化参数（从 config 导入）
        self.mel_mean = MEL_MEAN
        self.mel_std = MEL_STD

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """将 log mel 归一化到 N(0, 1)"""
        return (mel - self.mel_mean) / self.mel_std

    def denormalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """将归一化的 mel 还原"""
        return mel * self.mel_std + self.mel_mean

    def forward(self, batch: dict, device: torch.device) -> Dict[str, Any]:
        """
        Training forward pass with anti-leakage strategies.

        支持两种训练模式：
        1. 标准模式（带 prompt）：使用语义泄漏防护策略
        2. 无 prompt 模式：完全不使用 prompt conditioning，仅依赖 speaker embedding

        语义泄漏防护策略（标准模式）：
        1. 静音隔离带 (Silence Padding): 在 prompt 和 target 之间插入静音
        2. 动态 Prompt 长度: 随机选择不同比例的 prompt
        3. Prompt Dropout: 一定概率完全丢弃 prompt
        4. 边界 Loss 权重: 在边界区域施加更高的 loss 权重
        5. 跨样本训练 (Cross-Sample Prompting): 使用不同音频的 mel 作为 prompt
        6. 文本侧致盲 (Text Blinding): 在 prompt 区域将 encoder 输出置零
        """
        # 获取模型当前精度
        dtype = self.input_embedding.weight.dtype

        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device).to(dtype)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device).to(dtype)

        # 在线归一化 mel
        feat = self.normalize_mel(feat)

        # ========== 检查是否启用无 prompt 训练模式 ==========
        no_prompt_mode = NO_PROMPT_TRAINING_CONFIG.get('enabled', False)

        if no_prompt_mode:
            # ========== 无 Prompt 训练模式 ==========
            return self._forward_no_prompt(token, token_len, feat, feat_len, embedding, device, dtype)

        # ========== 标准模式（带 prompt）==========
        # 策略5: 跨样本训练 - 获取跨样本 mel
        cross_sample_mel = None
        cross_sample_mel_len = None
        if 'cross_sample_mel' in batch:
            cross_sample_mel = batch['cross_sample_mel'].to(device).to(dtype)
            cross_sample_mel = self.normalize_mel(cross_sample_mel)
            cross_sample_mel_len = batch.get('cross_sample_mel_len', None)
            if cross_sample_mel_len is not None:
                cross_sample_mel_len = cross_sample_mel_len.to(device)

        # Speaker embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Token embedding
        mask = (~make_pad_mask(token_len)).to(dtype).unsqueeze(-1).to(device)
        token_emb = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        assert self.encoder is not None
        h, h_lengths = self.encoder(token_emb, token_len)
        h = self.encoder_proj(h)
        assert self.length_regulator is not None
        h, h_lengths = self.length_regulator(h, feat_len)

        # ========== 语义泄漏防护策略配置 ==========
        silence_enabled = ANTI_LEAKAGE_CONFIG.get('silence_padding_enabled', False)
        dynamic_enabled = ANTI_LEAKAGE_CONFIG.get('dynamic_prompt_enabled', True)
        dropout_enabled = ANTI_LEAKAGE_CONFIG.get('prompt_dropout_enabled', True)
        text_blinding_enabled = ANTI_LEAKAGE_CONFIG.get('text_blinding_enabled', True)
        cross_sample_enabled = ANTI_LEAKAGE_CONFIG.get('cross_sample_enabled', True)

        prompt_min_ratio = ANTI_LEAKAGE_CONFIG.get('prompt_min_ratio', 0.10)
        prompt_max_ratio = ANTI_LEAKAGE_CONFIG.get('prompt_max_ratio', 0.30)
        dropout_prob = ANTI_LEAKAGE_CONFIG.get('prompt_dropout_prob', 0.10)
        text_blinding_prob = ANTI_LEAKAGE_CONFIG.get('text_blinding_prob', 0.7)

        silence_min = ANTI_LEAKAGE_CONFIG.get('silence_min_tokens', 5)
        silence_max = ANTI_LEAKAGE_CONFIG.get('silence_max_tokens', 10)
        silence_mel_value = ANTI_LEAKAGE_CONFIG.get('silence_mel_value', -11.5)
        # 归一化后的静音值
        silence_mel_normalized = (silence_mel_value - self.mel_mean) / self.mel_std

        # 构建条件和记录 prompt 长度
        conds = torch.zeros(feat.shape, device=device, dtype=dtype)
        prompt_lens = []

        for i, j in enumerate(feat_len):
            j = int(j.item())

            # ========== 策略3: Prompt Dropout ==========
            if dropout_enabled and random.random() < dropout_prob:
                prompt_lens.append(0)
                continue

            # ========== 策略2: 动态 Prompt 长度 ==========
            if dynamic_enabled:
                min_idx = max(1, int(prompt_min_ratio * j))
                max_idx = max(min_idx + 1, int(prompt_max_ratio * j))
                prompt_len = random.randint(min_idx, max_idx)
            else:
                prompt_len = max(1, int(0.3 * j))

            # ========== 策略5: 跨样本训练 - 选择 prompt 来源 ==========
            use_cross_sample = False
            if (cross_sample_enabled
                and cross_sample_mel is not None
                and cross_sample_mel_len is not None
                and cross_sample_mel_len[i].item() > 0):
                use_cross_sample = True
                cross_mel_len = int(cross_sample_mel_len[i].item())
                prompt_len = min(prompt_len, cross_mel_len)

            # ========== 策略1: 静音隔离带 ==========
            if silence_enabled:
                silence_tokens = random.randint(silence_min, silence_max)
                silence_mel_frames = int(silence_tokens * 22050 / 256 / self.input_frame_rate)
                silence_mel_frames = max(3, min(silence_mel_frames, 20))

                if prompt_len + silence_mel_frames < j:
                    if use_cross_sample and cross_sample_mel is not None:
                        conds[i, :prompt_len] = cross_sample_mel[i, :prompt_len]
                    else:
                        conds[i, :prompt_len] = feat[i, :prompt_len]
                    conds[i, prompt_len:prompt_len + silence_mel_frames] = silence_mel_normalized
                    prompt_lens.append(prompt_len + silence_mel_frames)
                else:
                    if use_cross_sample and cross_sample_mel is not None:
                        conds[i, :prompt_len] = cross_sample_mel[i, :prompt_len]
                    else:
                        conds[i, :prompt_len] = feat[i, :prompt_len]
                    prompt_lens.append(prompt_len)
            else:
                if use_cross_sample and cross_sample_mel is not None:
                    conds[i, :prompt_len] = cross_sample_mel[i, :prompt_len]
                else:
                    conds[i, :prompt_len] = feat[i, :prompt_len]
                prompt_lens.append(prompt_len)

            # ========== 策略6: 文本侧致盲 ==========
            if text_blinding_enabled and random.random() < text_blinding_prob:
                h[i, :prompt_len, :] = 0.0

        conds = conds.transpose(1, 2)

        # Compute loss（传入 prompt_lens 用于边界 loss 权重和 prompt 区域掩码）
        mask = (~make_pad_mask(feat_len)).to(h)
        assert self.decoder is not None
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            prompt_lens=prompt_lens
        )
        return {'loss': loss}

    def _forward_no_prompt(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        feat: torch.Tensor,
        feat_len: torch.Tensor,
        embedding: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[str, Any]:
        """
        无 Prompt 训练模式的 forward pass

        在此模式下：
        - conditioning 始终为全零（无 prompt 信息）
        - 模型学习仅依赖 speaker embedding 和文本生成语音
        - 推理时可以完全不需要参考音频
        """
        mode = NO_PROMPT_TRAINING_CONFIG.get('mode', 'full')
        no_prompt_ratio = NO_PROMPT_TRAINING_CONFIG.get('no_prompt_ratio', 0.8)

        # Speaker embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Token embedding
        mask = (~make_pad_mask(token_len)).to(dtype).unsqueeze(-1).to(device)
        token_emb = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        assert self.encoder is not None
        h, _ = self.encoder(token_emb, token_len)
        h = self.encoder_proj(h)
        assert self.length_regulator is not None
        h, _ = self.length_regulator(h, feat_len)

        # ========== 无 Prompt 模式：构建全零 conditioning ==========
        if mode == 'full':
            # 100% 无 prompt
            conds = torch.zeros(feat.shape, device=device, dtype=dtype)
            prompt_lens = [0] * feat.shape[0]
        else:  # mixed 模式
            conds = torch.zeros(feat.shape, device=device, dtype=dtype)
            prompt_lens = []

            for i, j in enumerate(feat_len):
                j = int(j.item())

                if random.random() < no_prompt_ratio:
                    # 无 prompt
                    prompt_lens.append(0)
                else:
                    # 有 prompt（使用少量 prompt）
                    prompt_len = random.randint(1, max(2, int(0.1 * j)))
                    conds[i, :prompt_len] = feat[i, :prompt_len]
                    prompt_lens.append(prompt_len)

        conds = conds.transpose(1, 2)

        # Compute loss
        # 注意：无 prompt 模式下，所有帧都参与 loss 计算（没有 prompt 区域需要屏蔽）
        loss_mask = (~make_pad_mask(feat_len)).to(h)
        assert self.decoder is not None
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            loss_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            prompt_lens=prompt_lens  # 全为 0，不会屏蔽任何区域
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        flow_cache=None,
    ):
        """Inference forward pass

        注意：此方法不做归一化处理，与 CosyVoice 原版保持一致。
        如果使用微调后的权重，调用方需要自行处理归一化（见 quick_inference.py 的 patch）
        """
        assert token.shape[0] == 1

        # Speaker embedding
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Concat prompt and target tokens
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        assert self.encoder is not None
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)

        assert self.length_regulator is not None
        # 使用 length_regulator.inference 方法（如果存在）
        if hasattr(self.length_regulator, 'inference'):
            h, _ = self.length_regulator.inference(
                h[:, :token_len1], h[:, token_len1:],
                mel_len1, mel_len2, self.input_frame_rate
            )
        else:
            h, _ = self.length_regulator(h, torch.tensor([mel_len1 + mel_len2], device=token.device))

        # Conditioning
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        # ========== 动态调整 ODE 步数 ==========
        # 长序列需要更多步数才能让 Flow Matching 收敛
        # 短序列 (<300 帧): 10 步
        # 中等序列 (300-500 帧): 15 步
        # 长序列 (>500 帧): 20 步
        total_mel_len = mel_len1 + mel_len2
        if total_mel_len > 500:
            n_timesteps = 20
        elif total_mel_len > 300:
            n_timesteps = 15
        else:
            n_timesteps = 10

        # Decode
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2], device=token.device))).to(h)
        assert self.decoder is not None
        feat, new_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            prompt_len=mel_len1,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        return feat.float(), new_cache

    @torch.inference_mode()
    def inference_like_training(
        self,
        token,
        token_len,
        feat_len,
        embedding,
        prompt_feat=None,
        prompt_len=0,
        n_timesteps=10,
    ):
        """
        Inference that matches training logic exactly.

        Key differences from inference():
        - Uses complete token sequence (no splitting into prompt/target)
        - prompt_feat is optional (like training where some have no prompt)
        - prompt_len should be 0~30% of feat_len (matching training)

        Args:
            token: (1, L) complete speech token sequence
            token_len: (1,) token length
            feat_len: (1,) or int, target mel frame count
            embedding: (1, 192) speaker embedding
            prompt_feat: (1, prompt_len, 80) optional prompt mel, can be None
            prompt_len: int, number of prompt frames (should be 0~30% of feat_len)
            n_timesteps: ODE solver steps

        Returns:
            feat: (1, 80, feat_len) generated mel spectrogram (full, including prompt region)
        """
        assert token.shape[0] == 1

        if isinstance(feat_len, torch.Tensor):
            feat_len_val = int(feat_len.item())
        else:
            feat_len_val = int(feat_len)

        # Speaker embedding
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Token embedding (complete sequence, no splitting)
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token_emb = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        assert self.encoder is not None
        h, h_lengths = self.encoder(token_emb, token_len)
        h = self.encoder_proj(h)

        # Length regulation to target mel length
        assert self.length_regulator is not None
        h, h_lengths = self.length_regulator(h, torch.tensor([feat_len_val], device=token.device))

        # Build conditioning (matching training logic)
        conds = torch.zeros([1, feat_len_val, self.output_size], device=token.device, dtype=h.dtype)
        if prompt_feat is not None and prompt_len > 0:
            actual_prompt_len = min(prompt_len, prompt_feat.shape[1], feat_len_val)
            conds[:, :actual_prompt_len] = prompt_feat[:, :actual_prompt_len]
        conds = conds.transpose(1, 2)  # (1, 80, T)

        # ========== 动态调整 ODE 步数 ==========
        # 与 inference() 保持一致
        if n_timesteps is None or n_timesteps == 10:
            if feat_len_val > 500:
                n_timesteps = 20
            elif feat_len_val > 300:
                n_timesteps = 15
            else:
                n_timesteps = 10

        # Decode
        mask = torch.ones([1, 1, feat_len_val], device=token.device, dtype=h.dtype)
        assert self.decoder is not None
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            prompt_len=prompt_len if prompt_feat is not None else 0,
            cache=None
        )

        return feat.float()


def build_flow_model(
    pretrained_path: Optional[str] = None,
    device: str = 'cuda',
    # Model architecture params
    input_size: int = 512,
    output_size: int = 80,
    spk_embed_dim: int = 192,
    vocab_size: int = 4096,
    # Encoder params
    encoder_attention_heads: int = 8,
    encoder_linear_units: int = 2048,
    encoder_num_blocks: int = 6,
    # Decoder params - MUST match pretrained weights exactly!
    decoder_channels: tuple = (256, 256),
    decoder_attention_head_dim: int = 64,
    decoder_n_blocks: int = 4,        # 4 transformer blocks per position (matches pretrained)
    decoder_num_mid_blocks: int = 12,  # 12 mid blocks (matches pretrained)
    decoder_num_heads: int = 8,
) -> MaskedDiffWithXvec:
    """Build and optionally load pretrained Flow model"""

    # Build encoder (matches CosyVoice-300M config)
    encoder = ConformerEncoder(
        input_size=input_size,
        output_size=input_size,
        attention_heads=encoder_attention_heads,
        linear_units=encoder_linear_units,
        num_blocks=encoder_num_blocks,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        cnn_module_kernel=15,
        use_cnn_module=False,
        macaron_style=False,
        causal=False,
    )

    # Build length regulator
    length_regulator = InterpolateRegulator(
        channels=output_size,
        sampling_ratios=(1, 1, 1, 1),
        out_channels=output_size,
        groups=1,
    )

    # Build decoder estimator
    # in_channels = 80*4 (x + mu + spk + cond)
    estimator = ConditionalDecoder(
        in_channels=320,  # 80*4 (x + mu + spk + cond)
        out_channels=80,
        channels=decoder_channels,
        dropout=0.0,
        attention_head_dim=decoder_attention_head_dim,
        n_blocks=decoder_n_blocks,
        num_mid_blocks=decoder_num_mid_blocks,
        num_heads=decoder_num_heads,
        act_fn='gelu',  # MUST be 'gelu' to match CosyVoice-300M pretrained weights!
    )

    # Build CFM decoder
    decoder = ConditionalCFM(
        in_channels=output_size,
        n_spks=1,
        spk_emb_dim=output_size,
        sigma_min=1e-6,
        t_scheduler='cosine',
        training_cfg_rate=0.2,
        inference_cfg_rate=0.7,
        estimator=estimator,
    )

    # Build full model
    model = MaskedDiffWithXvec(
        input_size=input_size,
        output_size=output_size,
        spk_embed_dim=spk_embed_dim,
        vocab_size=vocab_size,
        input_frame_rate=50,
        encoder=encoder,
        length_regulator=length_regulator,
        decoder=decoder,
    )

    # Load pretrained weights
    if pretrained_path is not None:
        import os
        weight_file = os.path.join(pretrained_path, 'flow.pt') if os.path.isdir(pretrained_path) else pretrained_path

        if os.path.exists(weight_file):
            print(f"Loading pretrained weights from: {weight_file}")
            state_dict = torch.load(weight_file, map_location='cpu')

            try:
                model.load_state_dict(state_dict, strict=True)
                print("Weights loaded successfully (strict=True)")
            except Exception as e:
                print(f"Strict loading failed: {e}")
                print("Attempting partial loading...")

                model_dict = model.state_dict()
                matched = {}
                unmatched = []

                for k, v in state_dict.items():
                    if k in model_dict:
                        if model_dict[k].shape == v.shape:
                            matched[k] = v
                        else:
                            unmatched.append(f"{k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
                    else:
                        unmatched.append(f"{k}: not in model")

                model_dict.update(matched)
                model.load_state_dict(model_dict, strict=False)
                print(f"Partial loading: {len(matched)}/{len(state_dict)} weights loaded")

                if unmatched[:5]:
                    print("Unmatched weights (first 5):")
                    for u in unmatched[:5]:
                        print(f"  {u}")
        else:
            print(f"Warning: Weight file not found: {weight_file}")
            print("Using random initialization")

    model = model.to(device)
    return model
