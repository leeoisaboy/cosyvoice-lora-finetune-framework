# Copyright (c) 2024 Alibaba Inc
# Self-contained Flow model for fine-tuning
# No external CosyVoice/Matcha-TTS dependencies

import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_pad_mask
from modules import ConformerEncoder, InterpolateRegulator, ConditionalDecoder


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
        estimator: nn.Module = None,
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

            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Compute flow matching loss"""
        b, _, t = mu.shape

        # Random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * 3.14159265359)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # CFG dropout
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.view(b), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
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
        encoder: nn.Module = None,
        length_regulator: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator

        # ========== Mel 归一化参数 ==========
        # Flow Matching 期望目标分布接近 N(0, 1)
        # Log mel 原始统计: mean ≈ -6.0, std ≈ 2.0
        self.mel_mean = -6.0
        self.mel_std = 2.0

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """将 log mel 归一化到 N(0, 1)"""
        return (mel - self.mel_mean) / self.mel_std

    def denormalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """将归一化的 mel 还原"""
        return mel * self.mel_std + self.mel_mean

    def forward(self, batch: dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """Training forward pass"""
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # ========== 在线归一化 mel ==========
        # 原始: mean ≈ -6, range ≈ [-11, 1]
        # 归一化后: mean ≈ 0, std ≈ 1
        feat = self.normalize_mel(feat)

        # Speaker embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Token embedding
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # Random conditioning (for training)
        # ========== Prompt 策略配置 ==========
        # 混合策略：让模型同时学习
        #   1. 从头生成（无 prompt）
        #   2. 续写生成（有 prompt，学习利用前文生成后文）
        #
        # 这样模型既能学习完整生成，也能学习长序列的连贯性
        NO_PROMPT_PROB = 0.3      # 30% 样本无 prompt（学习从头生成）
        PROMPT_MIN_RATIO = 0.1    # prompt 最小比例
        PROMPT_MAX_RATIO = 0.5    # prompt 最大比例（10%-50% 随机）

        conds = torch.zeros(feat.shape, device=device)
        for i, j in enumerate(feat_len):
            if random.random() < NO_PROMPT_PROB:
                # 无 prompt，从头生成
                continue
            # 有 prompt，随机选择 10%-50% 作为条件
            min_idx = max(1, int(PROMPT_MIN_RATIO * j))
            max_idx = max(min_idx + 1, int(PROMPT_MAX_RATIO * j))
            index = random.randint(min_idx, max_idx)
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        # Compute loss
        mask = (~make_pad_mask(feat_len)).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
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
        """Inference forward pass"""
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
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # Conditioning
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        # Decode
        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, new_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
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
        - prompt_feat is optional (like training where 50% have no prompt)
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
        h, h_lengths = self.encoder(token_emb, token_len)
        h = self.encoder_proj(h)

        # Length regulation to target mel length
        h, h_lengths = self.length_regulator(h, torch.tensor([feat_len_val], device=token.device))

        # Build conditioning (matching training logic)
        conds = torch.zeros([1, feat_len_val, self.output_size], device=token.device, dtype=h.dtype)
        if prompt_feat is not None and prompt_len > 0:
            actual_prompt_len = min(prompt_len, prompt_feat.shape[1], feat_len_val)
            conds[:, :actual_prompt_len] = prompt_feat[:, :actual_prompt_len]
        conds = conds.transpose(1, 2)  # (1, 80, T)

        # Decode
        mask = torch.ones([1, 1, feat_len_val], device=token.device, dtype=h.dtype)
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
    pretrained_path: str = None,
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
        attention_dropout_rate=0.1,  # 0.1 in pretrained config
        normalize_before=True,
        cnn_module_kernel=15,
        use_cnn_module=False,  # False in pretrained config
        macaron_style=False,   # False in pretrained config
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
