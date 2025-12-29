# Copyright (c) 2024 Alibaba Inc
# Dataset and DataLoader for Flow model fine-tuning

import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


# ============================================================
# 数据增强模块
# ============================================================

class MelAugmentation:
    """Mel spectrogram 数据增强

    对于 Flow 模型的 mel spectrogram 微调，适用的增强方式：
    1. 时间遮蔽 (Time Masking) - SpecAugment 风格
    2. 频率遮蔽 (Frequency Masking) - SpecAugment 风格
    3. 音量扰动 (Volume Perturbation) - 整体增益调整
    4. 时间拉伸 (Time Stretch) - 轻微的速度变化
    5. 添加噪声 (Add Noise) - 轻微的高斯噪声
    """

    def __init__(
        self,
        enable: bool = True,
        # 时间遮蔽
        time_mask_prob: float = 0.5,
        time_mask_max_ratio: float = 0.1,  # 最多遮蔽 10% 的时间
        num_time_masks: int = 2,
        # 频率遮蔽
        freq_mask_prob: float = 0.5,
        freq_mask_max_bins: int = 8,  # 最多遮蔽 8 个频率 bin
        num_freq_masks: int = 2,
        # 音量扰动
        volume_prob: float = 0.5,
        volume_range: tuple = (-0.2, 0.2),  # ±20% 音量变化（dB 空间）
        # 时间拉伸
        time_stretch_prob: float = 0.3,
        time_stretch_range: tuple = (0.95, 1.05),  # ±5% 速度变化
        # 噪声
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
    ):
        self.enable = enable
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_ratio = time_mask_max_ratio
        self.num_time_masks = num_time_masks
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_max_bins = freq_mask_max_bins
        self.num_freq_masks = num_freq_masks
        self.volume_prob = volume_prob
        self.volume_range = volume_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(self, mel: torch.Tensor, speech_token: torch.Tensor = None):
        """
        Args:
            mel: (T, n_mels) mel spectrogram
            speech_token: (T',) speech tokens (需要同步调整长度)
        Returns:
            augmented mel, augmented speech_token
        """
        if not self.enable:
            return mel, speech_token

        mel = mel.clone()

        # 1. 时间遮蔽
        if random.random() < self.time_mask_prob:
            mel = self._time_mask(mel)

        # 2. 频率遮蔽
        if random.random() < self.freq_mask_prob:
            mel = self._freq_mask(mel)

        # 3. 音量扰动
        if random.random() < self.volume_prob:
            mel = self._volume_perturb(mel)

        # 4. 时间拉伸 (同时调整 speech_token)
        if random.random() < self.time_stretch_prob and speech_token is not None:
            mel, speech_token = self._time_stretch(mel, speech_token)

        # 5. 添加噪声
        if random.random() < self.noise_prob:
            mel = self._add_noise(mel)

        return mel, speech_token

    def _time_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """时间遮蔽：随机遮蔽一段时间"""
        T, n_mels = mel.shape
        for _ in range(self.num_time_masks):
            t = int(T * self.time_mask_max_ratio * random.random())
            if t > 0:
                t0 = random.randint(0, max(0, T - t))
                mel[t0:t0+t, :] = mel.mean()  # 用均值填充
        return mel

    def _freq_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """频率遮蔽：随机遮蔽一些频率 bin"""
        T, n_mels = mel.shape
        for _ in range(self.num_freq_masks):
            f = random.randint(1, self.freq_mask_max_bins)
            f0 = random.randint(0, max(0, n_mels - f))
            mel[:, f0:f0+f] = mel.mean()  # 用均值填充
        return mel

    def _volume_perturb(self, mel: torch.Tensor) -> torch.Tensor:
        """音量扰动：在 log mel 空间加减（相当于乘除）"""
        gain = random.uniform(*self.volume_range)
        return mel + gain

    def _time_stretch(self, mel: torch.Tensor, speech_token: torch.Tensor):
        """时间拉伸：通过插值改变速度"""
        T, n_mels = mel.shape
        stretch_factor = random.uniform(*self.time_stretch_range)
        new_T = int(T * stretch_factor)

        if new_T < 10 or new_T > T * 2:  # 安全检查
            return mel, speech_token

        # 插值 mel
        mel_t = mel.t().unsqueeze(0)  # (1, n_mels, T)
        mel_stretched = F.interpolate(mel_t, size=new_T, mode='linear', align_corners=False)
        mel_stretched = mel_stretched.squeeze(0).t()  # (new_T, n_mels)

        # 插值 speech_token（需要保持整数）
        token_len = speech_token.shape[0]
        new_token_len = int(token_len * stretch_factor)
        if new_token_len > 0:
            indices = torch.linspace(0, token_len - 1, new_token_len).long()
            indices = indices.clamp(0, token_len - 1)
            speech_token = speech_token[indices]

        return mel_stretched, speech_token

    def _add_noise(self, mel: torch.Tensor) -> torch.Tensor:
        """添加轻微噪声"""
        noise = torch.randn_like(mel) * self.noise_std
        return mel + noise


# ============================================================
# Dataset
# ============================================================


class FlowFinetuneDataset(Dataset):
    """Dataset for Flow model fine-tuning"""

    def __init__(
        self,
        data_dir: str,
        max_duration: float = 15.0,
        target_sr: int = 22050,
        augmentation: bool = True,  # 是否启用数据增强
    ):
        self.data_dir = data_dir
        self.max_duration = max_duration
        self.target_sr = target_sr
        self.hop_size = 256
        self.n_mels = 80

        # 数据增强
        self.augmentation = MelAugmentation(enable=augmentation)
        self.augmentation_enabled = augmentation

        self.samples = []
        self._load_data()
        print(f"Dataset loaded: {len(self.samples)} samples")
        if augmentation:
            print(f"Data augmentation: ENABLED")

    def _load_data(self):
        """Load data from parquet files"""
        data_list_path = os.path.join(self.data_dir, 'data.list')
        parquet_files = []

        if os.path.exists(data_list_path):
            with open(data_list_path, 'r') as f:
                raw_paths = [line.strip() for line in f if line.strip()]
            print(f"Found {len(raw_paths)} entries in data.list")

            for raw_path in raw_paths:
                raw_path = raw_path.replace('\\', '/')
                candidate_paths = [
                    raw_path,
                    os.path.join(self.data_dir, os.path.basename(raw_path)),
                    os.path.join(self.data_dir, raw_path),
                ]
                path_parts = raw_path.split('/')
                if len(path_parts) > 1:
                    candidate_paths.append(os.path.join(self.data_dir, path_parts[-1]))
                    candidate_paths.append(os.path.join(self.data_dir, '/'.join(path_parts[1:])))
                    if len(path_parts) > 2:
                        candidate_paths.append(os.path.join(self.data_dir, '/'.join(path_parts[2:])))

                for cp in candidate_paths:
                    if os.path.exists(cp):
                        parquet_files.append(cp)
                        break
        else:
            print("data.list not found, searching for parquet files...")
            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if f.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, f))
            parquet_files = sorted(parquet_files)

        print(f"Found {len(parquet_files)} valid parquet files")

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                for idx, row in df.iterrows():
                    self.samples.append(row.to_dict())
            except Exception as e:
                print(f"Failed to read {pf}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Get speech_feat
            if 'speech_feat' not in sample:
                return None

            speech_feat = sample['speech_feat']
            speech_feat_shape = sample.get('speech_feat_shape', None)

            # Convert to tensor
            if isinstance(speech_feat, torch.Tensor):
                speech_feat = speech_feat.float()
            elif isinstance(speech_feat, np.ndarray):
                speech_feat = torch.from_numpy(speech_feat.copy()).float()
            elif isinstance(speech_feat, list):
                speech_feat = torch.tensor(speech_feat, dtype=torch.float32)
            else:
                # Try to convert unknown type
                speech_feat = torch.tensor(np.array(speech_feat), dtype=torch.float32)

            # Handle 1D case - reshape using saved shape or n_mels
            if speech_feat.dim() == 1:
                if speech_feat_shape is not None:
                    # Use saved shape
                    if isinstance(speech_feat_shape, (list, tuple)) and len(speech_feat_shape) == 2:
                        T, n_mels = speech_feat_shape
                        speech_feat = speech_feat.view(int(T), int(n_mels))
                    else:
                        # Fallback to n_mels
                        total = speech_feat.numel()
                        if total % self.n_mels == 0:
                            speech_feat = speech_feat.view(-1, self.n_mels)
                        else:
                            return None
                else:
                    # Fallback to n_mels
                    total = speech_feat.numel()
                    if total % self.n_mels == 0:
                        speech_feat = speech_feat.view(-1, self.n_mels)
                    else:
                        return None

            # Ensure shape is (T, n_mels)
            if speech_feat.dim() == 2:
                if speech_feat.shape[-1] != self.n_mels and speech_feat.shape[0] == self.n_mels:
                    speech_feat = speech_feat.transpose(0, 1)
            else:
                return None

            # Get speech_token
            if 'speech_token' not in sample:
                return None

            speech_token = sample['speech_token']

            # Convert to tensor
            if isinstance(speech_token, torch.Tensor):
                speech_token = speech_token.long()
            elif isinstance(speech_token, np.ndarray):
                speech_token = torch.from_numpy(speech_token.copy()).long()
            elif isinstance(speech_token, list):
                speech_token = torch.tensor(speech_token, dtype=torch.long)
            else:
                speech_token = torch.tensor(np.array(speech_token), dtype=torch.long)

            # Flatten if needed
            if speech_token.dim() > 1:
                speech_token = speech_token.flatten()

            # Get embedding
            embedding = None
            for key in ['utt_embedding', 'spk_embedding', 'embedding']:
                if key in sample and sample[key] is not None:
                    embedding = sample[key]
                    break

            if embedding is None:
                embedding = torch.randn(192, dtype=torch.float32)
            else:
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.float()
                elif isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding.copy()).float()
                elif isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                else:
                    embedding = torch.tensor(np.array(embedding), dtype=torch.float32)

                # Flatten if needed
                if embedding.dim() > 1:
                    embedding = embedding.flatten()

            # 应用数据增强
            if self.augmentation_enabled:
                speech_feat, speech_token = self.augmentation(speech_feat, speech_token)

            return {
                'speech_token': speech_token,
                'speech_feat': speech_feat,
                'embedding': embedding,
            }

        except Exception as e:
            # Debug: print error for first few failures
            if idx < 3:
                print(f"Error loading sample {idx}: {e}")
                print(f"  Keys: {list(sample.keys())}")
                if 'speech_feat' in sample:
                    sf = sample['speech_feat']
                    print(f"  speech_feat type: {type(sf)}, len: {len(sf) if hasattr(sf, '__len__') else 'N/A'}")
                if 'speech_feat_shape' in sample:
                    print(f"  speech_feat_shape: {sample['speech_feat_shape']}")
                if 'speech_token' in sample:
                    st = sample['speech_token']
                    print(f"  speech_token type: {type(st)}, len: {len(st) if hasattr(st, '__len__') else 'N/A'}")
            return None


def collate_fn(batch):
    """Collate function for DataLoader"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_token_len = max(b['speech_token'].shape[0] for b in batch)
    max_feat_len = max(b['speech_feat'].shape[0] for b in batch)

    # Mel spectrogram padding value
    # Log mel 的范围大约是 [-11.5, 1.0]，静音区域接近 -11.5
    # 注意：padding 发生在归一化之前，所以使用原始值 -11.5
    # 归一化后这个值会变成 (-11.5 - (-6.0)) / 2.0 ≈ -2.75
    MEL_PADDING_VALUE = -11.5

    speech_tokens = []
    speech_token_lens = []
    speech_feats = []
    speech_feat_lens = []
    embeddings = []

    for b in batch:
        token = b['speech_token']
        token_len = token.shape[0]
        padded_token = F.pad(token, (0, max_token_len - token_len), value=0)
        speech_tokens.append(padded_token)
        speech_token_lens.append(token_len)

        feat = b['speech_feat']
        feat_len = feat.shape[0]
        # 使用 MEL_PADDING_VALUE 而非 0
        padded_feat = F.pad(feat, (0, 0, 0, max_feat_len - feat_len), value=MEL_PADDING_VALUE)
        speech_feats.append(padded_feat)
        speech_feat_lens.append(feat_len)

        embeddings.append(b['embedding'])

    return {
        'speech_token': torch.stack(speech_tokens),
        'speech_token_len': torch.tensor(speech_token_lens),
        'speech_feat': torch.stack(speech_feats),
        'speech_feat_len': torch.tensor(speech_feat_lens),
        'embedding': torch.stack(embeddings),
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 2,
    max_duration: float = 15.0,
) -> DataLoader:
    """Create DataLoader for training"""
    dataset = FlowFinetuneDataset(
        data_dir=data_dir,
        max_duration=max_duration,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader
