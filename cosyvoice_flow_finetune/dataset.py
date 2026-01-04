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

# 导入跨样本训练配置
try:
    from config import ANTI_LEAKAGE_CONFIG
except ImportError:
    ANTI_LEAKAGE_CONFIG = {
        'cross_sample_enabled': True,
        'cross_sample_prob': 0.5,
    }


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

    def __call__(self, mel: torch.Tensor, speech_token: Optional[torch.Tensor] = None):
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
    """Dataset for Flow model fine-tuning with cross-sample prompting support"""

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

        # 跨样本训练配置
        self.cross_sample_enabled = ANTI_LEAKAGE_CONFIG.get('cross_sample_enabled', True)
        self.cross_sample_prob = ANTI_LEAKAGE_CONFIG.get('cross_sample_prob', 0.5)
        if self.cross_sample_enabled:
            print(f"Cross-sample prompting: ENABLED (prob={self.cross_sample_prob})")

    def _load_data(self):
        """Load data from parquet files"""
        data_list_path = os.path.join(self.data_dir, 'data.list')
        parquet_files = []

        if os.path.exists(data_list_path):
            with open(data_list_path, 'r', encoding='utf-8') as f:
                raw_paths = [line.strip() for line in f if line.strip()]
            print(f"Found {len(raw_paths)} entries in data.list")

            for raw_path in raw_paths:
                # 标准化路径分隔符
                raw_path = raw_path.replace('\\', '/')

                # 尝试多种路径解析方式
                candidate_paths = [
                    # 1. 原始路径（绝对路径）
                    raw_path,
                    # 2. 直接使用文件名
                    os.path.join(self.data_dir, os.path.basename(raw_path)),
                    # 3. 相对于 data_dir
                    os.path.join(self.data_dir, raw_path),
                ]

                # 4. 如果路径包含子目录，提取文件名和部分路径
                path_parts = raw_path.split('/')
                if len(path_parts) > 1:
                    candidate_paths.append(os.path.join(self.data_dir, path_parts[-1]))
                    candidate_paths.append(os.path.join(self.data_dir, '/'.join(path_parts[1:])))
                    if len(path_parts) > 2:
                        candidate_paths.append(os.path.join(self.data_dir, '/'.join(path_parts[2:])))

                # 查找第一个存在的文件
                found = False
                for cp in candidate_paths:
                    if os.path.exists(cp):
                        parquet_files.append(cp)
                        found = True
                        break

                if not found:
                    print(f"Warning: Could not find parquet file for: {raw_path}")
        else:
            print("data.list not found, searching for parquet files...")
            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if f.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, f))
            parquet_files = sorted(parquet_files)

        print(f"Found {len(parquet_files)} valid parquet files")

        # 加载所有 parquet 文件（使用更高效的方式）
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                self.samples.extend(df.to_dict('records'))
                del df  # 及时释放内存
            except Exception as e:
                print(f"Failed to read {pf}: {e}")

    def __len__(self):
        return len(self.samples)

    def _get_random_prompt_mel(self, exclude_idx: int, max_len: int = 100) -> Optional[torch.Tensor]:
        """
        获取随机样本的 mel 特征作为跨样本 prompt

        Args:
            exclude_idx: 需要排除的样本索引（当前样本）
            max_len: 最大返回长度，限制内存使用

        Returns:
            随机样本的 mel 特征 (T, 80) 或 None
        """
        if len(self.samples) < 2:
            return None

        # 随机选择一个不同的样本
        random_idx = exclude_idx
        attempts = 0
        while attempts < 10 and random_idx == exclude_idx:
            random_idx = random.randint(0, len(self.samples) - 1)
            attempts += 1

        if random_idx == exclude_idx:
            return None

        try:
            random_sample = self.samples[random_idx]

            if 'speech_feat' not in random_sample:
                return None

            speech_feat = random_sample['speech_feat']
            speech_feat_shape = random_sample.get('speech_feat_shape', None)

            # Convert to tensor
            if isinstance(speech_feat, torch.Tensor):
                speech_feat = speech_feat.float()
            elif isinstance(speech_feat, np.ndarray):
                speech_feat = torch.from_numpy(speech_feat.copy()).float()
            elif isinstance(speech_feat, list):
                speech_feat = torch.tensor(speech_feat, dtype=torch.float32)
            else:
                speech_feat = torch.tensor(np.array(speech_feat), dtype=torch.float32)

            # Handle 1D case
            if speech_feat.dim() == 1:
                if speech_feat_shape is not None:
                    if isinstance(speech_feat_shape, (list, tuple)) and len(speech_feat_shape) == 2:
                        T, n_mels = speech_feat_shape
                        speech_feat = speech_feat.view(int(T), int(n_mels))
                    else:
                        total = speech_feat.numel()
                        if total % self.n_mels == 0:
                            speech_feat = speech_feat.view(-1, self.n_mels)
                        else:
                            return None
                else:
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

            # 限制长度，减少内存使用
            if speech_feat.shape[0] > max_len:
                speech_feat = speech_feat[:max_len]

            return speech_feat

        except Exception:
            return None

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

            # ========== 跨样本训练：获取随机样本的 mel 作为 prompt ==========
            cross_sample_mel = None
            if self.cross_sample_enabled and random.random() < self.cross_sample_prob:
                cross_sample_mel = self._get_random_prompt_mel(idx)

            # ========== 获取 text_token（如果存在）==========
            text_token = None
            if 'text_token' in sample and sample['text_token'] is not None:
                text_token_data = sample['text_token']
                if isinstance(text_token_data, torch.Tensor):
                    text_token = text_token_data.long()
                elif isinstance(text_token_data, np.ndarray):
                    text_token = torch.from_numpy(text_token_data.copy()).long()
                elif isinstance(text_token_data, list):
                    text_token = torch.tensor(text_token_data, dtype=torch.long)
                else:
                    text_token = torch.tensor(np.array(text_token_data), dtype=torch.long)

                if text_token.dim() > 1:
                    text_token = text_token.flatten()

            return {
                'speech_token': speech_token,
                'speech_feat': speech_feat,
                'embedding': embedding,
                'cross_sample_mel': cross_sample_mel,  # 跨样本 prompt mel，可能为 None
                'text_token': text_token,  # 文本 token，可能为 None（兼容旧数据集）
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


def collate_fn(batch, max_feat_len_limit: Optional[int] = None):
    """Collate function for DataLoader with cross-sample prompting and text_token support

    Args:
        batch: list of samples
        max_feat_len_limit: 最大序列长度限制（帧数），超过会截断。
                           如果为 None，从 config 读取或不限制。
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # 从 config 获取 max_feat_len 限制
    if max_feat_len_limit is None:
        try:
            from config import JOINT_TRAINING_CONFIG
            max_feat_len_limit = JOINT_TRAINING_CONFIG.get('max_feat_len', 150)
        except ImportError:
            max_feat_len_limit = 150  # 默认限制 150 帧约 1.7 秒

    # 截断过长的样本
    for b in batch:
        feat_len = b['speech_feat'].shape[0]
        if feat_len > max_feat_len_limit:
            # 截断 speech_feat
            b['speech_feat'] = b['speech_feat'][:max_feat_len_limit]
            # 同步截断 speech_token（按比例）
            token_len = b['speech_token'].shape[0]
            new_token_len = int(token_len * max_feat_len_limit / feat_len)
            b['speech_token'] = b['speech_token'][:new_token_len]
            # 同步截断 text_token（如果有）
            if b.get('text_token') is not None:
                text_len = b['text_token'].shape[0]
                new_text_len = int(text_len * max_feat_len_limit / feat_len)
                b['text_token'] = b['text_token'][:new_text_len]

    max_token_len = max(b['speech_token'].shape[0] for b in batch)
    max_feat_len = max(b['speech_feat'].shape[0] for b in batch)

    # Mel spectrogram padding value
    # Log mel 的范围大约是 [-11.5, 1.0]，静音区域接近 -11.5
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
        padded_feat = F.pad(feat, (0, 0, 0, max_feat_len - feat_len), value=MEL_PADDING_VALUE)
        speech_feats.append(padded_feat)
        speech_feat_lens.append(feat_len)

        embeddings.append(b['embedding'])

    result = {
        'speech_token': torch.stack(speech_tokens),
        'speech_token_len': torch.tensor(speech_token_lens),
        'speech_feat': torch.stack(speech_feats),
        'speech_feat_len': torch.tensor(speech_feat_lens),
        'embedding': torch.stack(embeddings),
    }

    # ========== 处理 text_token（用于 LLM 联合训练）==========
    text_tokens = [b.get('text_token', None) for b in batch]
    valid_text_tokens = [t for t in text_tokens if t is not None]

    if valid_text_tokens and len(valid_text_tokens) == len(batch):
        # 所有样本都有 text_token
        max_text_len = max(t.shape[0] for t in valid_text_tokens)
        padded_text_tokens = []
        text_token_lens = []

        for text_token in text_tokens:
            if text_token is not None:
                text_len = text_token.shape[0]
                padded_text = F.pad(text_token, (0, max_text_len - text_len), value=0)
                padded_text_tokens.append(padded_text)
                text_token_lens.append(text_len)

        result['text_token'] = torch.stack(padded_text_tokens)
        result['text_token_len'] = torch.tensor(text_token_lens)

    # 只有当有跨样本 mel 时才处理（减少内存开销）
    cross_sample_mels = [b.get('cross_sample_mel', None) for b in batch]
    valid_cross_mels = [m for m in cross_sample_mels if m is not None]

    if valid_cross_mels:
        cross_sample_mel_lens = [m.shape[0] if m is not None else 0 for m in cross_sample_mels]
        max_cross_mel_len = max(cross_sample_mel_lens)

        padded_cross_mels = []
        for cross_mel, cross_len in zip(cross_sample_mels, cross_sample_mel_lens):
            if cross_mel is not None:
                padded_cross_mel = F.pad(cross_mel, (0, 0, 0, max_cross_mel_len - cross_len), value=MEL_PADDING_VALUE)
                padded_cross_mels.append(padded_cross_mel)
            else:
                padded_cross_mels.append(torch.full((max_cross_mel_len, 80), MEL_PADDING_VALUE))

        result['cross_sample_mel'] = torch.stack(padded_cross_mels)
        result['cross_sample_mel_len'] = torch.tensor(cross_sample_mel_lens)

    return result


def create_dataloader(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 0,  # Windows 下建议设为 0
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
