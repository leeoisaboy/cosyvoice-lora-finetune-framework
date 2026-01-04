#!/usr/bin/env python3
"""
LLM + Flow 联合训练数据准备脚本

生成包含 text_token 的数据集，用于联合训练 LLM 和 Flow。

数据格式要求:
    raw_audio/
    ├── audio1.wav
    ├── audio1.txt  (文本内容)
    ├── audio2.wav
    └── audio2.txt

使用方法:
    python prepare_joint_data.py

输出:
    data/
    ├── data_000000.parquet
    └── data.list
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
COSYVOICE_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, COSYVOICE_ROOT)

from config import PRETRAINED_MODEL_DIR, RAW_AUDIO_DIR, DATA_DIR


class JointDataPreparer:
    """联合训练数据准备器"""

    def __init__(self, pretrained_dir: str = PRETRAINED_MODEL_DIR, device: str = 'cuda'):
        """
        初始化

        Args:
            pretrained_dir: CosyVoice 预训练模型目录
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.pretrained_dir = pretrained_dir

        print("=" * 60)
        print("CosyVoice LLM + Flow 联合训练数据准备")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"预训练模型: {pretrained_dir}")

        # 加载 CosyVoice 模型（用于提取特征）
        self._load_cosyvoice()

    def _load_cosyvoice(self):
        """加载 CosyVoice 模型"""
        from cosyvoice.cli.cosyvoice import CosyVoice

        print("\n加载 CosyVoice 模型...")
        self.cosyvoice = CosyVoice(self.pretrained_dir, load_jit=False, load_trt=False)
        print(f"  ✓ 模型加载成功")
        print(f"  采样率: {self.cosyvoice.sample_rate}")

        # 获取 tokenizer 和 frontend
        self.frontend = self.cosyvoice.frontend
        self.tokenizer = self.frontend.tokenizer
        self.allowed_special = self.frontend.allowed_special

        print(f"  ✓ Tokenizer 加载成功")

    def text_to_tokens(self, text: str) -> List[int]:
        """
        将文本转换为 token

        Args:
            text: 输入文本

        Returns:
            token 列表
        """
        # 文本归一化
        normalized_text = self.frontend.text_normalize(text, split=False, text_frontend=True)

        # 编码为 token
        tokens = self.tokenizer.encode(normalized_text, allowed_special=self.allowed_special)

        return tokens

    def extract_speech_token(self, waveform: torch.Tensor, sample_rate: int) -> List[int]:
        """
        提取 speech token

        Args:
            waveform: 音频波形 (1, T)
            sample_rate: 采样率

        Returns:
            speech token 列表
        """
        # 重采样到 16kHz
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 使用 CosyVoice 的 speech tokenizer（私有方法 _extract_speech_token）
        try:
            speech_token, _ = self.frontend._extract_speech_token(waveform.to(self.device))
            return speech_token[0].cpu().tolist()
        except Exception as e:
            print(f"    [WARN] Speech token 提取失败: {e}")
            # 回退: 生成占位 token
            duration = waveform.shape[-1] / 16000
            num_tokens = max(1, int(duration * 50))  # 50 tokens/s
            return list(np.random.randint(0, 4096, size=num_tokens))

    def extract_embedding(self, waveform: torch.Tensor, sample_rate: int) -> List[float]:
        """
        提取说话人 embedding

        Args:
            waveform: 音频波形
            sample_rate: 采样率

        Returns:
            embedding 向量
        """
        # 重采样到 16kHz
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        try:
            embedding = self.frontend._extract_spk_embedding(waveform.to(self.device))
            return embedding[0].cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"    [WARN] Embedding 提取失败: {e}")
            return np.random.randn(192).astype(np.float32).tolist()

    def extract_mel(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        提取 mel 频谱

        Args:
            waveform: 音频波形
            sample_rate: 采样率

        Returns:
            mel 频谱 (T, 80)
        """
        # 重采样到模型需要的采样率（22050 或 24000）
        target_sr = self.cosyvoice.sample_rate
        if sample_rate != target_sr:
            waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        try:
            mel, _ = self.frontend._extract_speech_feat(waveform.to(self.device))
            mel = mel[0].cpu().numpy()  # (T, 80)
            return mel
        except Exception as e:
            print(f"    [WARN] Mel 提取失败: {e}")
            return None

    def read_samples(self, input_dir: str) -> List[Dict]:
        """
        读取样本

        Args:
            input_dir: 输入目录

        Returns:
            样本列表
        """
        samples = []
        input_path = Path(input_dir)

        # 查找所有音频文件
        audio_files = list(input_path.glob('*.wav'))
        audio_files = sorted(audio_files)

        print(f"\n找到 {len(audio_files)} 个音频文件")

        for audio_file in audio_files:
            # 查找对应的文本文件
            text_file = audio_file.with_suffix('.txt')

            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                print(f"  [WARN] 未找到文本文件: {audio_file.stem}")
                continue  # 跳过没有文本的样本

            if not text:
                print(f"  [WARN] 文本为空: {audio_file.stem}")
                continue

            samples.append({
                'utt': audio_file.stem,
                'wav': str(audio_file),
                'text': text,
            })

        print(f"有效样本: {len(samples)} 个")
        return samples

    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """
        处理单个样本

        Args:
            sample: 样本信息

        Returns:
            处理后的样本数据
        """
        try:
            wav_path = sample.get('wav')
            text = sample.get('text', '')

            if not wav_path or not os.path.exists(wav_path):
                return None

            # 加载音频
            waveform, sample_rate = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 检查时长
            duration = waveform.shape[1] / sample_rate
            if duration < 0.5 or duration > 30:
                print(f"  [WARN] 跳过 {sample['utt']}: 时长 {duration:.1f}s 超出范围 [0.5, 30]")
                return None

            # 提取 text token
            text_token = self.text_to_tokens(text)
            if len(text_token) == 0:
                print(f"  [WARN] 跳过 {sample['utt']}: text_token 为空")
                return None

            # 提取 speech token
            speech_token = self.extract_speech_token(waveform, sample_rate)

            # 提取 mel 频谱
            mel = self.extract_mel(waveform, sample_rate)
            if mel is None:
                return None

            # 提取 embedding
            embedding = self.extract_embedding(waveform, sample_rate)

            return {
                'utt': sample.get('utt', ''),
                'text': text,
                'text_token': text_token,
                'speech_token': speech_token,
                'speech_feat': mel.flatten().tolist(),
                'speech_feat_shape': (mel.shape[0], mel.shape[1]),
                'utt_embedding': embedding,
                'spk_embedding': embedding,
            }

        except Exception as e:
            print(f"  [ERROR] 处理 {sample.get('utt', 'unknown')} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare(
        self,
        input_dir: str,
        output_dir: str,
        samples_per_file: int = 100,
    ):
        """
        准备数据

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            samples_per_file: 每个 parquet 文件的样本数
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取样本
        samples = self.read_samples(input_dir)

        if not samples:
            print("错误: 没有找到有效样本")
            return

        # 处理数据
        processed = []
        file_list = []
        file_idx = 0

        print(f"\n处理 {len(samples)} 个样本...")
        for sample in tqdm(samples, desc="处理进度"):
            result = self.process_sample(sample)
            if result is not None:
                processed.append(result)

            # 保存
            if len(processed) >= samples_per_file:
                file_path = self._save_parquet(processed, output_dir, file_idx)
                file_list.append(file_path)
                processed = []
                file_idx += 1

                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 保存剩余
        if processed:
            file_path = self._save_parquet(processed, output_dir, file_idx)
            file_list.append(file_path)

        # 生成 data.list
        data_list_path = os.path.join(output_dir, 'data.list')
        with open(data_list_path, 'w') as f:
            for path in file_list:
                f.write(f"{path}\n")

        print(f"\n{'='*60}")
        print("数据准备完成!")
        print(f"{'='*60}")
        print(f"输出目录: {output_dir}")
        print(f"Parquet 文件数: {len(file_list)}")
        print(f"总样本数: {sum(1 for _ in open(data_list_path)) * samples_per_file if file_list else 0}")
        print(f"数据列表: {data_list_path}")
        print(f"\n数据包含:")
        print(f"  ✓ text_token (文本 token，用于 LLM 训练)")
        print(f"  ✓ speech_token (语音 token)")
        print(f"  ✓ speech_feat (mel 频谱)")
        print(f"  ✓ embedding (说话人向量)")
        print(f"\n下一步:")
        print(f"  python train_joint.py --mode joint")

    def _save_parquet(self, samples: List[Dict], output_dir: str, index: int) -> str:
        """保存到 parquet"""
        file_path = os.path.join(output_dir, f'data_{index:06d}.parquet')
        df = pd.DataFrame(samples)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
        print(f"  保存: {file_path} ({len(samples)} 样本)")
        return file_path


def main():
    parser = argparse.ArgumentParser(description='CosyVoice LLM + Flow 联合训练数据准备')
    parser.add_argument('--input_dir', type=str, default=RAW_AUDIO_DIR,
                        help=f'输入目录 (默认: {RAW_AUDIO_DIR})')
    parser.add_argument('--output_dir', type=str, default=DATA_DIR,
                        help=f'输出目录 (默认: {DATA_DIR})')
    parser.add_argument('--pretrained_dir', type=str, default=PRETRAINED_MODEL_DIR,
                        help=f'预训练模型目录 (默认: {PRETRAINED_MODEL_DIR})')
    parser.add_argument('--samples_per_file', type=int, default=100,
                        help='每个 parquet 文件的样本数 (默认: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')

    args = parser.parse_args()

    preparer = JointDataPreparer(
        pretrained_dir=args.pretrained_dir,
        device=args.device,
    )

    preparer.prepare(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        samples_per_file=args.samples_per_file,
    )


if __name__ == '__main__':
    main()
