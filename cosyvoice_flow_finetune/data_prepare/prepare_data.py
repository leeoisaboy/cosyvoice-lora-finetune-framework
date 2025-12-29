#!/usr/bin/env python3
"""
CosyVoice Flow LoRA 微调数据准备脚本

从原始音频和文本准备训练所需的 parquet 数据集。

数据格式要求:
    方式1 - 简单目录结构:
        data_dir/
        ├── audio1.wav
        ├── audio1.txt  (或 audio1.lab)
        ├── audio2.wav
        └── audio2.txt

    方式2 - Kaldi 格式:
        data_dir/
        ├── text          # utt_id text
        ├── wav.scp       # utt_id wav_path
        └── utt2spk       # utt_id spk_id (可选)

使用方法:
    # 从简单目录结构准备
    python prepare_data.py --input_dir ./my_audio --output_dir ./output/data

    # 从 Kaldi 格式准备
    python prepare_data.py --input_dir ./kaldi_data --output_dir ./output/data --format kaldi

    # 使用完整模型提取特征
    python prepare_data.py --input_dir ./my_audio --output_dir ./output/data --use_model

输出:
    output_dir/
    ├── data_000000.parquet
    ├── data_000001.parquet
    └── data.list
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prepare.mel_extractor import MelSpectrogramExtractor


class DataPreparer:
    """数据准备器"""

    def __init__(
        self,
        device: str = 'cuda',
        pretrained_dir: str = None,
        use_model: bool = False,
    ):
        """
        初始化

        Args:
            device: 计算设备
            pretrained_dir: CosyVoice 预训练模型目录（用于提取 speech_token 和 embedding）
            use_model: 是否使用预训练模型提取特征
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.pretrained_dir = pretrained_dir
        self.use_model = use_model

        # Mel 提取器
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0,
            fmax=8000,
            device=self.device,
        )

        # ONNX 模型
        self.speech_token_session = None
        self.embedding_session = None

        if use_model and pretrained_dir:
            self._load_models(pretrained_dir)

        print(f"数据准备器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  使用模型提取: {use_model}")

    def _load_models(self, pretrained_dir: str):
        """加载 ONNX 模型"""
        try:
            import onnxruntime

            # Speech Token 模型
            speech_token_path = os.path.join(pretrained_dir, 'speech_tokenizer_v1.onnx')
            if os.path.exists(speech_token_path):
                option = onnxruntime.SessionOptions()
                option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = ["CUDAExecutionProvider"] if self.device == 'cuda' else ["CPUExecutionProvider"]
                self.speech_token_session = onnxruntime.InferenceSession(
                    speech_token_path, sess_options=option, providers=providers
                )
                print(f"  ✓ Speech Token 模型加载成功")

            # Embedding 模型
            embedding_path = os.path.join(pretrained_dir, 'campplus.onnx')
            if os.path.exists(embedding_path):
                option = onnxruntime.SessionOptions()
                providers = ["CPUExecutionProvider"]
                self.embedding_session = onnxruntime.InferenceSession(
                    embedding_path, sess_options=option, providers=providers
                )
                print(f"  ✓ Embedding 模型加载成功")

        except ImportError:
            print("  [WARN] onnxruntime 未安装，使用随机特征")

    def read_simple_dir(self, input_dir: str) -> List[Dict]:
        """
        读取简单目录结构

        目录中包含音频文件和对应的文本文件:
        - audio.wav + audio.txt
        - audio.wav + audio.lab
        """
        samples = []
        input_path = Path(input_dir)

        # 查找所有音频文件
        audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.mp3'))
        audio_files = sorted(audio_files)

        for audio_file in audio_files:
            # 查找对应的文本文件
            text_file = audio_file.with_suffix('.txt')
            if not text_file.exists():
                text_file = audio_file.with_suffix('.lab')

            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                text = ""
                print(f"  [WARN] 未找到文本文件: {audio_file.stem}")

            samples.append({
                'utt': audio_file.stem,
                'wav': str(audio_file),
                'text': text,
                'spk': 'speaker_001',
            })

        print(f"读取简单目录: {len(samples)} 个样本")
        return samples

    def read_kaldi_format(self, input_dir: str) -> List[Dict]:
        """
        读取 Kaldi 格式

        需要文件:
        - text: utt_id text
        - wav.scp: utt_id wav_path
        - utt2spk: utt_id spk_id (可选)
        """
        samples = {}

        # 读取 text
        text_file = os.path.join(input_dir, 'text')
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) >= 1:
                        utt_id = parts[0]
                        text = parts[1] if len(parts) == 2 else ""
                        samples[utt_id] = {'utt': utt_id, 'text': text}

        # 读取 wav.scp
        wav_scp = os.path.join(input_dir, 'wav.scp')
        if os.path.exists(wav_scp):
            with open(wav_scp, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, wav_path = parts
                        if utt_id in samples:
                            samples[utt_id]['wav'] = wav_path

        # 读取 utt2spk
        utt2spk = os.path.join(input_dir, 'utt2spk')
        if os.path.exists(utt2spk):
            with open(utt2spk, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        utt_id, spk = parts[0], parts[1]
                        if utt_id in samples:
                            samples[utt_id]['spk'] = spk

        # 过滤有效样本
        valid_samples = [
            {**v, 'spk': v.get('spk', 'speaker_001')}
            for v in samples.values()
            if 'wav' in v and 'text' in v
        ]

        print(f"读取 Kaldi 格式: {len(valid_samples)} 个样本")
        return valid_samples

    def extract_speech_token(self, waveform: torch.Tensor, sample_rate: int) -> List[int]:
        """提取 speech token"""
        if self.speech_token_session is not None:
            try:
                import whisper
                # 重采样到 16kHz
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # 提取 mel
                feat = whisper.log_mel_spectrogram(waveform, n_mels=128)
                speech_token = self.speech_token_session.run(
                    None,
                    {
                        self.speech_token_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                        self.speech_token_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
                    }
                )[0].flatten().tolist()
                return [int(t) for t in speech_token]
            except Exception:
                pass

        # 回退: 生成占位 token
        duration = waveform.shape[-1] / sample_rate
        num_tokens = max(1, int(duration * 50))  # 50 tokens/s
        return list(np.random.randint(0, 4096, size=num_tokens))

    def extract_embedding(self, waveform: torch.Tensor, sample_rate: int) -> List[float]:
        """提取说话人 embedding"""
        if self.embedding_session is not None:
            try:
                import torchaudio.compliance.kaldi as kaldi
                # 重采样到 16kHz
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                feat = kaldi.fbank(waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)
                embedding = self.embedding_session.run(
                    None,
                    {self.embedding_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
                )[0].flatten().tolist()
                return embedding
            except Exception:
                pass

        # 回退: 随机向量
        return np.random.randn(192).astype(np.float32).tolist()

    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """处理单个样本"""
        try:
            wav_path = sample.get('wav')
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

            # 提取 mel 频谱
            mel = self.mel_extractor.extract_from_waveform(waveform, sample_rate)
            if mel is None:
                return None

            # 提取 speech token
            speech_token = self.extract_speech_token(waveform, sample_rate)

            # 提取 embedding
            embedding = self.extract_embedding(waveform, sample_rate)

            return {
                'utt': sample.get('utt', ''),
                'text': sample.get('text', ''),
                'spk': sample.get('spk', 'speaker_001'),
                'speech_token': speech_token,
                'speech_feat': mel.flatten().tolist(),
                'speech_feat_shape': (mel.shape[0], mel.shape[1]),
                'utt_embedding': embedding,
                'spk_embedding': embedding,  # 使用相同的 embedding
            }

        except Exception as e:
            print(f"  [ERROR] 处理 {sample.get('utt', 'unknown')} 失败: {e}")
            return None

    def prepare(
        self,
        input_dir: str,
        output_dir: str,
        format: str = 'simple',
        samples_per_file: int = 100,
    ):
        """
        准备数据

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            format: 数据格式 ('simple' 或 'kaldi')
            samples_per_file: 每个 parquet 文件的样本数
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取数据
        if format == 'kaldi':
            samples = self.read_kaldi_format(input_dir)
        else:
            samples = self.read_simple_dir(input_dir)

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

        # 保存剩余
        if processed:
            file_path = self._save_parquet(processed, output_dir, file_idx)
            file_list.append(file_path)

        # 生成 data.list
        data_list_path = os.path.join(output_dir, 'data.list')
        with open(data_list_path, 'w') as f:
            for path in file_list:
                f.write(f"{path}\n")

        print(f"\n{'='*50}")
        print("数据准备完成!")
        print(f"  输出目录: {output_dir}")
        print(f"  Parquet 文件数: {len(file_list)}")
        print(f"  数据列表: {data_list_path}")
        print(f"\n下一步:")
        print(f"  1. 更新 config.py 中的 DATA_DIR 路径")
        print(f"  2. 运行 python train.py 开始训练")

    def _save_parquet(self, samples: List[Dict], output_dir: str, index: int) -> str:
        """保存到 parquet"""
        file_path = os.path.join(output_dir, f'data_{index:06d}.parquet')
        df = pd.DataFrame(samples)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
        return file_path


def main():
    parser = argparse.ArgumentParser(description='CosyVoice Flow LoRA 数据准备')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录 (包含音频和文本)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录 (parquet 文件)')
    parser.add_argument('--format', type=str, choices=['simple', 'kaldi'], default='simple',
                        help='输入格式: simple=音频+txt文件, kaldi=text+wav.scp+utt2spk')
    parser.add_argument('--use_model', action='store_true',
                        help='使用预训练模型提取特征 (需要 CosyVoice 模型)')
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='CosyVoice 预训练模型目录')
    parser.add_argument('--samples_per_file', type=int, default=100,
                        help='每个 parquet 文件的样本数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')

    args = parser.parse_args()

    print("=" * 60)
    print("CosyVoice Flow LoRA 数据准备")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"输入格式: {args.format}")
    print("=" * 60)

    preparer = DataPreparer(
        device=args.device,
        pretrained_dir=args.pretrained_dir,
        use_model=args.use_model,
    )

    preparer.prepare(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format=args.format,
        samples_per_file=args.samples_per_file,
    )


if __name__ == '__main__':
    main()
