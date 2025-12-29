"""
Mel 频谱提取工具

使用与 CosyVoice 完全一致的 mel 提取方法 (librosa mel filterbank + STFT)
"""

import io
import torch
import torchaudio
import numpy as np

try:
    from librosa.filters import mel as librosa_mel_fn
except ImportError:
    raise ImportError("请安装 librosa: pip install librosa")


# 全局缓存
mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram_cosyvoice(
    y: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sampling_rate: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0,
    fmax: float = 8000,
    center: bool = False,
) -> torch.Tensor:
    """
    CosyVoice 官方 mel 频谱提取方法
    与 matcha.utils.audio.mel_spectrogram 完全一致

    Args:
        y: (1, T) 或 (T,) 音频波形，范围 [-1, 1]
        其他参数与 CosyVoice config 一致

    Returns:
        mel: (num_mels, T') log mel 频谱
    """
    global mel_basis_cache, hann_window_cache

    if y.dim() == 1:
        y = y.unsqueeze(0)

    device = y.device
    cache_key = f"{fmax}_{device}"

    # 创建 mel filterbank (使用 librosa，与 CosyVoice 一致)
    if cache_key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis_cache[cache_key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[str(device)] = torch.hann_window(win_size).to(device)

    # Padding (与 CosyVoice 一致: center=False, 手动 reflect padding)
    pad_size = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_size, pad_size), mode="reflect")
    y = y.squeeze(1)

    # STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window_cache[str(device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # 取幅度谱
    spec = torch.abs(spec) + 1e-9

    # 应用 mel filterbank
    spec = torch.matmul(mel_basis_cache[cache_key], spec)

    # Log 压缩 (dynamic range compression，与 CosyVoice 一致)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec.squeeze(0)  # (num_mels, T')


class MelSpectrogramExtractor:
    """Mel 频谱提取器，与 CosyVoice 完全兼容"""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0,
        fmax: float = 8000,
        device: str = 'cuda',
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 重采样器缓存
        self.resamplers = {}

    def extract_from_waveform(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """
        从波形提取 mel 频谱

        Args:
            waveform: (C, T) 音频波形
            sr: 采样率

        Returns:
            mel: (T, n_mels) mel 频谱
        """
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重采样到目标采样率
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            waveform = waveform.to(self.device)
            waveform = self.resamplers[sr](waveform)
        else:
            waveform = waveform.to(self.device)

        # 使用 CosyVoice 官方方法提取 mel 频谱
        mel = mel_spectrogram_cosyvoice(
            waveform,
            n_fft=self.n_fft,
            num_mels=self.n_mels,
            sampling_rate=self.sample_rate,
            hop_size=self.hop_length,
            win_size=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
            center=False,  # CosyVoice 使用 center=False
        )  # (num_mels, T)

        # 转换形状为 (T, n_mels)，移回 CPU
        mel = mel.transpose(0, 1).cpu().numpy()  # (T, n_mels)

        return mel

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """从音频文件提取 mel 频谱"""
        waveform, sr = torchaudio.load(audio_path)
        return self.extract_from_waveform(waveform, sr)

    def extract_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """从音频字节提取 mel 频谱"""
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(audio_buffer)
        return self.extract_from_waveform(waveform, sr)
