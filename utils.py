import torch
import torchaudio
import numpy as np
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioTextProcessor:
    """
    A class to process audio files into features and convert text to/from integer sequences.
    """
    def __init__(self):
        # Character mapping for text conversion
        self.char_map = {
            ' ': 0,
            "'": 1,
            **{chr(97 + i): 2 + i for i in range(26)}  # a-z mapped to 2-27
        }
        self.blank_idx = 28
        self.index_map = {v: k for k, v in self.char_map.items()}
        self.index_map[0] = ' '  # Ensure space mapping

    def compute_feature_dim(self, frame_length_ms: int, max_frequency: int) -> int:
        """
        Calculate the dimension of audio features based on frame length and max frequency.

        Args:
            frame_length_ms (int): Length of the FFT window in milliseconds.
            max_frequency (int): Maximum frequency to consider.

        Returns:
            int: Number of frequency bins.
        """
        frame_length_sec = frame_length_ms / 1000.0
        return int(frame_length_sec * max_frequency) + 1

    def calculate_conv_output_length(
        self,
        input_length: Optional[int],
        kernel_size: int,
        padding_mode: str,
        stride: int,
        dilation: int = 1
    ) -> Optional[int]:
        """
        Compute the output length after 1D convolution.

        Args:
            input_length (int, optional): Length of the input sequence.
            kernel_size (int): Size of the convolution kernel.
            padding_mode (str): Padding mode ("same" or "valid").
            stride (int): Convolution stride.
            dilation (int): Dilation factor.

        Returns:
            int or None: Output sequence length, or None if input_length is None.
        """
        if input_length is None:
            return None
        if padding_mode not in {"same", "valid"}:
            raise ValueError("padding_mode must be 'same' or 'valid'")
        
        effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
        if padding_mode == "same":
            output_len = input_length
        else:  # valid
            output_len = input_length - effective_kernel + 1
        return (output_len + stride - 1) // stride

    def extract_spectrogram(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        n_fft: int = 400,
        hop_length: int = 160
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the power spectrogram of an audio signal.

        Args:
            audio (torch.Tensor): Input audio signal (1D tensor).
            sample_rate (int): Sampling rate of the audio.
            n_fft (int): Number of FFT components.
            hop_length (int): Number of samples between successive frames.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Spectrogram (freq, time) and frequencies.
        """
        transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            normalized=False
        )
        spec = transform(audio)
        freqs = torch.linspace(0, sample_rate / 2, steps=spec.shape[0])
        return spec, freqs

    def spectrogram_from_audio(
        self,
        audio_path: str,
        frame_step_ms: int = 10,
        frame_length_ms: int = 25,
        max_freq: Optional[int] = None,
        epsilon: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute the log spectrogram from an audio file.

        Args:
            audio_path (str): Path to the audio file.
            frame_step_ms (int): Step size between frames in milliseconds.
            frame_length_ms (int): Length of each frame in milliseconds.
            max_freq (int, optional): Maximum frequency to include.
            epsilon (float): Small value for numerical stability in log.

        Returns:
            torch.Tensor: Log spectrogram (time, freq).
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)  # Convert to mono
            
            if max_freq is None:
                max_freq = sample_rate // 2
            if max_freq > sample_rate // 2:
                raise ValueError("max_freq cannot exceed Nyquist frequency")
            if frame_step_ms > frame_length_ms:
                raise ValueError("frame_step_ms must not exceed frame_length_ms")
            
            n_fft = int(frame_length_ms / 1000.0 * sample_rate)
            hop_length = int(frame_step_ms / 1000.0 * sample_rate)
            
            spec, freqs = self.extract_spectrogram(waveform, sample_rate, n_fft, hop_length)
            
            # Trim to max_freq
            freq_idx = torch.searchsorted(freqs, max_freq, right=True)
            spec = spec[:freq_idx, :]
            
            # Compute log spectrogram
            log_spec = torch.log(spec + epsilon)
            
            return log_spec.transpose(0, 1)  # (time, freq)
        
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return torch.zeros((100, self.compute_feature_dim(frame_length_ms, max_freq)))

    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert a text string to a sequence of integer indices.

        Args:
            text (str): Input text.

        Returns:
            List[int]: List of integer indices.
        """
        return [self.char_map.get(c.lower(), self.blank_idx) for c in text]

    def indices_to_text(self, indices: List[int]) -> str:
        """
        Convert a sequence of integer indices to text.

        Args:
            indices (List[int]): List of integer indices.

        Returns:
            str: Converted text string.
        """
        return ''.join(self.index_map.get(idx, '') for idx in indices if idx != self.blank_idx)
