import json
import numpy as np
import torchaudio
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Union
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioFeaturizer:
    """
    A class to process audio files, extract features (spectrogram or MFCC), and prepare batches
    for training, validation, or testing in a speech-related neural network.
    """
    def __init__(
        self,
        feature_type: str = "spectrogram",
        frame_length: int = 25,
        frame_step: int = 10,
        max_freq: int = 8000,
        n_mfcc: int = 13,
        batch_size: int = 32,
        max_clip_duration: float = 12.0,
        metadata_path: Union[str, Path] = None
    ):
        """
        Initialize the audio featurizer.

        Args:
            feature_type (str): Type of feature to extract ('spectrogram' or 'mfcc').
            frame_length (int): Window size in milliseconds for feature extraction.
            frame_step (int): Step size in milliseconds between windows.
            max_freq (int): Maximum frequency for spectrogram features.
            n_mfcc (int): Number of MFCC coefficients if using MFCC features.
            batch_size (int): Number of samples per batch.
            max_clip_duration (float): Maximum duration of audio clips in seconds.
            metadata_path (str or Path): Path to JSON metadata file with audio paths and labels.
        """
        self.feature_type = feature_type.lower()
        self.frame_length = frame_length / 1000  # Convert to seconds
        self.frame_step = frame_step / 1000      # Convert to seconds
        self.max_freq = max_freq
        self.n_mfcc = n_mfcc
        self.batch_size = batch_size
        self.max_clip_duration = max_clip_duration
        self.scaler = StandardScaler()
        
        # Feature dimension calculation
        self._sample_rate = 16000  # Default sample rate
        self.feature_dim = self._calculate_feature_dim()
        
        # Data storage
        self.data_splits = {
            "train": {"paths": [], "texts": [], "durations": []},
            "valid": {"paths": [], "texts": [], "durations": []},
            "test": {"paths": [], "texts": [], "durations": []}
        }
        self.current_indices = {"train": 0, "valid": 0, "test": 0}
        
        # Load metadata if provided
        if metadata_path:
            self.load_metadata(metadata_path)

    def _calculate_feature_dim(self) -> int:
        """
        Calculate the dimension of the extracted features.
        """
        if self.feature_type == "spectrogram":
            # Approximate number of frequency bins based on max_freq and sample rate
            return int(self.max_freq * self.frame_length / 2) + 1
        else:
            return self.n_mfcc

    def load_metadata(self, metadata_path: Union[str, Path], split: str = "all"):
        """
        Load metadata from a JSON file containing audio paths, durations, and transcriptions.

        Args:
            metadata_path (str or Path): Path to the JSON file.
            split (str): Data split to load ('train', 'valid', 'test', or 'all').
        """
        logger.info(f"Loading metadata from {metadata_path}")
        splits = [split] if split != "all" else ["train", "valid", "test"]
        
        with open(metadata_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                duration = float(entry.get("duration", 0))
                if duration > self.max_clip_duration:
                    continue
                
                audio_path = entry.get("audio_path")
                text = entry.get("transcription", "")
                target_split = entry.get("split", "train")
                
                if target_split in splits:
                    self.data_splits[target_split]["paths"].append(audio_path)
                    self.data_splits[target_split]["texts"].append(text)
                    self.data_splits[target_split]["durations"].append(duration)
        
        for s in splits:
            logger.info(f"Loaded {len(self.data_splits[s]['paths'])} samples for {s} split")

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract features from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: Extracted features (spectrogram or MFCC).
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self._sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, self._sample_rate)(waveform)
            waveform = waveform.mean(dim=0)  # Convert to mono
            
            if self.feature_type == "spectrogram":
                transform = torchaudio.transforms.Spectrogram(
                    n_fft=int(self.frame_length * self._sample_rate),
                    win_length=int(self.frame_length * self._sample_rate),
                    hop_length=int(self.frame_step * self._sample_rate),
                    power=2.0
                )
                features = transform(waveform)
                # Trim to max_freq
                freq_bins = int(self.max_freq * features.shape[0] / (self._sample_rate / 2))
                features = features[:freq_bins, :]
            else:
                transform = torchaudio.transforms.MFCC(
                    sample_rate=self._sample_rate,
                    n_mfcc=self.n_mfcc,
                    melkwargs={
                        "n_fft": int(self.frame_length * self._sample_rate),
                        "hop_length": int(self.frame_step * self._sample_rate)
                    }
                )
                features = transform(waveform)
            
            return features.numpy().T  # Time axis first
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return np.zeros((100, self.feature_dim))

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using the fitted scaler.

        Args:
            features (np.ndarray): Features to normalize.

        Returns:
            np.ndarray: Normalized features.
        """
        return self.scaler.transform(features)

    def fit_normalizer(self, split: str = "train", num_samples: int = 100):
        """
        Fit the feature normalizer using a sample of the data.

        Args:
            split (str): Data split to use for fitting.
            num_samples (int): Number of samples to use.
        """
        logger.info(f"Fitting normalizer using {num_samples} samples from {split} split")
        samples = np.random.choice(self.data_splits[split]["paths"], 
                                 min(num_samples, len(self.data_splits[split]["paths"])), 
                                 replace=False)
        all_features = []
        for path in samples:
            features = self.extract_features(path)
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)
        logger.info("Normalizer fitted")

    def _text_to_indices(self, text: str) -> List[int]:
        """
        Convert text to a sequence of integer indices.

        Args:
            text (str): Input text.

        Returns:
            List[int]: Sequence of character indices.
        """
        char_map = {c: i for i, c in enumerate(" abcdefghijklmnopqrstuvwxyz'")}
        return [char_map.get(c, 0) for c in text.lower()]

    def get_data_batch(self, split: str) -> Tuple[Dict, Dict]:
        """
        Generate a batch of data for the specified split.

        Args:
            split (str): Data split ('train', 'valid', or 'test').

        Returns:
            Tuple[Dict, Dict]: Inputs and outputs for the batch.
        """
        if split not in self.data_splits:
            raise ValueError(f"Invalid split: {split}")
        
        paths = self.data_splits[split]["paths"]
        texts = self.data_splits[split]["texts"]
        start_idx = self.current_indices[split]
        
        batch_paths = paths[start_idx:start_idx + self.batch_size]
        batch_texts = texts[start_idx:start_idx + self.batch_size]
        
        # Extract and normalize features
        batch_features = [self.normalize_features(self.extract_features(p)) for p in batch_paths]
        
        # Calculate lengths
        max_time = max(f.shape[0] for f in batch_features)
        max_text_len = max(len(t) for t in batch_texts)
        
        # Initialize arrays
        X = np.zeros((len(batch_features), max_time, self.feature_dim))
        labels = np.ones((len(batch_features), max_text_len)) * -1  # Padding with -1
        input_lengths = np.zeros((len(batch_features), 1))
        label_lengths = np.zeros((len(batch_features), 1))
        
        for i, (features, text) in enumerate(zip(batch_features, batch_texts)):
            X[i, :features.shape[0], :] = features
            input_lengths[i] = features.shape[0]
            text_indices = self._text_to_indices(text)
            labels[i, :len(text_indices)] = text_indices
            label_lengths[i] = len(text_indices)
        
        inputs = {
            "features": X,
            "labels": labels,
            "feature_lengths": input_lengths,
            "label_lengths": label_lengths
        }
        outputs = {"loss": np.zeros(len(batch_features))}  # Placeholder for CTC loss
        
        self.current_indices[split] += self.batch_size
        if self.current_indices[split] >= len(paths):
            self.current_indices[split] = 0
            if split != "test":
                self._shuffle_split(split)
        
        return inputs, outputs

    def _shuffle_split(self, split: str):
        """
        Shuffle the data for the specified split.

        Args:
            split (str): Data split to shuffle.
        """
        logger.info(f"Shuffling {split} split")
        indices = np.random.permutation(len(self.data_splits[split]["paths"]))
        self.data_splits[split]["paths"] = [self.data_splits[split]["paths"][i] for i in indices]
        self.data_splits[split]["texts"] = [self.data_splits[split]["texts"][i] for i in indices]
        self.data_splits[split]["durations"] = [self.data_splits[split]["durations"][i] for i in indices]

    def data_generator(self, split: str):
        """
        Create a generator for the specified split.

        Args:
            split (str): Data split ('train', 'valid', or 'test').

        Yields:
            Tuple[Dict, Dict]: Batch of inputs and outputs.
        """
        while True:
            yield self.get_data_batch(split)

    def visualize_sample(self, index: int = 0, split: str = "train"):
        """
        Visualize a sample from the specified split.

        Args:
            index (int): Index of the sample to visualize.
            split (str): Data split to use.
        """
        import matplotlib.pyplot as plt
        
        audio_path = self.data_splits[split]["paths"][index]
        text = self.data_splits[split]["texts"][index]
        features = self.extract_features(audio_path)
        
        logger.info(f"Visualizing sample: {audio_path}")
        logger.info(f"Transcription: {text}")
        
        plt.figure(figsize=(10, 4))
        plt.imshow(features.T, aspect="auto", origin="lower")
        plt.title(f"{self.feature_type.capitalize()} Features")
        plt.xlabel("Time Frames")
        plt.ylabel("Feature Dimension")
        plt.colorbar(label="Amplitude")
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Audio Featurizer for Speech Processing")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON file")
    parser.add_argument("--feature_type", type=str, default="spectrogram", choices=["spectrogram", "mfcc"],
                        help="Type of features to extract")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    # Initialize featurizer
    featurizer = AudioFeaturizer(
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        metadata_path=args.metadata
    )
    
    # Fit normalizer
    featurizer.fit_normalizer()
    
    # Example: Generate one batch and visualize a sample
    train_gen = featurizer.data_generator("train")
    inputs, outputs = next(train_gen)
    logger.info(f"Batch feature shape: {inputs['features'].shape}")
    
    # Visualize first sample
    featurizer.visualize_sample(index=0, split="train")

if __name__ == "__main__":
    main()