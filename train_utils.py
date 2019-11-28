import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path
import logging
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CTCLossWrapper(nn.Module):
    """
    Wrapper to compute CTC loss for a speech recognition model.
    """
    def __init__(self, model: nn.Module):
        """
        Args:
            model (nn.Module): The base model predicting softmax probabilities.
        """
        super(CTCLossWrapper, self).__init__()
        self.model = model
        self.ctc_loss = nn.CTCLoss(blank=28, zero_infinity=True)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        feature_lengths: torch.Tensor,
        label_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            features (torch.Tensor): Input features (batch, seq_len, input_dim).
            labels (torch.Tensor): Target label sequences (batch, max_label_len).
            feature_lengths (torch.Tensor): Lengths of input sequences (batch,).
            label_lengths (torch.Tensor): Lengths of label sequences (batch,).

        Returns:
            torch.Tensor: CTC loss value.
        """
        log_probs = self.model(features).log_softmax(dim=-1)  # (batch, seq_len, output_dim)
        log_probs = log_probs.transpose(0, 1)  # (seq_len, batch, output_dim) for CTC
        loss = self.ctc_loss(log_probs, labels, feature_lengths, label_lengths)
        return loss

class AudioDataLoader:
    """
    Simplified data loader for audio data, mimicking AudioGenerator.
    """
    def __init__(
        self,
        json_path: str,
        batch_size: int,
        feature_type: str = "spectrogram",
        max_duration: float = 12.0,
        is_training: bool = False
    ):
        """
        Args:
            json_path (str): Path to JSON-line file with audio metadata.
            batch_size (int): Number of samples per batch.
            feature_type (str): Type of features ("spectrogram" or "mfcc").
            max_duration (float): Maximum audio clip duration in seconds.
            is_training (bool): Whether to shuffle data for training.
        """
        self.batch_size = batch_size
        self.feature_type = feature_type
        self.max_duration = max_duration
        self.is_training = is_training
        self.data = self._load_metadata(json_path)
        self.indices = list(range(len(self.data)))

    def _load_metadata(self, json_path: str) -> list:
        """
        Load metadata from JSON-line file.

        Args:
            json_path (str): Path to JSON file.

        Returns:
            list: List of metadata entries.
        """
        data = []
        with open(json_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("duration", float('inf')) <= self.max_duration:
                    data.append(entry)
        logger.info(f"Loaded {len(data)} samples from {json_path}")
        return data

    def __len__(self) -> int:
        """
        Returns:
            int: Number of batches.
        """
        return len(self.data) // self.batch_size

    def _extract_features(self, audio_path: str) -> torch.Tensor:
        """
        Placeholder for feature extraction (simplified).

        Args:
            audio_path (str): Path to audio file.

        Returns:
            torch.Tensor: Dummy features (replace with actual extraction).
        """
        # TODO: Replace with actual feature extraction (e.g., torchaudio)
        return torch.randn(100, 13 if self.feature_type == "mfcc" else 161)

    def _text_to_indices(self, text: str) -> list:
        """
        Convert text to integer indices.

        Args:
            text (str): Input transcription.

        Returns:
            list: List of integer indices.
        """
        char_map = {c: i for i, c in enumerate(" abcdefghijklmnopqrstuvwxyz'", start=1)}
        char_map[""] = 0  # Blank index
        return [char_map.get(c.lower(), 0) for c in text]

    def __iter__(self):
        """
        Iterator for data batches.

        Yields:
            tuple: (features, labels, feature_lengths, label_lengths).
        """
        if self.is_training:
            torch.manual_seed(42)  # For reproducibility
            torch.randperm(len(self.indices))
        
        for start in range(0, len(self.data) - self.batch_size + 1, self.batch_size):
            batch_data = [self.data[i] for i in self.indices[start:start + self.batch_size]]
            
            max_seq_len = 0
            max_label_len = 0
            for entry in batch_data:
                features = self._extract_features(entry["audio_path"])
                max_seq_len = max(max_seq_len, features.shape[0])
                labels = self._text_to_indices(entry["transcription"])
                max_label_len = max(max_label_len, len(labels))
            
            features = torch.zeros(self.batch_size, max_seq_len, 
                                 13 if self.feature_type == "mfcc" else 161)
            labels = torch.full((self.batch_size, max_label_len), -1, dtype=torch.long)
            feature_lengths = torch.zeros(self.batch_size, dtype=torch.long)
            label_lengths = torch.zeros(self.batch_size, dtype=torch.long)
            
            for i, entry in enumerate(batch_data):
                feat = self._extract_features(entry["audio_path"])
                lab = self._text_to_indices(entry["transcription"])
                features[i, :feat.shape[0], :] = feat
                labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
                feature_lengths[i] = feat.shape[0]
                label_lengths[i] = len(lab)
            
            yield features, labels, feature_lengths, label_lengths

def train_speech_model(
    model: nn.Module,
    history_path: str,
    checkpoint_path: str,
    train_json: str = "train_corpus.json",
    valid_json: str = "valid_corpus.json",
    batch_size: int = 32,
    feature_type: str = "spectrogram",
    feature_dim: int = 161,
    optimizer_type: str = "adam",
    learning_rate: float = 0.001,
    epochs: int = 20,
    max_duration: float = 12.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train a speech recognition model with CTC loss.

    Args:
        model (nn.Module): The base model predicting softmax probabilities.
        history_path (str): Path to save training history (pickle file).
        checkpoint_path (str): Path to save model checkpoints.
        train_json (str): Path to training JSON-line file.
        valid_json (str): Path to validation JSON-line file.
        batch_size (int): Number of samples per batch.
        feature_type (str): Type of features ("spectrogram" or "mfcc").
        feature_dim (int): Dimension of input features.
        optimizer_type (str): Type of optimizer ("adam" or "sgd").
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        max_duration (float): Maximum audio clip duration in seconds.
        device (str): Device to train on ("cuda" or "cpu").
    """
    logger.info(f"Training on {device}")
    
    # Initialize data loaders
    train_loader = AudioDataLoader(
        json_path=train_json,
        batch_size=batch_size,
        feature_type=feature_type,
        max_duration=max_duration,
        is_training=True
    )
    valid_loader = AudioDataLoader(
        json_path=valid_json,
        batch_size=batch_size,
        feature_type=feature_type,
        max_duration=max_duration,
        is_training=False
    )
    
    # Wrap model with CTC loss
    ctc_model = CTCLossWrapper(model).to(device)
    
    # Set up optimizer
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(ctc_model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(
            ctc_model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True
        )
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Training history
    history = {"train_loss": [], "valid_loss": []}
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        ctc_model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in tqdm(train_loader, desc="Training"):
            features, labels, feature_lengths, label_lengths = batch
            features = features.to(device)
            labels = labels.to(device)
            feature_lengths = feature_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            optimizer.zero_grad()
            loss = ctc_model(features, labels, feature_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        ctc_model.eval()
        valid_loss = 0.0
        valid_steps = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                features, labels, feature_lengths, label_lengths = batch
                features = features.to(device)
                labels = labels.to(device)
                feature_lengths = feature_lengths.to(device)
                label_lengths = label_lengths.to(device)
                
                loss = ctc_model(features, labels, feature_lengths, label_lengths)
                valid_loss += loss.item()
                valid_steps += 1
        
        avg_valid_loss = valid_loss / valid_steps
        history["valid_loss"].append(avg_valid_loss)
        
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        # Save checkpoint
        torch.save(ctc_model.state_dict(), results_dir / checkpoint_path)
    
    # Save history
    with open(results_dir / history_path, "wb") as f:
        pickle.dump(history, f)
    
    logger.info("Training completed")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a speech recognition model")
    parser.add_argument("--model", type=str, required=True, help="Path to model file or model name")
    parser.add_argument("--history", type=str, default="history.pkl", help="Path to save training history")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Path to save model checkpoint")
    parser.add_argument("--train-json", type=str, default="train_corpus.json", help="Training JSON file")
    parser.add_argument("--valid-json", type=str, default="valid_corpus.json", help="Validation JSON file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--feature-type", type=str, default="spectrogram", choices=["spectrogram", "mfcc"])
    parser.add_argument("--feature-dim", type=int, default=161, help="Feature dimension")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--max-duration", type=float, default=12.0, help="Max audio duration")
    args = parser.parse_args()
    
    # Placeholder: Load model (replace with actual model loading)
    class DummyModel(nn.Module):
        def __init__(self, input_dim, output_dim=29):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel(input_dim=args.feature_dim)
    
    train_speech_model(
        model=model,
        history_path=args.history,
        checkpoint_path=args.checkpoint,
        train_json=args.train_json,
        valid_json=args.valid_json,
        batch_size=args.batch_size,
        feature_type=args.feature_type,
        feature_dim=args.feature_dim,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_duration=args.max_duration
    )

if __name__ == "__main__":
    main()