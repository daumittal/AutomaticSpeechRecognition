import torch
import torch.nn as nn
from typing import Callable, Optional

class BaseSpeechModel(nn.Module):
    """
    Base class for speech recognition models.
    """
    def __init__(self):
        super(BaseSpeechModel, self).__init__()
        self._output_length_fn = lambda x: x

    def output_length(self, input_length: int) -> int:
        """
        Compute the output sequence length.

        Args:
            input_length (int): Length of the input sequence.

        Returns:
            int: Length of the output sequence.
        """
        return self._output_length_fn(input_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward")

class BasicRecurrentModel(BaseSpeechModel):
    """
    A simple recurrent neural network for speech recognition.
    """
    def __init__(self, input_dim: int, output_dim: int = 29):
        """
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Number of output classes (default: 29).
        """
        super(BasicRecurrentModel, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        out, _ = self.rnn(x)
        return self.softmax(out)

class EnhancedRecurrentModel(BaseSpeechModel):
    """
    A recurrent neural network with LSTM and normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "tanh", output_dim: int = 29):
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Number of hidden units in LSTM.
            activation (str): Activation function for LSTM.
            output_dim (int): Number of output classes (default: 29).
        """
        super(EnhancedRecurrentModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = getattr(torch, activation, torch.tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        out, _ = self.lstm(x)
        out = self.activation(out)
        out = self.norm(out)
        out = self.dense(out)
        return self.softmax(out)

class ConvRecurrentModel(BaseSpeechModel):
    """
    A convolutional + recurrent neural network for speech recognition.
    """
    def __init__(
        self,
        input_dim: int,
        conv_filters: int,
        kernel_size: int,
        stride: int,
        padding: str,
        hidden_dim: int,
        output_dim: int = 29
    ):
        """
        Args:
            input_dim (int): Dimension of input features.
            conv_filters (int): Number of convolutional filters.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Convolution stride.
            padding (str): Padding mode ("same" or "valid").
            hidden_dim (int): Number of hidden units in GRU.
            output_dim (int): Number of output classes (default: 29).
        """
        super(ConvRecurrentModel, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same" if padding == "same" else 0
        )
        self.norm1 = nn.LayerNorm(conv_filters)
        self.gru = nn.GRU(
            input_size=conv_filters,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

        def compute_output_length(input_length: int) -> int:
            if padding == "same":
                return input_length
            return (input_length - kernel_size + 1 + stride - 1) // stride

        self._output_length_fn = compute_output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_filters)
        x = torch.relu(x)
        x = self.norm1(x)
        x, _ = self.gru(x)
        x = self.norm2(x)
        x = self.dense(x)
        return self.softmax(x)

class DeepRecurrentModel(BaseSpeechModel):
    """
    A deep recurrent neural network with multiple LSTM layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 29):
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Number of hidden units per LSTM layer.
            num_layers (int): Number of LSTM layers.
            output_dim (int): Number of output classes (default: 29).
        """
        super(DeepRecurrentModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=input_dim if i == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True
                )
            )
            self.layers.append(nn.LayerNorm(hidden_dim))
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        for i in range(0, len(self.layers), 2):
            x, _ = self.layers[i](x)
            x = torch.relu(x)
            x = self.layers[i + 1](x)
        x = self.dense(x)
        return self.softmax(x)

class BidirectionalRecurrentModel(BaseSpeechModel):
    """
    A bidirectional recurrent neural network for speech recognition.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 29):
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Number of hidden units per LSTM direction.
            output_dim (int): Number of output classes (default: 29).
        """
        super(BidirectionalRecurrentModel, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dense = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        x, _ = self.bilstm(x)
        x = self.dense(x)
        return self.softmax(x)

class AdvancedSpeechModel(BaseSpeechModel):
    """
    A comprehensive speech recognition model with convolution and deep recurrent layers.
    """
    def __init__(
        self,
        input_dim: int,
        conv_filters: int,
        kernel_size: int,
        stride: int,
        padding: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.4,
        cell_type: str = "gru",
        activation: str = "tanh",
        output_dim: int = 29
    ):
        """
        Args:
            input_dim (int): Dimension of input features.
            conv_filters (int): Number of convolutional filters.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Convolution stride.
            padding (str): Padding mode ("same" or "valid").
            hidden_dim (int): Number of hidden units per recurrent layer.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout rate for recurrent layers.
            cell_type (str): Type of recurrent cell ("gru" or "lstm").
            activation (str): Activation function for recurrent layers.
            output_dim (int): Number of output classes (default: 29).
        """
        super(AdvancedSpeechModel, self).__init__()
        self.cell = nn.GRU if cell_type.lower() == "gru" else nn.LSTM
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same" if padding == "same" else 0
        )
        self.norm1 = nn.LayerNorm(conv_filters)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                self.cell(
                    input_size=conv_filters if i == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout if i < num_layers - 1 else 0
                )
            )
            self.layers.append(nn.LayerNorm(hidden_dim))
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = getattr(torch, activation, torch.tanh)

        def compute_output_length(input_length: int) -> int:
            if padding == "same":
                return input_length
            return (input_length - kernel_size + 1 + stride - 1) // stride

        self._output_length_fn = compute_output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_filters)
        x = torch.relu(x)
        x = self.norm1(x)
        for i in range(0, len(self.layers), 2):
            x, _ = self.layers[i](x)
            x = self.activation(x)
            x = self.layers[i + 1](x)
        x = self.dense(x)
        return self.softmax(x)

def build_model(model_name: str, **kwargs) -> BaseSpeechModel:
    """
    Factory function to build a speech model.

    Args:
        model_name (str): Name of the model to build.
        **kwargs: Model-specific parameters.

    Returns:
        BaseSpeechModel: Instantiated model.
    """
    models = {
        "basic_recurrent": BasicRecurrentModel,
        "enhanced_recurrent": EnhancedRecurrentModel,
        "conv_recurrent": ConvRecurrentModel,
        "deep_recurrent": DeepRecurrentModel,
        "bidirectional_recurrent": BidirectionalRecurrentModel,
        "advanced_speech": AdvancedSpeechModel
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    model = models[model_name](**kwargs)
    print(f"Model {model_name} summary:")
    print(model)
    return model