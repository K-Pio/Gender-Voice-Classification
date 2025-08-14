import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    A PyTorch-based LSTM classifier for sequence data.

    Attributes:
        input_size (int): The number of features in the input sequence.
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of stacked LSTM layers.
        num_classes (int): The number of output classes for classification.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Initializes the LSTMClassifier.

        Args:
            input_size (int): Number of features in the input sequence.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of output classes.
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Defines the forward pass of the LSTMClassifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes), representing class probabilities.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.fc1(out)

        out = self.fc2(out)

        return out
