### 2004 ###
from .cuda_bn import o_dropout

import torch
from torch import nn

class OnionDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return o_dropout.custom_dropout(x, self.p, self.training)