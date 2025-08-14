import torch.nn as nn
import torch.nn.functional as F
from new_cuda_layeres.new_conv1d_module import CustomConv1d

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        # self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, 
        #                       padding=padding, dilation=dilation)
        self.conv = CustomConv1d(in_ch, out_ch, kernel_size,
                                         padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return self.dropout(out)

class TCNClassifier(nn.Module):
    def __init__(self, input_size=20, num_classes=2, channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_size
        dilation = 1
        for ch in channels:
            layers.append(TCNBlock(in_ch, ch, kernel_size, dilation, dropout))
            in_ch = ch
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_classes)
    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        x = self.tcn(x)          # [B, C, T]
        x = self.pool(x).squeeze(-1)  # [B, C]
        return self.fc(x)
