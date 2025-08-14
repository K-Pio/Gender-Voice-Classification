import torch
from torch import nn

class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, padding='same', bias=True, causal=False):
        super().__init__()
        assert kernel_size >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.padding_mode = padding
        self.causal = bool(causal)

        # Kaiming init for conv weights
        w = torch.empty(out_channels, in_channels, kernel_size)
        nn.init.kaiming_normal_(w, nonlinearity='relu')
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _pad(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        rf = (self.kernel_size - 1) * self.dilation + 1
        if self.causal:
            pad_left = rf - 1
            pad_right = 0
        elif self.padding_mode == 'same':
            total = rf - 1
            pad_left = total // 2
            pad_right = total - pad_left
        else:
            pad_left = pad_right = 0
        if pad_left == 0 and pad_right == 0:
            return x
        pad_l = torch.zeros(B, C, pad_left, dtype=x.dtype, device=x.device)
        pad_r = torch.zeros(B, C, pad_right, dtype=x.dtype, device=x.device)
        return torch.cat([pad_l, x, pad_r], dim=2)

    def forward(self, x):
        # x: (B, C_in, T)
        x = self._pad(x)  # (B, C_in, T_pad)
        B, C, T = x.shape
        rf = (self.kernel_size - 1) * self.dilation + 1

        windows = x.unfold(dimension=2, size=rf, step=self.stride)

        idx = torch.arange(0, rf, self.dilation, device=x.device)  # (K,)
        # (B, C, L_out, K)
        windows = windows.index_select(dim=3, index=idx)

        B, Cin, Lout, K = windows.shape
        assert K == self.kernel_size

        windows = windows.permute(0, 2, 1, 3).reshape(B, Lout, Cin * K)

        W = self.weight.reshape(self.out_channels, Cin * K)
        # out: (B, L_out, Cout)
        out = windows @ W.t()
        if self.bias is not None:
            out = out + self.bias

        # (B, Cout, L_out)
        return out.permute(0, 2, 1)


class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x: (B, C, T)
        if self.training:
            self.num_batches_tracked += 1

            mean = x.mean(dim=(0, 2))  # (C,)
            var = x.var(dim=(0, 2), unbiased=False)  # (C,)

            momentum = self.momentum
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean.detach()
            self.running_var = (1 - momentum) * self.running_var + momentum * var.detach()

            x_hat = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        else:
            x_hat = (x - self.running_mean[None, :, None]) / torch.sqrt(self.running_var[None, :, None] + self.eps)

        if self.affine:
            x_hat = x_hat * self.weight[None, :, None] + self.bias[None, :, None]
        return x_hat


class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return mask * x / (1.0 - self.p)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 dropout=0.1, causal=False):
        super().__init__()
        self.conv1 = CustomConv1d(in_channels, out_channels, kernel_size,
                                  stride=1, dilation=dilation, padding='same', causal=causal)
        self.bn1 = CustomBatchNorm1d(out_channels)
        self.conv2 = CustomConv1d(out_channels, out_channels, kernel_size,
                                  stride=1, dilation=dilation, padding='same', causal=causal)
        self.bn2 = CustomBatchNorm1d(out_channels)
        self.drop = CustomDropout(dropout)
        self.residual = None
        if in_channels != out_channels:
            self.residual = CustomConv1d(in_channels, out_channels, kernel_size=1, padding='none')

    def forward(self, x):
        # x: (B, C_in, T)
        residual = x if self.residual is None else self.residual(x)
        out = self.conv1(x)
        out = out.clamp_min_(0)      # ReLU
        out = self.bn1(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = out.clamp_min_(0)      # ReLU
        out = self.bn2(out)
        out = self.drop(out)
        return out + residual

class TinyTCN(nn.Module):
    def __init__(self, in_features, channels=(64, 128), ks=3, dilations=(1, 2), causal=False, num_classes=2):
        super().__init__()
        blocks = []
        c_in = in_features
        for c_out, d in zip(channels, dilations):
            blocks.append(TCNBlock(c_in, c_out, kernel_size=ks, dilation=d, dropout=0.2, causal=causal))
            c_in = c_out
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        # x: (B, T, F) → permutation (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)                       # (B, C, T)
        x = self.pool(x).squeeze(-1)          # (B, C)
        return self.fc(x)                     # (B, num_classes)
