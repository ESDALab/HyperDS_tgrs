import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)

        return x + residual


class ResidualInResidual(nn.Module):
    def __init__(self, channels):
        super(ResidualInResidual, self).__init__()

        self.residual_blocks = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels)
        )

    def forward(self, x):
        return x + self.residual_blocks(x)


class RCAN(nn.Module):
    def __init__(self, upscale_factor=4, n_resgroups=10, n_resblocks=20, n_feats=64,
                 h8_in_channel=8, era5_in_channel=5):
        super(RCAN, self).__init__()
        self.h8_head = nn.Conv2d(h8_in_channel, h8_in_channel, kernel_size=3, stride=1, padding=1)
        self.in_channel = h8_in_channel + era5_in_channel
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats

        self.head = nn.Sequential(
            nn.Conv2d(self.in_channel, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        _resgroups = [
            ResidualInResidual(n_feats) for _ in range(n_resgroups)
        ]
        _resgroups.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.body = nn.Sequential(*_resgroups)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ConvTranspose2d(n_feats, era5_in_channel, upscale_factor, stride=upscale_factor, bias=True)
        )

    def forward(self, h8_data, era5_data):
        BSize = h8_data.shape[0]
        target_size = h8_data.shape[-2:]
        h8_data = self.h8_head(h8_data.reshape(BSize, -1, h8_data.shape[3], h8_data.shape[4]))
        h8_data = nn.functional.adaptive_avg_pool2d(h8_data, (era5_data.shape[-2:]))
        x = torch.cat([h8_data, era5_data], dim=1)
        x = self.head(x)

        residual = self.body(x)
        residual += x

        out = self.tail(residual)
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        return out

# h8_data = torch.zeros(1,2,4,145,225)
# ear5_data = torch.zeros(1,1,5,37,57)
# model = RCAN()

# y = model(h8_data,ear5_data)
# import pdb
# pdb.set_trace()
