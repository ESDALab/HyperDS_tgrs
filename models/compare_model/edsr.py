import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class EDSR(nn.Module):
    def __init__(self, upscale_factor=4, num_res_blocks=16,h8_in_channel=8, era5_in_channel=5):
        super(EDSR, self).__init__()
        self.h8_head = nn.Conv2d(h8_in_channel, h8_in_channel, kernel_size=3, stride=1, padding=1)
        self.in_channel = h8_in_channel + era5_in_channel
        # First layer
        self.conv1 = nn.Conv2d(self.in_channel, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Upscale block
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor)
        )
        
        # Final output block
        self.conv3 = nn.Conv2d(64, out_channels=era5_in_channel, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, h8_data, era5_data):
        BSize = h8_data.shape[0]
        target_size = h8_data.shape[-2:]
        h8_data = self.h8_head(h8_data.reshape(BSize, -1, h8_data.shape[3], h8_data.shape[4]))
        h8_data = nn.functional.adaptive_avg_pool2d(h8_data, (era5_data.shape[-2:]))
        x = torch.cat([h8_data, era5_data], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        residual = x
        x = self.resblocks(x)
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.upscale(x)
        x = self.conv3(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x

# h8_data = torch.zeros(1,2,4,145,225)
# ear5_data = torch.zeros(1,1,5,37,57)
# model = EDSR(out_channel=5)

# y = model(h8_data,ear5_data)
# import pdb
# pdb.set_trace()