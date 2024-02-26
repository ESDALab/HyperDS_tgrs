import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

def upconv2x2(in_channels, out_channels, mode='interpolate'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else: # mode == 'interpolate'
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
            conv3x3(in_channels, out_channels)
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, h8_in_channel=8, era5_in_channel=5, upscale_factor=4):
        super().__init__()
        self.h8_head = nn.Conv2d(h8_in_channel, h8_in_channel, kernel_size=3, stride=1, padding=1)
        self.in_channel = h8_in_channel + era5_in_channel
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(self.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改 center 层以匹配输入尺寸
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=1),  # 1x1 卷积进行通道数减半
            nn.ReLU(inplace=True)
        )

        # 修改解码器以适应逐步上采样
        self.dec5 = DecoderBlock(1024, 512)
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = nn.Conv2d(64, era5_in_channel, kernel_size=1)  # 1x1 卷积以适配输出通道数

    def forward(self, h8_data, era5_data):
        BSize = h8_data.shape[0]
        target_size = h8_data.shape[-2:]
        h8_data = self.h8_head(h8_data.reshape(BSize, -1, h8_data.shape[3], h8_data.shape[4]))
        h8_data = nn.functional.adaptive_avg_pool2d(h8_data, (era5_data.shape[-2:]))
        x = torch.cat([h8_data, era5_data], dim=1)
        conv1 = self.encoder.conv1(x)
        conv2 = self.encoder.layer1(conv1)
        conv3 = self.encoder.layer2(conv2)
        conv4 = self.encoder.layer3(conv3)
        conv5 = self.encoder.layer4(conv4)

        center = self.center(conv5)

        dec5 = self.dec5(center)
        dec4 = self.dec4(dec5)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)

        return F.interpolate(dec1, size=target_size, mode='bilinear', align_corners=True)
