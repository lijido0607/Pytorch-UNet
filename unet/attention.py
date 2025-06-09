"""
Attention mechanisms for U-Net model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module as used in SENet
    
    Attention applied on channel dimension using global average pooling
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Attention applied on spatial dimensions (height and width)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        
        return self.sigmoid(y) * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Combines channel and spatial attention mechanisms
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-Attention Module
    
    Non-local self attention mechanism
    """
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1):
        super(SelfAttention, self).__init__()
        self.scale = scale
        
        self.in_channels = in_channels
        self.key_channels = key_channels or in_channels // 8
        self.value_channels = value_channels or in_channels
        
        self.query_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.value_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Query, Key, Value projections
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C
        key = self.key_conv(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        
        # Scaled dot-product attention
        energy = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = F.softmax(energy * self.scale, dim=-1)
        
        # Output
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, self.value_channels, height, width)
        
        out = self.gamma * out + x
        
        return out


class DoubleConvWithAttention(nn.Module):
    """Double Convolution with attention mechanism
    
    (convolution => [BN] => ReLU) * 2 + attention
    """
    def __init__(self, in_channels, out_channels, attention_type='cbam', mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Add attention module
        if attention_type == 'channel':
            self.attention = ChannelAttention(out_channels)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        elif attention_type == 'cbam':
            self.attention = CBAM(out_channels)
        elif attention_type == 'self_attention':
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = None
            
    def forward(self, x):
        x = self.double_conv(x)
        if self.attention:
            x = self.attention(x)
        return x


# Classes to replace Down and Up with attention versions
class AttentionDown(nn.Module):
    """Downscaling with maxpool then double conv with attention"""

    def __init__(self, in_channels, out_channels, attention_type='cbam'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWithAttention(in_channels, out_channels, attention_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionUp(nn.Module):
    """Upscaling then double conv with attention"""

    def __init__(self, in_channels, out_channels, bilinear=True, attention_type='cbam'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvWithAttention(in_channels, out_channels, attention_type, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvWithAttention(in_channels, out_channels, attention_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)