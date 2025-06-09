""" U-Net with Attention Mechanisms """

from .unet_parts import *
from .attention import *


class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention_type='cbam'):
        """
        U-Net with attention mechanisms
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
            bilinear (bool, optional): Whether to use bilinear upsampling. Defaults to False.
            attention_type (str, optional): Type of attention to use ('channel', 'spatial', 'cbam', 'self_attention').
                                            Defaults to 'cbam'.
        """
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention_type = attention_type

        # Same structure as the original U-Net but with attention modules
        self.inc = DoubleConvWithAttention(n_channels, 64, attention_type)
        self.down1 = AttentionDown(64, 128, attention_type)
        self.down2 = AttentionDown(128, 256, attention_type)
        self.down3 = AttentionDown(256, 512, attention_type)
        factor = 2 if bilinear else 1
        self.down4 = AttentionDown(512, 1024 // factor, attention_type)
        self.up1 = AttentionUp(1024, 512 // factor, bilinear, attention_type)
        self.up2 = AttentionUp(512, 256 // factor, bilinear, attention_type)
        self.up3 = AttentionUp(256, 128 // factor, bilinear, attention_type)
        self.up4 = AttentionUp(128, 64, bilinear, attention_type)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
