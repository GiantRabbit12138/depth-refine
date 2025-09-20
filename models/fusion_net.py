import torch
import torch.nn as nn
import torch.nn.functional as F # 导入 functional

def conv3d_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    """
    A simplified 3D U-Net for regularizing the fused volume.
    """
    def __init__(self, in_channels, base_channels=8):
        super().__init__()
        # Encoder
        self.enc1 = conv3d_block(in_channels, base_channels)
        self.enc2 = conv3d_block(base_channels, base_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # Pool only spatial dims

        # Bottleneck
        self.bottleneck = conv3d_block(base_channels * 2, base_channels * 4)

        # Decoder
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = conv3d_block(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = conv3d_block(base_channels * 2, base_channels)

        # Final layer to get probability volume
        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder
        d2 = self.upconv2(b)

        # --- 新增代码：对齐 d2 和 e2 的尺寸 ---
        # 获取 e2 的目标尺寸 (D, H, W)
        target_size = e2.shape[2:]
        # 使用 F.interpolate 将 d2 调整为目标尺寸
        # 这比手动裁剪更通用和安全
        d2 = F.interpolate(d2, size=target_size, mode='trilinear', align_corners=False)
        # --- 修改结束 ---

        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)

        # --- 新增代码：对齐 d1 和 e1 的尺寸 ---
        target_size = e1.shape[2:]
        d1 = F.interpolate(d1, size=target_size, mode='trilinear', align_corners=False)
        # --- 修改结束 ---

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Output
        prob_volume = self.out_conv(d1).squeeze(1) # (B, D, H, W)

        return prob_volume
