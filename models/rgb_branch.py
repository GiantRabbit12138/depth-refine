import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入我们已经定义好的 warp_features 函数
from utils.camera_utils import warp_features

class FeaturePyramidNetwork(nn.Module):
    """
    一个简化的特征金字塔网络(FPN)。
    在实际项目中，为了获得更好的性能，强烈建议使用一个在ImageNet上预训练过的
    ResNet + FPN 结构作为特征提取器。
    """
    def __init__(self, in_channels=3, out_channels_list=[16, 32, 64]):
        super().__init__()
        # 通常FPN包含从下到上和从上到下的路径，这里为了简化，我们只做一个多尺度提取器
        self.conv1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels_list[0])

        self.conv2 = nn.Conv2d(out_channels_list[0], out_channels_list[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels_list[1])

        self.conv3 = nn.Conv2d(out_channels_list[1], out_channels_list[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_list[2])

    def forward(self, x):
        # 返回多尺度的特征图，通常我们使用分辨率较低但语义信息更丰富的特征图
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        # 返回最深层的特征图，也可以返回一个列表 [c3, c2, c1] 用于多尺度处理
        return c3

class RGBBranch(nn.Module):
    """
    RGB分支:
    1. 从多视角RGB图像中提取2D特征。
    2. 通过特征翘曲(warping)和方差计算来构建代价体(Cost Volume)。
    """
    def __init__(self, fpn_channels=[16, 32, 64]):
        super().__init__()
        self.feature_extractor = FeaturePyramidNetwork(out_channels_list=fpn_channels)

    def forward(self, images, cam_intrinsics, cam_poses, depth_hypotheses):
        """
        Args:
            images (torch.Tensor): (B, V, 3, H, W)
            cam_intrinsics (torch.Tensor): (B, V, 3, 3)
            cam_poses (torch.Tensor): (B, V, 4, 4)
            depth_hypotheses (torch.Tensor): (B, D)
        """
        B, V, _, H, W = images.shape
        D = depth_hypotheses.shape[1]
        device = images.device

        # 1. 对所有视图提取特征
        # 将批次(B)和视角(V)维度合并，以便一次性通过2D CNN
        features = self.feature_extractor(images.view(B * V, 3, H, W))

        # --- 修改开始 ---
        # 从特征图 `features` 本身获取其通道数、高度和宽度
        _, C, f_H, f_W = features.shape

        # 使用特征图自己的尺寸 f_H 和 f_W 来恢复 B,V 维度
        features = features.view(B, V, C, f_H, f_W)
        # --- 修改结束 ---

        # 2. 划分参考视图和源视图
        ref_features = features[:, 0]  # (B, C, H, W)
        src_features = features[:, 1:]  # (B, V-1, C, H, W)

        # 3. 准备相机参数
        # 参考相机参数
        ref_cam_pose = cam_poses[:, 0]      # (B, 4, 4)
        ref_cam_intr = cam_intrinsics[:, 0]  # (B, 3, 3)

        # 源相机参数
        src_cam_poses = cam_poses[:, 1:]     # (B, V-1, 4, 4)
        src_cam_intrinsics = cam_intrinsics[:, 1:] # (B, V-1, 3, 3)

        # 4. 计算代价体
        # 初始化一个用于累加翘曲后特征的张量
        warped_features_sum = torch.zeros(B, C, D, f_H, f_W, device=device)

        # 遍历每一个源视图
        for i in range(V - 1):
            current_src_feat = src_features[:, i] # (B, C, H, W)

            # 提取当前源视图对应的相机参数
            current_src_pose = src_cam_poses[:, i]
            current_src_intr = src_cam_intrinsics[:, i]

            # 执行特征翘曲
            warped = warp_features(
                src_feat=current_src_feat,
                ref_cam_pose=ref_cam_pose,
                src_cam_pose=current_src_pose,
                ref_cam_intr=ref_cam_intr,
                src_cam_intr=current_src_intr,
                depth_hypotheses=depth_hypotheses
            ) # 返回 (B, C, D, H, W)

            warped_features_sum += warped

        # 计算平均的翘曲特征
        # V-1 是源视图的数量
        avg_warped_features = warped_features_sum / (V - 1)

        # 5. 计算方差作为代价
        # 将参考视图特征扩展到与深度假设维度匹配
        # ref_features: (B, C, H, W) -> (B, C, 1, H, W)
        # avg_warped_features: (B, C, D, H, W)
        # 广播机制会自动处理减法
        variance_volume = torch.mean((avg_warped_features - ref_features.unsqueeze(2))**2, dim=1, keepdim=True)
        # variance_volume 的形状: (B, 1, D, H, W)

        return variance_volume
