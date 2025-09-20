import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthFeatureNet(nn.Module):
    """
    A simple CNN to extract features from a single depth map.
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, out_channels, 3, 1, 1)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class DepthBranch(nn.Module):
    """
    Generates a 3D feature volume from an initial depth map.
    """
    def __init__(self, feature_channels):
        super().__init__()
        self.feature_extractor = DepthFeatureNet(out_channels=feature_channels)

    def forward(self, ref_init_depth, depth_hypotheses):
        """
        Args:
            ref_init_depth (torch.Tensor): Reference view initial depth (B, 1, H, W)
            depth_hypotheses (torch.Tensor): (B, D)

        Returns:
            torch.Tensor: Prior volume (B, C, D, H, W)
        """
        B, _, H, W = ref_init_depth.shape
        D = depth_hypotheses.shape[1]

        # 1. Extract 2D features from the initial depth map
        depth_features_2d = self.feature_extractor(ref_init_depth)
        C = depth_features_2d.shape[1]

        # 2. Build the 3D prior volume (Feature Splatting)
        prior_volume = torch.zeros(B, C, D, H, W, device=ref_init_depth.device)

        for i in range(B):
            depth_map = ref_init_depth[i, 0] # (H, W)
            hypotheses = depth_hypotheses[i] # (D)

            abs_diff = torch.abs(depth_map.unsqueeze(-1) - hypotheses)
            closest_idx = torch.argmin(abs_diff, dim=-1) # (H, W)

            # --- 修改开始：优化内存占用 ---

            # 使用scatter_方法，这是一种原地操作，比one_hot更节省内存
            # 创建一个 (D, H, W) 的零张量
            one_hot_volume = torch.zeros(D, H, W, device=depth_map.device)

            # 将 closest_idx 扩展到与 one_hot_volume 相同的维度数量，以便进行索引
            # closest_idx: (H, W) -> (1, H, W)
            # 在第0维（深度维）上，根据 closest_idx 将值1填充进去
            one_hot_volume.scatter_(0, closest_idx.unsqueeze(0), 1)

            # one_hot_volume 现在是一个稀疏的one-hot体积 (D, H, W)

            # 避免一次性创建 (C, D, H, W) 的巨大中间张量
            # 我们通过广播机制逐通道地填充 prior_volume
            # depth_features_2d[i]: (C, H, W) -> (C, 1, H, W)
            # one_hot_volume: (D, H, W) -> (1, D, H, W)
            # 这样广播后的结果直接写入 prior_volume[i]，避免了巨大的中间分配
            prior_volume[i] = depth_features_2d[i].unsqueeze(1) * one_hot_volume.unsqueeze(0)

            # --- 修改结束 ---

        return prior_volume
