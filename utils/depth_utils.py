import torch

def soft_argmax(prob_volume, depth_hypotheses):
    """
    Calculates the expected depth map from a probability volume.

    Args:
        prob_volume (torch.Tensor): Probability volume (B, D, H, W)
        depth_hypotheses (torch.Tensor): Depth values (B, D)

    Returns:
        torch.Tensor: Expected depth map (B, H, W)
    """
    B, D, H, W = prob_volume.shape

    # Reshape depth_hypotheses to (B, D, 1, 1) for broadcasting
    d_shape = (-1, D, 1, 1)
    depths = depth_hypotheses.view(d_shape)

    # Weighted sum along the depth dimension
    expected_depth = torch.sum(prob_volume * depths, dim=1)

    return expected_depth