import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from datasets.mvs_dataset import MVSDataset
from models.two_stream_mvs import TwoStreamMVS
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import gc # 导入垃圾回收模块

# compute_depth_metrics 和 save_depth_visualization 函数保持不变
def compute_depth_metrics(pred, gt):
    """计算单个深度图的评估指标"""
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    mask = gt > 0
    if not mask.any():
        return None
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    pred_valid = torch.clamp(pred_valid, min=1e-3)
    gt_valid = torch.clamp(gt_valid, min=1e-3)
    thresh = torch.max((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = torch.sqrt(((gt_valid - pred_valid) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt_valid) - torch.log(pred_valid)) ** 2).mean())
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    return {'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'rmse': rmse.item(),
            'rmse_log': rmse_log.item(), 'a1': a1.item(), 'a2': a2.item(), 'a3': a3.item()}

def save_depth_visualization(path, depth_map):
    """将深度图保存为彩色的可视化图像"""
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    colormap = plt.get_cmap('magma')
    depth_colored = (colormap(depth_normalized.numpy()) * 255).astype(np.uint8)
    Image.fromarray(depth_colored[:, :, :3]).save(path)


def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 修改：移除硬编码尺寸，保留分块大小 ---
    chunk_size = 10 # 每10个样本清理一次内存
    # ------------------------------------

    output_dir = os.path.join(config['train']['log_path'], "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Evaluation outputs will be saved to: {output_dir}")

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 修改：从config文件读取resize_scale，并传递给数据集 ---
    resize_scale = config['model'].get('resize_scale', 1.0)
    print(f"Using resize_scale: {resize_scale} for evaluation.")

    eval_dataset = MVSDataset(
        data_path=config['train']['data_path'],
        num_views=config['model']['num_views'],
        transform=eval_transform,
        resize_scale=resize_scale # 统一使用 resize_scale
    )
    # -----------------------------------------------------------

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    model = TwoStreamMVS(config).to(device)
    model_path = os.path.join(config['train']['log_path'], "final_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(config['train']['log_path'], "checkpoint_epoch_20.pth")
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 后续的分块评估和内存清理逻辑保持不变
    all_metrics = []
    chunk_metrics = []
    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            outputs = model(batch)
            refined_depth = outputs['refined_depth']
            gt_depth = batch['gt_depth'].squeeze(1)
            metrics = compute_depth_metrics(refined_depth, gt_depth)
            if metrics:
                all_metrics.append(metrics)
                chunk_metrics.append(metrics)
            if i < 50:
                pred_depth_cpu = refined_depth.squeeze(0).cpu()
                gt_depth_cpu = gt_depth.squeeze(0).cpu()
                save_depth_visualization(os.path.join(output_dir, f"{i:04d}_pred.png"), pred_depth_cpu)
                save_depth_visualization(os.path.join(output_dir, f"{i:04d}_gt.png"), gt_depth_cpu)
            if (i + 1) % chunk_size == 0 or (i + 1) == len(eval_loader):
                if chunk_metrics:
                    avg_chunk_metrics = {key: np.mean([m[key] for m in chunk_metrics]) for key in chunk_metrics[0]}
                    print(f"\n--- Chunk {i//chunk_size + 1} (samples {i-len(chunk_metrics)+2}-{i+1}) Avg Metrics ---")
                    print(f"  AbsRel: {avg_chunk_metrics['abs_rel']:.4f}, RMSE: {avg_chunk_metrics['rmse']:.4f}")
                    chunk_metrics = []
                print(f"--- Clearing CUDA cache at sample {i+1} ---\n")
                del outputs, refined_depth, gt_depth, batch
                gc.collect()
                torch.cuda.empty_cache()

    if all_metrics:
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
        print("\n" + "="*40 + "\nEvaluation Finished. Final Average Metrics:\n" + "="*40)
        print(f"Absolute Relative Error (AbsRel): {avg_metrics['abs_rel']:.4f}")
        print(f"Squared Relative Error (SqRel):  {avg_metrics['sq_rel']:.4f}")
        print(f"Root Mean Squared Error (RMSE):  {avg_metrics['rmse']:.4f}")
        print(f"Log RMSE (RMSElog):              {avg_metrics['rmse_log']:.4f}")
        print(f"Threshold δ < 1.25 (a1):         {avg_metrics['a1']:.4f}")
        print(f"Threshold δ < 1.25² (a2):        {avg_metrics['a2']:.4f}")
        print(f"Threshold δ < 1.25³ (a3):        {avg_metrics['a3']:.4f}")
        print("="*40)
    else:
        print("Evaluation could not be completed as no valid data was found.")

if __name__ == '__main__':
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    evaluate(config)
