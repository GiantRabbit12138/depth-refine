import torch
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # 添加项目根目录到 Python 路径
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.mvs_dataset import MVSDataset
from models.two_stream_mvs import TwoStreamMVS
from utils.losses import CombinedLoss

def main(config):
    # Setup
    torch.manual_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = MVSDataset(
        data_path=config['train']['data_path'],
        num_views=config['model']['num_views'],
        transform=train_transform,
        resize_scale=config['model']['resize_scale']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )

    # Model
    model = TwoStreamMVS(config).to(device)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = CombinedLoss(smoothness_weight=config['loss']['smoothness_weight'])

    # Training Loop
    for epoch in range(config['train']['epochs']):
        model.train()
        for i, batch in enumerate(train_loader):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            outputs = model(batch)
            loss_dict = criterion(outputs, batch)

            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{config['train']['epochs']}, Step {i}, Loss: {loss.item():.4f}")

    print("Training finished.")
    # Save model
    torch.save(model.state_dict(), f"{config['train']['log_path']}/final_model.pth")

if __name__ == '__main__':
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)
