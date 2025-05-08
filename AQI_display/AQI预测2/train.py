import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.cnn_gru import CNNGRU
from utils.data_processor import DataProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, y_pred, y_true):
        # 计算MSE和MAE损失
        mse_loss = self.mse(y_pred, y_true)
        mae_loss = self.mae(y_pred, y_true)
        
        # 添加时间衰减权重
        batch_size, seq_len = y_pred.shape[:2]
        time_weights = torch.exp(-torch.arange(seq_len, device=y_pred.device) * 0.1)
        time_weights = time_weights.view(1, -1, 1).expand_as(y_pred)
        
        # 组合损失
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
        weighted_loss = (combined_loss * time_weights).mean()
        
        return weighted_loss

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=500, device='cuda'):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch_features, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        total_test_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                total_test_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_targets.cpu().numpy())
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # 更新学习率
        scheduler.step(avg_test_loss)
        
        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Test Loss: {avg_test_loss:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            
            # 保存预测结果的可视化
            if (epoch + 1) % 50 == 0:
                plot_predictions(predictions, actuals, epoch)
    
    print(f'\nTraining completed! Best model was at epoch {best_epoch+1} with test loss: {best_test_loss:.4f}')
    return train_losses, test_losses

def plot_predictions(predictions, actuals, epoch):
    # 选择第一个样本的AQI预测进行可视化
    pred = predictions[0][:, 0]  # AQI是第一个特征
    actual = actuals[0][:, 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(pred, label='Predicted', marker='x')
    plt.title(f'AQI Prediction vs Actual (Epoch {epoch+1})')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'prediction_epoch_{epoch+1}.png')
    plt.close()

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.6)
    plt.plot(test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_plot.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'd_aqi_huizhou.json')
    
    # 数据处理
    data_processor = DataProcessor(data_path)
    X_train, X_test, y_train, y_test = data_processor.prepare_data()
    train_loader, test_loader = data_processor.create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32)
    
    # 打印数据形状
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # 模型初始化
    input_channels = X_train.shape[2]  # 特征数量
    sequence_length = X_train.shape[1]  # 序列长度
    output_dim = y_train.shape[2]  # 输出维度
    prediction_length = y_train.shape[1]  # 预测序列长度
    
    print(f"Model parameters:")
    print(f"input_channels: {input_channels}")
    print(f"sequence_length: {sequence_length}")
    print(f"output_dim: {output_dim}")
    print(f"prediction_length: {prediction_length}")
    
    model = CNNGRU(
        input_channels=input_channels,
        sequence_length=sequence_length,
        output_dim=output_dim,
        prediction_length=prediction_length
    ).to(device)
    
    # 损失函数和优化器
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期的倍数
        eta_min=1e-6  # 最小学习率
    )
    
    # 训练模型
    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # 绘制损失曲线
    plot_losses(train_losses, test_losses)

if __name__ == '__main__':
    main()
