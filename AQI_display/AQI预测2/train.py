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
        # 检查输入是否包含NaN或inf
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any() or \
           torch.isinf(y_pred).any() or torch.isinf(y_true).any():
            print("Warning: NaN or Inf detected in loss computation")
            return torch.tensor(1e6, requires_grad=True, device=y_pred.device)  # 返回一个大的损失值
        
        # 计算MSE和MAE损失
        mse_loss = self.mse(y_pred, y_true)
        mae_loss = self.mae(y_pred, y_true)
        
        # 添加时间衰减权重（降低权重的影响）
        batch_size, seq_len = y_pred.shape[:2]
        time_weights = torch.exp(-torch.arange(seq_len, device=y_pred.device) * 0.01)  # 进一步降低衰减率
        time_weights = time_weights.view(1, -1, 1).expand_as(y_pred)
        
        # 组合损失
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
        weighted_loss = (combined_loss * time_weights).mean()
        
        # 检查最终损失值
        if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
            print("Warning: NaN or Inf in final loss value")
            return torch.tensor(1e6, requires_grad=True, device=y_pred.device)
            
        return weighted_loss

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=500, device='cuda'):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_epoch = 0
    patience = 20  # 早停的耐心值
    no_improvement = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # 检查输入数据
            if torch.isnan(batch_features).any() or torch.isnan(batch_targets).any():
                print(f"Warning: NaN detected in input data at epoch {epoch+1}")
                continue
                
            outputs = model(batch_features)
            
            # 检查输出是否包含NaN
            if torch.isnan(outputs).any():
                print(f"Warning: NaN detected in model outputs at epoch {epoch+1}")
                continue
                
            loss = criterion(outputs, batch_targets)
            
            if loss.item() > 1e5:  # 如果损失值过大，跳过这个batch
                print(f"Warning: Large loss value ({loss.item()}) detected at epoch {epoch+1}")
                continue
                
            loss.backward()
            
            # 梯度裁剪（降低阈值）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # 检查梯度是否为NaN
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        param.grad.data.zero_()
            
            if not has_nan_grad:
                optimizer.step()
                total_train_loss += loss.item()
                batch_count += 1
        
        if batch_count == 0:
            print(f"Warning: All batches were skipped in epoch {epoch+1}")
            continue
            
        avg_train_loss = total_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        total_test_loss = 0
        test_batch_count = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                if loss.item() < 1e5:  # 只统计正常的损失值
                    total_test_loss += loss.item()
                    test_batch_count += 1
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_targets.cpu().numpy())
        
        if test_batch_count == 0:
            print(f"Warning: All test batches were skipped in epoch {epoch+1}")
            continue
            
        avg_test_loss = total_test_loss / test_batch_count
        test_losses.append(avg_test_loss)
        
        # 更新学习率
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Test Loss: {avg_test_loss:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement = 0
            
            # 保存预测结果的可视化
            if (epoch + 1) % 50 == 0 and len(predictions) > 0:
                plot_predictions(predictions, actuals, epoch)
        else:
            no_improvement += 1
        
        # 早停检查
        if no_improvement >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    print(f'\nTraining completed! Best model was at epoch {best_epoch+1} with test loss: {best_test_loss:.4f}')
    return train_losses, test_losses

def plot_predictions(predictions, actuals, epoch):
    # 选择第一个样本的AQI和hap预测进行可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制AQI预测
    pred_aqi = predictions[0][:, 0]  # AQI是第一个特征
    actual_aqi = actuals[0][:, 0]
    ax1.plot(actual_aqi, label='Actual', marker='o')
    ax1.plot(pred_aqi, label='Predicted', marker='x')
    ax1.set_title(f'AQI Prediction vs Actual (Epoch {epoch+1})')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('AQI Value')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制hap预测
    pred_hap = predictions[0][:, 7]  # hap是第8个特征
    actual_hap = actuals[0][:, 7]
    ax2.plot(actual_hap, label='Actual', marker='o')
    ax2.plot(pred_hap, label='Predicted', marker='x')
    ax2.set_title(f'HAP (Atmospheric Pressure) Prediction vs Actual (Epoch {epoch+1})')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('HAP Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
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
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
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
        prediction_length=prediction_length,
        hidden_dim=64,  # 减小隐藏层维度
        num_layers=2
    ).to(device)
    
    # 损失函数和优化器
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # 降低学习率
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6
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
