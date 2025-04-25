import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.cnn_gru import CNNGRU
from utils.data_processor import DataProcessor
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=500, device='cuda'):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
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
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
        
        # 每50个epoch进行一次测试
        if (epoch + 1) % 50 == 0:
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    total_test_loss += loss.item()
            
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            print(f'Test Loss: {avg_test_loss:.4f}')
            
            # 保存最佳模型
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, test_losses

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(range(0, len(train_losses), 50), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据处理
    data_processor = DataProcessor('data/d_aqi_huizhou.json')
    X_train, X_test, y_train, y_test = data_processor.prepare_data()
    train_loader, test_loader = data_processor.create_dataloaders(X_train, X_test, y_train, y_test)
    
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # 绘制损失曲线
    plot_losses(train_losses, test_losses)

if __name__ == '__main__':
    main()
