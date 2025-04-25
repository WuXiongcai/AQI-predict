import torch
import torch.nn as nn

class CNNGRU(nn.Module):
    def __init__(self, input_channels, sequence_length, hidden_dim=64, num_layers=3, output_dim=1, prediction_length=24):
        super(CNNGRU, self).__init__()
        
        self.prediction_length = prediction_length
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after CNN layers
        self.cnn_output_size = sequence_length // 4 * 64
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layers
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN forward pass
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Reshape for GRU
        x = x.permute(0, 2, 1)  # Change shape to (batch, seq_len, features)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # 使用最后的隐藏状态生成预测序列
        outputs = []
        current_input = gru_out[:, -1:, :]  # 使用最后一个时间步的输出
        
        for _ in range(self.prediction_length):
            # 通过全连接层生成当前时间步的预测
            current_output = self.relu(self.fc(current_input))
            current_output = self.output(current_output)
            outputs.append(current_output)
            
            # 更新输入为当前预测
            current_input = self.relu(self.fc(current_input))
        
        # 将所有预测拼接在一起
        outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, prediction_length, output_dim]
        
        return outputs 