import torch
import torch.nn as nn

class CNNGRU(nn.Module):
    def __init__(self, input_channels, sequence_length, hidden_dim=128, num_layers=2, output_dim=8, prediction_length=24):
        super(CNNGRU, self).__init__()
        
        self.prediction_length = prediction_length
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # CNN layers
        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, padding=1)  # 固定输入通道为6
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False  # 改为单向GRU
        )
        
        # Output layers
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)  # 调整维度
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # 调整维度
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 确保输入只有6个特征
        x = x[:, :, :6] if x.size(2) > 6 else x
        
        # CNN forward pass
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Reshape for GRU
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        
        # GRU forward pass
        _, hidden = self.gru(x)
        
        # 使用GRU的最后一个隐藏状态生成预测序列
        outputs = []
        current_hidden = hidden[-1]  # 只使用最后一层的隐藏状态
        
        for _ in range(self.prediction_length):
            # 通过全连接层生成当前时间步的预测
            current_hidden = self.dropout(current_hidden)
            current_hidden = self.relu(self.fc_hidden(current_hidden))
            current_output = self.fc_out(current_hidden)
            outputs.append(current_output.unsqueeze(1))
        
        # 将所有预测拼接在一起
        outputs = torch.cat(outputs, dim=1)  # [batch_size, prediction_length, output_dim]
        
        return outputs