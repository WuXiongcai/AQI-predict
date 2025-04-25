import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class AQIDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        # 确保targets是浮点数类型
        if isinstance(targets, np.ndarray):
            targets = targets.astype(np.float32)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, data_path, sequence_length=6, prediction_length=24):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.load_data(data_path)
        
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        self.df = pd.DataFrame(data)
        
        # 确保时间顺序
        self.df['timestamp'] = pd.to_datetime(self.df.index)
        self.df = self.df.sort_values('timestamp')
        
        # 将Quality转换为数值
        if 'Quality' in self.df.columns:
            self.df['Quality'] = self.label_encoder.fit_transform(self.df['Quality'])
        
    def prepare_data(self):
        features = ['PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        targets = ['AQI'] + features + ['Quality']
        
        # 标准化特征
        for feature in features + ['AQI']:
            self.scalers[feature] = MinMaxScaler()
            self.df[feature] = self.scalers[feature].fit_transform(self.df[feature].values.reshape(-1, 1))
        
        # 准备序列数据
        X, y = [], []
        for i in range(len(self.df) - self.sequence_length - self.prediction_length + 1):
            X.append(self.df[features].iloc[i:i+self.sequence_length].values)
            y.append(self.df[targets].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_length].values)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        train_dataset = AQIDataset(X_train, y_train)
        test_dataset = AQIDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def inverse_transform(self, data, feature_name):
        return self.scalers[feature_name].inverse_transform(data) 