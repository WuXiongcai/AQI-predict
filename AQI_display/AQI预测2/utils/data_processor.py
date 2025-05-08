import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

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

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        # 确保数据类型为float32
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scalers = {}
        self.load_and_process_data()
    
    def load_and_process_data(self):
        # 读取JSON数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        self.df = pd.DataFrame(data)
        
        # 确保时间列存在
        if 'time' not in self.df.columns:
            self.df['time'] = self.df.index
        
        # 数值列
        self.numeric_columns = ['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        
        # 对每个数值特征进行归一化
        for column in self.numeric_columns:
            if column in self.df.columns:
                # 将字符串转换为浮点数
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype(np.float32)
                scaler = MinMaxScaler()
                self.df[column] = scaler.fit_transform(self.df[column].values.reshape(-1, 1)).astype(np.float32)
                self.scalers[column] = scaler
        
        # 将Quality转换为数值
        quality_mapping = {
            '优': 0,
            '良': 1,
            '轻度污染': 2,
            '中度污染': 3,
            '重度污染': 4,
            '严重污染': 5
        }
        if 'Quality' in self.df.columns:
            self.df['Quality'] = self.df['Quality'].map(quality_mapping).astype(np.float32)
    
    def prepare_sequences(self, sequence_length=72, prediction_length=24):
        features = []
        targets = []
        
        # 获取所有特征列
        feature_columns = self.numeric_columns + ['Quality']
        
        # 创建序列
        for i in range(len(self.df) - sequence_length - prediction_length + 1):
            feature_seq = self.df[feature_columns].iloc[i:i+sequence_length].values
            target_seq = self.df[feature_columns].iloc[i+sequence_length:i+sequence_length+prediction_length].values
            
            features.append(feature_seq)
            targets.append(target_seq)
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def prepare_data(self, sequence_length=72, prediction_length=24, train_ratio=0.8):
        X, y = self.prepare_sequences(sequence_length, prediction_length)
        
        # 分割训练集和测试集
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def get_last_sequence(self, sequence_length=72):
        """获取最新的序列数据用于预测"""
        feature_columns = self.numeric_columns + ['Quality']
        last_sequence = self.df[feature_columns].iloc[-sequence_length:].values.astype(np.float32)
        return last_sequence
    
    def get_current_features(self):
        """获取最新的特征值"""
        feature_columns = self.numeric_columns + ['Quality']
        return self.df[feature_columns].iloc[-1].to_dict()
    
    def update_data(self, new_data):
        """更新数据集，添加新的观测值"""
        # 将新数据添加到DataFrame
        new_df = pd.DataFrame([new_data])
        
        # 对新数据进行归一化
        for column in self.numeric_columns:
            if column in new_df.columns:
                new_df[column] = pd.to_numeric(new_df[column], errors='coerce').astype(np.float32)
                new_df[column] = self.scalers[column].transform(new_df[column].values.reshape(-1, 1)).astype(np.float32)
        
        # 将Quality转换为数值
        quality_mapping = {
            '优': 0,
            '良': 1,
            '轻度污染': 2,
            '中度污染': 3,
            '重度污染': 4,
            '严重污染': 5
        }
        if 'Quality' in new_df.columns:
            new_df['Quality'] = new_df['Quality'].map(quality_mapping).astype(np.float32)
        
        # 添加新数据
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def inverse_transform(self, data, feature_name):
        """反归一化数据"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return self.scalers[feature_name].inverse_transform(data.reshape(-1, 1)).astype(np.float32) 