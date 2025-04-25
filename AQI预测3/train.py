import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib

from utils import (
    load_data, create_time_features, prepare_features,
    evaluate_regression, evaluate_classification, scale_features,
    encode_quality
)
from models.lstm_model import AQIPredictor
from models.ensemble_model import EnsembleRegressor, QualityClassifier

def main():
    # 加载和预处理数据
    df = load_data('d_aqi_huizhou.json')
    df = create_time_features(df)
    df, label_encoder = encode_quality(df)
    
    # 准备特征和目标变量
    target_cols = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 'Quality_encoded']
    X, y = prepare_features(df, target_cols)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 训练LSTM模型
    print("训练LSTM模型...")
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    lstm_model = AQIPredictor(input_dim=X_train.shape[-1])
    lstm_model.train(train_loader, num_epochs=500)
    
    # 训练集成回归模型
    print("\n训练集成回归模型...")
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
    
    ensemble_regressor = EnsembleRegressor(n_targets=7)  # 不包括Quality
    ensemble_regressor.train(X_train_reshaped, y_train[:, :7])
    
    # 训练空气质量分类器
    print("\n训练空气质量分类器...")
    quality_classifier = QualityClassifier()
    quality_classifier.train(X_train_reshaped, y_train[:, 7])
    
    # 评估模型性能
    print("\n评估模型性能...")
    # LSTM评估
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    test_dataset = TensorDataset(X_test_tensor, torch.zeros_like(X_test_tensor[:, 0]))
    test_loader = DataLoader(test_dataset, batch_size=32)
    lstm_predictions = lstm_model.predict(test_loader)
    
    # 集成模型评估
    ensemble_predictions = ensemble_regressor.predict(X_test_reshaped)
    quality_predictions = quality_classifier.predict(X_test_reshaped)
    
    # 输出评估结果
    target_names = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2']
    print("\nLSTM模型性能：")
    lstm_results = evaluate_regression(y_test[:, :7], lstm_predictions[:, :7], target_names)
    for target, metrics in lstm_results.items():
        print(f"\n{target}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    print("\n集成模型性能：")
    ensemble_results = evaluate_regression(y_test[:, :7], ensemble_predictions, target_names)
    for target, metrics in ensemble_results.items():
        print(f"\n{target}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    print("\n空气质量分类性能：")
    quality_labels = label_encoder.classes_
    print(evaluate_classification(y_test[:, 7], quality_predictions, quality_labels))
    
    # 保存模型
    print("\n保存模型...")
    torch.save(lstm_model.model.state_dict(), 'models/lstm_model.pth')
    joblib.dump(ensemble_regressor, 'models/ensemble_regressor.pkl')
    joblib.dump(quality_classifier, 'models/quality_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print("训练完成！")

if __name__ == '__main__':
    main()
