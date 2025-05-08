import torch
import numpy as np
from models.cnn_gru import CNNGRU
from utils.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions(y_true, y_pred, feature_names):
    metrics = {}
    for i, feature in enumerate(feature_names):
        # 重塑数组以匹配维度
        true_values = y_true[:, :, i].reshape(-1)
        pred_values = y_pred[:, :, i].reshape(-1)
        
        metrics[feature] = {
            'MSE': mean_squared_error(true_values, pred_values),
            'MAE': mean_absolute_error(true_values, pred_values),
            'R2': r2_score(true_values, pred_values)
        }
    return metrics

def plot_predictions(y_true, y_pred, feature_name, time_steps):
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_true, label='True Values', marker='o')
    plt.plot(time_steps, y_pred, label='Predictions', marker='x')
    plt.xlabel('Time Step')
    plt.ylabel(feature_name)
    plt.title(f'{feature_name} Predictions vs True Values')
    plt.legend()
    plt.savefig(f'{feature_name}_predictions.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据处理
    data_processor = DataProcessor('data/d_aqi_huizhou.json')
    X_train, X_test, y_train, y_test = data_processor.prepare_data()
    _, test_loader = data_processor.create_dataloaders(X_train, X_test, y_train, y_test)
    
    # 加载模型
    input_channels = X_test.shape[2]
    sequence_length = X_test.shape[1]
    output_dim = y_test.shape[2]
    prediction_length = y_test.shape[1]
    
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
    
    # 加载训练好的模型
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Successfully loaded model from best_model.pth")
    except:
        print("Error loading model. Please ensure best_model.pth exists and is valid.")
        return
        
    model.eval()
    
    # 进行预测
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.numpy())
    
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"True values shape: {y_true.shape}")
    
    # 评估预测结果
    feature_names = ['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Quality']
    metrics = evaluate_predictions(y_true, y_pred, feature_names)
    
    # 打印评估指标
    for feature, feature_metrics in metrics.items():
        print(f'\nMetrics for {feature}:')
        for metric_name, value in feature_metrics.items():
            print(f'{metric_name}: {value:.4f}')
    
    # 绘制预测结果
    for i, feature in enumerate(feature_names):
        plot_predictions(
            y_true[0, :, i],
            y_pred[0, :, i],
            feature,
            range(y_true.shape[1])
        )

if __name__ == '__main__':
    main()

