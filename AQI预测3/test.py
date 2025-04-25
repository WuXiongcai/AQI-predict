import numpy as np
import torch
import joblib
import pandas as pd
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data, create_time_features, get_quality_info
from models.lstm_model import AQIPredictor

def load_models():
    """加载训练好的模型和相关组件"""
    lstm_model = AQIPredictor(input_dim=11, hidden_dim=64, num_layers=2)  # 修改为正确的参数
    lstm_model.model.load_state_dict(torch.load('models/lstm_model.pth'))
    ensemble_regressor = joblib.load('models/ensemble_regressor.pkl')
    quality_classifier = joblib.load('models/quality_classifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    return lstm_model, ensemble_regressor, quality_classifier, scaler, label_encoder

def get_latest_data(df, current_time):
    """获取当前时间之前最近的24小时数据"""
    # 确保current_time是datetime类型
    if isinstance(current_time, str):
        current_time = pd.to_datetime(current_time)
    
    # 获取最近的数据点
    df_before = df[df['time_point'] <= current_time].copy()
    if len(df_before) < 24:
        raise ValueError("没有足够的历史数据用于预测")
    
    # 获取最后24小时的数据
    latest_data = df_before.iloc[-24:].copy()
    if len(latest_data) < 24:
        raise ValueError("没有足够的历史数据用于预测")
    
    return latest_data

def predict_next_24h(df, models, current_time=None):
    """预测未来24小时的空气质量指标"""
    if current_time is None:
        current_time = datetime.now()
    elif isinstance(current_time, str):
        current_time = pd.to_datetime(current_time)

    lstm_model, ensemble_regressor, quality_classifier, scaler, label_encoder = models
    
    try:
        # 获取最近24小时数据
        recent_df = get_latest_data(df, current_time)
        
        # 准备特征数据
        feature_cols = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 
                       'hour', 'day', 'month', 'weekday']
        recent_data = recent_df[feature_cols].values
        
        # 生成预测时间点
        predictions_list = []
        current_input = recent_data.copy()
        
        for i in range(24):
            future_time = current_time + timedelta(hours=i)
            
            # 准备输入数据
            input_sequence = current_input[-24:].reshape(1, 24, -1)
            input_scaled = scaler.transform(input_sequence.reshape(-1, input_sequence.shape[-1]))
            input_scaled = input_scaled.reshape(input_sequence.shape)
            
            # LSTM预测
            X_tensor = torch.FloatTensor(input_scaled)
            test_dataset = TensorDataset(X_tensor, torch.zeros(1))
            test_loader = DataLoader(test_dataset, batch_size=1)
            lstm_pred = lstm_model.predict(test_loader)
            
            # 集成模型预测
            X_reshaped = input_scaled.reshape(1, -1)
            ensemble_pred = ensemble_regressor.predict(X_reshaped)
            quality_pred = quality_classifier.predict(X_reshaped)
            quality_label = label_encoder.inverse_transform(quality_pred.astype(int))[0]
            
            # 获取空气质量描述信息
            quality_info = get_quality_info(quality_label)
            
            # 合并预测结果
            avg_predictions = (lstm_pred[0, :7] + ensemble_pred[0]) / 2
            
            # 保存预测结果
            predictions_list.append({
                'time': future_time,
                'AQI': round(float(avg_predictions[0]), 2),
                'CO': round(float(avg_predictions[1]), 2),
                'NO2': round(float(avg_predictions[2]), 2),
                'O3': round(float(avg_predictions[3]), 2),
                'PM10': round(float(avg_predictions[4]), 2),
                'PM2_5': round(float(avg_predictions[5]), 2),
                'SO2': round(float(avg_predictions[6]), 2),
                'Quality': quality_label,
                'measure': quality_info['measure'],
                'unheathful': quality_info['unheathful']
            })
            
            # 更新输入序列
            new_row = np.zeros(len(feature_cols))
            new_row[:7] = avg_predictions
            new_row[7:] = [future_time.hour, future_time.day, 
                          future_time.month, future_time.weekday()]
            current_input = np.vstack([current_input[1:], new_row])
        
        # 创建DataFrame并格式化
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df['time'] = pd.to_datetime(predictions_df['time'])
        
        # 设置列的顺序
        columns_order = ['time', 'AQI', 'Quality', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 
                        'measure', 'unheathful']
        predictions_df = predictions_df[columns_order]
        
        return predictions_df
    
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise

def main():
    try:
        # 加载数据和模型
        print(f"\n正在加载数据和模型...")
        df = load_data('d_aqi_huizhou.json')
        df = create_time_features(df)
        
        # 获取数据集的时间范围
        start_time = df['time_point'].min()
        end_time = df['time_point'].max()
        print(f"\n数据集时间范围: {start_time} 至 {end_time}")
        
        # 设置预测起始时间（使用数据集中的最后一个有效时间点）
        current_time = df['time_point'].max()
        if current_time.year >= 2025:  # 如果时间戳异常，使用当前时间
            print("\n警告：数据集时间戳异常，使用当前时间作为预测起点")
            current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        print(f"\n使用时间点作为预测起点: {current_time}")
        
        # 加载模型
        models = load_models()
        
        # 预测未来24小时
        print(f"\n开始预测从 {current_time.strftime('%Y-%m-%d %H:%M')} 起的未来24小时数据...")
        predictions_df = predict_next_24h(df, models, current_time)
        
        # 保存预测结果
        output_file = f'predictions_{current_time.strftime("%Y%m%d_%H%M")}.csv'
        predictions_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n预测结果已保存到文件：{output_file}")
        print("\n预测结果预览：")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(predictions_df.to_string())
        
    except Exception as e:
        print(f"\n错误：{str(e)}")
        print("请检查数据集和模型文件是否正确。")

if __name__ == '__main__':
    main()
