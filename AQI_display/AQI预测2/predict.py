import torch
import numpy as np
import pandas as pd
from models.cnn_gru import CNNGRU
from utils.data_processor import DataProcessor
from datetime import datetime, timedelta
import time
import os

def predict_future(model, last_sequence, device, data_processor):
    model.eval()
    with torch.no_grad():
        # 将输入数据转换为tensor并移到正确的设备上
        input_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        # 生成预测
        predictions = model(input_sequence)
        # 将预测移回CPU并转换为numpy数组
        predictions = predictions.cpu().numpy()[0]
    
    return predictions

def calculate_measure(row):
    # 根据AQI值计算measure（活动建议）
    aqi = row['AQI']
    if aqi <= 50:
        return "各类人群可正常活动"
    elif aqi <= 100:
        return "极少数异常敏感人群应减少户外活动"
    elif aqi <= 150:
        return "儿童、老年人及心脏病、呼吸系统疾病患者应减少长时间、高强度的户外锻炼"
    elif aqi <= 200:
        return "儿童、老年人及心脏病、呼吸系统疾病患者应避免长时间、高强度的户外锻炼，一般人群适量减少户外运动"
    elif aqi <= 300:
        return "儿童、老年人及心脏病、呼吸系统疾病患者应停留在室内，停止户外运动，一般人群减少户外运动"
    else:
        return "儿童、老年人和病人应当留在室内，避免体力消耗，一般人群应避免户外活动"

def calculate_unhealthful(row):
    # 根据AQI值计算unhealthful（空气质量状况描述）
    aqi = row['AQI']
    if aqi <= 50:
        return "空气质量令人满意，基本无空气污染"
    elif aqi <= 100:
        return "空气质量可以接受，但某些污染物可能对极少数异常敏感人群健康有较弱影响"
    elif aqi <= 150:
        return "易感人群症状有轻度加剧，健康人群出现刺激症状"
    elif aqi <= 200:
        return "进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响"
    elif aqi <= 300:
        return "心脏病和肺病患者症状显著加剧，运动耐受力降低，健康人群普遍出现症状"
    else:
        return "健康人群运动耐受力降低，有明显强烈症状，提前出现某些疾病"

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'd_aqi_huizhou.json')
    model_path = os.path.join(current_dir, 'best_model.pth')
    
    # 数据处理
    data_processor = DataProcessor(data_path)
    
    # 获取最后72小时的序列数据
    last_sequence = data_processor.get_last_sequence()
    
    # 使用当前时间作为预测起点
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    print(f"\n使用当前时间 {current_time.strftime('%Y-%m-%d %H:00:00')} 作为预测起点")
    
    # 模型参数
    input_channels = last_sequence.shape[1]  # 特征数量
    sequence_length = last_sequence.shape[0]  # 序列长度
    output_dim = 9  # AQI + 6个特征 + Quality + hap
    prediction_length = 24  # 预测未来24小时
    
    print(f"\n模型参数:")
    print(f"输入特征数: {input_channels}")
    print(f"输入序列长度: {sequence_length}")
    print(f"输出维度: {output_dim}")
    print(f"预测长度: {prediction_length}")
    
    # 初始化模型
    model = CNNGRU(
        input_channels=input_channels,
        sequence_length=sequence_length,
        hidden_dim=64,  # 使用与训练时相同的隐藏层维度
        num_layers=2,
        output_dim=output_dim,
        prediction_length=prediction_length
    ).to(device)
    
    # 加载训练好的模型
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Successfully loaded model from best_model.pth")
        print(f"Model parameters loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Model path: {model_path}")
        print(f"Device: {device}")
        return
    
    # 生成预测
    predictions = predict_future(model, last_sequence, device, data_processor)
    
    # 反归一化预测结果
    feature_names = ['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'hap', 'Quality']
    predictions_dict = {}
    
    for i, feature in enumerate(feature_names[:-1]):  # 除了Quality
        if feature in data_processor.scalers:
            predictions_dict[feature] = data_processor.scalers[feature].inverse_transform(
                predictions[:, i].reshape(-1, 1)
            ).flatten()
            # 对hap进行四舍五入处理为整数
            if feature == 'hap':
                predictions_dict[feature] = np.round(predictions_dict[feature]).astype(int)
        else:
            predictions_dict[feature] = predictions[:, i]
    
    # Quality预测（取最接近的整数作为分类）
    predictions_dict['Quality'] = np.round(predictions[:, -1]).astype(int)
    
    # 创建时间索引（从当前时间开始）
    time_index = [current_time + timedelta(hours=i) for i in range(1, prediction_length + 1)]
    
    # 创建DataFrame
    df_predictions = pd.DataFrame(predictions_dict, index=time_index)
    
    # 将Quality数值转换回文本标签
    quality_labels = {
        0: '优',
        1: '良',
        2: '轻度污染',
        3: '中度污染',
        4: '重度污染',
        5: '严重污染'
    }
    df_predictions['Quality'] = df_predictions['Quality'].map(quality_labels)
    
    # 添加measure和unhealthful列
    df_predictions['measure'] = df_predictions.apply(calculate_measure, axis=1)
    df_predictions['unhealthful'] = df_predictions.apply(calculate_unhealthful, axis=1)
    
    # 保存预测结果
    output_file = os.path.join(current_dir, 'predictions_24h.csv')
    df_predictions.to_csv(output_file, encoding='utf-8-sig')
    print(f"\n预测结果已保存到 {output_file}")
    
    # 打印预测结果
    print(f"\n从当前时间开始的未来24小时预测结果:")
    print(df_predictions)

if __name__ == '__main__':
    main() 