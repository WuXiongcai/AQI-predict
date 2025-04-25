import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report

def load_data(file_path):
    """加载数据并进行基础处理"""
    df = pd.read_json(file_path)
    # 将字符串类型的数值转换为float类型
    numeric_cols = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2']
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    
    # 正确处理时间戳
    df['time_point'] = pd.to_datetime(df['time_point'], unit='s')
    # 按时间排序
    df = df.sort_values('time_point')
    return df

def create_time_features(df):
    """创建时间特征"""
    df['hour'] = df['time_point'].dt.hour
    df['day'] = df['time_point'].dt.day
    df['month'] = df['time_point'].dt.month
    df['weekday'] = df['time_point'].dt.weekday
    return df

def prepare_features(df, target_cols, sequence_length=24):
    """准备特征和目标变量"""
    feature_cols = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 
                   'hour', 'day', 'month', 'weekday']
    
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[feature_cols].iloc[i:i+sequence_length].values)
        y.append(df[target_cols].iloc[i+sequence_length].values)
    
    return np.array(X), np.array(y)

def evaluate_regression(y_true, y_pred, target_names):
    """评估回归模型性能"""
    results = {}
    for i, name in enumerate(target_names):
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'R2': r2_score(y_true[:, i], y_pred[:, i])
        }
    return results

def evaluate_classification(y_true, y_pred, target_names):
    """评估分类模型性能"""
    return classification_report(y_true, y_pred, target_names=target_names)

def scale_features(X_train, X_test):
    """特征标准化"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
    
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    return X_train_scaled, X_test_scaled, scaler

def encode_quality(df):
    """对空气质量等级进行编码"""
    le = LabelEncoder()
    df['Quality_encoded'] = le.fit_transform(df['Quality'])
    return df, le

def get_quality_info(quality_level):
    """
    根据空气质量等级返回对应的措施和健康影响信息
    
    Args:
        quality_level (str): 空气质量等级
        
    Returns:
        dict: 包含measure和unheathful信息的字典
    """
    quality_info = {
        '优': {
            'measure': '各类人群可正常活动',
            'unheathful': '空气质量令人满意，基本无空气污染'
        },
        '良': {
            'measure': '极少数异常敏感人群应减少户外活动',
            'unheathful': '空气质量可接受，但某些污染物可能对极少数异常敏感人群健康有较弱影响'
        },
        '轻度污染': {
            'measure': '儿童、老年人及心脏病、呼吸系统疾病患者应减少长时间、高强度的户外锻炼',
            'unheathful': '易感人群症状有轻度加剧，健康人群出现刺激症状'
        },
        '中度污染': {
            'measure': '儿童、老年人及心脏病、呼吸系统疾病患者应避免长时间、高强度的户外锻炼，一般人群适量减少户外运动',
            'unheathful': '进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响'
        },
        '重度污染': {
            'measure': '儿童、老年人及心脏病、呼吸系统疾病患者应停留在室内，停止户外运动，一般人群减少户外运动',
            'unheathful': '心脏病和肺病患者症状显著加剧，运动耐受力降低，健康人群普遍出现症状'
        },
        '严重污染': {
            'measure': '儿童、老年人和病人应停留在室内，避免体力消耗，一般人群应避免户外活动',
            'unheathful': '健康人群运动耐受力降低，有明显强烈症状，提前出现某些疾病'
        }
    }
    
    # 如果找不到对应的等级，返回默认值
    if quality_level not in quality_info:
        return {
            'measure': '暂无建议',
            'unheathful': '暂无影响信息'
        }
    
    return quality_info[quality_level]
