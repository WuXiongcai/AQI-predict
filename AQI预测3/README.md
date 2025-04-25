# 空气质量预测系统

这是一个基于深度学习和机器学习的空气质量预测系统，能够预测未来24小时的空气质量指标，包括AQI、各项污染物浓度以及空气质量等级。

## 功能特点

- 使用LSTM深度学习模型和集成学习模型进行预测
- 预测指标包括：
  - AQI（空气质量指数）
  - CO（一氧化碳）
  - NO2（二氧化氮）
  - O3（臭氧）
  - PM10（可吸入颗粒物）
  - PM2.5（细颗粒物）
  - SO2（二氧化硫）
  - 空气质量等级
- 提供空气质量等级对应的健康建议和影响说明
- 支持自定义预测起始时间
- 预测结果以CSV格式导出，便于后续分析和使用

## 项目结构

```
.
├── models/                 # 模型文件夹
│   ├── lstm_model.py      # LSTM模型定义
│   ├── ensemble_model.py  # 集成学习模型定义
│   ├── lstm_model.pth     # 训练好的LSTM模型
│   ├── ensemble_regressor.pkl    # 训练好的集成回归模型
│   ├── quality_classifier.pkl    # 训练好的空气质量分类器
│   ├── scaler.pkl         # 特征标准化器
│   └── label_encoder.pkl  # 标签编码器
├── utils.py               # 工具函数
├── train.py              # 模型训练脚本
├── test.py               # 模型预测脚本
├── d_aqi_huizhou.json    # 训练数据集
└── README.md             # 项目说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- scikit-learn 0.24+
- pandas
- numpy

## 安装依赖

```bash
pip install torch pandas numpy scikit-learn
```

## 使用方法

### 1. 训练模型

如果需要重新训练模型，运行：

```bash
python train.py
```

训练完成后，模型文件会自动保存在 `models/` 目录下。

### 2. 预测未来24小时空气质量

运行预测脚本：

```bash
python test.py
```

预测结果将保存为CSV文件，文件名格式为：`predictions_YYYYMMDD_HHMM.csv`

### 预测结果说明

预测结果CSV文件包含以下字段：
- time: 预测时间点
- AQI: 空气质量指数
- Quality: 空气质量等级（优、良、轻度污染等）
- CO: 一氧化碳浓度
- NO2: 二氧化氮浓度
- O3: 臭氧浓度
- PM10: 可吸入颗粒物浓度
- PM2.5: 细颗粒物浓度
- SO2: 二氧化硫浓度
- measure: 相应的活动建议
- unheathful: 健康影响说明

## 注意事项

1. 确保 `d_aqi_huizhou.json` 数据文件位于项目根目录
2. 首次运行时需要先训练模型
3. 预测时会自动使用最新的历史数据作为起始点
4. 如果数据时间戳异常，系统会自动使用当前时间作为预测起点

## 开发说明

- LSTM模型参数：
  - hidden_dim: 64
  - num_layers: 2
  - sequence_length: 24（使用前24小时数据预测下一个时间点）

- 预测结果会同时结合LSTM模型和集成学习模型的输出，以提高预测准确性 