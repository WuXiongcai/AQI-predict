# AQI预测系统

这是一个基于深度学习的空气质量指数（AQI）预测系统，使用CNN-GRU混合模型对未来24小时的空气质量进行预测。

## 项目特点

- 使用CNN-GRU混合模型进行时序预测
- 预测包含AQI、PM2.5、PM10、SO2、NO2、CO、O3等多个空气质量指标
- 提供空气质量等级、活动建议和健康影响评估
- 支持实时预测未来24小时的空气质量数据
- 预测结果包含详细的空气质量描述和活动建议

## 环境要求

- Python 3.7+
- PyTorch
- CUDA（可选，用于GPU加速）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd AQI预测
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

运行以下命令开始训练模型：

```bash
python main.py --mode train
```

训练过程中会：
- 自动加载和处理数据
- 训练CNN-GRU模型
- 保存最佳模型到 `best_model.pth`
- 生成训练过程的损失曲线图

### 预测未来24小时数据

运行以下命令进行预测：

```bash
python predict.py
```

预测结果将：
- 生成未来24小时的空气质量预测数据
- 保存为CSV文件（predictions_24h.csv）
- 包含以下信息：
  - 时间点
  - AQI指数
  - PM2.5、PM10浓度
  - SO2、NO2、CO、O3浓度
  - 空气质量等级
  - 活动建议
  - 健康影响评估

### 测试模型

运行以下命令测试模型性能：

```bash
python main.py --mode test
```

## 项目结构

```
AQI预测/
│
├── data/                   # 数据目录
│   └── d_aqi_huizhou.json # 训练数据
│
├── models/                 # 模型目录
│   └── cnn_gru.py         # CNN-GRU模型定义
│
├── utils/                  # 工具目录
│   └── data_processor.py  # 数据处理工具
│
├── main.py                # 主程序
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明文档
```

## 输出说明

预测结果（predictions_24h.csv）包含以下字段：
- 时间：预测时间点
- AQI：空气质量指数
- PM2.5：PM2.5浓度（μg/m³）
- PM10：PM10浓度（μg/m³）
- SO2：二氧化硫浓度（μg/m³）
- NO2：二氧化氮浓度（μg/m³）
- CO：一氧化碳浓度（mg/m³）
- O3：臭氧浓度（μg/m³）
- Quality：空气质量等级（优、良、轻度污染等）
- measure：活动建议（如：各类人群可正常活动）
- unheathful：空气质量影响描述（如：空气质量令人满意，基本无空气污染）

## 注意事项

- 确保data目录下有正确的训练数据文件
- GPU训练需要CUDA支持
- 首次运行需要完整训练过程
- 预测时需要已训练好的模型文件（best_model.pth） 