# XGBoost 沪深300指数预测与时间评估工具

这个项目使用XGBoost模型对沪深300指数进行预测，并提供了模型训练时间评估、结果可视化等功能。项目包含了几个关键组件：XGBoost训练与评估工具、时间测量工具以及基于Plotly的交互式可视化组件。

## 项目结构

```
Xgboost_timing/
│
├── utils/                                # 工具函数库
│   ├── xgb_helperfunc.py                 # XGBoost辅助函数
│   └── visualization_utils.py            # 可视化工具函数
│
├── Output/                               # 输出文件夹
│
├── HS300_Xgboost.ipynb                   # 主要Jupyter笔记本
├── hs300_visualization_example.py        # 可视化示例脚本
├── hs300_timing_example.py               # 时间评估示例脚本
├── hs300_index.csv                       # 沪深300指数数据
└── README.md                             # 项目说明文档
```

## 功能介绍

### 1. XGBoost模型训练与评估

- 数据准备与划分：提供了训练集、验证集和测试集的划分功能
- 模型训练：包含XGBoost模型的训练及超参数优化
- 模型评估：提供了混淆矩阵、准确率、精确率、召回率等评估指标

### 2. 训练时间评估

- 单次训练时间测量：记录单次训练模型所需时间
- 多次训练统计：通过多次重复训练，计算平均、最小、最大训练时间
- 参数对比实验：比较不同参数配置下的训练时间差异

### 3. 交互式可视化

- 带按钮的交互式图表：可在真实值、预测值、全部显示之间切换
- 2x1子图显示：在同一页面上同时显示真实值和预测值
- 混淆矩阵热图：直观显示模型预测结果与真实值的对比

## 使用指南

### 安装依赖

```bash
pip install numpy pandas matplotlib seaborn xgboost scikit-learn plotly optuna
```

### 运行示例脚本

1. 时间评估示例：

```bash
python hs300_timing_example.py
```

2. 可视化示例：

```bash
python hs300_visualization_example.py
```

### 在Jupyter Notebook中使用

可以通过`HS300_Xgboost.ipynb`笔记本文件查看完整的分析流程。

### 示例代码

使用可视化工具：

```python
import utils.visualization_utils as vis_utils

# 评估模型性能
metrics, conf_matrix = vis_utils.evaluate_model_performance(pred_df)

# 创建交互式可视化
interactive_fig = vis_utils.create_interactive_plot(
    price_series=price_series, 
    pred_df=pred_df, 
    title='沪深300指数预测信号可视化',
    filename='interactive_plot.html'
)

# 创建子图可视化
subplot_fig = vis_utils.create_subplot_visualization(
    price_series=price_series, 
    pred_df=pred_df,
    title='沪深300指数真实值与预测值对比',
    filename='subplot_visualization.html'
)
```

测量训练时间：

```python
from utils.xgb_helperfunc import measure_training_time

# 多次重复测量训练时间
timing_results = measure_training_time(
    params,             # 模型参数
    num_rounds,         # 训练轮次 
    dtrain,             # 训练数据
    dval,               # 验证数据
    early_stopping_round=20,
    verbose_eval=False, # 关闭详细输出
    n_repeats=5         # 重复次数
)

# 打印结果
print(f"平均训练时间: {timing_results['average']:.4f} 秒")
```

## 注意事项

- 所有可视化结果将保存在`Output`文件夹中
- 默认使用时间序列数据的前80%作为训练集，后20%作为测试集
- 建议针对具体数据调整特征工程和模型参数
- 可视化结果会保存为HTML文件，可以使用浏览器打开并进行交互 