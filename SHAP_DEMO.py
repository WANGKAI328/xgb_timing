"""
SHAP (SHapley Additive exPlanations) TreeExplainer 演示
===============================================

本演示文件详细介绍了如何使用SHAP库中的TreeExplainer来解释基于树的模型(如XGBoost)

作者: Claude
日期: 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os

# 确保输出目录存在
os.makedirs('Output/shap_plots', exist_ok=True)

# =============================================
# 第一部分: 加载示例数据和训练简单模型
# =============================================
print("第一部分: 准备数据和模型")

# 加载数据
print("加载沪深300指数数据...")
try:
    df = pd.read_csv('hs300_index.csv')
    df['trade_day'] = pd.to_datetime(df['trade_day'], format='%Y%m%d')
    df.set_index('trade_day', inplace=True)
    
    # 简单数据处理
    df['returns'] = df['close'].pct_change()
    df['target'] = np.where(df['returns'].shift(-1) > 0, 1, -1)  # 预测下一天涨跌
    
    # 特征工程
    for window in [5, 10, 20, 30]:
        df[f'ret_ma{window}'] = df['returns'].rolling(window).mean()
        df[f'ret_std{window}'] = df['returns'].rolling(window).std()
    
    df['price_ma5'] = df['close'].rolling(5).mean()
    df['price_ma20'] = df['close'].rolling(20).mean()
    df['ma_cross'] = df['price_ma5'] - df['price_ma20']
    df['vol_change'] = df['vol'].pct_change()
    df['high_low'] = df['high'] - df['low']
    
    # 删除缺失值
    df = df.dropna()
    
    # 定义特征和目标变量
    features = ['ret_ma5', 'ret_ma10', 'ret_ma20', 'ret_ma30', 
                'ret_std5', 'ret_std10', 'ret_std20', 'ret_std30',
                'ma_cross', 'vol_change', 'high_low']
    X = df[features]
    y = df['target']
    
except FileNotFoundError:
    print("找不到hs300_index.csv文件，使用boston housing数据集作为示例...")
    # 如果找不到特定数据，使用波士顿房价数据
    from sklearn.datasets import fetch_california_housing
    
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    features = housing.feature_names

# 训练/测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
print("训练XGBoost模型...")
model_type = "分类" if np.unique(y).size <= 10 else "回归"

if model_type == "分类":
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
else:
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")

# =============================================
# 第二部分: 初始化SHAP解释器
# =============================================
print("\n第二部分: 初始化SHAP TreeExplainer")

# 创建SHAP解释器
print("创建TreeExplainer...")
print("TreeExplainer是SHAP库专门用于解释基于树的模型的工具，如XGBoost、LightGBM、CatBoost等")

explainer = shap.TreeExplainer(model)
print("解释器初始化完成!")

# 计算SHAP值
print("\n计算SHAP值...")
print("SHAP值表示每个特征对模型预测的贡献度")

# 对训练集的样本计算SHAP值
shap_values_train = explainer.shap_values(X_train)

# 对测试集的样本计算SHAP值
shap_values_test = explainer.shap_values(X_test)

# 判断是否是分类问题，检查shap_values的维度
is_multiclass = isinstance(shap_values_test, list)
print(f"是否为多分类问题: {is_multiclass}")

# =============================================
# 第三部分: 基础SHAP值分析
# =============================================
print("\n第三部分: 基础SHAP值分析")

# 对于回归或二分类，shap_values直接是每个特征的SHAP值
# 对于多分类，shap_values是一个列表，每个元素对应一个类别的SHAP值

if is_multiclass:
    # 多分类情况，取第一个类别的SHAP值进行演示
    print("多分类问题，展示第一个类别的SHAP值...")
    current_shap_values = shap_values_test[0]
    print(f"随机样本的SHAP值 (类别0):\n{current_shap_values[0]}")
    
    # SHAP基准值 (预测的起点)
    expected_value = explainer.expected_value[0]
    print(f"SHAP基准值 (类别0): {expected_value:.4f}")
else:
    # 回归或二分类
    current_shap_values = shap_values_test
    print(f"随机样本的SHAP值:\n{current_shap_values[0]}")
    
    # SHAP基准值 (预测的起点)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    print(f"SHAP基准值: {expected_value:.4f}")

# 验证SHAP值的和等于预测值与基准值之差
sample_idx = 0
if model_type == "分类":
    # 获取样本的原始预测值（概率）
    pred_prob = model.predict_proba(X_test.iloc[[sample_idx]])[0]
    if is_multiclass:
        # 多分类，计算第一个类别
        pred_value = pred_prob[0]
        sum_shap = np.sum(shap_values_test[0][sample_idx])
    else:
        # 二分类
        pred_value = pred_prob[1]  # 正类的概率
        sum_shap = np.sum(current_shap_values[sample_idx])
    
    print(f"预测概率: {pred_value:.4f}")
    print(f"SHAP值总和: {sum_shap:.4f}")
    print(f"差值: {pred_value - (expected_value + sum_shap):.4f}")
else:
    # 回归问题
    pred_value = model.predict(X_test.iloc[[sample_idx]])[0]
    sum_shap = np.sum(current_shap_values[sample_idx])
    
    print(f"预测值: {pred_value:.4f}")
    print(f"SHAP基准值 + SHAP值总和: {expected_value + sum_shap:.4f}")
    print(f"差值: {pred_value - (expected_value + sum_shap):.4f}")

# =============================================
# 第四部分: SHAP可视化 - 摘要图
# =============================================
print("\n第四部分: SHAP摘要图")
print("摘要图展示了所有特征对模型输出的影响")

# 绘制摘要图
plt.figure(figsize=(10, 8))
shap.summary_plot(current_shap_values, X_test, feature_names=features, show=False)
plt.title("SHAP 摘要图")
plt.tight_layout()
plt.savefig('Output/shap_plots/summary_plot.png')
plt.close()

print("生成了SHAP摘要图，保存到Output/shap_plots/summary_plot.png")
print("解读摘要图:")
print("- 每一行代表一个特征")
print("- 横轴是SHAP值，表示特征对预测的影响程度")
print("- 点的颜色表示特征值的大小（红色高，蓝色低）")
print("- 特征按照重要性排序")

# 条形图摘要
plt.figure(figsize=(10, 8))
shap.summary_plot(current_shap_values, X_test, plot_type="bar", feature_names=features, show=False)
plt.title("SHAP 特征重要性")
plt.tight_layout()
plt.savefig('Output/shap_plots/feature_importance.png')
plt.close()

print("生成了SHAP特征重要性图，保存到Output/shap_plots/feature_importance.png")
print("解读特征重要性图:")
print("- 条形图展示了每个特征的平均绝对SHAP值")
print("- 最重要的特征在顶部")

# =============================================
# 第五部分: SHAP可视化 - 依赖图
# =============================================
print("\n第五部分: SHAP依赖图")
print("依赖图展示了特定特征的SHAP值如何随着特征值变化")

# 选择最重要的特征（通过平均绝对SHAP值来确定）
importance = np.abs(current_shap_values).mean(0)
top_feature_idx = np.argmax(importance)
top_feature = features[top_feature_idx]

# 绘制依赖图
plt.figure(figsize=(10, 8))
shap.dependence_plot(top_feature_idx, current_shap_values, X_test, 
                     feature_names=features, show=False)
plt.title(f"{top_feature} 的SHAP依赖图")
plt.tight_layout()
plt.savefig(f'Output/shap_plots/dependence_plot_{top_feature}.png')
plt.close()

print(f"生成了{top_feature}的SHAP依赖图，保存到Output/shap_plots/dependence_plot_{top_feature}.png")
print("解读依赖图:")
print("- 横轴是特征值")
print("- 纵轴是SHAP值")
print("- 每个点代表一个样本")
print("- 点的颜色表示另一个特征的值（与主特征有交互作用的特征）")

# 绘制带交互效应的依赖图
# 寻找与顶级特征有交互的特征
if len(features) > 1:
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(top_feature_idx, current_shap_values, X_test, 
                         feature_names=features, interaction_index="auto", show=False)
    plt.title(f"{top_feature} 的SHAP依赖图（带交互）")
    plt.tight_layout()
    plt.savefig(f'Output/shap_plots/dependence_plot_{top_feature}_interaction.png')
    plt.close()
    
    print(f"生成了{top_feature}的SHAP依赖图（带交互），保存到Output/shap_plots/dependence_plot_{top_feature}_interaction.png")
    print("解读交互依赖图:")
    print("- 颜色表示另一个特征的值，该特征被自动选择为与主特征有最强交互效应的特征")
    print("- 可以观察到特征之间的交互效应")

# =============================================
# 第六部分: SHAP可视化 - 力图(Force Plot)
# =============================================
print("\n第六部分: SHAP力图")
print("力图直观地展示了每个特征对单个预测的贡献")

# 创建一个样本的力图
sample_idx = 0
plt.figure(figsize=(20, 3))
shap.force_plot(expected_value, current_shap_values[sample_idx], 
               X_test.iloc[sample_idx], feature_names=features, 
               matplotlib=True, show=False)
plt.title("单个样本的SHAP力图")
plt.tight_layout()
plt.savefig('Output/shap_plots/force_plot_single.png')
plt.close()

print("生成了单个样本的SHAP力图，保存到Output/shap_plots/force_plot_single.png")
print("解读力图:")
print("- 红色推高预测值，蓝色拉低预测值")
print("- 条形的宽度表示特征影响的大小")
print("- 基准值(E[f(x)])是模型的平均输出")

# 多样本力图
plt.figure(figsize=(20, 10))
sample_size = min(20, len(X_test))
shap.force_plot(expected_value, current_shap_values[:sample_size], 
               X_test.iloc[:sample_size], feature_names=features, 
               matplotlib=True, show=False)
plt.title("多个样本的SHAP力图")
plt.tight_layout()
plt.savefig('Output/shap_plots/force_plot_multiple.png')
plt.close()

print(f"生成了{sample_size}个样本的SHAP力图，保存到Output/shap_plots/force_plot_multiple.png")

# =============================================
# 第七部分: SHAP可视化 - 决策图(Decision Plot)
# =============================================
print("\n第七部分: SHAP决策图")
print("决策图展示了预测如何从基准值构建")

# 单样本决策图
plt.figure(figsize=(10, 8))
sample_idx = 0
shap.decision_plot(expected_value, current_shap_values[sample_idx], 
                  X_test.iloc[sample_idx], feature_names=features, show=False)
plt.title("单个样本的SHAP决策图")
plt.tight_layout()
plt.savefig('Output/shap_plots/decision_plot_single.png')
plt.close()

print("生成了单个样本的SHAP决策图，保存到Output/shap_plots/decision_plot_single.png")

# 多样本决策图
plt.figure(figsize=(10, 8))
sample_indices = range(min(10, len(X_test)))
shap.decision_plot(expected_value, current_shap_values[sample_indices], 
                  X_test.iloc[sample_indices], feature_names=features, show=False)
plt.title("多个样本的SHAP决策图")
plt.tight_layout()
plt.savefig('Output/shap_plots/decision_plot_multiple.png')
plt.close()

print("生成了多个样本的SHAP决策图，保存到Output/shap_plots/decision_plot_multiple.png")
print("解读决策图:")
print("- 每一行是一个特征")
print("- 特征按照对预测的绝对影响排序")
print("- 从底部的基准值开始，逐步加上各特征的贡献，最终达到预测值")

# =============================================
# 第八部分: SHAP可视化 - 瀑布图(Waterfall Plot)
# =============================================
print("\n第八部分: SHAP瀑布图")
print("瀑布图展示了特征如何建立单个预测")

plt.figure(figsize=(10, 8))
sample_idx = 0
shap.plots._waterfall.waterfall_legacy(expected_value, current_shap_values[sample_idx], 
                           feature_names=features, show=False)
plt.title("SHAP瀑布图")
plt.tight_layout()
plt.savefig('Output/shap_plots/waterfall_plot.png')
plt.close()

print("生成了SHAP瀑布图，保存到Output/shap_plots/waterfall_plot.png")
print("解读瀑布图:")
print("- 从基准值开始，逐步加上各特征的影响")
print("- 红色表示增加预测值，蓝色表示减少预测值")
print("- 最终累积到模型的预测值")

# =============================================
# 第九部分: 特征群组分析
# =============================================
print("\n第九部分: 特征群组分析")
print("分析特征群组的综合影响")

if len(features) >= 6:  # 至少需要几个特征才能分组
    # 创建特征群组
    feature_groups = {
        'Group1': features[:len(features)//2],
        'Group2': features[len(features)//2:]
    }
    
    # 打印群组信息
    for group_name, group_features in feature_groups.items():
        print(f"{group_name}: {', '.join(group_features)}")
    
    # 计算群组SHAP值
    group_shap_values = {}
    for group_name, group_features in feature_groups.items():
        # 获取组内特征的索引
        indices = [features.index(f) for f in group_features]
        # 计算组的总SHAP值
        if is_multiclass:
            group_shap_values[group_name] = current_shap_values[:, indices].sum(axis=1)
        else:
            group_shap_values[group_name] = current_shap_values[:, indices].sum(axis=1)
    
    # 比较群组的重要性
    group_importance = {group: np.abs(values).mean() for group, values in group_shap_values.items()}
    for group, importance in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{group} 平均绝对SHAP值: {importance:.4f}")
    
    # 绘制群组SHAP值的分布
    plt.figure(figsize=(10, 6))
    for group, values in group_shap_values.items():
        plt.hist(values, bins=30, alpha=0.5, label=group)
    plt.xlabel('SHAP值')
    plt.ylabel('样本数量')
    plt.title('特征群组SHAP值分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Output/shap_plots/group_shap_distribution.png')
    plt.close()
    
    print("生成了特征群组SHAP值分布图，保存到Output/shap_plots/group_shap_distribution.png")

# =============================================
# 第十部分: 高级功能 - 树状图(Treemap)
# =============================================
print("\n第十部分: SHAP树状图")
print("树状图用于可视化特征重要性的层次结构")

plt.figure(figsize=(12, 8))
importance = np.abs(current_shap_values).mean(0)
importance_norm = importance / importance.sum()

# 设置大小与重要性成比例，同时避免太小的区块
sizes = importance_norm * 1000 + 10

# 创建一个简单的树状图
shap_df = pd.DataFrame({
    'feature': features,
    'importance': importance,
    'size': sizes
})
shap_df = shap_df.sort_values('importance', ascending=False)

# 使用matplotlib创建简单的树状图
from matplotlib.patches import Rectangle

# 这里我们用一个简单的Grid布局代替复杂的treemap
fig, ax = plt.subplots(figsize=(12, 8))
n_features = len(features)
cols = int(np.ceil(np.sqrt(n_features)))
rows = int(np.ceil(n_features / cols))

for i, (_, row) in enumerate(shap_df.iterrows()):
    if i >= n_features:
        break
    r, c = i // cols, i % cols
    width = height = row['size'] / 50
    x = c * (1.2)
    y = r * (1.2)
    rect = Rectangle((x, y), width, height, 
                     facecolor=plt.cm.viridis(row['importance'] / importance.max()),
                     edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, row['feature'], 
            ha='center', va='center', fontsize=10, 
            color='white' if row['importance'] / importance.max() > 0.5 else 'black')

ax.set_xlim(0, cols * 1.2 + 1)
ax.set_ylim(0, rows * 1.2 + 1)
ax.set_title('SHAP特征重要性树状图')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Output/shap_plots/treemap.png')
plt.close()

print("生成了SHAP树状图，保存到Output/shap_plots/treemap.png")
print("解读树状图:")
print("- 矩形大小与特征重要性成正比")
print("- 颜色深浅也表示重要性")

# =============================================
# 第十一部分: SHAP全局解释与本地解释
# =============================================
print("\n第十一部分: 全局与本地解释")
print("SHAP值可以用于全局解释(整体模型行为)和本地解释(单个预测)")

print("全局解释:")
print("- 摘要图和特征重要性图展示了特征对所有预测的平均影响")
print("- 依赖图展示了特征如何影响模型的整体行为")

print("\n本地解释:")
print("- 力图、瀑布图和决策图展示了特征如何影响单个预测")
print("- 通过这些图可以理解模型为什么对特定样本做出特定预测")

print("\n全局解释示例: 打印所有特征的平均绝对SHAP值")
mean_abs_shap = np.abs(current_shap_values).mean(0)
for i, feature in enumerate(features):
    print(f"{feature}: {mean_abs_shap[i]:.4f}")

print("\n本地解释示例: 打印单个样本的SHAP值")
sample_idx = 0
sample_values = X_test.iloc[sample_idx].values
sample_shap = current_shap_values[sample_idx]

print(f"样本 #{sample_idx} 的特征值和SHAP值:")
for i, feature in enumerate(features):
    print(f"{feature}: 值 = {sample_values[i]:.4f}, SHAP值 = {sample_shap[i]:.4f}")

# =============================================
# 第十二部分: 总结与最佳实践
# =============================================
print("\n第十二部分: 总结与最佳实践")

print("SHAP值的优势:")
print("1. 理论基础: 基于博弈论中的Shapley值，有坚实的数学基础")
print("2. 一致性: 高SHAP值总是表示正贡献，低SHAP值总是表示负贡献")
print("3. 局部准确性: SHAP值的总和加上基准值等于模型输出")
print("4. 全局洞察: 平均SHAP值提供了全局特征重要性")

print("\n使用SHAP的最佳实践:")
print("1. 从摘要图开始，了解整体特征重要性")
print("2. 使用依赖图探索特征的非线性关系和交互")
print("3. 对特定预测使用力图或瀑布图进行深入分析")
print("4. 考虑特征群组的整体贡献，而不仅仅是单个特征")
print("5. 对比不同群组或子群组的SHAP值分布")

print("\nSHAP的局限性:")
print("1. 计算成本: 复杂模型或大型数据集上计算SHAP值可能很慢")
print("2. 特征相关性: 高度相关的特征可能会分散重要性")
print("3. 解释不等于因果关系: SHAP解释的是模型的行为，不一定反映真实的因果关系")

# =============================================
# 结束
# =============================================
print("\n演示结束!")
print("所有图表已保存到Output/shap_plots/目录")
print("探索SHAP值，深入理解您的模型!")

# 提供API参考
print("\nSHAP TreeExplainer常用API参考:")
print("- explainer = shap.TreeExplainer(model): 创建解释器")
print("- shap_values = explainer.shap_values(X): 计算SHAP值")
print("- expected_value = explainer.expected_value: 获取基准值")
print("- shap.summary_plot(): 绘制摘要图")
print("- shap.dependence_plot(): 绘制依赖图")
print("- shap.force_plot(): 绘制力图")
print("- shap.decision_plot(): 绘制决策图")

print("\n更多信息请参考SHAP官方文档: https://shap.readthedocs.io/") 