import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from typing import Optional, List, Union

def create_interactive_plot(price_series, pred_df, title='沪深300指数预测信号可视化', save = False, save_path = None, auto_open=False):
    """
    创建带有按钮的交互式图表，可以切换显示真实值和预测值的散点
    
    参数:
    price_series: 价格序列数据 (pandas Series，索引为日期)
    pred_df: 预测结果DataFrame，包含'test_sign'真实值和'pred_sign'预测值
    title: 图表标题
    filename: 保存的HTML文件名
    auto_open: 是否自动打开浏览器
    
    返回:
    fig: plotly图表对象
    """
    # 准备数据
    down_trend_signals_real = pred_df[pred_df['test_sign'] == -1]
    up_trend_signals_real = pred_df[pred_df['test_sign'] == 1]
    
    down_trend_signals_pred = pred_df[pred_df['pred_sign'] == -1]
    up_trend_signals_pred = pred_df[pred_df['pred_sign'] == 1]
    
    # 创建基础图形
    fig = go.Figure()
    
    # 添加价格曲线
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name='沪深300',
        line=dict(color='black', width=2)
    ))
    
    # 添加真实下跌信号
    fig.add_trace(go.Scatter(
        x=down_trend_signals_real.index,
        y=price_series.loc[down_trend_signals_real.index].values,
        mode='markers',
        name='真实下跌',
        marker=dict(color='blue', size=10, symbol='circle'),
        visible=False
    ))
    
    # 添加真实上涨信号
    fig.add_trace(go.Scatter(
        x=up_trend_signals_real.index,
        y=price_series.loc[up_trend_signals_real.index].values,
        mode='markers',
        name='真实上涨',
        marker=dict(color='red', size=10, symbol='circle'),
        visible=False
    ))
    
    # 添加预测下跌信号
    fig.add_trace(go.Scatter(
        x=down_trend_signals_pred.index,
        y=price_series.loc[down_trend_signals_pred.index].values,
        mode='markers',
        name='预测下跌',
        marker=dict(color='green', size=10, symbol='triangle-down'),
        visible=True
    ))
    
    # 添加预测上涨信号
    fig.add_trace(go.Scatter(
        x=up_trend_signals_pred.index,
        y=price_series.loc[up_trend_signals_pred.index].values,
        mode='markers',
        name='预测上涨',
        marker=dict(color='red', size=10, symbol='triangle-up'),
        visible=True
    ))
    
    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='归一化价格',
        legend_title='信号类型',
        template='plotly_white',
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.1,
                y=1.15,
                buttons=list([
                    dict(
                        label="预测值",
                        method="update",
                        args=[{"visible": [True, False, False, True, True]},
                              {"title": "沪深300指数预测信号可视化"}]
                    ),
                    dict(
                        label="真实值",
                        method="update",
                        args=[{"visible": [True, True, True, False, False]},
                              {"title": "沪深300指数真实信号可视化"}]
                    ),
                    dict(
                        label="全部显示",
                        method="update",
                        args=[{"visible": [True, True, True, True, True]},
                              {"title": "沪深300指数真实与预测信号对比"}]
                    )
                ]),
            )
        ]
    )
    
    # 保存为HTML文件
    if save:
        if save_path:
            plot(fig, filename=save_path, auto_open=auto_open)
        else:
            raise ValueError('save_path is not provided')
    return fig

def create_subplot_visualization(price_series, pred_df, title='沪深300指数真实值与预测值对比', save = False, save_path = None, auto_open=False):
    """
    创建2x1子图，分别显示真实值和预测值
    
    参数:
    price_series: 价格序列数据 (pandas Series，索引为日期)
    pred_df: 预测结果DataFrame，包含'test_sign'真实值和'pred_sign'预测值
    title: 图表标题
    filename: 保存的HTML文件名
    auto_open: 是否自动打开浏览器
    
    返回:
    fig: plotly图表对象
    """
    # 准备数据
    down_trend_signals_real = pred_df[pred_df['test_sign'] == -1]
    up_trend_signals_real = pred_df[pred_df['test_sign'] == 1]
    
    down_trend_signals_pred = pred_df[pred_df['pred_sign'] == -1]
    up_trend_signals_pred = pred_df[pred_df['pred_sign'] == 1]
    
    # 创建子图
    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('真实涨跌信号', '预测涨跌信号')
    )
    
    # 第一个子图：真实值
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name='沪深300',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=down_trend_signals_real.index,
            y=price_series.loc[down_trend_signals_real.index].values,
            mode='markers',
            name='真实下跌',
            marker=dict(color='blue', size=10, symbol='circle')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=up_trend_signals_real.index,
            y=price_series.loc[up_trend_signals_real.index].values,
            mode='markers',
            name='真实上涨',
            marker=dict(color='red', size=10, symbol='circle')
        ),
        row=1, col=1
    )
    
    # 第二个子图：预测值
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name='沪深300',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=down_trend_signals_pred.index,
            y=price_series.loc[down_trend_signals_pred.index].values,
            mode='markers',
            name='预测下跌',
            marker=dict(color='green', size=10, symbol='triangle-down')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=up_trend_signals_pred.index,
            y=price_series.loc[up_trend_signals_pred.index].values,
            mode='markers',
            name='预测上涨',
            marker=dict(color='red', size=10, symbol='triangle-up')
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text=title,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='日期', row=2, col=1)
    fig.update_yaxes(title_text='归一化价格', row=1, col=1)
    fig.update_yaxes(title_text='归一化价格', row=2, col=1)
    
    # 保存为HTML文件
    if save:
        if save_path:
            plot(fig, filename=save_path, auto_open=auto_open)
        else:
            raise ValueError('save_path is not provided')
    
    return fig

def evaluate_model_performance(y_true,y_pred, show_heatmap = True, heatmap_cmap = 'viridis',heatmap_title = 'Confusion Matrix',figsize = (5,3)):
    """
    评估模型性能，计算并显示各项指标
    
    参数:
    pred_df: 预测结果DataFrame，包含'test'真实值和'pred'预测值
    display: 是否显示评估结果
    
    返回:
    metrics_dict: 包含各项评估指标的字典
    """
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    # 计算各项指标
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 计算回归指标
    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    
    # 汇总所有指标
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (TPR)': recall,
        'Specificity (TNR)': specificity,
        'F1 Score': f1,
        'RMSE': rmse,
        'R2 Score': r2,
        'Confusion Matrix': conf_matrix
    }

    if show_heatmap:
        plt.figure(figsize = figsize)
        sns.heatmap(conf_matrix, annot = True, cmap = heatmap_cmap, fmt = 'd', cbar = False)
        plt.xlabel('Predicted',fontsize = 12)
        plt.ylabel('Real',fontsize = 12)
        plt.title(heatmap_title,fontsize = 14)
        plt.show()

    return metrics_dict

def plot_custom_shap_force(
    base_value: float,
    shap_values: np.ndarray,
    features: pd.Series,
    feature_names: List[str],
    true_value: Optional[float] = None,
    pred_value: Optional[float] = None,
    sample_date: Optional[str] = None,
    top_features_num: int = 10,
    shap_decimals: int = 2,
    pred_decimals: int = 4,
    figsize: tuple = (12, 8),
    x_label: str = 'SHAP值 (对模型输出的影响)',
    y_label: str = '特征 = 特征值',
    title: Optional[str] = None,
    pos_color: str = 'red',
    neg_color: str = 'blue',
    grid_alpha: float = 0.6,
    show_values: bool = True,
    value_fontweight: str = 'bold',
    title_fontsize: int = 14,
    include_base_value_in_title: bool = True,
    ascending: bool = False,  # 控制特征排序方向，True表示最重要的特征在底部，False表示在顶部
    debug: bool = False       # 开启调试模式，显示数据与标签的映射关系
) -> plt.Figure:
    """
    绘制自定义SHAP力图，允许精确控制小数点位数和显示方式
    
    参数:
        base_value (float): 模型的基准预测值
        shap_values (np.ndarray): 特征的SHAP值数组
        features (pd.Series): 特征值Series
        feature_names (List[str]): 特征名称列表
        true_value (float, optional): 样本的真实值
        pred_value (float, optional): 样本的预测值
        sample_date (str, optional): 样本日期，将显示在标题中
        top_features_num (int): 要显示的最重要特征数量，默认10
        shap_decimals (int): SHAP值显示的小数位数，默认2
        pred_decimals (int): 预测值、真实值和基准值显示的小数位数，默认4
        figsize (tuple): 图像大小，默认(12, 8)
        x_label (str): x轴标签
        y_label (str): y轴标签
        title (str, optional): 自定义标题，设置后将覆盖默认标题
        pos_color (str): 正面贡献的颜色，默认红色
        neg_color (str): 负面贡献的颜色，默认蓝色
        grid_alpha (float): 网格线的透明度，默认0.6
        show_values (bool): 是否在条形末端显示数值，默认True
        value_fontweight (str): 数值文本的字体粗细，默认'bold'
        title_fontsize (int): 标题字体大小，默认14
        include_base_value_in_title (bool): 是否在标题中包含基准值，默认True
        ascending (bool): 控制特征排序方向，True表示最重要的特征在底部，False表示在顶部
        debug (bool): 开启调试模式，显示数据与标签的映射关系
        
    返回:
        plt.Figure: matplotlib图像对象
    """
    # 确保特征名称和SHAP值一致
    if isinstance(features, pd.Series):
        features_series = features
    else:
        features_series = pd.Series(features, index=feature_names)
    
    # 打印调试信息
    if debug:
        print("原始SHAP值:")
        for name, value in zip(feature_names, shap_values):
            print(f"{name}: {value:.{shap_decimals}f}")
        print("\n")
    
    # 创建特征重要度Series
    feature_importance = pd.Series(shap_values, index=feature_names).abs().sort_values(ascending=False)
    
    # 选择最重要的特征
    top_features = feature_importance.head(top_features_num).index
    top_shap_values = pd.Series(shap_values, index=feature_names).loc[top_features]
    
    # 打印调试信息
    if debug:
        print(f"选择的前{top_features_num}个重要特征:")
        for name, value in top_shap_values.items():
            print(f"{name}: {value:.{shap_decimals}f} (特征值: {features_series[name]})")
        print("\n")
    
    # 按照绝对值大小排序特征
    sorted_indices = top_shap_values.abs().sort_values(ascending=ascending).index
    
    # 打印调试信息
    if debug:
        print(f"排序后的特征 (ascending={ascending}):")
        for i, name in enumerate(sorted_indices):
            print(f"位置{i}: {name}, SHAP值: {top_shap_values[name]:.{shap_decimals}f}")
        print("\n")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置颜色
    colors = [pos_color if x > 0 else neg_color for x in top_shap_values[sorted_indices]]
    
    # 绘制水平条形图
    bars = ax.barh(range(len(sorted_indices)), top_shap_values[sorted_indices], color=colors)
    
    # 打印调试信息
    if debug:
        print("绘制的条形图数据:")
        for i, (name, value) in enumerate(zip(sorted_indices, top_shap_values[sorted_indices])):
            print(f"Y位置{i}: {name}, SHAP值: {value:.{shap_decimals}f}")
        print("\n")
    
    # 在条形末端添加精确的数值
    if show_values:
        for i, (feat, v) in enumerate(zip(sorted_indices, top_shap_values[sorted_indices])):
            ax.text(v + (0.01 if v >= 0 else -0.01), 
                    i, 
                    f"{v:.{shap_decimals}f}", 
                    ha='left' if v >= 0 else 'right', 
                    va='center',
                    fontweight=value_fontweight)
    
    # 添加特征值标注
    feature_labels = []
    for feat in sorted_indices:
        value = features_series[feat]
        
        # 根据值的类型格式化显示
        if isinstance(value, (int, float)):
            # 数值型数据保留小数
            feature_display = f"{feat} = {value:.{shap_decimals}f}"
        elif isinstance(value, str) and len(value) > 15:
            # 长字符串截断
            feature_display = f"{feat} = {value[:15]}..."
        else:
            # 其他类型直接转字符串
            feature_display = f"{feat} = {value}"
            
        feature_labels.append(feature_display)
    
    # 打印调试信息
    if debug:
        print("Y轴标签:")
        for i, label in enumerate(feature_labels):
            print(f"位置{i}: {label}")
        print("\n")
    
    # 设置y轴刻度和标签
    ax.set_yticks(range(len(sorted_indices)))
    ax.set_yticklabels(feature_labels)
    
    # 为了更清晰地显示对应关系，在调试模式下添加网格线
    if debug:
        ax.grid(True, axis='y', linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
    
    # 构建标题
    if title is None:
        title_parts = []
        if sample_date:
            title_parts.append(f"样本日期: {sample_date}")
        
        subtitle_parts = []
        if true_value is not None:
            subtitle_parts.append(f"真实值: {true_value:.{pred_decimals}f}")
        if pred_value is not None:
            subtitle_parts.append(f"预测值: {pred_value:.{pred_decimals}f}")
        if include_base_value_in_title:
            subtitle_parts.append(f"基准值: {base_value:.{pred_decimals}f}")
        
        if title_parts and subtitle_parts:
            full_title = f"{' | '.join(title_parts)}\n{', '.join(subtitle_parts)}"
        elif subtitle_parts:
            full_title = f"{', '.join(subtitle_parts)}"
        elif title_parts:
            full_title = f"{' | '.join(title_parts)}"
        else:
            full_title = "SHAP力图"
    else:
        full_title = title
    
    # 设置标题和标签
    plt.title(full_title, fontsize=title_fontsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # 添加网格和零线
    plt.grid(axis='x', linestyle='--', alpha=grid_alpha)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    return fig

def plot_shap_summary(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    max_display: int = 20,
    plot_type: str = "bar",
    figsize: tuple = (10, 12),
    title: str = "SHAP特征重要性",
    title_fontsize: int = 14
) -> plt.Figure:
    """
    绘制SHAP特征重要性总结图
    
    参数:
        shap_values (np.ndarray): 所有样本的SHAP值数组
        features (pd.DataFrame): 特征数据框
        max_display (int): 要显示的最重要特征数量，默认20
        plot_type (str): 图表类型，'bar'或'dot'
        figsize (tuple): 图像大小，默认(10, 12)
        title (str): 图表标题
        title_fontsize (int): 标题字体大小，默认14
        
    返回:
        plt.Figure: matplotlib图像对象
    """
    # 计算每个特征的平均绝对SHAP值
    feature_names = features.columns
    feature_importance = pd.DataFrame(
        {'feature': feature_names,
         'importance': np.abs(shap_values).mean(axis=0)
        }
    ).sort_values('importance', ascending=False)
    
    # 选择前N个特征
    top_features = feature_importance.head(max_display)['feature'].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "bar":
        # 绘制条形图
        sns.barplot(x='importance', y='feature', data=feature_importance.head(max_display), ax=ax)
        plt.xlabel('平均|SHAP值|')
        plt.ylabel('特征')
    else:  # 点图
        # 收集所有要绘制的特征的SHAP值
        plot_data = []
        for i, feature in enumerate(top_features):
            feature_idx = np.where(feature_names == feature)[0][0]
            for j in range(len(shap_values)):
                plot_data.append({
                    'feature': feature,
                    'shap_value': shap_values[j, feature_idx],
                    'feature_value': features.iloc[j, feature_idx]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # 对于每个特征，计算其特征值的分位数
        for feature in top_features:
            feature_values = features[feature].values
            feature_df = plot_df[plot_df['feature'] == feature]
            
            # 对每个样本的特征值计算分位数
            quantiles = []
            for idx in feature_df.index:
                value = plot_df.loc[idx, 'feature_value']
                q = np.searchsorted(np.sort(feature_values), value) / len(feature_values)
                quantiles.append(q)
            
            plot_df.loc[feature_df.index, 'quantile'] = quantiles
        
        # 绘制散点图
        plt.figure(figsize=figsize)
        for i, feature in enumerate(top_features):
            feature_df = plot_df[plot_df['feature'] == feature].sort_values('shap_value')
            plt.scatter(
                feature_df['shap_value'], 
                np.ones(len(feature_df)) * i,
                c=feature_df['quantile'], 
                cmap='coolwarm',
                s=30
            )
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('SHAP值')
        plt.ylabel('特征')
        plt.colorbar(label='特征值分位数')
    
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    
    return fig

def plot_interactive_shap_force(
    base_value: float,
    shap_values: np.ndarray,
    features: pd.Series,
    feature_names: List[str],
    true_value: Optional[float] = None,
    pred_value: Optional[float] = None,
    sample_date: Optional[str] = None,
    top_features_num: int = 10,
    shap_decimals: int = 2,
    pred_decimals: int = 4,
    plot_width: int = 1200,
    plot_height: int = 1000,
    pos_color: str = 'rgba(255, 0, 0, 0.8)',
    neg_color: str = 'rgba(0, 0, 255, 0.8)',
    save: bool = False,
    save_path: Optional[str] = None,
    auto_open: bool = False
) -> go.Figure:
    """
    创建交互式SHAP力图，使用Plotly，可显示预测值和真实值，允许用户通过按钮控制显示
    
    参数:
        base_value (float): 模型的基准预测值
        shap_values (np.ndarray): 特征的SHAP值数组
        features (pd.Series): 特征值Series
        feature_names (List[str]): 特征名称列表
        true_value (float, optional): 样本的真实值
        pred_value (float, optional): 样本的预测值
        sample_date (Optional[str]): 样本日期
        top_features_num (int): 要显示的最重要特征数量，默认10
        shap_decimals (int): SHAP值显示的小数位数，默认2
        pred_decimals (int): 预测值、真实值和基准值显示的小数位数，默认4
        plot_width (int): 图表宽度，默认1000
        plot_height (int): 图表高度，默认600
        pos_color (str): 正面贡献的颜色，默认红色
        neg_color (str): 负面贡献的颜色，默认蓝色
        save (bool): 是否保存图表为HTML文件
        save_path (str, optional): 保存路径
        auto_open (bool): 是否自动打开保存的HTML文件

    返回:
        go.Figure: Plotly图表对象
    """
    # 确保特征名称和SHAP值一致
    if isinstance(features, pd.Series):
        features_series = features
    else:
        features_series = pd.Series(features, index=feature_names)
    
    # 创建特征重要度Series
    feature_importance = pd.Series(shap_values, index=feature_names).abs().sort_values(ascending=False)
    
    # 选择最重要的特征
    top_features = feature_importance.head(top_features_num).index
    top_shap_values = pd.Series(shap_values, index=feature_names).loc[top_features]
    top_features_values = features_series.loc[top_features]
    
    # 按重要性降序排序
    sorted_indices = top_shap_values.abs().sort_values(ascending=False).index
    sorted_shap_values = top_shap_values[sorted_indices]
    sorted_feature_values = top_features_values[sorted_indices]
    
    # 创建特征标签
    feature_labels = []
    for feat, value in sorted_feature_values.items():
        if isinstance(value, (int, float)):
            feature_label = f"{feat} = {value:.{shap_decimals}f}"
        elif isinstance(value, str) and len(value) > 15:
            feature_label = f"{feat} = {value[:15]}..."
        else:
            feature_label = f"{feat} = {value}"
        feature_labels.append(feature_label)
    
    # 准备基础图的数据
    base_to_pred_fig = go.Figure()
    
    # 添加累积贡献瀑布图
    cumulative_values = [base_value]
    cumulative_labels = ['基准值']
    
    # 计算每个特征的累积值
    running_total = base_value
    for i, (feat, value) in enumerate(sorted_shap_values.items()):
        running_total += value
        cumulative_values.append(running_total)
        cumulative_labels.append(f"+ {feat}")
    
    # 瀑布图点
    base_to_pred_fig.add_trace(go.Scatter(
        x=cumulative_values,
        y=cumulative_labels,
        mode='markers',
        marker=dict(size=12, color='black'),
        name='累积预测值',
        hoverinfo='text',
        hovertext=[f"{label}: {val:.{pred_decimals}f}" for label, val in zip(cumulative_labels, cumulative_values)]
    ))
    
    # 瀑布图线
    base_to_pred_fig.add_trace(go.Scatter(
        x=cumulative_values,
        y=cumulative_labels,
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 添加真实值点（初始隐藏）
    if true_value is not None:
        base_to_pred_fig.add_trace(go.Scatter(
            x=[true_value],
            y=['真实值'],
            mode='markers',
            marker=dict(size=12, color='green', symbol='star'),
            name='真实值',
            visible=False,
            hoverinfo='text',
            hovertext=[f"真实值: {true_value:.{pred_decimals}f}"]
        ))
    
    # 创建特征贡献条形图
    feature_fig = go.Figure()
    
    # 添加特征贡献条形图
    for i, (feat, value) in enumerate(sorted_shap_values.items()):
        color = pos_color if value > 0 else neg_color
        feature_fig.add_trace(go.Bar(
            x=[value],
            y=[feature_labels[i]],
            orientation='h',
            marker=dict(color=color),
            name=feat,
            hoverinfo='text',
            hovertext=[f"{feat} = {sorted_feature_values[feat]}<br>SHAP值: {value:.{shap_decimals}f}"]
        ))
    
    # 创建组合子图
    fig = sp.make_subplots(
        rows=2, cols=1,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.1,
        subplot_titles=('累积SHAP值瀑布图', '特征重要性贡献')
    )
    
    # 将瀑布图和真实值点添加到子图
    for trace in base_to_pred_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # 将特征贡献条形图添加到子图
    for trace in feature_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # 为基准值添加垂直线
    fig.add_shape(
        type="line",
        x0=base_value, y0=0, x1=base_value, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash"),
        row=1, col=1
    )
    
    # 为预测值添加垂直线
    if pred_value is not None:
        fig.add_shape(
            type="line",
            x0=pred_value, y0=0, x1=pred_value, y1=1,
            yref="paper",
            line=dict(color="red", width=2),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=pred_value,
            y=1,
            yref="paper",
            text=f"预测值: {pred_value:.{pred_decimals}f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            row=1, col=1
        )
    
    # 添加基准值注释
    fig.add_annotation(
        x=base_value,
        y=1,
        yref="paper",
        text=f"基准值: {base_value:.{pred_decimals}f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-60,
        row=1, col=1
    )
    
    # 添加SHAP值为零的垂直线
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=0, y1=1,
        yref="paper",
        line=dict(color="black", width=1),
        row=2, col=1
    )
    
    # 构建标题
    title_parts = []
    if sample_date:
        title_parts.append(f"样本日期: {sample_date}")
    
    if true_value is not None:
        title_parts.append(f"真实值: {true_value:.{pred_decimals}f}")
    
    if pred_value is not None:
        title_parts.append(f"预测值: {pred_value:.{pred_decimals}f}")
    
    title = " | ".join(title_parts) if title_parts else "SHAP力图"
    
    # 更新布局
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(t=150, b=50, l=100, r=50),
        template="plotly_white",
        width=plot_width,
        height=plot_height,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.5,
                y=1.15,
                xanchor="center",
                buttons=[
                    dict(
                        label="隐藏真实值",
                        method="update",
                        args=[{"visible": [True, True, False]}, {}]
                    ),
                    dict(
                        label="显示真实值",
                        method="update",
                        args=[{"visible": [True, True, True]}, {}]
                    )
                ],
            ),
        ] if true_value is not None else []
    )
    
    # 更新X和Y轴
    fig.update_xaxes(title_text="预测值", row=1, col=1)
    fig.update_yaxes(title_text="累积特征", row=1, col=1)
    
    fig.update_xaxes(title_text="SHAP值", row=2, col=1)
    fig.update_yaxes(title_text="特征 = 特征值", row=2, col=1)
    
    # 保存为HTML文件
    if save:
        if save_path:
            plot(fig, filename=save_path, auto_open=auto_open)
        else:
            raise ValueError('save_path is not provided')
    else:
        plot(fig, auto_open=auto_open)
    
    return fig

def plot_interactive_shap_summary(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_names: List[str] = None,
    max_display: int = 20,
    plot_width: int = 1000,
    plot_height: int = 700,
    title: str = "SHAP特征重要性",
    save: bool = False,
    save_path: Optional[str] = None,
    auto_open: bool = False
) -> go.Figure:
    """
    创建交互式SHAP特征重要性总结图，使用Plotly
    
    参数:
        shap_values (np.ndarray): 所有样本的SHAP值数组
        features (pd.DataFrame): 特征数据框
        feature_names (List[str]): 特征名称列表，如果为None则使用features.columns
        max_display (int): 要显示的最重要特征数量，默认20
        plot_width (int): 图表宽度，默认1000
        plot_height (int): 图表高度，默认700
        title (str): 图表标题
        save (bool): 是否保存图表为HTML文件
        save_path (str, optional): 保存路径
        auto_open (bool): 是否自动打开保存的HTML文件
        
    返回:
        go.Figure: Plotly图表对象
    """
    # 使用DataFrame的列名作为特征名称（如果未提供）
    if feature_names is None and isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
    
    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    # 选择前N个特征
    top_features = feature_importance.head(max_display)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加条形图
    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale='RdBu_r',
            colorbar=dict(title="平均|SHAP值|")
        ),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    
    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title="平均|SHAP值|",
        yaxis_title="特征",
        template="plotly_white",
        width=plot_width,
        height=plot_height,
        margin=dict(t=100, b=50, l=100, r=50)
    )
    
    # 保存为HTML文件
    if save:
        if save_path:
            plot(fig, filename=save_path, auto_open=auto_open)
        else:
            raise ValueError('save_path is not provided')
    else:
        plot(fig, auto_open=auto_open)
    
    return fig

def plot_period_shap_force(
    base_value: float,
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_names: List[str] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    sample_start_index: int = 0,
    sample_end_index: Optional[int] = None,
    dates: Optional[List[str]] = None,
    top_features_num: int = 10,
    shap_decimals: int = 2,
    pred_decimals: int = 4,
    feature_decimals: int = 4,
    plot_width: int = 1200,
    plot_height: int = 800,
    pos_color: str = 'rgba(255, 0, 0, 0.7)',
    neg_color: str = 'rgba(0, 0, 255, 0.7)',
    save: bool = False,
    save_path: Optional[str] = None,
    auto_open: bool = False
) -> go.Figure:
    """
    创建特定时间段的SHAP力图，展示多个样本的SHAP值分布和贡献情况
    
    参数:
        base_value (float): 模型的基准预测值
        shap_values (np.ndarray): 所有样本的SHAP值数组，形状为(n_samples, n_features)
        features (pd.DataFrame): 特征数据框，形状为(n_samples, n_features)
        feature_names (List[str]): 特征名称列表，如果为None则使用features.columns
        y_true (np.ndarray, optional): 真实值数组
        y_pred (np.ndarray, optional): 预测值数组
        sample_start_index (int): 样本起始索引，默认为0
        sample_end_index (int, optional): 样本结束索引，如果为None则使用所有样本
        dates (List[str], optional): 日期列表，用于标注样本
        top_features_num (int): 要显示的最重要特征数量，默认10
        shap_decimals (int): SHAP值显示的小数位数，默认2
        pred_decimals (int): 预测值、真实值和基准值显示的小数位数，默认4
        feature_decimals (int): 特征值显示的小数位数，默认4
        plot_width (int): 图表宽度，默认1200
        plot_height (int): 图表高度，默认800
        pos_color (str): 正面贡献的颜色，默认红色
        neg_color (str): 负面贡献的颜色，默认蓝色
        save (bool): 是否保存图表为HTML文件
        save_path (str, optional): 保存路径
        auto_open (bool): 是否自动打开保存的HTML文件
        
    返回:
        go.Figure: Plotly图表对象
    """
    # 使用DataFrame的列名作为特征名称（如果未提供）
    if feature_names is None and isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
    
    # 计算每个特征的平均绝对SHAP值（全局重要性）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    # 选择前N个最重要的特征
    top_features_idx = [feature_names.index(feat) for feat in feature_importance.head(top_features_num)['feature']]
    top_feature_names = feature_importance.head(top_features_num)['feature'].tolist()
    
    # 确定要展示的样本范围
    if sample_end_index is None:
        sample_end_index = shap_values.shape[0]
    
    # 确保索引范围有效
    sample_start_index = max(0, min(sample_start_index, shap_values.shape[0]-1))
    sample_end_index = max(sample_start_index+1, min(sample_end_index, shap_values.shape[0]))
    
    # 生成样本索引列表
    sample_indices = list(range(sample_start_index, sample_end_index))
    
    # 创建子图布局
    fig = sp.make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.1,
        subplot_titles=('预测值与真实值分布', 'SHAP值分布 (按特征)')
    )
    
    # 第一个子图：预测值与真实值的散点图（如果提供）
    if y_pred is not None:
        sample_texts = []
        for i in sample_indices:
            if dates is not None and i < len(dates):
                sample_texts.append(f"样本 {i}<br>日期: {dates[i]}<br>预测值: {y_pred[i]:.{pred_decimals}f}")
            else:
                sample_texts.append(f"样本 {i}<br>预测值: {y_pred[i]:.{pred_decimals}f}")
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sample_indices))),
                y=[y_pred[i] for i in sample_indices],
                mode='markers',
                name='预测值',
                marker=dict(
                    size=10,
                    color='blue',
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=sample_texts
            ),
            row=1, col=1
        )
        
        # 添加基准值线
        fig.add_shape(
            type="line",
            x0=0, y0=base_value, x1=len(sample_indices)-1, y1=base_value,
            line=dict(color="gray", width=2, dash="dash"),
            row=1, col=1
        )
        
        # 添加基准值标注
        fig.add_annotation(
            x=len(sample_indices)-1,
            y=base_value,
            text=f"基准值: {base_value:.{pred_decimals}f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=0,
            row=1, col=1
        )
        
        if y_true is not None:
            # 为每个真实值点创建hover文本
            true_hover_texts = []
            for i in sample_indices:
                if dates is not None and i < len(dates):
                    true_hover_texts.append(f"样本 {i}<br>日期: {dates[i]}<br>真实值: {y_true[i]:.{pred_decimals}f}")
                else:
                    true_hover_texts.append(f"样本 {i}<br>真实值: {y_true[i]:.{pred_decimals}f}")
                
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sample_indices))),
                    y=[y_true[i] for i in sample_indices],
                    mode='markers',
                    name='真实值',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='cross'
                    ),
                    hoverinfo='text',
                    hovertext=true_hover_texts
                ),
                row=1, col=1
            )
    
    # 第二个子图：SHAP值分布图
    for i, feature_idx in enumerate(top_features_idx):
        feature_name = top_feature_names[i]
        
        # 所有样本在该特征上的SHAP值
        feature_shap_values = [shap_values[idx, feature_idx] for idx in sample_indices]
        
        # 添加每个样本在该特征上的SHAP值散点
        hover_texts = []
        for j, sample_idx in enumerate(sample_indices):
            shap_value = shap_values[sample_idx, feature_idx]
            feature_value = features.iloc[sample_idx, feature_idx] if isinstance(features, pd.DataFrame) else features[sample_idx, feature_idx]
            
            if dates is not None and sample_idx < len(dates):
                hover_texts.append(f"样本 {sample_idx}<br>日期: {dates[sample_idx]}<br>特征: {feature_name}<br>特征值: {feature_value:.{feature_decimals}f}<br>SHAP值: {shap_value:.{shap_decimals}f}")
            else:
                hover_texts.append(f"样本 {sample_idx}<br>特征: {feature_name}<br>特征值: {feature_value:.{feature_decimals}f}<br>SHAP值: {shap_value:.{shap_decimals}f}")
        
        # 为每个特征使用不同颜色
        feature_color = 'rgba(' + ','.join([str(int(50 + 200 * i / len(top_features_idx))), 
                                          str(int(50 + 150 * (len(top_features_idx) - i) / len(top_features_idx))), 
                                          str(int(100 + 150 * i / len(top_features_idx)))] + ['0.7']) + ')'
        
        fig.add_trace(
            go.Box(
                y=[i] * len(feature_shap_values),
                x=feature_shap_values,
                name=feature_name,
                orientation='h',
                marker=dict(color=feature_color),
                boxpoints='all',
                jitter=0.5,
                pointpos=0,
                hoveron="points",
                hoverinfo='text',
                hovertext=hover_texts,
                # 调整点大小
                marker_size=8
            ),
            row=2, col=1
        )
    
    # 为基准值添加垂直线
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=0, y1=1,
        yref="paper",
        line=dict(color="black", width=1),
        row=2, col=1
    )
    
    # 添加Y轴标签（特征名称）
    fig.update_yaxes(
        tickvals=list(range(len(top_feature_names))),
        ticktext=top_feature_names,
        row=2, col=1
    )
    
    # 创建日期标签（如果提供）
    if dates is not None and len(dates) > 0:
        date_labels = []
        for i in sample_indices:
            if i < len(dates):
                date_labels.append(dates[i])
            else:
                date_labels.append(f"样本 {i}")
        
        # 更新X轴刻度标签为日期（每隔几个显示一个，避免拥挤）
        step = max(1, len(sample_indices) // 10)  # 最多显示约10个标签
        tick_vals = list(range(0, len(sample_indices), step))
        tick_text = [date_labels[i] for i in tick_vals]
        
        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45,
            row=1, col=1
        )
    
    # 更新X和Y轴标题
    if y_pred is not None or y_true is not None:
        fig.update_xaxes(title_text="样本索引/日期", row=1, col=1)
        fig.update_yaxes(title_text="值", row=1, col=1)
    
    fig.update_xaxes(title_text="SHAP值", row=2, col=1)
    fig.update_yaxes(title_text="特征", row=2, col=1)
    
    # 构建标题
    period_info = ""
    if dates is not None and len(dates) > 0:
        if sample_start_index < len(dates) and sample_end_index-1 < len(dates):
            period_info = f"时间段: {dates[sample_start_index]} 至 {dates[sample_end_index-1]}"
    
    # 更新布局
    fig.update_layout(
        title=f"时期SHAP值分布{' | ' + period_info if period_info else ''}",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        template="plotly_white",
        width=plot_width,
        height=plot_height,
        margin=dict(t=150, b=50, l=150, r=50),
        boxmode='group'
    )
    
    # 保存为HTML文件
    if save:
        if save_path:
            plot(fig, filename=save_path, auto_open=auto_open)
        else:
            raise ValueError('save_path is not provided')
    else:
        plot(fig, auto_open=auto_open)
    
    return fig

def plot_global_shap_force(
    base_value: float,
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_names: List[str] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    line_series: Optional[pd.Series] = None,
    dates: Optional[List[str]] = None,
    color_by: str = 'sign',  # 'sign' 或 'category'
    category_values: Optional[np.ndarray] = None,
    category_colors: Optional[dict] = None,
    top_features_num: int = 10,
    shap_decimals: int = 2,
    pred_decimals: int = 4,
    feature_decimals: int = 4,
    plot_width: int = 1800,
    plot_height: int = 1600,
    point_size: int = 6,
    save: bool = False,
    save_path: Optional[str] = None,
    auto_open: bool = False
) -> go.Figure:
    """
    创建全局SHAP力图，展示所有样本的预测值分布和贡献情况
    
    参数:
        base_value (float): 模型的基准预测值
        shap_values (np.ndarray): 所有样本的SHAP值数组，形状为(n_samples, n_features)
        features (pd.DataFrame): 特征数据框，形状为(n_samples, n_features)
        feature_names (List[str]): 特征名称列表，如果为None则使用features.columns
        y_true (np.ndarray, optional): 真实值数组
        y_pred (np.ndarray, optional): 预测值数组
        line_series (pd.Series, optional): 时间序列数据（如收盘价），索引应与dates对应
        dates (List[str], optional): 日期列表，用于标注样本
        color_by (str): 散点颜色依据，'sign'表示根据预测值正负，'category'表示根据类别
        category_values (np.ndarray, optional): 类别值数组，当color_by='category'时使用
        category_colors (dict, optional): 类别对应的颜色字典，如{0: 'blue', 1: 'red'}
        top_features_num (int): 要显示的最重要特征数量，默认10
        shap_decimals (int): SHAP值显示的小数位数，默认2
        pred_decimals (int): 预测值、真实值和基准值显示的小数位数，默认4
        feature_decimals (int): 特征值显示的小数位数，默认4
        plot_width (int): 图表宽度，默认1800
        plot_height (int): 图表高度，默认1600
        point_size (int): 散点大小，默认6
        save (bool): 是否保存图表为HTML文件
        save_path (str, optional): 保存路径
        auto_open (bool): 是否自动打开保存的HTML文件
        
    返回:
        go.Figure: Plotly图表对象
    """
    # 使用DataFrame的列名作为特征名称（如果未提供）
    if feature_names is None and isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
    
    # 检查color_by参数
    if color_by not in ['sign', 'category']:
        raise ValueError("color_by参数必须是'sign'或'category'")
    
    if color_by == 'category' and category_values is None:
        raise ValueError("当color_by='category'时，必须提供category_values参数")
    
    # 设置默认类别颜色
    if color_by == 'category' and category_colors is None:
        unique_categories = np.unique(category_values)
        category_colors = {}
        # 生成不同类别的颜色
        cmap = px.colors.qualitative.Plotly
        for i, cat in enumerate(unique_categories):
            category_colors[cat] = cmap[i % len(cmap)]
    
    # 计算每个特征的平均绝对SHAP值（全局重要性）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    # 选择前N个最重要的特征
    top_features = feature_importance.head(top_features_num)['feature'].tolist()
    top_features_idx = [feature_names.index(feat) for feat in top_features]
    
    # 确定样本索引
    sample_indices = list(range(shap_values.shape[0]))
    
    # 准备hover信息
    hover_texts = []
    for i in sample_indices:
        hover_text = ""
        if dates is not None and i < len(dates):
            hover_text += f"样本 {i}<br>日期: {dates[i]}<br>"
        else:
            hover_text += f"样本 {i}<br>"
            
        # 添加预测值和真实值
        if y_pred is not None:
            hover_text += f"预测值: {y_pred[i]:.{pred_decimals}f}<br>"
        if y_true is not None:
            hover_text += f"真实值: {y_true[i]:.{pred_decimals}f}<br>"
        
        hover_text += f"基准值: {base_value:.{pred_decimals}f}<br><br>"
        
        # 添加每个重要特征的值和SHAP贡献
        hover_text += "<b>特征贡献:</b><br>"
        for feat in top_features:
            feat_idx = feature_names.index(feat)
            feat_val = features.iloc[i, feat_idx] if isinstance(features, pd.DataFrame) else features[i, feat_idx]
            shap_val = shap_values[i, feat_idx]
            
            # 使用HTML格式化，为特征名称使用黑色，为正SHAP值使用红色，为负SHAP值使用蓝色
            color = "red" if shap_val > 0 else "blue"
            # 特征名黑色，值和SHAP彩色
            hover_text += f"<span style='color:black'>{feat}</span>: {feat_val:.{feature_decimals}f} | <span style='color:{color}'>SHAP: {shap_val:.{shap_decimals}f}</span><br>"
            
        hover_texts.append(hover_text)
    
    # 创建子图布局
    # 根据是否有line_series决定子图数量和布局
    if line_series is not None:
        fig = sp.make_subplots(
            rows=3, cols=1,
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.08,
            subplot_titles=('时间序列与预测', '预测值与真实值分布', '预测值分布')
        )
    else:
        fig = sp.make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.1,
            subplot_titles=('预测值与真实值分布', '预测值分布')
        )
    
    # 如果提供了line_series，添加第一个子图
    if line_series is not None and dates is not None:
        # 确保line_series索引与dates一致
        if len(line_series) != len(dates):
            # 尝试将line_series重新索引到与dates相同的索引
            try:
                if isinstance(line_series.index, pd.DatetimeIndex):
                    # 将dates转换为日期时间索引
                    date_index = pd.DatetimeIndex(dates)
                    # 尝试重新索引line_series
                    line_series = line_series.reindex(date_index)
                else:
                    print("警告：line_series的索引与dates不匹配，可能导致绘图错误")
            except:
                print("警告：无法重新索引line_series，可能导致绘图错误")
        
        # 绘制时间序列线
        fig.add_trace(
            go.Scatter(
                x=list(range(len(dates))),
                y=line_series.values,
                mode='lines',
                name='时间序列',
                line=dict(color='gray', width=2),
            ),
            row=1, col=1
        )
        
        # 准备散点颜色
        if color_by == 'sign' and y_pred is not None:
            # 根据预测值的符号设置颜色
            colors = ['red' if pred > 0 else 'blue' for pred in y_pred]
        elif color_by == 'category' and category_values is not None:
            # 根据类别设置颜色
            colors = [category_colors.get(cat, 'gray') for cat in category_values]
        else:
            # 默认颜色
            colors = ['blue'] * len(sample_indices)
        
        # 添加预测值散点
        if y_pred is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sample_indices))),
                    y=line_series.values,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=colors,
                    ),
                    name='预测点',
                    hoverinfo='text',
                    hovertext=hover_texts
                ),
                row=1, col=1
            )
            
            # 添加图例项，显示不同颜色的含义
            if color_by == 'sign':
                # 添加正负图例
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name='正值预测',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        name='负值预测',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            elif color_by == 'category':
                # 添加类别图例
                for cat, color in category_colors.items():
                    if cat in category_values:  # 只添加存在的类别
                        fig.add_trace(
                            go.Scatter(
                                x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color=color),
                                name=f'类别 {cat}',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
    
    # 第二个子图：预测值与真实值的散点图（如果提供）
    row_for_pred_true = 2 if line_series is not None else 1
    if y_pred is not None:
        # 绘制预测值点
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sample_indices))),
                y=y_pred,
                mode='markers',
                name='预测值',
                marker=dict(
                    size=point_size,
                    color='blue',
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=hover_texts
            ),
            row=row_for_pred_true, col=1
        )
        
        # 添加基准值水平线
        fig.add_shape(
            type="line",
            x0=0, y0=base_value, x1=len(sample_indices)-1, y1=base_value,
            line=dict(color="gray", width=2, dash="dash"),
            row=row_for_pred_true, col=1
        )
        
        # 添加基准值标注
        fig.add_annotation(
            x=len(sample_indices)-1,
            y=base_value,
            text=f"基准值: {base_value:.{pred_decimals}f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=0,
            row=row_for_pred_true, col=1
        )
        
        # 如果提供真实值，添加真实值点
        if y_true is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sample_indices))),
                    y=y_true,
                    mode='markers',
                    name='真实值',
                    marker=dict(
                        size=point_size,
                        color='red',
                        symbol='cross'
                    ),
                    hoverinfo='text',
                    hovertext=[f"样本 {i}{f'<br>日期: {dates[i]}' if dates is not None and i < len(dates) else ''}<br>真实值: {y_true[i]:.{pred_decimals}f}" for i in sample_indices]
                ),
                row=row_for_pred_true, col=1
            )
    
    # 第三个子图（或第二个子图如果没有line_series）：预测值散点图与SHAP值详情
    row_for_pred = 3 if line_series is not None else 2
    if y_pred is not None:
        # 绘制预测值点
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sample_indices))),
                y=y_pred,
                mode='markers',
                name='预测值',
                marker=dict(
                    size=point_size,
                    color='blue',
                    symbol='circle'
                ),
                hoverinfo='text',
                hovertext=hover_texts,
                showlegend=False
            ),
            row=row_for_pred, col=1
        )
        
        # 添加基准值水平线
        fig.add_shape(
            type="line",
            x0=0, y0=base_value, x1=len(sample_indices)-1, y1=base_value,
            line=dict(color="gray", width=2, dash="dash"),
            row=row_for_pred, col=1
        )
    
    # 创建日期标签（如果提供）
    if dates is not None and len(dates) > 0:
        # 更新X轴刻度标签为日期（每隔几个显示一个，避免拥挤）
        step = max(1, len(sample_indices) // 10)  # 最多显示约10个标签
        tick_vals = list(range(0, len(sample_indices), step))
        tick_text = [dates[i] for i in tick_vals if i < len(dates)]
        
        # 为所有子图设置X轴标签
        for row in range(1, row_for_pred + 1):
            fig.update_xaxes(
                tickvals=tick_vals[:len(tick_text)],
                ticktext=tick_text,
                tickangle=45,
                row=row, col=1
            )
    
    # 更新X和Y轴标题
    if line_series is not None:
        fig.update_xaxes(title_text="样本索引/日期", row=1, col=1)
        fig.update_yaxes(title_text="值", row=1, col=1)
    
    fig.update_xaxes(title_text="样本索引/日期", row=row_for_pred_true, col=1)
    fig.update_yaxes(title_text="值", row=row_for_pred_true, col=1)
    
    fig.update_xaxes(title_text="样本索引/日期", row=row_for_pred, col=1)
    fig.update_yaxes(title_text="预测值", row=row_for_pred, col=1)
    
    # 更新布局
    fig.update_layout(
        title="全局SHAP值分析",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        template="plotly_white",
        width=plot_width,
        height=plot_height,
        margin=dict(t=150, b=50, l=100, r=50)
    )
    
    # 保存为HTML文件
    if save:
        if save_path:
            plot(fig, filename=save_path, auto_open=auto_open)
        else:
            raise ValueError('save_path is not provided')
    else:
        plot(fig, auto_open=auto_open)
    
    return fig