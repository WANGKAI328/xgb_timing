# Backtrader演示示例

本文件夹包含两个Backtrader的演示示例，用于学习如何使用这个Python回测框架。

## 安装必要的包

在运行这些示例之前，请确保已安装所需的包：

```bash
pip install backtrader pandas numpy matplotlib
```

## 示例文件

### 1. basic_strategy.py

这是一个基础的Backtrader示例，展示了如何：
- 创建一个简单的移动平均线交叉策略
- 加载市场数据
- 执行回测
- 可视化结果

该示例使用两条简单移动平均线（SMA）的交叉作为交易信号：
- 快速SMA上穿慢速SMA时产生买入信号
- 快速SMA下穿慢速SMA时产生卖出信号

运行方法：
```bash
python basic_strategy.py
```

### 2. custom_signal_strategy.py

这个高级示例展示了如何：
- 向Backtrader添加自定义信号数据
- 基于这些自定义信号构建交易策略
- 使用自定义数据执行回测
- 生成模拟数据用于测试

该示例包含两个主要功能：
1. `prepare_custom_data()`: 生成带有自定义信号列的模拟数据
2. `run_strategy_with_real_data()`: 使用包含自定义信号的实际CSV数据执行回测

运行方法：
```bash
python custom_signal_strategy.py
```

## 使用自己的数据

### 使用CSV数据文件

如果你想使用自己的CSV数据，请确保文件格式正确：
- 日期应作为索引列
- 需要包含`open`, `high`, `low`, `close`, `volume`列
- 对于`custom_signal_strategy.py`，还需要包含`signal1`和`signal2`列

### 使用其他数据源

Backtrader支持多种数据源，包括：
- Yahoo Finance
- Alpha Vantage
- Interactive Brokers
- 以及更多

## 自定义策略

### 创建自己的策略

要创建自己的交易策略，只需继承`bt.Strategy`类并实现必要的方法：

```python
class MyStrategy(bt.Strategy):
    params = (
        ('param1', default_value1),
        ('param2', default_value2),
    )
    
    def __init__(self):
        # 初始化指标和变量
        pass
        
    def next(self):
        # 主要的交易逻辑
        pass
```

### 关键方法

Backtrader策略中最重要的方法是：

- `__init__()`: 初始化策略，设置指标和变量
- `next()`: 每个bar都会调用，包含主要交易逻辑
- `notify_order()`: 处理订单状态变化
- `notify_trade()`: 处理交易结束事件

## 添加自定义信号

如`custom_signal_strategy.py`所示，添加自定义信号的步骤是：

1. 创建一个继承自`bt.feeds.PandasData`的自定义数据源类：
```python
class CustomSignalData(bt.feeds.PandasData):
    lines = ('signal1', 'signal2',)
    params = (
        ('signal1', 'signal1'),
        ('signal2', 'signal2'),
    )
```

2. 准备包含这些信号的数据：
```python
df = pd.DataFrame({
    'open': open_values,
    'high': high_values,
    'low': low_values,
    'close': close_values,
    'volume': volume_values,
    'signal1': signal1_values,
    'signal2': signal2_values
}, index=date_index)
```

3. 创建数据源并添加到Cerebro：
```python
data = CustomSignalData(dataname=df)
cerebro.adddata(data)
```

4. 在策略中使用这些信号：
```python
def __init__(self):
    self.signal1 = self.data.signal1
    self.signal2 = self.data.signal2
```

## 参考资源

- [Backtrader官方文档](https://www.backtrader.com/docu/)
- [Backtrader GitHub](https://github.com/mementum/backtrader)
- [Backtrader社区](https://community.backtrader.com/)

## 注意事项

- 这些示例仅用于教育目的，不构成投资建议
- 在实际交易之前，请进行彻底的策略测试
- 过度拟合历史数据可能导致策略在实际市场中表现不佳 