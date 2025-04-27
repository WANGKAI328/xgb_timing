# BackTrader 完整演示 - 多资产回测系统

这个演示项目展示了BackTrader的全面功能，包括多资产回测、自定义指标、多种策略比较以及性能分析。适合初学者全面学习BackTrader的各个组件及其使用方法。

## 目录

1. [安装](#安装)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [运行示例](#运行示例)
5. [自定义组件](#自定义组件)
6. [多资产处理](#多资产处理)
7. [扩展指南](#扩展指南)

## 安装

运行此演示程序前，需要安装以下依赖：

```bash
pip install backtrader pandas numpy matplotlib
```

## 项目结构

`BackTrader_DEMO.py` 文件被组织为六个主要部分：

1. **数据处理和加载**：展示如何从不同源获取和处理数据
2. **自定义指标**：创建自定义技术指标
3. **策略定义**：实现多个交易策略
4. **多资产回测**：演示如何同时对多个资产进行回测
5. **自定义Observer和Writer**：监控和记录回测结果
6. **完整演示**：整合以上组件的主函数

## 核心组件

### 1. 数据源

演示程序提供了三种获取数据的方法：

- **Yahoo Finance**: 直接从网络获取股票数据
  ```python
  data = get_data_from_yahoo('AAPL', start_date, end_date)
  ```

- **CSV文件**: 从本地CSV文件加载数据
  ```python
  data = prepare_csv_data('data.csv', start_date, end_date)
  ```

- **模拟数据**: 生成测试用的模拟数据
  ```python
  data = generate_mock_data('AAPL', start_date, end_date)
  ```

### 2. 自定义指标

演示了两种自定义指标的创建方法：

- **动量指标**: 计算当前价格与N天前价格的比率
  ```python
  momentum = MomentumIndicator(data, period=20)
  ```

- **波动率指标**: 计算N天收盘价的标准差
  ```python
  volatility = VolatilityIndicator(data, period=20)
  ```

### 3. 交易策略

实现了三种不同的交易策略：

- **均线交叉策略**: 基于快速和慢速移动平均线的交叉
- **动量策略**: 基于价格动量指标的突破
- **波动率突破策略**: 基于价格相对于波动率范围的突破

所有策略都继承自`BaseStrategy`，它提供了日志记录、订单管理等基础功能。

### 4. 分析器

展示了如何使用分析器评估策略性能：

- **夏普比率**: 评估收益相对于风险的效率
- **最大回撤**: 评估策略的风险
- **收益率**: 计算投资回报率

## 运行示例

直接运行脚本即可执行完整演示：

```bash
python BackTrader_DEMO.py
```

这将会：
1. 运行三种不同的策略
2. 绘制每种策略的回测结果
3. 显示和比较每种策略的性能指标

## 自定义组件

### 创建自定义指标

要创建新的技术指标，继承`bt.Indicator`类：

```python
class MyNewIndicator(bt.Indicator):
    lines = ('myline',)
    params = (('period', 20),)
    
    def __init__(self):
        self.lines.myline = self.data.close - bt.indicators.SMA(self.data.close, period=self.params.period)
```

### 创建自定义策略

要创建新的交易策略，继承`BaseStrategy`类：

```python
class MyNewStrategy(BaseStrategy):
    params = (('param1', 10), ('param2', 20))
    
    def __init__(self):
        # 初始化指标
        # ...
    
    def next(self):
        # 定义交易逻辑
        # ...
```

### 创建自定义Observer

要创建新的观察器，继承`bt.Observer`类：

```python
class MyNewObserver(bt.Observer):
    lines = ('myline',)
    
    def next(self):
        # 更新观察值
        # ...
```

## 多资产处理

演示程序展示了如何同时对多个资产进行回测：

1. **数据加载**: 为每个资产创建单独的数据源
2. **策略应用**: 在策略中使用字典存储每个资产的指标
3. **持仓管理**: 对每个资产分别计算仓位和订单
4. **资金分配**: 示例中每次交易使用总资金的10%

关键实现在于`run_multi_asset_backtest`函数，它接收多个资产的数据并进行回测。

## 扩展指南

### 添加新的资产类型

要支持更多资产，可以扩展数据加载函数：

```python
def get_data_for_forex(symbol, start_date, end_date):
    # 加载外汇数据的逻辑
    # ...
    return data
```

### 实现更复杂的策略

可以在策略中添加更多的指标和信号：

```python
# 在__init__中添加更多指标
self.rsi = bt.indicators.RSI(self.data, period=14)
self.bollinger = bt.indicators.BollingerBands(self.data, period=20)

# 在next中组合多个信号
if self.rsi < 30 and self.data.close < self.bollinger.lines.bot:
    # 超卖信号
    self.buy()
```

### 添加风险管理

可以扩展策略来包含止损和头寸规模管理：

```python
# 动态计算头寸规模
risk_per_trade = 0.02  # 每次交易风险2%的资金
price = self.data.close[0]
stop_price = price * 0.95  # 5%止损
risk_amount = price - stop_price
position_size = (self.broker.get_cash() * risk_per_trade) / risk_amount

# 使用止损下单
self.buy(size=position_size, exectype=bt.Order.StopLimit, price=price, plimit=price+0.01)
self.sell(exectype=bt.Order.Stop, price=stop_price)
```

## 参考资源

- [BackTrader官方文档](https://www.backtrader.com/docu/)
- [BackTrader GitHub](https://github.com/mementum/backtrader)
- [BackTrader社区](https://community.backtrader.com/) 