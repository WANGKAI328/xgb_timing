#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BackTrader 全面演示程序
- 展示BackTrader的核心组件
- 演示如何处理多个资产
- 展示如何添加不同类型的数据
- 实现多种不同的策略
"""

import datetime
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt


##########################################
# 第一部分：数据处理和加载
##########################################

def get_data_from_yahoo(ticker, start_date, end_date):
    """从Yahoo获取数据"""
    data = bt.feeds.YahooFinanceData(
        dataname=ticker,
        fromdate=start_date,
        todate=end_date,
        reverse=False
    )
    return data


def prepare_csv_data(filename, start_date, end_date):
    """从CSV文件加载数据"""
    data = bt.feeds.GenericCSVData(
        dataname=filename,
        fromdate=start_date,
        todate=end_date,
        nullvalue=0.0,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1  # -1表示不使用该列
    )
    return data


def generate_mock_data(ticker, start_date, end_date):
    """生成模拟数据用于测试"""
    print(f"为 {ticker} 生成模拟数据...")
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 设置随机种子，确保可重现性
    np.random.seed(42 + hash(ticker) % 100)  # 不同的股票有不同的种子
    
    n = len(date_index)
    # 生成更具有真实市场特征的价格数据
    # 使用随机游走模型生成有趋势的价格序列
    # 基础价格变动
    price_changes = np.random.normal(0.0005, 0.015, n)
    
    # 添加一些趋势和波动性周期
    trend = np.linspace(0, 0.2, n) * np.sin(np.linspace(0, 10, n))
    price_changes = price_changes + trend
    
    # 生成价格序列
    close = 100.0 * np.exp(np.cumsum(price_changes))
    
    # 生成开盘价、最高价、最低价
    daily_volatility = np.random.uniform(0.005, 0.02, n)
    high = close * (1 + daily_volatility)
    low = close * (1 - daily_volatility)
    open_price = low + np.random.random(n) * (high - low)
    
    # 确保价格序列有正确的时间顺序关系
    for i in range(n):
        high[i] = max(close[i], open_price[i], high[i])
        low[i] = min(close[i], open_price[i], low[i])
    
    # 生成交易量，与波动性相关
    volume = np.random.normal(1000000, 200000, n) * (1 + daily_volatility * 5)
    volume = np.abs(volume)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=date_index)
    
    # 保存到临时CSV文件 - 修复路径问题
    csv_file = f'temp_{ticker.replace(".", "_")}.csv'
    df.to_csv(csv_file)
    print(f"已保存数据到 {csv_file}, 包含 {len(df)} 行")
    
    # 创建并返回数据源
    data = bt.feeds.GenericCSVData(
        dataname=csv_file,
        fromdate=start_date,
        todate=end_date,
        nullvalue=0.0,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )
    return data


##########################################
# 第二部分：自定义指标
##########################################

class MomentumIndicator(bt.Indicator):
    """
    自定义动量指标：当前收盘价与N天前收盘价的比率
    """
    # 指标的参数
    params = (
        ('period', 20),  # 默认周期为20天
    )
    
    # 指标的线
    lines = ('momentum',)
    
    def __init__(self):
        # 定义指标的计算方式
        self.lines.momentum = self.data.close / self.data.close(-self.params.period)
        
        # 设置指标绘图的相关属性
        self.plotinfo.subplot = True  # 使用子图
        self.plotinfo.plotname = '动量指标'  # 指标名称
        self.plotinfo.plotabove = False  # 在主图的下方显示
        
        # 设置线条的属性
        self.plotlines.momentum.plotname = 'MOM'
        self.plotlines.momentum.color = 'green'


class VolatilityIndicator(bt.Indicator):
    """
    自定义波动率指标：N天收盘价的标准差
    """
    # 指标的参数
    params = (
        ('period', 20),  # 默认计算20天的波动率
    )
    
    # 指标的线
    lines = ('volatility',)
    
    def __init__(self):
        # 计算波动率
        self.lines.volatility = bt.indicators.StdDev(self.data.close, period=self.params.period)
        
        # 设置指标绘图的相关属性
        self.plotinfo.subplot = True
        self.plotinfo.plotname = '波动率指标'
        self.plotinfo.plotabove = False
        
        # 设置线条的属性
        self.plotlines.volatility.plotname = 'VOL'
        self.plotlines.volatility.color = 'red'


##########################################
# 第三部分：策略定义
##########################################

class BaseStrategy(bt.Strategy):
    """
    基础策略类：包含所有策略共用的功能
    """
    def log(self, txt, dt=None):
        """日志功能"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
            else:
                self.log(f'卖出执行: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单被取消/保证金不足/拒绝')
            
        # 重置订单引用
        self.order = None
    
    def notify_trade(self, trade):
        """交易状态通知"""
        if not trade.isclosed:
            return
            
        self.log(f'交易利润: 毛利润 {trade.pnl:.2f}, 净利润 {trade.pnlcomm:.2f}')


class MovingAverageCrossover(BaseStrategy):
    """
    均线交叉策略：快速均线上穿慢速均线买入，下穿卖出
    """
    params = (
        ('fast_period', 10),  # 快速均线周期
        ('slow_period', 30),  # 慢速均线周期
        ('printlog', False),   # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        # 初始化指标
        self.fast_ma = {}
        self.slow_ma = {}
        self.crossover = {}
        
        # 对每个数据源分别初始化指标
        for i, d in enumerate(self.datas):
            # 获取数据名称
            data_name = d._name
            
            # 初始化均线
            self.fast_ma[data_name] = bt.indicators.SMA(d.close, 
                                                        period=self.params.fast_period)
            self.slow_ma[data_name] = bt.indicators.SMA(d.close, 
                                                        period=self.params.slow_period)
            
            # 初始化交叉信号
            self.crossover[data_name] = bt.indicators.CrossOver(
                self.fast_ma[data_name], 
                self.slow_ma[data_name]
            )
        
        # 跟踪订单
        self.orders = {d._name: None for d in self.datas}
    
    def next(self):
        """每个bar执行一次"""
        # 遍历每个数据源
        for i, d in enumerate(self.datas):
            data_name = d._name
            
            # 如果有未完成的订单，跳过该数据
            if self.orders[data_name]:
                continue
                
            # 检查持仓
            pos = self.getposition(d).size
            
            # 如果没有持仓且有买入信号
            if not pos and self.crossover[data_name] > 0:
                self.log(f'{data_name} - 买入信号', d.datetime.date(0))
                # 计算资金的10%用于买入
                cash = self.broker.get_cash() * 0.1
                size = int(cash / d.close[0])
                
                # 执行买入
                if size > 0:
                    self.orders[data_name] = self.buy(data=d, size=size)
            
            # 如果有持仓且有卖出信号
            elif pos and self.crossover[data_name] < 0:
                self.log(f'{data_name} - 卖出信号', d.datetime.date(0))
                # 执行卖出
                self.orders[data_name] = self.sell(data=d, size=pos)


class MomentumStrategy(BaseStrategy):
    """
    动量策略：动量指标上穿阈值买入，下穿阈值卖出
    """
    params = (
        ('momentum_period', 20),     # 动量指标周期
        ('buy_threshold', 1.05),     # 买入阈值
        ('sell_threshold', 0.95),    # 卖出阈值
        ('printlog', False),          # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        # 初始化指标
        self.momentum = {}
        
        # 对每个数据源分别初始化
        for i, d in enumerate(self.datas):
            data_name = d._name
            # 初始化动量指标
            self.momentum[data_name] = MomentumIndicator(d,  # d是数据源
                                                         period=self.params.momentum_period)
        
        # 跟踪订单
        self.orders = {d._name: None for d in self.datas}
    
    def next(self):
        """每个bar执行一次"""
        # 遍历每个数据源
        for i, d in enumerate(self.datas):
            data_name = d._name
            
            # 如果有未完成的订单，跳过该数据
            if self.orders[data_name]:
                continue
                
            # 检查持仓
            pos = self.getposition(d).size
            
            # 计算当前动量值
            momentum_value = self.momentum[data_name].momentum[0]
            
            # 如果没有持仓且动量高于买入阈值
            if not pos and momentum_value > self.params.buy_threshold:
                self.log(f'{data_name} - 动量买入信号: {momentum_value:.4f}', d.datetime.date(0))
                # 计算资金的10%用于买入
                cash = self.broker.get_cash() * 0.1
                size = int(cash / d.close[0])
                
                # 执行买入
                if size > 0:
                    self.orders[data_name] = self.buy(data=d, size=size)
            
            # 如果有持仓且动量低于卖出阈值
            elif pos and momentum_value < self.params.sell_threshold:
                self.log(f'{data_name} - 动量卖出信号: {momentum_value:.4f}', d.datetime.date(0))
                # 执行卖出
                self.orders[data_name] = self.sell(data=d, size=pos)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    波动率突破策略：价格突破前N天波动率范围时产生信号
    """
    params = (
        ('volatility_period', 20),   # 波动率计算周期
        ('breakout_factor', 1.5),    # 突破因子
        ('printlog', False),          # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        # 初始化指标
        self.volatility = {}
        self.average_price = {}
        
        # 对每个数据源分别初始化
        for i, d in enumerate(self.datas):
            data_name = d._name
            # 初始化波动率指标
            self.volatility[data_name] = VolatilityIndicator(d, 
                                                           period=self.params.volatility_period)
            # 初始化平均价格指标
            self.average_price[data_name] = bt.indicators.SMA(d.close, 
                                                            period=self.params.volatility_period)
        
        # 跟踪订单
        self.orders = {d._name: None for d in self.datas}
    
    def next(self):
        """每个bar执行一次"""
        # 遍历每个数据源
        for i, d in enumerate(self.datas):
            data_name = d._name
            
            # 如果有未完成的订单，跳过该数据
            if self.orders[data_name]:
                continue
                
            # 检查持仓
            pos = self.getposition(d).size
            
            # 计算突破范围
            volatility_value = self.volatility[data_name].volatility[0]
            avg_price = self.average_price[data_name][0]
            
            upper_band = avg_price + volatility_value * self.params.breakout_factor
            lower_band = avg_price - volatility_value * self.params.breakout_factor
            
            # 如果没有持仓且价格突破上轨
            if not pos and d.close[0] > upper_band:
                self.log(f'{data_name} - 向上突破信号: {d.close[0]:.2f} > {upper_band:.2f}', d.datetime.date(0))
                # 计算资金的10%用于买入
                cash = self.broker.get_cash() * 0.1
                size = int(cash / d.close[0])
                
                # 执行买入
                if size > 0:
                    self.orders[data_name] = self.buy(data=d, size=size)
            
            # 如果有持仓且价格突破下轨
            elif pos and d.close[0] < lower_band:
                self.log(f'{data_name} - 向下突破信号: {d.close[0]:.2f} < {lower_band:.2f}', d.datetime.date(0))
                # 执行卖出
                self.orders[data_name] = self.sell(data=d, size=pos)


##########################################
# 第四部分：多资产回测示例
##########################################

def run_multi_asset_backtest(strategy_class, start_date, end_date, tickers=None, params=None):
    """
    运行多资产回测
    
    参数:
        strategy_class: 策略类
        start_date: 回测开始日期
        end_date: 回测结束日期
        tickers: 股票代码列表，默认为None，使用模拟数据
        params: 策略参数
    """
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加自定义Observer
    # 注释掉可能导致错误的Observer
    # cerebro.addobserver(PositionValue)
    # cerebro.addobserver(PositionObserver)
    
    # 添加策略
    if params:
        cerebro.addstrategy(strategy_class, **params)
    else:
        cerebro.addstrategy(strategy_class)
    
    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 使用默认的模拟数据
    if not tickers:
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    
    print(f"加载 {len(tickers)} 个股票的数据...")
    
    # 添加数据 - 使用模拟数据
    for ticker in tickers:
        data = generate_mock_data(ticker, start_date, end_date)
        data._name = ticker  # 设置数据源名称
        cerebro.adddata(data)
    
    # 添加分析器，设置合适的参数
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 打印起始资金
    print(f'起始资金: {cerebro.broker.getvalue():.2f}')
    
    # 设置交易日志显示
    cerebro.stdstats = True  # 使用标准统计信息
    cerebro.broker.set_slippage_perc(0.001)  # 设置滑点为0.1%
    
    # 运行回测
    print("开始运行策略回测...")
    results = cerebro.run()
    
    # 打印最终资金
    final_value = cerebro.broker.getvalue()
    print(f'最终资金: {final_value:.2f}')
    print(f'收益率: {((final_value / 100000.0) - 1.0) * 100:.2f}%')
    
    # 打印分析结果
    strat = results[0]
    
    # 添加安全的结果提取
    try:
        # 尝试获取交易分析结果
        if hasattr(strat.analyzers, 'trades'):
            trades_info = strat.analyzers.trades.get_analysis()
            
            # 安全地提取交易次数
            if isinstance(trades_info, dict):
                total_section = trades_info.get('total', {})
                if isinstance(total_section, dict):
                    total_trades = total_section.get('total', 0)
                else:
                    total_trades = 0
            else:
                total_trades = 0
                
            print(f'总交易次数: {total_trades}')
            
            if total_trades > 0:
                # 只在有交易时打印详细信息
                print("详细交易信息:")
                
                # 安全地提取盈利和亏损交易
                if isinstance(trades_info, dict):
                    won_section = trades_info.get('won', {})
                    lost_section = trades_info.get('lost', {})
                    
                    if isinstance(won_section, dict) and isinstance(lost_section, dict):
                        won = won_section.get('total', 0)
                        lost = lost_section.get('total', 0)
                        print(f'  盈利交易: {won}, 亏损交易: {lost}')
                
                # 安全地提取夏普比率
                if hasattr(strat.analyzers, 'sharpe'):
                    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
                    if isinstance(sharpe_analysis, dict):
                        sharpe = sharpe_analysis.get('sharperatio', None)
                        if sharpe is not None:
                            print(f'夏普比率: {sharpe:.3f}')
                        else:
                            print('夏普比率: 无法计算 (可能需要更长的回测期间)')
                
                # 安全地提取最大回撤
                if hasattr(strat.analyzers, 'drawdown'):
                    dd_analysis = strat.analyzers.drawdown.get_analysis()
                    if isinstance(dd_analysis, dict):
                        max_dd = dd_analysis.get('max', {})
                        if isinstance(max_dd, dict):
                            drawdown = max_dd.get('drawdown', 0.0)
                            print(f'最大回撤: {drawdown:.2f}%')
                
                # 安全地提取年化收益率
                if hasattr(strat.analyzers, 'returns'):
                    returns_analysis = strat.analyzers.returns.get_analysis()
                    if isinstance(returns_analysis, dict):
                        returns = returns_analysis.get('ravg', 0.0)
                        if returns is not None:
                            print(f'年化收益率: {returns * 100:.2f}%')
            else:
                print('警告: 回测期间没有发生交易，无法计算性能指标')
                print('可能的原因:')
                print('1. 策略参数不符合当前市场情况')
                print('2. 回测时间太短')
                print('3. 策略逻辑问题')
        else:
            print('警告: 未找到交易分析器')
            
    except Exception as e:
        print(f'分析结果计算错误: {str(e)}')
        import traceback
        traceback.print_exc()
    
    # 绘制结果
    try:
        print("生成回测图表...")
        figs = cerebro.plot(style='candle', barup='red', bardown='green', volume=True)
        # 可以在这里保存图表到文件
        # 例如：plt.savefig('backtest_result.png')
    except Exception as e:
        print(f'绘图错误: {str(e)}')
        import traceback
        traceback.print_exc()
    
    return results


##########################################
# 第五部分：自定义Observer和Writer
##########################################

class PositionValue(bt.Observer):
    """
    自定义Observer，跟踪每个资产的持仓价值
    """
    lines = ('value',)
    
    plotinfo = dict(plot=True, subplot=True, plotname='持仓价值')
    
    def next(self):
        # 计算所有资产的持仓价值总和
        total_value = 0
        for d in self.datas:
            # 获取该数据源的持仓
            position = self.owner.getposition(d)
            if position.size:
                # 计算持仓价值 = 持仓量 * 当前价格
                value = position.size * d.close[0]
                total_value += value
        
        # 更新持仓价值线
        self.lines.value[0] = total_value


class PositionObserver(bt.Observer):
    """
    自定义Observer，显示每个资产的持仓状态
    - 使用正确的方式定义动态线
    """
    # 定义一个空的lines属性，稍后我们会动态添加线
    lines = ()
    plotinfo = dict(plot=True, subplot=True, plotname='持仓状态')
    
    def __init__(self):
        # 使用这种方式无法在运行时动态添加线，所以我们需要简化处理方式
        # 创建一个单一的总持仓线
        self.plotlines.pos = dict(_name='总持仓', color='blue')

    def next(self):
        # 计算所有资产的总持仓
        total_pos = 0
        for d in self.datas:
            position = self.owner.getposition(d)
            if position.size:
                total_pos += position.size
        
        # 更新持仓线
        self.lines.pos[0] = total_pos


##########################################
# 第六部分：完整演示
##########################################

def main():
    """主函数，运行完整演示"""
    # 设置更长的回测期间，确保有足够的交易信号
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    
    # 测试股票列表
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    
    print("=" * 50)
    print("均线交叉策略回测")
    print("=" * 50)
    # 运行均线交叉策略回测，调整参数使其产生更多信号
    ma_params = {
        'fast_period': 5,  # 更短的快速均线
        'slow_period': 20,  # 更短的慢速均线
        'printlog': True
    }
    ma_results = run_multi_asset_backtest(
        MovingAverageCrossover, 
        start_date, 
        end_date, 
        tickers, 
        ma_params
    )
    
    print("\n" + "=" * 50)
    print("动量策略回测")
    print("=" * 50)
    # 运行动量策略回测
    momentum_params = {
        'momentum_period': 20,
        'buy_threshold': 1.05,
        'sell_threshold': 0.95,
        'printlog': True
    }
    momentum_results = run_multi_asset_backtest(
        MomentumStrategy, 
        start_date, 
        end_date, 
        tickers, 
        momentum_params
    )
    
    print("\n" + "=" * 50)
    print("波动率突破策略回测")
    print("=" * 50)
    # 运行波动率突破策略回测
    volatility_params = {
        'volatility_period': 20,
        'breakout_factor': 1.5,
        'printlog': True
    }
    volatility_results = run_multi_asset_backtest(
        VolatilityBreakoutStrategy, 
        start_date, 
        end_date, 
        tickers, 
        volatility_params
    )
    
    print("\n" + "=" * 50)
    print("策略对比")
    print("=" * 50)
    
    # 安全地提取和比较结果
    try:
        # 检查是否有足够的交易来进行比较
        ma_trades = ma_results[0].analyzers.trades.get_analysis().get('total', {}).get('total', 0)
        mom_trades = momentum_results[0].analyzers.trades.get_analysis().get('total', {}).get('total', 0)
        vol_trades = volatility_results[0].analyzers.trades.get_analysis().get('total', {}).get('total', 0)
        
        if ma_trades > 0 and mom_trades > 0 and vol_trades > 0:
            # 获取夏普比率
            ma_sharpe = ma_results[0].analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
            mom_sharpe = momentum_results[0].analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
            vol_sharpe = volatility_results[0].analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0
            
            # 获取年化收益率
            ma_returns = (ma_results[0].analyzers.returns.get_analysis().get('ravg', 0.0) or 0.0) * 100
            mom_returns = (momentum_results[0].analyzers.returns.get_analysis().get('ravg', 0.0) or 0.0) * 100
            vol_returns = (volatility_results[0].analyzers.returns.get_analysis().get('ravg', 0.0) or 0.0) * 100
            
            print(f'均线交叉策略 - 夏普比率: {ma_sharpe:.3f}, 年化收益率: {ma_returns:.2f}%, 交易次数: {ma_trades}')
            print(f'动量策略 - 夏普比率: {mom_sharpe:.3f}, 年化收益率: {mom_returns:.2f}%, 交易次数: {mom_trades}')
            print(f'波动率突破策略 - 夏普比率: {vol_sharpe:.3f}, 年化收益率: {vol_returns:.2f}%, 交易次数: {vol_trades}')
            
            # 找出最佳策略
            strategies = ['均线交叉策略', '动量策略', '波动率突破策略']
            sharpes = [ma_sharpe, mom_sharpe, vol_sharpe]
            returns = [ma_returns, mom_returns, vol_returns]
            
            best_sharpe_idx = sharpes.index(max(sharpes))
            best_return_idx = returns.index(max(returns))
            
            print(f'\n基于夏普比率的最佳策略: {strategies[best_sharpe_idx]}')
            print(f'基于收益率的最佳策略: {strategies[best_return_idx]}')
        else:
            print('一个或多个策略没有产生足够的交易来进行有效比较')
            
    except Exception as e:
        print(f'策略比较过程中出错: {str(e)}')


if __name__ == '__main__':
    main() 