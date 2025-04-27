#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的BackTrader示例
展示BackTrader的基本功能，非常适合初学者
"""

import datetime
import os.path
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt


# 1. 定义一个简单策略
class SimpleStrategy(bt.Strategy):
    """
    简单的移动平均线交叉策略
    适合展示BackTrader的基本功能
    """
    # 策略参数，可以在创建策略时进行修改
    params = (
        ('fast_length', 10),  # 快速移动平均线周期
        ('slow_length', 30),  # 慢速移动平均线周期
        ('printlog', True),   # 是否打印日志
    )

    def __init__(self):
        """初始化策略（在回测开始前调用一次）"""
        # 初始化移动平均线指标
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, 
            period=self.params.fast_length
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, 
            period=self.params.slow_length
        )
        
        # 计算移动平均线交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # 用于追踪订单
        self.order = None
        
        # 追踪当前收盘价
        self.dataclose = self.data.close
        
        # 显示策略参数
        print('策略参数:')
        print(f'- 快速均线周期: {self.params.fast_length}')
        print(f'- 慢速均线周期: {self.params.slow_length}')

    def log(self, txt, dt=None):
        """记录信息"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """订单状态变化通知"""
        # 如果订单已提交/接受，不做任何事
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 订单已完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
            else:
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
        
        # 订单被取消/拒绝/保证金不足
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

        # 重置订单变量
        self.order = None

    def notify_trade(self, trade):
        """交易状态变化通知"""
        if trade.isclosed:
            self.log(f'交易利润: 总利润={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}')

    def next(self):
        """
        策略核心逻辑（每个bar都会调用一次）
        这是实现交易规则的地方
        """
        # 记录当前收盘价
        self.log(f'收盘价: {self.dataclose[0]:.2f}')

        # 如果有未完成的订单，不进行新的交易
        if self.order:
            return

        # 如果没有持仓
        if not self.position:
            # 如果快速均线上穿慢速均线，买入信号
            if self.crossover > 0:
                self.log(f'买入信号: {self.dataclose[0]:.2f}')
                # 执行买入，使用全部资金的90%
                size = int(self.broker.getcash() * 0.9 / self.dataclose[0])
                self.log(f'买入 {size} 股')
                self.order = self.buy(size=size)
        
        # 如果已经有持仓
        else:
            # 如果快速均线下穿慢速均线，卖出信号
            if self.crossover < 0:
                self.log(f'卖出信号: {self.dataclose[0]:.2f}')
                # 执行卖出，卖出全部持仓
                self.log(f'卖出 {self.position.size} 股')
                self.order = self.sell(size=self.position.size)
    
    def stop(self):
        """回测结束时调用"""
        self.log('回测结束')
        self.log(f'期初资金: {self.broker.startingcash:.2f}')
        self.log(f'期末资金: {self.broker.getvalue():.2f}')
        self.log(f'总收益率: {(self.broker.getvalue() / self.broker.startingcash - 1) * 100:.2f}%')


# 2. 生成模拟数据
def create_sample_data(filename='sample_data.csv', 
                       start_date=datetime.datetime(2010, 1, 1),
                       end_date=datetime.datetime(2020, 1, 1)):
    """生成样本数据用于回测"""
    print("生成样本数据...")
    
    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 生成模拟价格
    np.random.seed(42)  # 设置随机种子保证可重复性
    
    n = len(date_range)
    
    # 创建有趋势的价格
    price_trend = np.linspace(100, 200, n) + np.random.normal(0, 10, n).cumsum()
    
    # 生成价格
    close = np.maximum(price_trend, 1)  # 确保价格为正
    high = close * (1 + np.random.uniform(0, 0.03, n))
    low = close * (1 - np.random.uniform(0, 0.03, n))
    open_price = low + np.random.random(n) * (high - low)
    
    # 生成交易量
    volume = np.random.normal(1000000, 200000, n)
    volume = np.maximum(volume, 0)  # 确保交易量为正
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': date_range,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    # 设置日期为索引
    data.set_index('date', inplace=True)
    
    # 保存到CSV文件
    data.to_csv(filename)
    print(f"数据已保存到 {filename}")
    
    return filename


# 3. 执行回测
def run_backtest(data_file, strategy_params=None):
    """执行回测"""
    print("开始回测...")
    
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加数据
    print(f"加载数据: {data_file}")
    data = bt.feeds.GenericCSVData(
        dataname=data_file,
        dtformat='%Y-%m-%d',
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )
    cerebro.adddata(data)
    
    # 设置初始资金
    initial_cash = 100000.0
    cerebro.broker.setcash(initial_cash)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 添加策略
    if strategy_params:
        cerebro.addstrategy(SimpleStrategy, **strategy_params)
    else:
        cerebro.addstrategy(SimpleStrategy)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 显示启动信息
    print(f"初始资金: {initial_cash:.2f}")
    
    # 运行回测
    strategies = cerebro.run()
    strategy = strategies[0]
    
    # 打印回测结果
    print("\n==== 回测结果 ====")
    
    # 打印最终资金
    final_value = cerebro.broker.getvalue()
    print(f"最终资金: {final_value:.2f}")
    print(f"总收益率: {(final_value / initial_cash - 1) * 100:.2f}%")
    
    # 打印交易统计
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    print(f"总交易次数: {total_trades}")
    
    if total_trades > 0:
        win_trades = trade_analysis.get('won', {}).get('total', 0)
        loss_trades = trade_analysis.get('lost', {}).get('total', 0)
        print(f"盈利交易: {win_trades}, 亏损交易: {loss_trades}")
        
        if win_trades > 0:
            win_pct = win_trades / total_trades * 100
            print(f"胜率: {win_pct:.2f}%")
        
        # 打印分析指标
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        if sharpe:
            print(f"夏普比率: {sharpe:.3f}")
        
        max_dd = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
        print(f"最大回撤: {max_dd:.2f}%")
        
        annual_return = strategy.analyzers.returns.get_analysis().get('ravg', 0.0) * 100
        print(f"年化收益率: {annual_return:.2f}%")
    
    # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    print("\n绘制回测图表...")
    cerebro.plot(style='candle', barup='red', bardown='green',
                plotdist=0.5, volume=True, figsize=(15, 10))
    
    return strategy


# 4. 主函数
def main():
    """主函数"""
    print("=" * 50)
    print("简单BackTrader回测示例")
    print("=" * 50)
    
    # 生成样本数据
    data_file = create_sample_data()
    
    # 执行回测
    strategy_params = {
        'fast_length': 10,
        'slow_length': 30,
        'printlog': True
    }
    
    # 运行回测
    strategy = run_backtest(data_file, strategy_params)
    
    print("\n回测完成!")


if __name__ == '__main__':
    main() 