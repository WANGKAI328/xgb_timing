#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义信号的backtrader示例
展示如何添加自定义信号列并基于这些信号进行交易
"""

import datetime
import os.path
import pandas as pd
import numpy as np
import backtrader as bt


class CustomSignalData(bt.feeds.PandasData):
    """
    自定义数据源，扩展了标准的PandasData，添加了额外的信号列
    """
    # 添加自定义的列名
    lines = ('signal1', 'signal2',)
    
    # 定义参数，指明额外的列在DataFrame中的列名
    params = (
        ('signal1', 'signal1'),
        ('signal2', 'signal2'),
    )


class CustomSignalStrategy(bt.Strategy):
    """
    基于自定义信号的交易策略
    """
    params = (
        ('signal1_threshold', 0.5),  # 信号1的阈值
        ('signal2_threshold', 0.7),  # 信号2的阈值
    )

    def __init__(self):
        """
        初始化策略
        """
        # 保存收盘价的引用
        self.dataclose = self.data.close
        
        # 保存信号值的引用
        self.signal1 = self.data.signal1
        self.signal2 = self.data.signal2
        
        # 用于跟踪订单
        self.order = None
        
        # 保存信号的布尔条件
        self.signal1_condition = bt.indicators.CrossOver(self.signal1, self.params.signal1_threshold)
        self.signal2_condition = bt.indicators.CrossOver(self.signal2, self.params.signal2_threshold)
        
    def log(self, txt, dt=None):
        """
        记录信息的辅助函数
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """
        当订单状态改变时收到通知
        """
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交/已接受 - 无行动
            return

        # 检查订单是否已完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行，价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
            else:  # 卖出
                self.log(f'卖出执行，价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单被取消/保证金不足/拒绝')

        # 订单处理完毕，重置订单变量
        self.order = None

    def notify_trade(self, trade):
        """
        当交易结束时收到通知
        """
        if not trade.isclosed:
            return

        self.log(f'交易利润, 毛利润: {trade.pnl:.2f}, 净利润: {trade.pnlcomm:.2f}')

    def next(self):
        """
        每个bar都会调用的核心策略方法
        """
        # 记录数据
        self.log(f'收盘价: {self.dataclose[0]:.2f}, 信号1: {self.signal1[0]:.2f}, 信号2: {self.signal2[0]:.2f}')

        # 如果有未完成的订单，不采取行动
        if self.order:
            return

        # 检查是否持仓
        if not self.position:
            # 没有持仓
            
            # 生成买入信号的条件：两个信号都上穿各自的阈值
            if self.signal1_condition > 0 and self.signal2_condition > 0:
                self.log(f'买入信号触发 - 信号1: {self.signal1[0]:.2f}, 信号2: {self.signal2[0]:.2f}')
                # 买入
                self.order = self.buy()
        
        else:
            # 已有持仓
            
            # 生成卖出信号的条件：两个信号都下穿各自的阈值
            if self.signal1_condition < 0 and self.signal2_condition < 0:
                self.log(f'卖出信号触发 - 信号1: {self.signal1[0]:.2f}, 信号2: {self.signal2[0]:.2f}')
                # 卖出
                self.order = self.sell()


def prepare_custom_data(filename=''):
    """
    准备带有自定义信号的数据
    
    可以从CSV文件加载数据，或者为演示创建模拟数据
    """
    if filename and os.path.exists(filename):
        # 从文件加载数据
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    
    # 生成模拟数据用于演示
    # 创建日期索引
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 创建价格和交易量
    np.random.seed(42)  # 固定随机种子，保证可重复性
    
    n = len(date_index)
    close = np.random.normal(100, 1, n).cumsum() + 1000  # 模拟价格
    high = close + np.random.normal(0, 1, n).cumsum() * 0.5  # 最高价
    low = close - np.random.normal(0, 1, n).cumsum() * 0.5  # 最低价
    low = np.maximum(low, 0)  # 确保最低价不为负
    open_price = close + np.random.normal(0, 1, n)  # 开盘价
    volume = np.random.normal(1000000, 100000, n)  # 交易量
    
    # 创建自定义信号
    # 信号1：基于价格的相对强度指标 (RSI简化版)
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    window = 14
    for i in range(window, n):
        avg_gain[i] = np.mean(gain[i-window+1:i+1])
        avg_loss[i] = np.mean(loss[i-window+1:i+1])
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    signal1 = 100 - (100 / (1 + rs))
    signal1 = signal1 / 100  # 归一化到0-1
    
    # 信号2：模拟随机的技术指标
    signal2 = np.random.normal(0.5, 0.15, n)
    signal2 = np.clip(signal2, 0, 1)  # 限制在0-1之间
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'signal1': signal1,
        'signal2': signal2
    }, index=date_index)
    
    return df


def run_strategy():
    """
    执行回测
    """
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(CustomSignalStrategy)

    # 设置现金起始值
    cerebro.broker.setcash(100000.0)
    
    # 设置佣金
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 准备自定义数据
    df = prepare_custom_data()
    
    # 创建数据源
    data = CustomSignalData(
        dataname=df,
        fromdate=df.index[0],
        todate=df.index[-1]
    )
    
    # 添加数据到引擎
    cerebro.adddata(data)
    
    # 打印起始资金
    print(f'起始资金: {cerebro.broker.getvalue():.2f}')

    # 运行回测
    cerebro.run()

    # 打印最终资金
    print(f'最终资金: {cerebro.broker.getvalue():.2f}')
    
    # 绘制结果
    cerebro.plot(style='candle')


def run_strategy_with_real_data(filename):
    """
    使用实际数据文件执行回测
    
    参数:
        filename (str): CSV文件路径
    """
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(CustomSignalStrategy)

    # 设置现金起始值
    cerebro.broker.setcash(100000.0)
    
    # 设置佣金
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 读取CSV数据并确保有必要的列
    if not os.path.exists(filename):
        print(f"错误: 文件 {filename} 不存在")
        return
    
    # 读取数据，假设CSV文件有日期索引和需要的OHLCV和信号列
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    
    # 确保数据包含必要的列
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'signal1', 'signal2']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"错误: CSV文件缺少以下列: {missing_columns}")
        return
    
    # 创建数据源
    data = CustomSignalData(
        dataname=df,
        fromdate=df.index[0],
        todate=df.index[-1]
    )
    
    # 添加数据到引擎
    cerebro.adddata(data)
    
    # 打印起始资金
    print(f'起始资金: {cerebro.broker.getvalue():.2f}')

    # 运行回测
    cerebro.run()

    # 打印最终资金
    print(f'最终资金: {cerebro.broker.getvalue():.2f}')
    
    # 绘制结果
    cerebro.plot(style='candle')


if __name__ == '__main__':
    # 使用示例数据运行
    run_strategy()
    
    # 如果有实际数据文件，取消下面的注释并提供文件路径
    # run_strategy_with_real_data('your_data_with_signals.csv') 