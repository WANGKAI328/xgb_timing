#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础的backtrader示例
展示如何加载数据和执行简单的移动平均策略
"""

import datetime
import os.path
import pandas as pd
import backtrader as bt


class SmaCrossStrategy(bt.Strategy):
    """
    移动平均线交叉策略
    """
    # 定义参数
    params = (
        ('fast_period', 10),  # 快速移动平均线周期
        ('slow_period', 30),  # 慢速移动平均线周期
    )

    def __init__(self):
        """
        初始化策略
        """
        # 初始化移动平均线指标
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
        
        # 用于跟踪订单
        self.order = None
        
        # 记录初始值
        self.dataclose = self.data.close
        
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
        # 简单记录收盘价
        self.log(f'收盘价, {self.dataclose[0]:.2f}')

        # 如果有未完成的订单，不采取行动
        if self.order:
            return

        # 检查是否持仓
        if not self.position:
            # 没有持仓
            
            # 如果快速移动平均线上穿慢速移动平均线 - 买入信号
            if self.crossover > 0:
                self.log(f'买入信号, {self.dataclose[0]:.2f}')
                # 买入
                self.order = self.buy()
        
        else:
            # 已有持仓
            
            # 如果快速移动平均线下穿慢速移动平均线 - 卖出信号
            if self.crossover < 0:
                self.log(f'卖出信号, {self.dataclose[0]:.2f}')
                # 卖出
                self.order = self.sell()


def run_strategy():
    """
    执行回测
    """
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(SmaCrossStrategy)

    # 设置现金起始值
    cerebro.broker.setcash(100000.0)
    
    # 设置佣金
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 准备示例数据
    # 如果有实际数据文件，可以替换下面的代码
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    
    # 方法1：使用Yahoo数据源(需要联网)
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',  # 股票代码
        fromdate=start_date,
        todate=end_date,
        reverse=False
    )
    
    # 方法2：如果有CSV文件，可以使用以下方式加载
    # data = bt.feeds.GenericCSVData(
    #     dataname='data_sample.csv',
    #     fromdate=start_date,
    #     todate=end_date,
    #     nullvalue=0.0,
    #     dtformat=('%Y-%m-%d'),
    #     datetime=0,
    #     open=1,
    #     high=2,
    #     low=3,
    #     close=4,
    #     volume=5,
    #     openinterest=-1  # -1表示不使用该列
    # )
    
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
    run_strategy() 