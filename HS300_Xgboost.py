import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import os
import sys
import pathlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from datetime import datetime
import time
from tqdm import tqdm,trange
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import MachineLearningforAssetManagers as mlam
import utils.xgb_helperfunc as xgb_helperfunc
import utils.visualization_utils as vis_utils
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import optuna
import shap
import joblib

# 设置随机种子确保结果可重复
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

if __name__ == "__main__":
    # 读取hs300 指数
    hs300_index = pd.read_csv("hs300_Index.csv",index_col = 0, parse_dates = ['trade_day'])
    # hs300 趋势
    hs300_trend_result = mlam.getTrendLabel(price_series = hs300_index.close,l_span = np.arange(5,20))
    hs300_close_trend = mlam.get_trend_label_for_plotting(hs300_index.close,hs300_trend_result)
    # visualize
    # hs300_close_trend_plot = hs300_close_trend.dropna()
    # plt.figure(figsize=(16,8))
    # cmap = plt.cm.bwr
    # sns.scatterplot(data=hs300_close_trend_plot, 
    #                 x='trade_day', 
    #                 y='close',
    #                 hue='normed_t_value',  # Use the continuous values directly
    #                 palette=cmap,
    #                 legend=False)
    # plt.show()

    # hs300 收益率
    data_df = hs300_index.copy()
    data_df['intra_rtn'] = data_df['close'] / data_df['open'] - 1 # 日内收益率
    data_df['overnight_rtn'] = data_df['open'] / data_df['close'].shift(1) - 1 # 隔夜
    data_df['close_rtn'] = data_df['close'].pct_change()
    data_df = data_df.dropna()

    # Lookback日累计收益率和夏普动量 (短期)
    zipped_data = zip(
        ['intra','overnight','close'],
        [np.log(data_df['intra_rtn'] + 1),np.log(data_df['overnight_rtn'] + 1), np.log(data_df['close_rtn'] + 1)]
    )
    dt_index = data_df.index

    X_short_list = [] # 短期特征列表
    short_term_start = 10
    short_term_end = 30
    short_term_step = 5
    for rtn_category,rtn_series in zipped_data:
        for lookback in trange(short_term_start,short_term_end + short_term_step,short_term_step,position = 0, leave = True):
            # 因子名称
            cumrtn_factor_name = f"{rtn_category}cumrtn_LB{lookback}"
            sharpe_factor_name = f"{rtn_category}sharpe_LB{lookback}"
            # 因子数据列表
            cumrtn_list = []
            sharpe_list = []
            dt_list = []
            for end_dt in dt_index:
                end_dt_idx = np.searchsorted(dt_index,end_dt)
                start_dt_idx = end_dt_idx - lookback
            
                if start_dt_idx < 0:
                    continue
                else:
                    rtn_series_slice = rtn_series.iloc[start_dt_idx:end_dt_idx] # 选中的收益率序列
                    cumrtn = rtn_series_slice.sum() # 累计收益率
                    sharpe = rtn_series_slice.mean() / rtn_series_slice.std() # 夏普动量
                    '''添加数据到list中'''
                    cumrtn_list.append(cumrtn)
                    sharpe_list.append(sharpe)
                    dt_list.append(end_dt)
            '''生成series'''
            cumrtn_df_slice = pd.DataFrame(index = dt_list, data = cumrtn_list, columns = [cumrtn_factor_name])
            sharpe_df_slice = pd.DataFrame(index = dt_list, data = sharpe_list, columns = [sharpe_factor_name])
            X_short_list.append(cumrtn_df_slice)
            X_short_list.append(sharpe_df_slice)        
    X_short = pd.concat(X_short_list,axis = 1)

    # Lookback日累计收益率和夏普动量 (中期)
    zipped_data = zip(
        ['intra','overnight','close'],
        [np.log(data_df['intra_rtn'] + 1),np.log(data_df['overnight_rtn'] + 1), np.log(data_df['close_rtn'] + 1)]
    )
    dt_index = data_df.index

    X_medium_list = [] # 中期期特征列表
    mid_term_start = 35
    mid_term_end = 90
    mid_term_step = 5
    mid_term_ex_K = 5 # 排除最近K日
    for rtn_category,rtn_series in zipped_data:
        for lookback in trange(mid_term_start,mid_term_end + mid_term_step,mid_term_step,position = 0, leave = True):
            # 因子名称
            cumrtn_factor_name = f"{rtn_category}cumrtn_LB{lookback}_ex{mid_term_ex_K}"
            sharpe_factor_name = f"{rtn_category}sharpe_LB{lookback}_ex{mid_term_ex_K}"
            # 因子数据列表
            cumrtn_list = []
            sharpe_list = []
            dt_list = []
            for end_dt in dt_index:
                end_dt_idx = np.searchsorted(dt_index,end_dt) - mid_term_ex_K
                start_dt_idx = end_dt_idx - lookback
            
                if start_dt_idx < 0:
                    continue
                else:
                    rtn_series_slice = rtn_series.iloc[start_dt_idx:end_dt_idx] # 选中的收益率序列
                    cumrtn = rtn_series_slice.sum() # 累计收益率
                    sharpe = rtn_series_slice.mean() / rtn_series_slice.std() # 夏普动量
                    '''添加数据到list中'''
                    cumrtn_list.append(cumrtn)
                    sharpe_list.append(sharpe)
                    dt_list.append(end_dt)
            '''生成series'''
            cumrtn_df_slice = pd.DataFrame(index = dt_list, data = cumrtn_list, columns = [cumrtn_factor_name])
            sharpe_df_slice = pd.DataFrame(index = dt_list, data = sharpe_list, columns = [sharpe_factor_name])
            X_medium_list.append(cumrtn_df_slice)
            X_medium_list.append(sharpe_df_slice)  
    X_medium = pd.concat(X_medium_list,axis = 1)

    # Lookback日累计收益率和夏普动量 (长期)
    zipped_data = zip(
        ['intra','overnight','close'],
        [np.log(data_df['intra_rtn'] + 1),np.log(data_df['overnight_rtn'] + 1), np.log(data_df['close_rtn'] + 1)]
    )
    dt_index = data_df.index

    X_long_list = [] # 中期期特征列表
    long_term_start = 95
    long_term_end = 245
    long_term_step = 10
    long_term_ex_K = 20 # 排除最近K日
    for rtn_category,rtn_series in zipped_data:
        for lookback in trange(long_term_start,long_term_end + long_term_step,long_term_step,position = 0, leave = True):
            # 因子名称
            cumrtn_factor_name = f"{rtn_category}cumrtn_LB{lookback}_ex{long_term_ex_K}"
            sharpe_factor_name = f"{rtn_category}sharpe_LB{lookback}_ex{long_term_ex_K}"
            # 因子数据列表
            cumrtn_list = []
            sharpe_list = []
            dt_list = []
            for end_dt in dt_index:
                end_dt_idx = np.searchsorted(dt_index,end_dt) - mid_term_ex_K
                start_dt_idx = end_dt_idx - lookback
            
                if start_dt_idx < 0:
                    continue
                else:
                    rtn_series_slice = rtn_series.iloc[start_dt_idx:end_dt_idx] # 选中的收益率序列
                    cumrtn = rtn_series_slice.sum() # 累计收益率
                    sharpe = rtn_series_slice.mean() / rtn_series_slice.std() # 夏普动量
                    '''添加数据到list中'''
                    cumrtn_list.append(cumrtn)
                    sharpe_list.append(sharpe)
                    dt_list.append(end_dt)
            '''生成series'''
            cumrtn_df_slice = pd.DataFrame(index = dt_list, data = cumrtn_list, columns = [cumrtn_factor_name])
            sharpe_df_slice = pd.DataFrame(index = dt_list, data = sharpe_list, columns = [sharpe_factor_name])
            X_long_list.append(cumrtn_df_slice)
            X_long_list.append(sharpe_df_slice)  
    X_long = pd.concat(X_long_list,axis = 1)

    # y label
    y =  hs300_close_trend[['t_value']].dropna()
    X_short_shift1 = X_short.shift(1) # 滞后一期
    X_medium_shift1 = X_medium.shift(1) # 滞后一期
    X_long_shift1 = X_long.shift(1) # 滞后一期

    # 训练(以及验证)集 和 测试集 划分
    X_short_train,X_short_test,y_short_train,y_short_test = xgb_helperfunc.prepare_train_test_data(X_short_shift1,y)
    X_medium_train,X_medium_test,y_medium_train,y_medium_test = xgb_helperfunc.prepare_train_test_data(X_medium_shift1,y)
    X_long_train,X_long_test,y_long_train,y_long_test = xgb_helperfunc.prepare_train_test_data(X_long_shift1,y)
    # 训练集 和 验证集 划分
    X_short_train,X_short_val,y_short_train,y_short_val = xgb_helperfunc.prepare_train_val_data_bydate(X_short_train,y_short_train,val_date = 'random')
    X_medium_train,X_medium_val,y_medium_train,y_medium_val = xgb_helperfunc.prepare_train_val_data_bydate(X_medium_train,y_medium_train,val_date = 'random')
    X_long_train,X_long_val,y_long_train,y_long_val = xgb_helperfunc.prepare_train_val_data_bydate(X_long_train,y_long_train,val_date = 'random')


    # 准备DMatrix
    # 短期
    dtrain_short = xgb_helperfunc.prepare_dmatrix(X_short_train,y_short_train)
    dval_short = xgb_helperfunc.prepare_dmatrix(X_short_val,y_short_val)
    dtest_short = xgb_helperfunc.prepare_dmatrix(X_short_test,y_short_test)
    # 中期
    dtrain_medium = xgb_helperfunc.prepare_dmatrix(X_medium_train,y_medium_train) 
    dval_medium = xgb_helperfunc.prepare_dmatrix(X_medium_val,y_medium_val)
    dtest_medium = xgb_helperfunc.prepare_dmatrix(X_medium_test,y_medium_test)
    # 长期
    dtrain_long = xgb_helperfunc.prepare_dmatrix(X_long_train,y_long_train)
    dval_long = xgb_helperfunc.prepare_dmatrix(X_long_val,y_long_val)
    dtest_long = xgb_helperfunc.prepare_dmatrix(X_long_test,y_long_test)

    # 可视化 训练集 和 验证集 和 测试集
    # visual_data_df = xgb_helperfunc.visual_set(train = y_short_train, val = y_short_val, test = y_long_test)

    # Optuna 调参 + 训练模型
    TODAY = datetime.now().strftime('%Y-%m-%d')
    MODEL_SAVE_PTH = f'Output\\xgboost_models\\{TODAY}'
    os.makedirs(MODEL_SAVE_PTH,exist_ok = True)

    SHORT_TERM_MODEL_SAVE_PTH = f'{MODEL_SAVE_PTH}\\xgboost_short_term_model.json'
    short_best_params,short_best_num_round,short_best_score = xgb_helperfunc.optimize_hyperparameters(dtrain = dtrain_short, dval = dval_short, n_trials = 50, verbose_eval = 100)
    short_best_params_model,short_evals_result = xgb_helperfunc.train_with_best_params(short_best_params,short_best_num_round,dtrain = dtrain_short, dval = dval_short)
    short_best_params_model.save_model(SHORT_TERM_MODEL_SAVE_PTH)
    # 读取短期模型
    short_best_params_model = xgb.Booster() 
    short_best_params_model.load_model(SHORT_TERM_MODEL_SAVE_PTH)
    short_y_pred = short_best_params_model.predict(dtest_short) # 预测值
    short_pred_df = xgb_helperfunc.concat_pred_to_test(short_y_pred,y_short_test) # 合并预测值与真实值


    # 中期特征模型训练与预测
    # Optuna 调参 + 训练模型 (中期特征)
    MEDIUM_TERM_MODEL_SAVE_PTH = f'{MODEL_SAVE_PTH}\\xgboost_medium_term_model.json'
    medium_best_params,medium_best_num_round,medium_best_score = xgb_helperfunc.optimize_hyperparameters(dtrain = dtrain_medium, dval = dval_medium, n_trials = 50,verbose_eval = 100)
    medium_best_params_model,medium_evals_result = xgb_helperfunc.train_with_best_params(medium_best_params,medium_best_num_round,dtrain = dtrain_medium, dval = dval_medium)
    medium_best_params_model.save_model(MEDIUM_TERM_MODEL_SAVE_PTH)
    # 读取中期模型
    medium_best_params_model = xgb.Booster()
    medium_best_params_model.load_model(MEDIUM_TERM_MODEL_SAVE_PTH)
    medium_y_pred = medium_best_params_model.predict(dtest_medium) # 预测值
    medium_pred_df = xgb_helperfunc.concat_pred_to_test(medium_y_pred,y_medium_test) # 合并预测值与真实值


    # 长期特征模型训练与预测 
    # Optuna 调参 + 训练模型 (长期特征)
    LONG_TERM_MODEL_SAVE_PTH = f'{MODEL_SAVE_PTH}\\xgboost_long_term_model.json'
    long_best_params,long_best_num_round,long_best_score = xgb_helperfunc.optimize_hyperparameters(dtrain = dtrain_long, dval = dval_long, n_trials = 50,verbose_eval = 100)
    long_best_params_model,long_evals_result = xgb_helperfunc.train_with_best_params(long_best_params,long_best_num_round,dtrain = dtrain_long, dval = dval_long)
    long_best_params_model.save_model(LONG_TERM_MODEL_SAVE_PTH)
    # 读取长期模型
    long_best_params_model = xgb.Booster()
    long_best_params_model.load_model(LONG_TERM_MODEL_SAVE_PTH)
    long_y_pred = long_best_params_model.predict(dtest_long) # 预测值
    long_pred_df = xgb_helperfunc.concat_pred_to_test(long_y_pred,y_long_test) # 合并预测值与真实值
