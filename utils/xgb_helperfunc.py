import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import time

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

def prepare_train_test_data(X,y,test_size = 0.2):
    '''准备XGBoost的数据'''
    X = X.dropna() # X_lag: X的滞后阶数
    X_y = pd.merge(X,y, left_index = True, right_index = True, how = 'left').dropna() # 合并
    split_idx = int(X_y.shape[0] * (1 - test_size)) # 分割index
    X_train,X_test = X_y.iloc[:split_idx,:-1],X_y.iloc[split_idx:,:-1]
    y_train,y_test = X_y.iloc[:split_idx,-1],X_y.iloc[split_idx:,-1]
    
    return X_train,X_test,y_train,y_test

def prepare_dmatrix(X,y):
    '''准备XGBoost的Dmatrix数据格式'''
    if isinstance(X,pd.DataFrame): # 如果X是DataFrame
        feature_names = X.columns.tolist() # 特征名
    else: # 如果X是numpy数组
        feature_names = [f"feature_{i}" for i in range(1,X.shape[1] + 1)] # 特征名
    dmatrix = xgb.DMatrix(X, label = y, feature_names = feature_names) # 数据格式
    return dmatrix

def get_weekday(grouped,val_date):
    '''选择weekday'''
    if val_date == 'random': # 随机抽
        random_weekday = np.random.randint(0,4 + 1)
        if random_weekday not in grouped.weekday.values:
            return 
        else:
            return grouped[grouped['weekday'] == random_weekday]
    elif isinstance(val_date,int):
        if val_date not in grouped.weekday.values:
            return 
        else:
            return grouped[grouped['weekday'] == val_date]
    else:
        raise ValueError(f"{val_date} not valid")

def prepare_train_val_data_bydate(X,y,val_date = 'random'):
    '''划分训练集和验证集, 按照日期来划分'''
    X = X.dropna() # X_lag: X的滞后阶数
    X_col = X.columns
    y_label = y.columns[0] if type(y) == pd.DataFrame else y.name
    X_y = pd.merge(X,y, left_index = True, right_index = True, how = 'left').dropna()
    X_y.index.name = 'date'
    date_index = X_y.index.to_frame()
    date_index['year'] = date_index['date'].dt.year
    date_index['month'] = date_index['date'].dt.month
    date_index['day'] = date_index['date'].dt.day
    date_index['weekday'] = date_index['date'].dt.weekday # 提取星期几
    date_index['week_num'] = date_index['date'].dt.isocalendar().week # 提取 ISO 周数
    # 验证集天数
    val_dates_df = date_index.groupby(['year','month','week_num']).apply(lambda grouped: get_weekday(grouped,val_date = val_date),include_groups=False)
    # 划分 训练集和验证集
    X_y = X_y.reset_index()
    val_X_y = X_y[X_y['date'].isin(val_dates_df['date'].values)].set_index('date').sort_index() # 验证集
    train_X_y = X_y[~X_y['date'].isin(val_dates_df['date'].values)].set_index('date').sort_index() # 训练集
    X_train,X_val = train_X_y[X_col],val_X_y[X_col] # 训练集和验证集
    y_train,y_val = train_X_y[y_label],val_X_y[y_label] # 训练集和验证集
    return X_train,X_val,y_train,y_val

def visual_set(**kwargs):
    visual_data_df_list = [] # 可视化数据列表
    value = 0 # 值
    for set_name,set_data in kwargs.items():
        set_data_index = set_data.index.tolist() # 索引
        set_data_df = pd.DataFrame(index = set_data_index, columns = ['type','value']) # 数据框
        set_data_df['type'] = set_name # 类型
        set_data_df['value'] = value # 值
        visual_data_df_list.append(set_data_df) # 添加到列表
        print(f"{set_name}: sample num: {len(set_data_index)}") # 打印样本数量
        value += 1
    visual_data_df = pd.concat(visual_data_df_list,axis = 0) # 合并
    visual_data_df.index = pd.to_datetime(visual_data_df.index) # 转换为日期
    visual_data_df.index.name = 'date' # 索引名
    visual_data_df = visual_data_df.sort_index().reset_index() # 排序并重置索引
    plt.figure(figsize = (18,6)) # 设置图形大小
    sns.scatterplot(data = visual_data_df, x = 'date', y = 'value', hue = 'type') # 绘制散点图
    plt.grid(True) # 显示网格
    plt.show() # 显示图形
    
    return visual_data_df

def objective_dmatrix(trial, dtrain, dvalid, verbose_eval = 50,early_stopping_rounds = 50):
    """Optuna目标函数，使用DMatrix进行超参数优化"""
    param = {
        'objective': 'reg:squarederror', # 目标函数
        'eval_metric': 'rmse', # 评估指标
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']), # 提升器
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # 正则化
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True), # 正则化
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # 采样率
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0), # 列采样率
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0), # 列采样率 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # 最小子权重
        'seed': RANDOM_SEED # 随机种子
    }
    
    # gbtree和dart共用的参数
    param['max_depth'] = trial.suggest_int('max_depth', 3, 15)
    param['eta'] = trial.suggest_float('eta', 0.005, 0.5, log=True) # 学习率
    param['gamma'] = trial.suggest_float('gamma', 1e-8, 2.0, log=True) # 惩罚项力度
    param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']) # 生长策略
    
    if param['grow_policy'] == 'lossguide':
        param['max_leaves'] = trial.suggest_int('max_leaves', 0, 100) # 最大叶子节点数
    
    param['tree_method'] = trial.suggest_categorical('tree_method', ['exact', 'hist', 'approx']) # 树方法
    
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted']) # 采样类型
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest']) # 归一化类型
        param['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.9) # 丢弃率
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.9) # 跳过率
    
    # 使用early stopping避免过拟合
    num_round = trial.suggest_int('num_round', 10, 500) # 训练迭代轮数
    
    try:
        # 训练模型
        evals_result = {} # 记录每一轮,每个数据集，每个指标的评估结果
        bst = xgb.train(
            param, 
            dtrain, 
            num_round, 
            evals=[(dtrain, 'train'), (dvalid, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose_eval # 每 verbose_eval 轮输出一次
        )
        
        # 返回验证集上的最佳RMSE
        return min(evals_result['validation']['rmse'])
    except ValueError as e:
        if "'-nan(ind)'" in str(e) or 'nan' in str(e).lower():
            # 如果遇到nan错误，返回一个非常大的值，这样Optuna会避开这组参数
            print(f"遇到NaN错误，使用的booster: {param['booster']}, 跳过此次尝试...")
            return float('inf')  # 返回无穷大使得这次参数组合被视为非常差
        else:
            # 其他错误正常抛出
            raise

def optimize_hyperparameters(dtrain, dval, n_trials=50, early_stopping_rounds=50, verbose_eval=50, study_name=None, study_storage=None):    
    '''用optuna调参'''
    study = optuna.create_study(direction='minimize', study_name=study_name, storage=study_storage)
    
    try:
        # 优化超参数
        study.optimize(lambda trial: objective_dmatrix(trial, dtrain, dval, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval), n_trials=n_trials)
        best_score = study.best_value # 最佳得分
        best_params = study.best_params # 最佳参数
        if 'num_round' in best_params:
            best_num_round = study.best_params['num_round'] # 最佳迭代轮数
            best_params.pop('num_round') # 删除num_round
        else:
            best_num_round = None # 最佳迭代轮数 
        return best_params, best_num_round, best_score
    except Exception as e:
        print(f"优化过程中遇到错误: {str(e)}")
        raise

def train_with_best_params(best_params, best_num_round, dtrain, dval, early_stopping_round=50, verbose_eval=50, measure_time=False):
    '''用最佳参数训练模型'''
    evals_result = {}
    
    # 如果需要计时，记录开始时间
    if measure_time:
        start_time = time.time()
    
    # 训练模型
    best_params_model = xgb.train(
        params = best_params,
        dtrain = dtrain,
        num_boost_round = best_num_round,
        evals = [(dtrain,'train'),(dval,'validation')],
        early_stopping_rounds = early_stopping_round,
        evals_result = evals_result,
        verbose_eval = verbose_eval
    )
    
    # 如果需要计时，计算并打印训练时间
    if measure_time:
        end_time = time.time()
        training_time = end_time - start_time
        print(f"模型训练完成，耗时: {training_time:.4f} 秒")
        return best_params_model, evals_result, training_time
    
    return best_params_model, evals_result

def measure_training_time(best_params, best_num_round, dtrain, dval, early_stopping_round=50, verbose_eval=50, n_repeats=5):
    '''测量XGBoost模型训练的时间'''
    times = []
    for i in range(n_repeats):
        start_time = time.time()
        
        # 训练模型，但不保存结果
        _ = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=best_num_round,
            evals=[(dtrain,'train'),(dval,'validation')],
            early_stopping_rounds=early_stopping_round,
            evals_result={},
            verbose_eval=verbose_eval
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        times.append(training_time)
        print(f"训练轮次 {i+1}/{n_repeats}: {training_time:.4f} 秒")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n训练时间统计 (重复 {n_repeats} 次):")
    print(f"平均时间: {avg_time:.4f} 秒")
    print(f"最短时间: {min_time:.4f} 秒")
    print(f"最长时间: {max_time:.4f} 秒")
    
    return {
        'times': times,
        'average': avg_time,
        'min': min_time,
        'max': max_time
    }

def concat_pred_to_test(y_pred,y_test):
    '''将预测和训练合并为DataFrame'''
    y_pred_test = pd.DataFrame(index = y_test.index) # 预测
    y_pred_test['pred'] = y_pred # 预测
    y_pred_test['test'] = y_test.values # 真实
    y_pred_test['pred_sign'] = np.sign(y_pred).astype(int) # 预测符号
    y_pred_test['test_sign'] = np.sign(y_test).astype(int) # 真实符号
    y_pred_test = y_pred_test.sort_index() # 按照index排序
    return y_pred_test
