import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import time
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional, Union
import os

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

def prepare_train_test_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    将特征和目标数据拆分为训练集和测试集
    
    参数:
        X: 特征数据框
        y: 目标变量
        test_size: 测试集比例
        random_state: 随机种子
    
    返回:
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def prepare_dmatrix(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: Optional[pd.DataFrame] = None, 
    y_val: Optional[pd.Series] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED
) -> Tuple[xgb.DMatrix, Optional[xgb.DMatrix]]:
    """
    准备XGBoost的DMatrix对象
    
    如果不提供验证集，则从训练集中分割出一部分作为验证集
    
    参数:
        X_train: 训练特征数据框
        y_train: 训练目标变量
        X_val: 验证特征数据框
        y_val: 验证目标变量
        test_size: 如果不提供验证集，从训练集中分割的比例
        random_state: 随机种子
    
    返回:
        (dtrain, dval)，如果不需要验证集，dval将为None
    """
    if X_val is None or y_val is None:
        # 从训练集分割出验证集
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state
        )
        dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
        dval = xgb.DMatrix(X_val_split, label=y_val_split)
    else:
        # 使用提供的验证集
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
    
    return dtrain, dval

def get_weekday(grouped,val_date,exclude_monday = True):
    '''选择weekday'''
    if val_date == 'random': # 随机抽
        random_weekday = np.random.randint(1,4 + 1) if exclude_monday else np.random.randint(0,4 + 1)
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

def prepare_train_val_data_bydate(X,y,val_date = 'random',exclude_monday = True):
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
    val_dates_df = date_index.groupby(['year','month','week_num']).apply(lambda grouped: get_weekday(grouped,val_date = val_date, exclude_monday = exclude_monday),include_groups=False)
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

def train_xgb_model(
    dtrain: xgb.DMatrix,
    dval: Optional[xgb.DMatrix] = None,
    params: Dict = None,
    num_rounds: int = 100,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 10
) -> Tuple[xgb.Booster, Dict]:
    """
    训练XGBoost模型
    
    参数:
        dtrain: 训练数据集的DMatrix对象
        dval: 验证数据集的DMatrix对象
        params: XGBoost参数字典
        num_rounds: 训练迭代次数
        early_stopping_rounds: 早停轮数
        verbose_eval: 打印评估信息的频率
    
    返回:
        (训练好的模型, 训练结果字典)
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': RANDOM_SEED
        }
    
    watchlist = [(dtrain, 'train')]
    if dval is not None:
        watchlist.append((dval, 'validation'))
    
    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )
    
    # 收集训练结果
    results = {
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else num_rounds,
        'best_score': model.best_score if hasattr(model, 'best_score') else None,
        'feature_importance': model.get_score(importance_type='weight'),
        'params': params,
        'num_rounds': num_rounds
    }
    
    return model, results

def evaluate_model(
    model: xgb.Booster,
    dtest: xgb.DMatrix,
    y_test: pd.Series
) -> Dict:
    """
    评估XGBoost模型性能
    
    参数:
        model: 训练好的XGBoost模型
        dtest: 测试数据集的DMatrix对象
        y_test: 测试集的真实目标值
    
    返回:
        包含评估指标的字典
    """
    # 预测测试集
    y_pred = model.predict(dtest)
    
    # 计算均方根误差
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    
    # 计算平均绝对误差
    mae = np.mean(np.abs(y_pred - y_test))
    
    # 计算R方
    y_mean = np.mean(y_test)
    ss_total = np.sum((y_test - y_mean) ** 2)
    ss_residual = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def plot_feature_importance(
    model: xgb.Booster,
    importance_type: str = 'weight',
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    绘制特征重要性图表
    
    参数:
        model: 训练好的XGBoost模型
        importance_type: 特征重要性类型，可选 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        top_n: 显示的顶部特征数量
        figsize: 图形大小
        save_path: 保存图形的路径，如果为None则不保存
    """
    # 获取特征重要性
    importance = model.get_score(importance_type=importance_type)
    
    # 转换为DataFrame并排序
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importance ({importance_type})')
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_prediction_vs_actual(
    y_test: pd.Series,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    绘制预测值与实际值的对比散点图
    
    参数:
        y_test: 测试集的真实目标值
        y_pred: 模型预测值
        figsize: 图形大小
        save_path: 保存图形的路径，如果为None则不保存
    """
    plt.figure(figsize=figsize)
    
    # 绘制散点图
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # 添加对角线 (完美预测线)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 设置图形属性
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加R方文本
    r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(
    model: xgb.Booster,
    model_path: str,
    model_dir: str = '../../Models'
):
    """
    保存XGBoost模型
    
    参数:
        model: 训练好的XGBoost模型
        model_path: 模型文件名，不包含路径
        model_dir: 模型目录
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    file_path = os.path.join(model_dir, model_path)
    model.save_model(file_path)
    print(f"模型已保存到: {file_path}")

def load_model(model_path: str, model_dir: str = '../../Models') -> xgb.Booster:
    """
    加载XGBoost模型
    
    参数:
        model_path: 模型文件名，不包含路径
        model_dir: 模型目录
    
    返回:
        加载的XGBoost模型
    """
    file_path = os.path.join(model_dir, model_path)
    model = xgb.Booster()
    model.load_model(file_path)
    print(f"模型已从 {file_path} 加载")
    return model
